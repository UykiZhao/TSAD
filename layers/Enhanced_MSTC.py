import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class EnhancedMSTC(nn.Module):
    """增强的多尺度时间卷积模块，专门针对时序异常检测优化"""
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilations: List[int] = [1, 2, 4, 8, 16],
        residual: bool = True,
        anomaly_sensitive: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilations = dilations
        self.residual = residual
        self.anomaly_sensitive = anomaly_sensitive
        
        # 多尺度时间卷积分支
        self.branches = nn.ModuleList()
        for dilation in dilations:
            # 计算有效的padding以保持输出大小
            effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
            padding = (effective_kernel_size - 1) // 2
            
            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels, in_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=in_channels,
                    bias=False
                ),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels, in_channels, 1, bias=False),
                nn.BatchNorm1d(in_channels)
            )
            self.branches.append(branch)
        
        # 异常敏感分支 - 使用较小的卷积核捕获突变
        if self.anomaly_sensitive:
            self.anomaly_branch = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(in_channels)
            )
        
        # 自适应融合权重
        self.fusion_weights = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels * (len(dilations) + (1 if anomaly_sensitive else 0)), 
                     len(dilations) + (1 if anomaly_sensitive else 0), 1),
            nn.Softmax(dim=1)
        )
        
        # 特征融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, T) 输入特征
        Returns:
            (B, C, T) 输出特征
        """
        # 保存残差
        residual = x
        
        # 多尺度分支输出
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # 异常敏感分支
        if self.anomaly_sensitive:
            branch_outputs.append(self.anomaly_branch(x))
        
        # 堆叠所有分支输出用于计算融合权重
        stacked_outputs = torch.stack(branch_outputs, dim=1)  # (B, num_branches, C, T)
        
        # 计算自适应融合权重
        pooled_features = torch.cat([
            F.adaptive_avg_pool1d(out, 1) for out in branch_outputs
        ], dim=1)  # (B, num_branches*C, 1)
        
        fusion_weights = self.fusion_weights(pooled_features)  # (B, num_branches, 1)
        fusion_weights = fusion_weights.unsqueeze(2)  # (B, num_branches, 1, 1)
        
        # 加权融合
        fused = torch.sum(stacked_outputs * fusion_weights, dim=1)  # (B, C, T)
        
        # 最终融合
        out = self.fusion_conv(fused)
        
        # 残差连接
        if self.residual:
            out = out + residual
            
        return out


class TrendExtractor(nn.Module):
    """趋势提取器 - 捕获长期趋势变化"""
    def __init__(self, d_model: int, window_sizes: List[int] = [5, 11, 21]):
        super().__init__()
        self.d_model = d_model
        self.window_sizes = window_sizes
        
        # 多个不同窗口大小的趋势提取器
        self.trend_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=ws, padding=ws//2, groups=d_model, bias=False),
                nn.BatchNorm1d(d_model),
                nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True)
            )
            for ws in window_sizes
        ])
        
        # 趋势融合
        self.trend_fusion = nn.Sequential(
            nn.Conv1d(d_model * len(window_sizes), d_model, 1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """提取多尺度趋势特征"""
        trends = []
        for extractor in self.trend_extractors:
            trends.append(extractor(x))
        
        # 拼接并融合
        trend_concat = torch.cat(trends, dim=1)
        trend_fused = self.trend_fusion(trend_concat)
        
        return trend_fused


class LocalPatternExtractor(nn.Module):
    """局部模式提取器 - 捕获短期异常模式"""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # 短期模式提取
        self.short_term = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(d_model * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model * 2, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model)
        )
        
        # 差分特征提取（捕获突变）
        self.diff_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=2, stride=1, padding=0, groups=d_model, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model)
        )
        
        # 局部统计特征
        self.local_stats = nn.Sequential(
            nn.Conv1d(d_model * 2, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """提取局部异常模式"""
        # 短期模式
        short_pattern = self.short_term(x)
        
        # 差分特征（需要padding以保持长度）
        x_padded = F.pad(x, (0, 1), mode='replicate')
        diff_pattern = self.diff_conv(x_padded)
        
        # 局部均值和方差
        kernel_size = 5
        padding = kernel_size // 2
        
        # 局部均值
        local_mean = F.avg_pool1d(x, kernel_size=kernel_size, stride=1, padding=padding)
        
        # 局部方差的近似（使用平方的均值 - 均值的平方）
        x_squared = x * x
        local_mean_squared = F.avg_pool1d(x_squared, kernel_size=kernel_size, stride=1, padding=padding)
        local_var = local_mean_squared - local_mean * local_mean
        
        # 融合统计特征
        stats_concat = torch.cat([local_mean, local_var], dim=1)
        stats_features = self.local_stats(stats_concat)
        
        # 组合所有局部特征
        local_features = short_pattern + diff_pattern + stats_features
        
        return local_features


class EnhancedAperiodicBranch(nn.Module):
    """增强的非周期性分支 - 顺序处理MSTC、趋势提取、局部模式提取"""
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.d_model = configs.d_model
        
        # 阶段1：增强的MSTC模块
        self.enhanced_mstc = EnhancedMSTC(
            in_channels=self.d_model,
            kernel_size=3,
            dilations=[1, 2, 4, 8, 16],
            residual=True,
            anomaly_sensitive=True
        )
        
        # 阶段2：趋势提取器
        self.trend_extractor = TrendExtractor(
            d_model=self.d_model,
            window_sizes=[5, 11, 21]
        )
        
        # 阶段3：局部模式提取器
        self.local_pattern_extractor = LocalPatternExtractor(d_model=self.d_model)
        
        # 阶段4：特征融合器
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model * 2),
            nn.LayerNorm(self.d_model * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # 异常增强门控机制
        self.anomaly_gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model // 2, self.d_model),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        顺序处理流程：
        1. MSTC多尺度卷积
        2. 趋势提取
        3. 局部模式提取
        4. 特征融合
        
        Args:
            x: (B, T, C) 输入特征
        Returns:
            (B, T, C) 输出特征
        """
        B, T, C = x.shape
        
        # 转换为卷积格式 (B, C, T)
        x_conv = x.permute(0, 2, 1)
        
        # 阶段1：MSTC多尺度时间特征提取
        mstc_features = self.enhanced_mstc(x_conv)  # (B, C, T)
        
        # 阶段2：趋势提取
        trend_features = self.trend_extractor(mstc_features)  # (B, C, T)
        
        # 阶段3：局部模式提取
        local_features = self.local_pattern_extractor(mstc_features)  # (B, C, T)
        
        # 转换回(B, T, C)格式
        mstc_features = mstc_features.permute(0, 2, 1)
        trend_features = trend_features.permute(0, 2, 1)
        local_features = local_features.permute(0, 2, 1)
        
        # 阶段4：特征融合
        # 拼接所有特征
        combined_features = torch.cat([mstc_features, trend_features, local_features], dim=-1)  # (B, T, 3*C)
        
        # 融合为最终特征
        fused_features = self.feature_fusion(combined_features)  # (B, T, C)
        
        # 异常增强门控
        anomaly_gate = self.anomaly_gate(fused_features)  # (B, T, C)
        enhanced_features = fused_features * anomaly_gate + fused_features
        
        return enhanced_features