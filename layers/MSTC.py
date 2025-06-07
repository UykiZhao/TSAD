import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class MultiScale_TemporalConv(nn.Module):
    """
    Multi-Scale Temporal Convolution (MSTC) module for capturing temporal dependencies
    at different scales for non-periodic patterns in time series.
    """
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):
        super(MultiScale_TemporalConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels  # 保持输入输出通道数一致
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilations = dilations
        self.residual = residual
        self.residual_kernel_size = residual_kernel_size

        # 多尺度时间卷积分支
        self.branches = nn.ModuleList()
        for dilation in dilations:
            effective_kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
            padding = (effective_kernel_size - 1) // 2  # 确保padding是整数
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 
                         kernel_size=(kernel_size, 1), 
                         padding=(padding, 0), 
                         dilation=dilation,
                         groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ))

        # 最大池化分支用于捕获不同的特征
        self.maxpool_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 
                     kernel_size=1, stride=1, padding=0, groups=in_channels),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 1x1卷积分支
        self.conv1x1_branch = nn.Conv2d(in_channels, in_channels, 
                                       kernel_size=1, stride=1, padding=0, 
                                       groups=in_channels)

        # 残差连接
        if self.residual:
            if self.residual_kernel_size == 1:
                self.residual_connection = nn.Identity()
            else:
                self.residual_connection = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 
                             kernel_size=(residual_kernel_size, 1), 
                             stride=stride, padding=0, groups=in_channels),
                    nn.BatchNorm2d(in_channels)
                )

        # 初始化权重
        self.apply(weights_init)

    def forward(self, x):
        """
        Forward pass of MSTC module
        
        Args:
            x: Input tensor of shape (B, C, T, 1)
            
        Returns:
            Output tensor of shape (B, C, T, 1)
        """
        residual = self.residual_connection(x)  # 保存残差信息: (B,C,T,1)

        branch_outputs = []
        
        # 多尺度膨胀卷积分支
        for branch in self.branches:
            branch_outputs.append(branch(x))  # (B,C,T,1) -> (B,C,T,1)
            
        # 最大池化分支
        branch_outputs.append(self.maxpool_branch(x))  # (B,C,T,1) -> (B,C,T,1)
        
        # 1x1卷积分支
        branch_outputs.append(self.conv1x1_branch(x))  # (B,C,T,1) -> (B,C,T,1)

        # 合并所有分支的输出，取平均值
        out = sum(branch_outputs) / len(branch_outputs)

        # 添加残差连接
        if self.residual:
            out += residual

        return out


class AperiodicBranch(nn.Module):
    """
    Non-periodic branch for capturing aperiodic temporal patterns
    使用MSTC + 趋势提取 + 局部模式提取
    """
    def __init__(self, configs):
        super(AperiodicBranch, self).__init__()
        
        self.configs = configs
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # MSTC模块 - 核心多尺度时间卷积
        self.mstc = MultiScale_TemporalConv(
            in_channels=self.d_model,
            kernel_size=3,
            dilations=[1, 2, 4, 8],  # 多尺度膨胀
            residual=True
        )
        
        # 趋势提取器 - 使用1D卷积处理时间维度
        self.trend_extractor = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=5, padding=2, groups=self.d_model),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=1)
        )
        
        # 局部模式提取器 - 捕获短期波动
        self.local_extractor = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Conv1d(self.d_model, self.d_model, kernel_size=1)
        )
        
        # 非周期特征融合器
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model * 2),
            nn.ReLU(),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
    def forward(self, x):
        """
        Forward pass of aperiodic branch
        
        Args:
            x: Input tensor of shape (B, T, C)  - 来自embedding的特征
            
        Returns:
            Output tensor of shape (B, T, C)
        """
        B, T, C = x.shape
        
        # 1. MSTC处理 - 转换为(B, C, T, 1)格式
        x_mstc = x.permute(0, 2, 1).unsqueeze(-1)  # (B, T, C) -> (B, C, T, 1)
        mstc_out = self.mstc(x_mstc)  # (B, C, T, 1)
        mstc_feat = mstc_out.squeeze(-1).permute(0, 2, 1)  # (B, C, T, 1) -> (B, T, C)
        
        # 2. 趋势提取 - 转换为(B, C, T)格式
        x_trend = x.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
        trend_out = self.trend_extractor(x_trend)  # (B, C, T)
        trend_feat = trend_out.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
        
        # 3. 局部模式提取
        local_out = self.local_extractor(x_trend)  # (B, C, T)
        local_feat = local_out.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
        
        # 4. 非周期特征融合
        # 拼接三种特征
        combined_feat = torch.cat([mstc_feat, trend_feat, local_feat], dim=-1)  # (B, T, 3*C)
        
        # 融合为最终特征
        final_feat = self.feature_fusion(combined_feat)  # (B, T, C)
        
        return final_feat