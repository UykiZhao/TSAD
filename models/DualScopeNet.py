import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.PKI_CAA import PeriodicBranch
from layers.Enhanced_MSTC import EnhancedAperiodicBranch
from layers.Adaptive_Fusion import EnhancedAdaptiveFusion, AnomalyAwareFusion


class DualBranchBlock(nn.Module):
    """
    双分支块：顺序处理的核心模块
    - 周期分支：FFT → 2D转换 → PKI → CAA → 2D还原
    - 非周期分支：MSTC → 趋势提取 → 局部模式 → 特征融合
    - 自适应融合：交叉注意力 + 动态门控 + 异常感知
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        
        # 周期性分支（顺序处理）
        self.periodic_branch = PeriodicBranch(configs)
        
        # 非周期性分支（顺序处理）
        self.aperiodic_branch = EnhancedAperiodicBranch(configs)
        
        # 增强的自适应融合
        self.adaptive_fusion = EnhancedAdaptiveFusion(
            d_model=configs.d_model,
            dropout=configs.dropout
        )
        
        # 异常感知融合（可选）
        self.use_anomaly_fusion = True  # 默认启用
        if self.use_anomaly_fusion:
            self.anomaly_fusion = AnomalyAwareFusion(d_model=configs.d_model)
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.LayerNorm(configs.d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(configs.dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C) 输入特征
        Returns:
            output: (B, T, C) 输出特征
            anomaly_scores: (B, T, 1) 异常分数（如果启用）
        """
        # 保存输入用于残差连接
        residual = x
        
        # 1. 周期性分支处理（顺序：FFT→2D→PKI→CAA→1D）
        periodic_out = self.periodic_branch(x)
        
        # 2. 非周期性分支处理（顺序：MSTC→趋势→局部→融合）
        aperiodic_out = self.aperiodic_branch(x)
        
        # 3. 自适应融合
        fused_features = self.adaptive_fusion(periodic_out, aperiodic_out)
        
        # 4. 异常感知融合（可选）
        anomaly_scores = None
        if self.use_anomaly_fusion:
            anomaly_aware_features, anomaly_scores = self.anomaly_fusion(
                periodic_out, aperiodic_out
            )
            # 结合两种融合结果
            fused_features = (fused_features + anomaly_aware_features) * 0.5
        
        # 5. 输出投影
        output = self.output_projection(fused_features)
        
        # 6. 残差连接
        output = output + residual
        
        return output, anomaly_scores


class Model(nn.Module):
    """
    DualScopeNet: 融合周期性与非周期性特征的双分支时序异常检测模型
    
    架构特点：
    1. 多层DualBranchBlock堆叠，每层包含完整的双分支处理
    2. 每个分支内部采用顺序处理策略
    3. 层间通过残差连接和LayerNorm连接
    4. 支持深度配置（2-6层）
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_layers = configs.e_layers
        
        # 数据嵌入层
        self.enc_embedding = DataEmbedding(
            configs.enc_in, 
            configs.d_model, 
            configs.embed, 
            configs.freq,
            configs.dropout
        )
        
        # 多层DualBranchBlock
        self.dual_branch_blocks = nn.ModuleList([
            DualBranchBlock(configs) for _ in range(self.num_layers)
        ])
        
        # 层标准化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(configs.d_model) for _ in range(self.num_layers)
        ])
        
        # 异常检测输出层
        if self.task_name == 'anomaly_detection':
            # 多尺度特征聚合（可选）
            self.multi_scale_fusion = nn.Sequential(
                nn.Conv1d(configs.d_model * self.num_layers, configs.d_model, 1),
                nn.BatchNorm1d(configs.d_model),
                nn.ReLU(inplace=True)
            )
            
            # 最终投影层
            self.projection = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model // 2, configs.c_out)
            )
            
            # 异常分数聚合（如果使用异常感知融合）
            if True:  # 默认启用
                self.anomaly_score_fusion = nn.Sequential(
                    nn.Conv1d(self.num_layers, 1, 1),
                    nn.Sigmoid()
                )
    
    def anomaly_detection(self, x_enc):
        """异常检测前向传播"""
        # 数据标准化
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc /= stdev
        
        # 嵌入
        enc_out = self.enc_embedding(x_enc, None)  # (B, T, C)
        
        # 多层处理
        layer_outputs = []
        anomaly_scores_list = []
        
        for i, (block, norm) in enumerate(zip(self.dual_branch_blocks, self.layer_norms)):
            # DualBranchBlock处理
            block_out, anomaly_scores = block(enc_out)
            
            # 残差连接和层标准化
            enc_out = norm(block_out + enc_out)
            
            # 保存每层输出（用于多尺度融合）
            layer_outputs.append(enc_out)
            
            # 保存异常分数
            if anomaly_scores is not None:
                anomaly_scores_list.append(anomaly_scores)
        
        # 多尺度特征融合（可选）
        if hasattr(self, 'multi_scale_fusion') and len(layer_outputs) > 1:
            # 拼接所有层的输出
            multi_scale_features = torch.cat([
                out.permute(0, 2, 1) for out in layer_outputs
            ], dim=1)  # (B, C*num_layers, T)
            
            # 融合
            fused_features = self.multi_scale_fusion(multi_scale_features)
            fused_features = fused_features.permute(0, 2, 1)  # (B, T, C)
            
            # 与最后一层输出结合
            enc_out = (enc_out + fused_features) * 0.5
        
        # 投影到输出维度
        dec_out = self.projection(enc_out)
        
        # 反标准化
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # 聚合异常分数（如果有）
        aggregated_anomaly_scores = None
        if anomaly_scores_list:
            # (B, T, num_layers)
            stacked_scores = torch.cat(anomaly_scores_list, dim=-1)
            # (B, num_layers, T)
            stacked_scores = stacked_scores.permute(0, 2, 1)
            # (B, 1, T)
            aggregated_anomaly_scores = self.anomaly_score_fusion(stacked_scores)
            # (B, T, 1)
            aggregated_anomaly_scores = aggregated_anomaly_scores.permute(0, 2, 1)
        
        return dec_out, aggregated_anomaly_scores
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """模型前向传播"""
        if self.task_name == 'anomaly_detection':
            dec_out, anomaly_scores = self.anomaly_detection(x_enc)
            # 训练和测试时都只返回重构输出，保持与其他模型一致
            return dec_out  # [B, L, D]
        return None