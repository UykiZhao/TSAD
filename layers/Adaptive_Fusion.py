import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossAttention(nn.Module):
    """交叉注意力机制，用于两个分支之间的信息交互"""
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (B, T, C) 查询特征（来自一个分支）
            key: (B, T, C) 键特征（来自另一个分支）
            value: (B, T, C) 值特征（来自另一个分支）
            mask: Optional attention mask
        Returns:
            (B, T, C) 输出特征
        """
        B, T, C = query.shape
        
        # 线性投影并重塑为多头格式
        Q = self.q_proj(query).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 应用mask（如果提供）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)
        
        # 重塑并投影输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.out_proj(attn_output)
        
        return output


class DynamicGating(nn.Module):
    """动态门控机制，根据输入特征自适应调整融合权重"""
    def __init__(self, d_model: int, num_experts: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        
        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(d_model * num_experts, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # 特征变换网络（用于对齐不同分支的特征空间）
        self.transform_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(inplace=True)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, features_list):
        """
        Args:
            features_list: List of tensors [(B, T, C), ...] 来自不同分支的特征
        Returns:
            (B, T, C) 融合后的特征
        """
        assert len(features_list) == self.num_experts
        
        # 变换各分支特征
        transformed_features = []
        for i, (features, transform) in enumerate(zip(features_list, self.transform_networks)):
            transformed_features.append(transform(features))
        
        # 拼接特征用于计算门控权重
        concat_features = torch.cat(features_list, dim=-1)  # (B, T, num_experts*C)
        
        # 计算门控权重
        gate_weights = self.gate_network(concat_features)  # (B, T, num_experts)
        
        # 加权融合
        output = torch.zeros_like(transformed_features[0])
        for i, features in enumerate(transformed_features):
            weight = gate_weights[..., i:i+1]  # (B, T, 1)
            output = output + features * weight
            
        return output


class EnhancedAdaptiveFusion(nn.Module):
    """增强的自适应融合模块，结合多种融合策略"""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1. 交叉注意力机制 - 双向信息交互
        self.cross_attn_p2a = CrossAttention(d_model, n_heads=8, dropout=dropout)
        self.cross_attn_a2p = CrossAttention(d_model, n_heads=8, dropout=dropout)
        
        # 2. 动态门控融合
        self.dynamic_gating = DynamicGating(d_model, num_experts=2)
        
        # 3. 特征级别的重要性评估
        self.importance_evaluator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1)
        )
        
        # 4. 互补性增强
        self.complementary_enhance = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 5. 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 残差权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, periodic_feat, aperiodic_feat):
        """
        Args:
            periodic_feat: (B, T, C) 周期性分支特征
            aperiodic_feat: (B, T, C) 非周期性分支特征
        Returns:
            (B, T, C) 融合后的特征
        """
        B, T, C = periodic_feat.shape
        
        # 1. 交叉注意力 - 双向信息交互
        # 周期性特征增强（使用非周期性信息）
        periodic_enhanced = self.cross_attn_p2a(
            query=periodic_feat,
            key=aperiodic_feat,
            value=aperiodic_feat
        )
        
        # 非周期性特征增强（使用周期性信息）
        aperiodic_enhanced = self.cross_attn_a2p(
            query=aperiodic_feat,
            key=periodic_feat,
            value=periodic_feat
        )
        
        # 2. 动态门控融合增强后的特征
        gated_fusion = self.dynamic_gating([periodic_enhanced, aperiodic_enhanced])
        
        # 3. 计算特征重要性权重
        concat_original = torch.cat([periodic_feat, aperiodic_feat], dim=-1)
        importance_weights = self.importance_evaluator(concat_original)  # (B, T, 2)
        
        # 应用重要性权重到原始特征
        weighted_periodic = periodic_feat * importance_weights[..., 0:1]
        weighted_aperiodic = aperiodic_feat * importance_weights[..., 1:2]
        
        # 4. 互补性增强
        concat_weighted = torch.cat([weighted_periodic, weighted_aperiodic], dim=-1)
        complementary_feat = self.complementary_enhance(concat_weighted)
        
        # 5. 最终融合
        # 结合三种融合结果：门控融合、互补增强、原始加权和
        all_features = torch.cat([
            gated_fusion,
            complementary_feat,
            weighted_periodic + weighted_aperiodic
        ], dim=-1)
        
        fused_output = self.final_fusion(all_features)
        
        # 添加残差连接（使用可学习权重）
        residual = (periodic_feat + aperiodic_feat) * 0.5 * self.residual_weight
        final_output = fused_output + residual
        
        return final_output


class AnomalyAwareFusion(nn.Module):
    """异常感知的融合模块，专门针对异常检测任务优化"""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # 异常分数预测器
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # 异常感知的特征调制
        self.anomaly_modulation = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model * 2),
            nn.GLU(dim=-1)
        )
        
    def forward(self, periodic_feat, aperiodic_feat):
        """
        根据异常分数自适应调整融合策略
        """
        # 计算异常分数
        concat_feat = torch.cat([periodic_feat, aperiodic_feat], dim=-1)
        anomaly_scores = self.anomaly_scorer(concat_feat)  # (B, T, 1)
        
        # 异常感知的特征调制
        concat_with_score = torch.cat([concat_feat, anomaly_scores], dim=-1)
        modulated_feat = self.anomaly_modulation(concat_with_score)  # (B, T, C)
        
        # 根据异常分数调整融合权重
        # 异常分数高时，更依赖非周期性特征（捕获异常模式）
        # 异常分数低时，更依赖周期性特征（正常模式）
        periodic_weight = 1.0 - anomaly_scores
        aperiodic_weight = anomaly_scores
        
        # 加权融合
        weighted_fusion = periodic_feat * periodic_weight + aperiodic_feat * aperiodic_weight
        
        # 结合调制特征
        final_output = weighted_fusion + modulated_feat
        
        return final_output, anomaly_scores