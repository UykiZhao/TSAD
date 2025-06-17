import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from layers.MSTC import AperiodicBranch


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    """
    原始TimesNet的TimesBlock，作为周期分支使用
    """
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class AdaptiveFusion(nn.Module):
    """
    自适应融合模块，智能融合周期性和非周期性特征
    """
    def __init__(self, d_model):
        super(AdaptiveFusion, self).__init__()
        self.d_model = d_model
        
        # 重要性评估器 - 评估两个分支的相对重要性
        self.importance_evaluator = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),  # 输出两个分支的重要性权重
            nn.Softmax(dim=-1)
        )
        
        # 门控融合机制 - 控制信息流
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        # 特征变换层 - 对齐不同分支的特征空间
        self.periodic_transform = nn.Linear(d_model, d_model)
        self.aperiodic_transform = nn.Linear(d_model, d_model)
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, periodic_feat, aperiodic_feat):
        """
        Args:
            periodic_feat: (B, T, C) 周期分支特征
            aperiodic_feat: (B, T, C) 非周期分支特征
        Returns:
            fused_feat: (B, T, C) 融合后的特征
        """
        # 特征变换
        periodic_transformed = self.periodic_transform(periodic_feat)
        aperiodic_transformed = self.aperiodic_transform(aperiodic_feat)
        
        # 连接特征用于重要性评估
        combined_feat = torch.cat([periodic_transformed, aperiodic_transformed], dim=-1)  # (B, T, 2*C)
        
        # 计算重要性权重
        importance_weights = self.importance_evaluator(combined_feat)  # (B, T, 2)
        periodic_weight = importance_weights[:, :, 0:1]  # (B, T, 1)
        aperiodic_weight = importance_weights[:, :, 1:2]  # (B, T, 1)
        
        # 门控机制
        gate_weights = self.gate(combined_feat)  # (B, T, C)
        
        # 自适应加权融合
        weighted_periodic = periodic_transformed * periodic_weight * gate_weights
        weighted_aperiodic = aperiodic_transformed * aperiodic_weight * (1 - gate_weights)
        
        # 最终融合
        final_combined = torch.cat([weighted_periodic, weighted_aperiodic], dim=-1)  # (B, T, 2*C)
        fused_feat = self.final_fusion(final_combined)  # (B, T, C)
        
        return fused_feat


class DualBranchBlock(nn.Module):
    """
    双分支模块：
    - 周期分支：原始TimesNet的TimesBlock (FFT分析 → 2D转换 → Inception卷积)
    - 非周期分支：MSTC模块 (多尺度时间卷积 → 趋势提取 → 局部模式提取)
    """
    def __init__(self, configs):
        super(DualBranchBlock, self).__init__()
        self.configs = configs
        
        # 周期分支：原始TimesNet模块
        self.periodic_branch = TimesBlock(configs)
        
        # 非周期分支：MSTC模块
        self.aperiodic_branch = AperiodicBranch(configs)
        
        # 自适应融合模块
        self.adaptive_fusion = AdaptiveFusion(configs.d_model)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, T, C)
        Returns:
            Output tensor of shape (B, T, C)
        """
        # 复制输入数据给两个分支
        x_periodic = x.clone()    # 给周期分支
        x_aperiodic = x.clone()   # 给非周期分支
        
        # 周期分支处理：FFT分析 → 2D转换 → Inception卷积
        periodic_out = self.periodic_branch(x_periodic)  # (B, T, C)
        
        # 非周期分支处理：MSTC → 趋势提取 → 局部模式提取
        aperiodic_out = self.aperiodic_branch(x_aperiodic)  # (B, T, C)
        
        # 自适应融合两个分支的特征
        fused_out = self.adaptive_fusion(periodic_out, aperiodic_out)  # (B, T, C)
        
        return fused_out


class Model(nn.Module):
    """
    增强版TimesNet，采用双分支架构：
    - 周期分支：保持原TimesNet的FFT+2D卷积能力
    - 非周期分支：新增MSTC多尺度时间卷积能力
    - 自适应融合：智能融合两种特征
    
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        
        # 使用双分支模块替代原始TimesBlock
        self.model = nn.ModuleList([DualBranchBlock(configs)
                                    for _ in range(configs.e_layers)])
        
        # 嵌入层
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        
        # 异常检测专用投影层
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        
        # 双分支处理：每一层都包含周期分支和非周期分支
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
            
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        return None