import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

# 导入必要的模块
try:
    from layers.PKI_CAA import PeriodicBranch
    from layers.Enhanced_MSTC import EnhancedAperiodicBranch
    from layers.Adaptive_Fusion import EnhancedAdaptiveFusion, AnomalyAwareFusion
except ImportError:
    print("Warning: Some PKI modules not found. Please check your imports.")


def FFT_for_Period(x, k=2):
    """原始TimesNet的FFT周期分析"""
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class OriginalTimesBlock(nn.Module):
    """原始TimesNet的TimesBlock，用于消融实验"""
    def __init__(self, configs):
        super(OriginalTimesBlock, self).__init__()
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
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class SimpleFusion(nn.Module):
    """简单的融合模块，用于消融实验"""
    def __init__(self, d_model):
        super(SimpleFusion, self).__init__()
        self.fusion = nn.Linear(d_model * 2, d_model)
        
    def forward(self, periodic_feat, aperiodic_feat):
        concat_feat = torch.cat([periodic_feat, aperiodic_feat], dim=-1)
        return self.fusion(concat_feat)


class DualBranchBlock_Ablation(nn.Module):
    """
    消融实验的双分支块，支持不同的配置：
    - ablation_mode: 消融模式
        - 'full': 完整模型（默认）
        - 'no_pki': 周期分支使用原始TimesNet
        - 'no_mstc': 非周期分支使用简单卷积
        - 'no_adaptive_fusion': 使用简单融合
        - 'no_anomaly_fusion': 不使用异常感知融合
        - 'periodic_only': 只使用周期分支
        - 'aperiodic_only': 只使用非周期分支
    """
    def __init__(self, configs, ablation_mode='full'):
        super().__init__()
        self.configs = configs
        self.ablation_mode = ablation_mode
        
        # 周期性分支
        if ablation_mode in ['full', 'no_mstc', 'no_adaptive_fusion', 'no_anomaly_fusion', 'periodic_only']:
            if ablation_mode == 'no_pki':
                self.periodic_branch = OriginalTimesBlock(configs)
            else:
                self.periodic_branch = PeriodicBranch(configs)
        else:
            self.periodic_branch = None
            
        # 非周期性分支
        if ablation_mode in ['full', 'no_pki', 'no_adaptive_fusion', 'no_anomaly_fusion', 'aperiodic_only']:
            if ablation_mode == 'no_mstc':
                # 使用简单的1D卷积代替MSTC
                self.aperiodic_branch = nn.Sequential(
                    nn.Conv1d(configs.d_model, configs.d_model, 3, padding=1),
                    nn.BatchNorm1d(configs.d_model),
                    nn.ReLU(),
                    nn.Conv1d(configs.d_model, configs.d_model, 3, padding=1),
                    nn.BatchNorm1d(configs.d_model),
                    nn.ReLU()
                )
            else:
                self.aperiodic_branch = EnhancedAperiodicBranch(configs)
        else:
            self.aperiodic_branch = None
            
        # 融合模块
        if ablation_mode not in ['periodic_only', 'aperiodic_only']:
            if ablation_mode == 'no_adaptive_fusion':
                self.adaptive_fusion = SimpleFusion(configs.d_model)
                self.use_anomaly_fusion = False
            else:
                self.adaptive_fusion = EnhancedAdaptiveFusion(
                    d_model=configs.d_model,
                    dropout=configs.dropout
                )
                # 异常感知融合
                self.use_anomaly_fusion = (ablation_mode != 'no_anomaly_fusion')
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
        """前向传播"""
        residual = x
        
        # 根据消融模式处理
        if self.ablation_mode == 'periodic_only':
            # 只使用周期分支
            output = self.periodic_branch(x)
        elif self.ablation_mode == 'aperiodic_only':
            # 只使用非周期分支
            if isinstance(self.aperiodic_branch, nn.Sequential):
                # 简单卷积需要转换维度
                x_conv = x.permute(0, 2, 1)
                output = self.aperiodic_branch(x_conv)
                output = output.permute(0, 2, 1)
            else:
                output = self.aperiodic_branch(x)
        else:
            # 双分支处理
            periodic_out = self.periodic_branch(x)
            
            if isinstance(self.aperiodic_branch, nn.Sequential):
                # 简单卷积需要转换维度
                x_conv = x.permute(0, 2, 1)
                aperiodic_out = self.aperiodic_branch(x_conv)
                aperiodic_out = aperiodic_out.permute(0, 2, 1)
            else:
                aperiodic_out = self.aperiodic_branch(x)
            
            # 融合
            if hasattr(self, 'adaptive_fusion'):
                fused_features = self.adaptive_fusion(periodic_out, aperiodic_out)
                
                # 异常感知融合（如果启用）
                if self.use_anomaly_fusion and hasattr(self, 'anomaly_fusion'):
                    anomaly_aware_features, _ = self.anomaly_fusion(periodic_out, aperiodic_out)
                    fused_features = (fused_features + anomaly_aware_features) * 0.5
                    
                output = fused_features
            else:
                # 简单平均（备用）
                output = (periodic_out + aperiodic_out) * 0.5
        
        # 输出投影
        output = self.output_projection(output)
        
        # 残差连接
        output = output + residual
        
        return output, None  # 返回None作为异常分数占位符


class Model(nn.Module):
    """
    DualScopeNet消融实验模型
    通过ablation_mode参数控制消融哪个组件
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_layers = configs.e_layers
        
        # 获取消融模式（从配置中读取，默认为'full'）
        self.ablation_mode = getattr(configs, 'ablation_mode', 'full')
        print(f"Ablation mode: {self.ablation_mode}")
        
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
            DualBranchBlock_Ablation(configs, ablation_mode=self.ablation_mode) 
            for _ in range(self.num_layers)
        ])
        
        # 层标准化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(configs.d_model) for _ in range(self.num_layers)
        ])
        
        # 异常检测输出层
        if self.task_name == 'anomaly_detection':
            # 最终投影层
            self.projection = nn.Sequential(
                nn.Linear(configs.d_model, configs.d_model // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model // 2, configs.c_out)
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
        for i, (block, norm) in enumerate(zip(self.dual_branch_blocks, self.layer_norms)):
            # DualBranchBlock处理
            block_out, _ = block(enc_out)
            
            # 残差连接和层标准化
            enc_out = norm(block_out + enc_out)
        
        # 投影到输出维度
        dec_out = self.projection(enc_out)
        
        # 反标准化
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1)
        
        return dec_out, None
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """模型前向传播"""
        if self.task_name == 'anomaly_detection':
            dec_out, _ = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        return None