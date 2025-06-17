import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Sequence


def make_divisible(value, divisor=8, min_value=None, min_ratio=0.9):
    """确保通道数可以被divisor整除"""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


def autopad(kernel_size: int, padding: int = None, dilation: int = 1):
    """自动计算padding以保持特征图大小不变"""
    assert kernel_size % 2 == 1, 'kernel size must be odd for autopad'
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1
    if padding is None:
        padding = kernel_size // 2
    return padding


class DepthwiseConv2d(nn.Module):
    """深度可分离卷积"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class CAA(nn.Module):
    """Context Anchor Attention - 用于捕获长程依赖关系"""
    def __init__(
        self,
        channels: int,
        h_kernel_size: int = 11,
        v_kernel_size: int = 11,
        reduction_ratio: int = 4
    ):
        super().__init__()
        # 平均池化层用于提取局部特征
        self.avg_pool = nn.AvgPool2d(7, stride=1, padding=3)
        
        # 通道压缩
        reduced_channels = make_divisible(channels // reduction_ratio)
        
        # 1x1卷积进行特征变换
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # 水平方向的条带卷积
        self.h_conv = nn.Conv2d(
            reduced_channels, reduced_channels, 
            kernel_size=(1, h_kernel_size),
            stride=1,
            padding=(0, h_kernel_size // 2),
            groups=reduced_channels,
            bias=False
        )
        
        # 垂直方向的条带卷积
        self.v_conv = nn.Conv2d(
            reduced_channels, reduced_channels,
            kernel_size=(v_kernel_size, 1),
            stride=1,
            padding=(v_kernel_size // 2, 0),
            groups=reduced_channels,
            bias=False
        )
        
        # 最终的1x1卷积和激活函数
        self.conv2 = nn.Sequential(
            nn.Conv2d(reduced_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 平均池化提取局部特征
        out = self.avg_pool(x)
        
        # 通道变换
        out = self.conv1(out)
        
        # 水平和垂直方向的长程依赖建模
        out = self.h_conv(out)
        out = self.v_conv(out)
        
        # 生成注意力权重
        attn_weights = self.conv2(out)
        
        return attn_weights


class PKIModule(nn.Module):
    """Poly Kernel Inception Module - 多尺度特征提取"""
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
        dilations: Sequence[int] = (1, 1, 1, 1, 1),
        expansion: float = 1.0,
        add_identity: bool = True,
        with_caa: bool = True,
        caa_kernel_size: int = 11
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion))
        
        # 预处理1x1卷积
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 基础3x3深度卷积
        self.dw_conv_base = DepthwiseConv2d(
            hidden_channels, hidden_channels,
            kernel_sizes[0], 1,
            autopad(kernel_sizes[0], None, dilations[0]),
            dilations[0]
        )
        
        # 多尺度深度卷积分支
        self.dw_convs = nn.ModuleList([
            DepthwiseConv2d(
                hidden_channels, hidden_channels,
                kernel_sizes[i], 1,
                autopad(kernel_sizes[i], None, dilations[i]),
                dilations[i]
            )
            for i in range(1, len(kernel_sizes))
        ])
        
        # 特征融合的1x1卷积
        self.pw_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # CAA注意力模块
        self.with_caa = with_caa
        if with_caa:
            self.caa = CAA(hidden_channels, caa_kernel_size, caa_kernel_size)
        
        # 是否添加恒等映射
        self.add_identity = add_identity and in_channels == out_channels
        
        # 后处理1x1卷积
        self.post_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        identity = x
        
        # 预处理
        x = self.pre_conv(x)
        
        # 保存用于CAA的特征
        caa_input = x
        
        # 基础深度卷积
        base_out = self.dw_conv_base(x)
        
        # 多尺度特征提取和融合
        multi_scale_out = base_out
        for dw_conv in self.dw_convs:
            multi_scale_out = multi_scale_out + dw_conv(x)
        
        # 特征融合
        x = self.pw_conv(multi_scale_out)
        
        # 应用CAA注意力
        if self.with_caa:
            attn_weights = self.caa(caa_input)
            x = x * attn_weights
            
        # 后处理
        x = self.post_conv(x)
        
        # 残差连接
        if self.add_identity:
            x = x + identity
            
        return x


class PeriodicBranch(nn.Module):
    """周期性分支 - 顺序处理FFT分析、2D转换、PKI特征提取和CAA建模"""
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.top_k = configs.top_k
        
        # PKI特征提取模块
        self.pki_module = PKIModule(
            in_channels=self.d_model,
            out_channels=self.d_model,
            kernel_sizes=(3, 5, 7, 9, 11),
            expansion=1.0,
            with_caa=True,
            caa_kernel_size=11
        )
        
        # 额外的CAA模块用于增强异常检测
        self.anomaly_caa = CAA(
            channels=self.d_model,
            h_kernel_size=15,
            v_kernel_size=15,
            reduction_ratio=2
        )
        
        # 特征增强层
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model * 2, 1, bias=False),
            nn.BatchNorm2d(self.d_model * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model * 2, self.d_model, 1, bias=False),
            nn.BatchNorm2d(self.d_model),
        )
        
    def FFT_for_Period(self, x):
        """FFT分析提取主要周期"""
        # x: [B, T, C]
        xf = torch.fft.rfft(x, dim=1)
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, self.top_k)
        top_list = top_list.detach().cpu().numpy()
        period = x.shape[1] // top_list
        return period, abs(xf).mean(-1)[:, top_list]
    
    def forward(self, x):
        """
        顺序处理流程：
        1. FFT周期分析
        2. 1D→2D转换
        3. PKI特征提取
        4. CAA长程建模
        5. 2D→1D还原
        """
        B, T, C = x.size()
        
        # 阶段1：FFT周期分析
        period_list, period_weight = self.FFT_for_Period(x)
        
        res = []
        for i in range(self.top_k):
            period = period_list[i]
            
            # 阶段2：1D→2D转换
            # padding处理
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([B, (length - (self.seq_len + self.pred_len)), C]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
                
            # reshape为2D: (B, C, H, W)
            out = out.reshape(B, length // period, period, C).permute(0, 3, 1, 2).contiguous()
            
            # 阶段3：PKI特征提取
            out = self.pki_module(out)
            
            # 阶段4：CAA长程建模（异常增强）
            anomaly_attn = self.anomaly_caa(out)
            out = out * anomaly_attn + out
            
            # 特征增强
            out = self.feature_enhance(out) + out
            
            # 阶段5：2D→1D还原
            out = out.permute(0, 2, 3, 1).reshape(B, -1, C)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
            
        # 多周期结果的自适应聚合
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, C, 1)
        res = torch.sum(res * period_weight, -1)
        
        # 残差连接
        res = res + x
        
        return res