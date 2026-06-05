import torch
import torch.nn as nn
from torch import Tensor
import math
import numpy as np
from torch.nn import init
from itertools import repeat
from torch.nn import functional as F
from torch._jit_internal import Optional
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import collections

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)

class Up(nn.Module):
    def __init__(self):
        super(Up, self).__init__()

    def forward(self, x1, x):
        # 自适应上采样 x1 至与 skip connection x 相同的空间尺寸
        x2 = F.interpolate(x1, size=x.shape[2:], mode='bilinear', align_corners=False)
        diffY = x.size()[2] - x2.size()[2]
        diffX = x.size()[3] - x2.size()[3]
        x3 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return x3

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ADown(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 1, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


# ==========================================================================================
# (a) DOConvV3 模块
# ==========================================================================================
class DOConv2d(Module):
    """
       DOConv2d can be used as an alternative for torch.nn.Conv2d.
       The interface is similar to that of Conv2d, with one exception:
            1. D_mul: the depth multiplier for the over-parameterization.
       Note that the groups parameter switchs between DO-Conv (groups=1),
       DO-DConv (groups=in_channels), DO-GConv (otherwise).
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size', 'D_mul']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self, in_channels, out_channels, kernel_size, D_mul=None, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):
        super(DOConv2d, self).__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))

        #################################### Initailization of D & W ###################################
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        self.D_mul = M * N if D_mul is None or M * N <= 1 else D_mul
        self.W = Parameter(torch.Tensor(out_channels, in_channels // groups, self.D_mul))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))

        if M * N > 1:
            self.D = Parameter(torch.Tensor(in_channels, M * N, self.D_mul))
            init_zero = np.zeros([in_channels, M * N, self.D_mul], dtype=np.float32)
            self.D.data = torch.from_numpy(init_zero)

            eye = torch.reshape(torch.eye(M * N, dtype=torch.float32), (1, M * N, M * N))
            d_diag = eye.repeat((in_channels, 1, self.D_mul // (M * N)))
            if self.D_mul % (M * N) != 0:  # the cases when D_mul > M * N
                zeros = torch.zeros([in_channels, M * N, self.D_mul % (M * N)])
                self.d_diag = Parameter(torch.cat([d_diag, zeros], dim=2), requires_grad=False)
            else:  # the case when D_mul = M * N
                self.d_diag = Parameter(d_diag, requires_grad=False)
        ##################################################################################################

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(DOConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        DoW_shape = (self.out_channels, self.in_channels // self.groups, M, N)
        if M * N > 1:
            ######################### Compute DoW #################
            # (input_channels, D_mul, M * N)
            D = self.D + self.d_diag
            W = torch.reshape(self.W, (self.out_channels // self.groups, self.in_channels, self.D_mul))

            # einsum outputs (out_channels // groups, in_channels, M * N),
            # which is reshaped to
            # (out_channels, in_channels // groups, M, N)
            DoW = torch.reshape(torch.einsum('ims,ois->oim', D, W), DoW_shape)
            #######################################################
        else:
            # in this case D_mul == M * N
            # reshape from
            # (out_channels, in_channels // groups, D_mul)
            # to
            # (out_channels, in_channels // groups, M, N)
            DoW = torch.reshape(self.W, DoW_shape)
        return self._conv_forward(input, DoW)


# ==========================================================================================
# (c) CRAB 模块
# ==========================================================================================
class CRAB(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(CRAB, self).__init__()
        # 3组连续的残差结构 (Conv -> ReLU -> Conv)
        self.blk1_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.blk1_relu = nn.ReLU(inplace=True)
        self.blk1_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)

        self.blk2_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.blk2_relu = nn.ReLU(inplace=True)
        self.blk2_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)

        self.blk3_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)
        self.blk3_relu = nn.ReLU(inplace=True)
        self.blk3_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)

        # 3组残差结束后的独立蓝色 Conv 块
        self.mid_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)

        # 混合注意力分支
        self.global_max_pooling = nn.AdaptiveMaxPool2d(1)
        self.global_average_pooling = nn.AdaptiveAvgPool2d(1)
        self.concat_conv = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1, bias=bias)
        self.sigmoid = nn.Sigmoid()

        # 尾部独立蓝色 Conv 块
        self.tail_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        # 3级局部残差
        out = x + self.blk1_conv2(self.blk1_relu(self.blk1_conv1(x)))
        out = out + self.blk2_conv2(self.blk2_relu(self.blk2_conv1(out)))
        out = out + self.blk3_conv2(self.blk3_relu(self.blk3_conv1(out)))

        # 经过中间蓝色 Conv
        x_mid = self.mid_conv(out)

        # 提取全局混合池化特征并拼合 (Concat)
        max_pool = self.global_max_pooling(x_mid)
        avg_pool = self.global_average_pooling(x_mid)
        attn = torch.cat([max_pool, avg_pool], dim=1)
        
        # 映射并经过 Sigmoid 生成通道注意力图
        attn = self.sigmoid(self.concat_conv(attn))
        
        # 对应图中的相乘点 ⊗ 
        fused = x_mid * attn
        
        # 加上最下方的输入长残差 ⊕ 
        fused = fused + x
        
        # 最后经过尾部蓝色 Conv 输出
        return self.tail_conv(fused)


# ==========================================================================================
# (d) HDCRAB 模块
# ==========================================================================================
class HDCRAB(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8, bias=True):
        super(HDCRAB, self).__init__()
        # Block 1: 1-DConv -> ReLU -> 2-DConv (扩张率 1 和 2)
        self.blk1_dconv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=bias)
        self.blk1_relu = nn.ReLU(inplace=True)
        self.blk1_dconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=bias)

        # Block 2: 3-DConv -> ReLU -> 4-DConv (扩张率 3 和 4)
        self.blk2_dconv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=bias)
        self.blk2_relu = nn.ReLU(inplace=True)
        self.blk2_dconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4, bias=bias)

        # Block 3: 3-DConv -> ReLU -> 2-DConv (扩张率 3 和 2)
        self.blk3_dconv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3, dilation=3, bias=bias)
        self.blk3_relu = nn.ReLU(inplace=True)
        self.blk3_dconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=bias)

        # 橙色中间过渡 1-DConv 块
        self.mid_dconv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=bias)

        # SE通道注意力分支 (Global Average Pooling -> Conv -> ReLU -> Conv -> Sigmoid)
        self.global_average_pooling = nn.AdaptiveAvgPool2d(1)
        self.attn_conv1 = nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1, bias=bias)
        self.attn_relu = nn.ReLU(inplace=True)
        self.attn_conv2 = nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=bias)
        self.sigmoid = nn.Sigmoid()

        # 尾部独立蓝色 Conv 块
        self.tail_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x):
        # 3级局部混合扩张率残差
        out = x + self.blk1_dconv2(self.blk1_relu(self.blk1_dconv1(x)))
        out = out + self.blk2_dconv2(self.blk2_relu(self.blk2_dconv1(out)))
        out = out + self.blk3_dconv2(self.blk3_relu(self.blk3_dconv1(out)))

        # 橙色 1-DConv
        x_mid = self.mid_dconv(out)

        # 挤压与激励全局注意力计算
        attn = self.global_average_pooling(x_mid)
        attn = self.attn_conv2(self.attn_relu(self.attn_conv1(attn)))
        attn = self.sigmoid(attn)

        # 注意力点乘 ⊗ 与底部的输入长残差 ⊕ 融合
        fused = (x_mid * attn) + x

        # 尾部蓝色 Conv
        return self.tail_conv(fused)


# ==========================================================================================
# (b) SDFB 模块 (完全精确复现 SDFB.png 内结构)
# ==========================================================================================
class SDFB(nn.Module):
    def __init__(self, channels, bias=True):
        super(SDFB, self).__init__()
        # 上分支门控映射 (Conv -> ReLU -> Conv -> Sigmoid)
        self.top_blk_conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)
        self.top_blk_relu = nn.ReLU(inplace=True)
        self.top_blk_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)
        self.top_sigmoid = nn.Sigmoid()

        # 下分支门控映射 (Conv -> ReLU -> Conv -> Sigmoid)
        self.bot_blk1_conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)
        self.bot_blk1_relu = nn.ReLU(inplace=True)
        self.bot_blk1_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)
        self.bot_sigmoid = nn.Sigmoid()

        # 下分支正中间的独立蓝色 Conv 块
        self.bot_mid_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)

        # 下分支尾部的增强模块 (Conv -> ReLU -> Conv)
        self.bot_blk2_conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)
        self.bot_blk2_relu = nn.ReLU(inplace=True)
        self.bot_blk2_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias)

    def forward(self, x_top, x_bottom):
        # 1. 顶部全图上分支的自注意力门控计算
        t_attn = self.top_sigmoid(self.top_blk_conv2(self.top_blk_relu(self.top_blk_conv1(x_top))))
        top_gated = x_top * t_attn

        # 2. 底部下分支的自注意力门控与独立卷积计算
        b_attn = self.bot_sigmoid(self.bot_blk1_conv2(self.bot_blk1_relu(self.bot_blk1_conv1(x_bottom))))
        bot_gated = x_bottom * b_attn
        bot_mid = self.bot_mid_conv(bot_gated)

        # 3. 对应图中正中央的上下两路特征跨分支相乘 ⊗ 点
        fused = bot_mid * top_gated

        # 4. 融合特征经过最后的尾部残差块与大跨度最长残差边 ⊕ 整合
        out = self.bot_blk2_conv2(self.bot_blk2_relu(self.bot_blk2_conv1(fused)))
        return out + x_bottom


# ==========================================================================================
# 完整主干网络
# ==========================================================================================
class UDCNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=1, nc=48, bias=True):
        super(UDCNet, self).__init__()
        kernel_size = 3
 
        # 将所有的旧 Conv 替换为全新的 DOConvV3 对应图(a)
        self.conv_head = DOConv2d(in_nc, nc, kernel_size=kernel_size, padding=1, bias=bias)
        self.doconvv3 = DOConv2d(nc, nc, kernel_size=kernel_size, padding=1, bias=bias)
        self.conv_tail = DOConv2d(nc, 16, kernel_size=kernel_size, padding=1, bias=bias)
        self.dual_tail2 = DOConv2d(16, out_nc, kernel_size=kernel_size, padding=1, bias=bias)

        # HDCRAB 分支使用独立的输入卷积，避免与 CRAB 分支共享 conv_head
        self.conv_head2 = DOConv2d(in_nc, nc, kernel_size=kernel_size, padding=1, bias=bias)

        # 精确替换双分支内的特征提取块
        self.crab = CRAB(nc, nc, bias=bias)
        self.hdcrab = HDCRAB(nc, nc, reduction=8, bias=bias)

        self.down = ADown(nc, nc)
        self.up = Up()
        self.up2 = Up()

        # 融合网络的核心替换为全新的 SDFB 对应图(b)
        self.sdfb = SDFB(channels=16, bias=bias)

    def forward(self, x):
        # --- CRAB 经典残差注意力分支 ---
        x1 = self.conv_head(x)         # 1072×1920
        x2 = self.doconvv3(x1)         # 1072×1920
        x2_1 = self.down(x2)           # 536×960   (第一次下采样)
        x3 = self.doconvv3(x2_1)       # 536×960
        x3_1 = self.down(x3)           # 268×480   (第二次下采样)
        x4 = self.crab(x3_1)           # 268×480   (最深层)
        x4_1 = self.up(x4, x3)         # 536×960   (第一次上采样, skip→x3)
        x5 = self.crab(x4_1 + x3)      # 536×960
        x5_1 = self.up2(x5, x2)        # 1072×1920 (第二次上采样, skip→x2)
        x6 = self.crab(x5_1 + x2)      # 1072×1920
        x7 = self.conv_tail(x6 + x1)   # 1072×1920 (skip→x1, 同尺寸直接相加)

        # --- HDCRAB 混合扩张卷积注意力分支 ---
        y1 = self.conv_head2(x)
        y2 = self.hdcrab(y1)
        y3 = self.hdcrab(y2)
        y4 = self.hdcrab(y3)
        y5 = self.hdcrab(y4 + y3)
        y6 = self.hdcrab(y5 + y2)
        y7 = self.conv_tail(y6 + y1)

        # --- SDFB 空间细节特征深度融合 ---
        # 此时由 y7 接入上分支引导流，x7 接入下分支主干流进行混合计算
        z1 = self.sdfb(x_top=y7, x_bottom=x7)
        z = self.dual_tail2(z1)

        return z


if __name__ == "__main__":
    model = UDCNet(2, 1)
    from torchinfo import summary
    summary(model, input_size=(1, 2, 1072, 1920), device='cuda', verbose=1)