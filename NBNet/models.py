# models
import logging
import math
import os
import time
from collections import Counter
from collections import OrderedDict
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim.lr_scheduler import _LRScheduler
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_

from DWT_IDWT_layer import DWT_2D, IDWT_2D
from utils import *

from atten import GFEB
from edge import CannyDetector
from einops import rearrange

class LFEB(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, down_scale, clamp=1.):
        super(LFEB, self).__init__()
        self.scale = down_scale
        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num
        # if self.split_len2 <= 0:
        #     # fallback: 将通道按半分
        #     self.split_len1 = channel_num // 2
        #     self.split_len2 = channel_num - self.split_len1
        #     import warnings
        #     warnings.warn(f"LFEB: channel_split_num too large, fallback to half split: "
        #                 f"split_len1={self.split_len1}, split_len2={self.split_len2}")
        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * self.scale - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * self.scale - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


sz_max_idx = []
class ConvMapping(nn.Module):
    def __init__(self, channel_in, scale=3):
        super(ConvMapping, self).__init__()
        self.scale = scale
        self.channel_in = channel_in
    def forward(self, x, rev=False):
        if not rev:
            x = x.reshape(x.shape[0], self.channel_in * self.scale ** 2, x.shape[2] // self.scale, x.shape[3] // self.scale)
        else:
            x = x.reshape(x.shape[0], self.channel_in, x.shape[2] * self.scale, x.shape[3] * self.scale)
        return x
# CWQRNet
class CWQRNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_constructor=None, block_num=[], attention=None,
                 down_scale=4, wavelet='DCWML'):
        super(CWQRNet, self).__init__()

        self.attention = attention

        operations = []

        current_channel = channel_in
        down_num = len(block_num)
        if down_num > 1:
            assert down_scale % down_num == 0
            down_scale //= down_num

        for i in range(down_num):
            # TODO: 切换小波映射
            if wavelet == 'haar':
                b = HaarWavelet()
            if wavelet == 'DCWML':
                b = Db2Wavelet()
            if wavelet == 'db3':
                b = Db3Wavelet()
            if wavelet == 'ch22':
                b = Ch2p2Wavelet()
            if wavelet == 'ch33':
                b = Ch3p3Wavelet()

            # b = ConvMapping(current_channel, down_scale)#普通维度映射
            operations.append(b)
            current_channel *= down_scale ** 2
            # 添加QDAB块
            for j in range(block_num[i]):
                b = LFEB(subnet_constructor, current_channel, channel_out, down_scale)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, x, rev=False, cal_jacobian=False):
        if self.attention is not None and x.shape[1] == 3:
            x = self.attention(x)
        out = x
        jacobian = 0
        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, jacobian
        else:
            return out


class ReconstructionLoss(nn.Module):
    def __init__(self, losstype='l2', eps=1e-6):
        super(ReconstructionLoss, self).__init__()
        self.losstype = losstype
        self.eps = eps

    def forward(self, x, target):
        if self.losstype == 'l2':
            return torch.mean(torch.sum((x - target) ** 2, (1, 2, 3)))
        elif self.losstype == 'l1':
            diff = x - target
            return torch.mean(torch.sum(torch.sqrt(diff * diff + self.eps), (1, 2, 3)))
        # elif self.losstype == 'KL':
        #     return self.edg(x, target)
        else:
            print("reconstruction loss type error!")
            return 0


# 分布边缘指导损失 Distribution Edge Guidance
class DEGLoss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5, device = 'cpu'):
        super(DEGLoss, self).__init__()
        self.device = device
        self.edge = CannyDetector(detach=True, device=device).to(device)
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def guassian_kernel(self, distance, n_samples):
        bandwidth = torch.sum(distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def separate_channel(self, source, target):
        n_samples = int(source.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        kernels = self.guassian_kernel(L2_distance, n_samples * 2)
        if torch.isnan(kernels).any(): return torch.tensor(0.)
        XX = kernels[:n_samples, :n_samples]
        YY = kernels[n_samples:, n_samples:]
        XY = kernels[:n_samples, n_samples:]
        YX = kernels[n_samples:, :n_samples]
        ret = torch.mean(XX + YY - XY - YX)
        return ret

    def forward(self, source, target):  # SR, HR
        assert source.shape[0] == target.shape[0] and (source.shape[1] == 3 and target.shape[1] == 3)
        res = torch.tensor(0.).to(self.device)
        for bs in range(source.shape[0]):
            res += self.separate_channel(source[bs, 0, :, :], target[bs, 0, :, :])
            res += self.separate_channel(source[bs, 1, :, :], target[bs, 1, :, :])
            res += self.separate_channel(source[bs, 2, :, :], target[bs, 2, :, :])
        return torch.div(res, source.shape[0] * source.shape[1])



def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        input = torch.clamp(input, 0, 1)
        output = (input * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)


class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.conv(x)
        # out = self.bn(out)
        out = self.gelu(out)
        return out
    

# Deep Future Extraction Layer 
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Deep Future Extraction Layer, MSGC

def conv_layer2(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)
    
class MSGC(nn.Module):
    def __init__(self, channel_in, channel_out, mid_channels=30, bias=True):
        super(MSGC, self).__init__()

        self.gelu = nn.GELU()
        
        # 入口
        self.conv_entry = nn.Conv2d(channel_in, mid_channels, kernel_size=1, bias=bias)

        # 分支 1：标准特征提取
        self.conv_b1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=bias)
        
        # 分支 2：深层特征提取 (串行在 b1 之后)
        self.conv_b2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=bias)
        
        # 分支 3 (替代 QDA)：门控注意力分支
        # 使用 1x1 卷积从 b2 生成权重，模拟 Attention
        self.conv_gate = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=bias)
        self.sigmoid = nn.Sigmoid()

        # 融合
        # 这里的输入也是 30*3 = 90
        self.pointwise_conv = nn.Conv2d(mid_channels * 3, channel_out, kernel_size=1, bias=bias)

    def forward(self, x):
        x_in = self.conv_entry(x)
        
        # 1. 基础特征
        x1 = self.gelu(self.conv_b1(x_in))
        
        # 2. 进阶特征
        x2_pre = self.gelu(self.conv_b2(x1))
        
        # 3. 门控操作
        # 用 x2 自己去计算一个 Gate，然后对自己进行加权
        # 这是一种自注意力 (Self-Gating) 机制
        gate = self.sigmoid(self.conv_gate(x2_pre))
        x3 = x2_pre * gate  # 门控激活
        
        # 注意：这里 x2 我们取原始卷积结果，x3 取门控结果，x1 取浅层结果
        # 这样保持了特征的多样性
        c_out = torch.cat([x1, x2_pre, x3], dim=1)
        
        out = self.pointwise_conv(c_out)
        return out


def subnet(net_structure, init='xavier', use_norm=True, device='cpu', gc=32):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return MSGC(channel_in, channel_out)
            else:
                return MSGC(channel_in, channel_out)
        else:
            return None

    return constructor


class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        # self.device = torch.device('cuda' if opt['gpu_ids'] is not None and torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def _set_lr(self, lr_groups_l):
        ''' set learning rate for warmup,
        lr_groups_l: list for lr_groups. each for a optimizer'''
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        # get the initial lr, which is set by the scheduler
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, cur_iter, warmup_iter=-1):
        for scheduler in self.schedulers:
            scheduler.step()
        #### set up warm up learning rate
        if cur_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * cur_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        # return self.schedulers[0].get_lr()[0]
        return self.optimizers[0].param_groups[0]['lr']

    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n

    def save_network(self, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(self.opt['path']['models'], save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith('module.'):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v
        network.load_state_dict(load_net_clean, strict=strict)

    def save_training_state(self, epoch, iter_step):
        '''Saves training state during training, which will be used for resuming'''
        state = {'epoch': epoch, 'iter': iter_step, 'schedulers': [], 'optimizers': []}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(iter_step)
        save_path = os.path.join(self.opt['path']['training_state'], save_filename)
        torch.save(state, save_path)

    def resume_training(self, resume_state):
        '''Resume the optimizers and schedulers for training'''
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers'
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers'
        for i, o in enumerate(resume_optimizers):
            # o['param_groups'][0]['lr'] = 0.000013
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)


class MultiStepLR_Restart(_LRScheduler):
    def __init__(self, optimizer, milestones, restarts=None, weights=None, gamma=0.1,
                 clear_state=False, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.clear_state = clear_state
        self.restarts = restarts if restarts else [0]
        self.restart_weights = weights if weights else [1]
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [
            group['lr'] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]


class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(self, optimizer, T_period, restarts=None, weights=None, eta_min=0, last_epoch=-1):
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        assert len(self.restarts) == len(
            self.restart_weights), 'restarts and their weights do not match.'
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group['lr'] + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [(1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) /
                (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) *
                (group['lr'] - self.eta_min) + self.eta_min
                for group in self.optimizer.param_groups]


class DownSampleWavelet(nn.Module):
    def __init__(self, wavename='haar'):
        super(DownSampleWavelet, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)

    def forward(self, input):
        LL = self.dwt(input)
        return LL

class UpSampleWavelet(nn.Module):
    def __init__(self, wavename='haar'):
        super(UpSampleWavelet, self).__init__()
        self.dwt = IDWT_2D(wavename=wavename)

    def forward(self, LL, LH, HL, HH):
        in_x = self.dwt(LL, LH, HL, HH)
        return in_x

class HaarWavelet(nn.Module):
    def __init__(self):
        super(HaarWavelet, self).__init__()
        wavename = 'haar'
        self.down = DownSampleWavelet(wavename)
        self.up = UpSampleWavelet(wavename)
        self.stb1 = NAFBlock_Stabilizer_Opt(c=12)
        self.stb2 = NAFBlock_Stabilizer_Opt(c=48)
    def forward(self, x, rev=False):
        if not rev:
            x = torch.cat(self.down(x), dim=1)
            nnn = x.shape[1]
            if nnn == 12:
                x = self.stb1(x)
            elif nnn == 48:
                x = self.stb2(x)
        else:
            c = x.shape[1] // 2 // 2
            x = self.up(x[:, 0*c:1*c],
                        x[:, 1*c:2*c],
                        x[:, 2*c:3*c],
                        x[:, 3*c:4*c], )
            # 1 48 36 36
            # 1 12 72 72
            # 1 3 144 144
        return x

from waveatten import NAFBlock_Stabilizer_Opt

class WaveletGatedTransformer(nn.Module):
    """
    专为小波拼接特征设计的“去噪与平滑” Transformer 变体。
    
    特点：
    1. 极度轻量：基于通道注意力 (Restormer 风格)。
    2. 门控去噪：使用 Gated-FFN 模拟小波软阈值操作，抑制高频干扰。
    3. 空间平滑：内置 Depthwise Conv 解决小波造成的空间不连续问题。
    """
    def __init__(self, dim, num_heads=4, ffn_expansion=2.66, bias=False):
        super(WaveletGatedTransformer, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        
        # 1. 频带位置编码 (Frequency Embedding)
        # 输入是 [LL, LH, HL, HH] 拼接，给它们打上可学习的标签
        self.freq_embedding = nn.Parameter(torch.zeros(1, dim, 1, 1))
        
        # 2. 输入归一化 (LayerNorm 的通道维度变体)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=dim)

        # 3. 模块 A: 跨频带全局注意力 (Cross-Band Global Attention)
        # 作用：利用频带间的协方差来校准特征
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 4. 模块 B: 门控前馈网络 (Gated-FFN) - 核心去噪组件
        # 作用：模拟软阈值去噪，并平滑空间特征
        hidden_dim = int(dim * ffn_expansion)
        
        # 4.1 升维
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=bias)
        
        # 4.2 空间平滑 (Spatial Smoothing) 
        # 关键！用于消除小波变换的“硬”边缘，实现到下一层的平滑过渡
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, 
                                groups=hidden_dim * 2, bias=bias)
        
        # 4.3 降维
        self.project_ffn_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=bias)
        
        # 初始化
        nn.init.trunc_normal_(self.freq_embedding, std=0.02)

    def forward(self, x):
        """
        x: [Batch, 4*C, H/2, W/2] (小波四子带拼接)
        """
        b, c, h, w = x.shape
        
        # --- 步骤 0: 注入频带信息 ---
        x = x + self.freq_embedding
        shortcut = x

        # --- 步骤 1: 跨频带注意力 (Global Context) ---
        x_norm = self.norm1(x)
        
        # 生成 Q, K, V
        qkv = self.qkv_dwconv(self.qkv(x_norm))
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape 为通道注意力形式: (B, Heads, C_per_head, HW)
        q = q.view(b, self.num_heads, c // self.num_heads, -1)
        k = k.view(b, self.num_heads, c // self.num_heads, -1)
        v = v.view(b, self.num_heads, c // self.num_heads, -1)

        # 计算通道协方差 (Attention Map)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        # 应用注意力
        out_attn = (attn @ v).view(b, c, h, w)
        out_attn = self.project_out(out_attn)
        
        # 残差连接 1
        x = shortcut + out_attn
        shortcut = x

        # --- 步骤 2: 门控去噪 FFN (Gated Denoising) ---
        x_norm = self.norm2(x)
        
        # 升维 -> 深度卷积平滑
        x_ffn = self.project_in(x_norm)
        x_ffn = self.dwconv(x_ffn)
        
        # 拆分为两部分：特征部分(x1) 和 门控部分(x2)
        x1, x2 = x_ffn.chunk(2, dim=1)
        
        # **门控机制 (Gating)**
        # GELU 作为一个平滑的非线性函数，配合乘法，起到类似“软阈值”的作用
        # 抑制噪声(低响应值)，保留特征(高响应值)
        x_ffn = x1 * F.gelu(x2) 
        
        # 降维输出
        x_ffn = self.project_ffn_out(x_ffn)
        
        # 残差连接 2
        out = shortcut + x_ffn
        
        return out

class Db2Wavelet(nn.Module):
    def __init__(self):
        super(Db2Wavelet, self).__init__()
        wavename = 'db2'
        self.down = DownSampleWavelet(wavename)
        self.up = UpSampleWavelet(wavename)
        # self.stb1 = NAFBlock_Stabilizer_Opt(c=12)
        # self.stb2 = NAFBlock_Stabilizer_Opt(c=48)

    def forward(self, x, rev=False):
        if not rev:
            x = torch.cat(self.down(x), dim=1)
            # nnn = x.shape[1]
            # if nnn == 12:
            #     x = self.stb1(x)
            # elif nnn == 48:
            #     x = self.stb2(x)
        else:
            c = x.shape[1] // 2 // 2
            x = self.up(x[:, 0*c:1*c],
                        x[:, 1*c:2*c],
                        x[:, 2*c:3*c],
                        x[:, 3*c:4*c], )
            # 1 48 36 36
            # 1 12 72 72
            # 1 3 144 144
        return x

class Db3Wavelet(nn.Module):
    def __init__(self):
        super(Db3Wavelet, self).__init__()
        wavename = 'db3'
        self.down = DownSampleWavelet(wavename)
        self.up = UpSampleWavelet(wavename)
    def forward(self, x, rev=False):
        if not rev:
            x = torch.cat(self.down(x), dim=1)
        else:
            c = x.shape[1] // 2 // 2
            x = self.up(x[:, 0*c:1*c],
                        x[:, 1*c:2*c],
                        x[:, 2*c:3*c],
                        x[:, 3*c:4*c], )
            # 1 48 36 36
            # 1 12 72 72
            # 1 3 144 144
        return x

class Ch2p2Wavelet(nn.Module):
    def __init__(self):
        super(Ch2p2Wavelet, self).__init__()
        wavename = 'bior2.2'
        self.down = DownSampleWavelet(wavename)
        self.up = UpSampleWavelet(wavename)
    def forward(self, x, rev=False):
        if not rev:
            x = torch.cat(self.down(x), dim=1)
        else:
            c = x.shape[1] // 2 // 2
            x = self.up(x[:, 0*c:1*c],
                        x[:, 1*c:2*c],
                        x[:, 2*c:3*c],
                        x[:, 3*c:4*c], )
            # 1 48 36 36
            # 1 12 72 72
            # 1 3 144 144
        return x

class Ch3p3Wavelet(nn.Module):
    def __init__(self):
        super(Ch3p3Wavelet, self).__init__()
        wavename = 'bior3.3'
        self.down = DownSampleWavelet(wavename)
        self.up = UpSampleWavelet(wavename)
        # self.stb1 = NAFBlock_Stabilizer_Opt(c=12)
        # self.stb2 = NAFBlock_Stabilizer_Opt(c=48)
    def forward(self, x, rev=False):
        if not rev:
            x = torch.cat(self.down(x), dim=1)
            # nnn = x.shape[1]
            # if nnn == 12:
            #     x = self.stb1(x)
            # elif nnn == 48:
            #     x = self.stb2(x)
        else:
            c = x.shape[1] // 2 // 2
            x = self.up(x[:, 0*c:1*c],
                        x[:, 1*c:2*c],
                        x[:, 2*c:3*c],
                        x[:, 3*c:4*c], )
            # 1 48 36 36
            # 1 12 72 72
            # 1 3 144 144
        return x


class MLRNModel(BaseModel):
    def __init__(self, opt):
        super(MLRNModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
            print(self.rank)
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']
        test_opt = opt['test']
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.use_KL = opt['use_KL_Loss']

        self.netG = define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()],
                                                broadcast_buffers=False, find_unused_parameters=True)
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        self.Quantization = Quantization()

        if self.is_train:
            self.netG.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw']) # L2
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back']) # l1
            self.Reconstruction_kl = DEGLoss(device=self.device)
            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger = logging.getLogger('base{}'.format(loggerIdx.log_idx))
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                            restarts=train_opt['restarts'],
                                            weights=train_opt['restart_weights'],
                                            gamma=train_opt['lr_gamma'],
                                            clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  # LQ
        self.real_H = data['GT'].to(self.device)  # GT

    def gaussian_batch(self, dims):
        return torch.randn(tuple(dims)).to(self.device)

    def loss_forward(self, out, y, z):
        # down
        l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)  # L2
        return l_forw_fit

    def loss_backward(self, x, y):
        # up re
        x_samples = self.netG(x=y, rev=True)
        x_samples_image = x_samples[:, :3, :, :]
        l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(x, x_samples_image)  # L1

        # TODO: 切换边缘 - 损失函数
        l_back_kl = self.train_opt['lambda_rec_kl'] * self.Reconstruction_kl(x, x_samples_image) ##########
        return l_back_rec, l_back_kl

    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        # forward downscaling
        self.input = self.real_H
        self.output = self.netG(x=self.input)

        zshape = self.output[:, 3:, :, :].shape
        LR_ref = self.ref_L.detach()
        if self.output[:, :3, :, :].shape[2:] != LR_ref.shape[2:]:
            raise RuntimeError(f"Spatial size mismatch: output {self.output[:, :3, :, :].shape[2:]} != LR_ref {LR_ref.shape[2:]}")
        l_forw_fit = self.loss_forward(self.output[:, :3, :, :], LR_ref, self.output[:, 3:, :, :])

        # backward upscaling
        LR = self.Quantization(self.output[:, :3, :, :])
        gaussian_scale = self.train_opt['gaussian_scale'] if self.train_opt['gaussian_scale'] != None else 1
        y_ = torch.cat((LR, gaussian_scale * self.gaussian_batch(zshape)), dim=1)  # 拼接 成 Z = [输出, 和高斯那啥]

        l_back_rec, l_back_kl = self.loss_backward(self.real_H, y_)

        loss = l_forw_fit + l_back_rec + l_back_kl

        loss.backward()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()

        # set log
        self.log_dict['l_forw_fit'] = l_forw_fit.item()
        self.log_dict['l_back_rec'] = l_back_rec.item()
        self.log_dict['l_back_kl'] = l_back_kl.item()
        self.log_dict['total_loss'] = l_forw_fit.item() + l_back_rec.item() + l_back_kl.item()

    def test(self):
        Lshape = self.ref_L.shape

        input_dim = Lshape[1]
        self.input = self.real_H

        zshape = [Lshape[0], input_dim * (self.opt['scale'] ** 2) - Lshape[1], Lshape[2], Lshape[3]]

        gaussian_scale = 1
        if self.test_opt and self.test_opt['gaussian_scale'] != None:
            gaussian_scale = self.test_opt['gaussian_scale']

        self.netG.eval()
        with torch.no_grad():
            self.forw_L = self.netG(x=self.input)[:, :3, :, :]
            self.forw_L = self.Quantization(self.forw_L)

            y_forw = torch.cat((self.forw_L, gaussian_scale * self.gaussian_batch(zshape)), dim=1)
            self.fake_H = self.netG(x=y_forw, rev=True)[:, :3, :, :]

        self.netG.train()

    def downscale(self, HR_img):
        self.netG.eval()
        with torch.no_grad():
            LR_img = self.netG(x=HR_img)[:, :3, :, :]
            LR_img = self.Quantization(LR_img)
        self.netG.train()

        return LR_img

    def upscale(self, LR_img, scale, gaussian_scale=1):
        Lshape = LR_img.shape
        zshape = [Lshape[0], Lshape[1] * (scale ** 2 - 1), Lshape[2], Lshape[3]]
        y_ = torch.cat((LR_img, gaussian_scale * self.gaussian_batch(zshape)), dim=1)

        self.netG.eval()
        with torch.no_grad():
            HR_img = self.netG(x=y_, rev=True)[:, :3, :, :]
        self.netG.train()

        return HR_img

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if not self.train_opt['save_pic']:
            out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
            out_dict['LR_ref'] = self.ref_L.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger = logging.getLogger('base{}'.format(loggerIdx.log_idx))
            logger.info(s)
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger = logging.getLogger('base{}'.format(loggerIdx.log_idx))
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)


def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    subnet_type = which_model['subnet_type']
    device = torch.device('cuda' if opt['gpu_ids'] is not None and torch.cuda.is_available() else 'cpu')
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'

    upscale = opt_net['scale']
    window_size = opt_net['window_size']
    height, width = 144, 144
    # 添加注意力模块
    attention = None
    cnt = 2
    if cnt == 1:
        attention = GFEB(upscale=upscale, img_size=(height, width),
                         window_size=window_size, img_range=1., depths=[cnt],
                         embed_dim=20, num_heads=[cnt], mlp_ratio=2, upsampler='pixelshuffledirect')
    if cnt == 2:
        attention = GFEB(upscale=upscale, img_size=(height, width),
                           window_size=window_size, img_range=1., depths=[2, 2],
                           embed_dim=60, num_heads=[2, 2], mlp_ratio=2, upsampler='pixelshuffledirect')

    if cnt == 3:
        attention = GFEB(upscale=upscale, img_size=(height, width),
                           window_size=window_size, img_range=1., depths=[3, 3, 3],
                           embed_dim=21, num_heads=[3, 3, 3], mlp_ratio=2, upsampler='pixelshuffledirect')

    # 消融参数 已禁用
    if opt_net['scale'] == 2:
        gc = 32
    elif opt_net['scale'] == 3:
        gc = 32 # 36 ?
    elif opt_net['scale'] == 4:
        gc = 32

    netG = CWQRNet(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init, opt['use_Norm_Layer'], device, gc=gc),
                         opt_net['block_num'], attention=attention, down_scale=opt_net['scale'], wavelet=opt['ab_wavelet'])
    return netG


def create_model(opt):
    # model = opt['model']  # No use
    m = MLRNModel(opt)
    logger = logging.getLogger('base{}'.format(loggerIdx.log_idx))
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
