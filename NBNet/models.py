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

from DWT_IDWT_layer import DWT_2D, IDWT_2D
from utils import *

from atten import GFEB
from edge import CannyDetector


class LFEB(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, down_scale, clamp=1.):
        super(LFEB, self).__init__()
        self.scale = down_scale
        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

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
                 down_scale=4, wavelet='wfml'):
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
# Deep Future Extraction Layer, QDAB
def conv_layer2(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)

class QDAB(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True, use_norm=True, device='cpu', scale=2):
        super(QDAB, self).__init__()

        self.gelu = nn.GELU()
        self.conv1 = conv_layer2(channel_in, 30, 1)
        self.aQDA=QDA()
        self.pointwise_conv = nn.Conv2d(30*3, channel_out, kernel_size=1)
        self.depthwise_conv12 = nn.Conv2d(30, 30, kernel_size=3, padding=1, dilation=1)
        self.depthwise_conv13 = nn.Conv2d(30, 30, kernel_size=3, padding=1, dilation=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x1=self.gelu(self.depthwise_conv12(x))
        #qda开始
        QDA_c1 = self.gelu(self.depthwise_conv13(x1))
        c2 =  torch.cat([x1, QDA_c1], dim=1)
        QDA_c2 = self.aQDA(QDA_c1)
        x4 = QDA_c2
        c3 =  torch.cat([c2, x4], dim=1)
        x = self.pointwise_conv(c3)
        return x

#qda核心
class QDA(nn.Module):
    def __init__(self, in_ch=30, codebook_size=6, reduction=5):
        super().__init__()
        self.in_ch = in_ch
        self.codebook_size = codebook_size
        # 可学习码本 [C, K]
        self.codebook = nn.Parameter(
            torch.randn(in_ch, codebook_size))

        # 特征压缩与编码
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_ch // reduction, codebook_size, 1)
        )

        # 残差权重控制
        self.gamma = nn.Parameter(torch.zeros(1))

        # 初始化
        nn.init.normal_(self.codebook, mean=0, std=0.02)
        nn.init.kaiming_normal_(self.encoder[0].weight, mode='fan_out')

    def forward(self, x):
        """
        输入: [1, 64, H, W]
        输出: [1, 64, H, W]
        """
        B, C, H, W = x.shape


        # 阶段1：特征编码与量化

        # 生成编码logits [1, K, H, W]
        logits = self.encoder(x)

        # Gumbel-Softmax量化（可微分）
        code_weights = F.gumbel_softmax(
            logits.view(B, self.codebook_size, -1),
            tau=0.5,
            hard=True,
            dim=1
        )  # [B, K, H*W]


        # 阶段2：码本查询与聚合

        # 码本查询 [C, K] x [B, K, HW] -> [B, C, HW]
        quantized = torch.matmul(
            self.codebook,
            code_weights
        ).view(B, C, H, W)


        # 阶段3：残差增强

        return x + (self.gamma) * quantized



def subnet(net_structure, init='xavier', use_norm=True, device='cpu', gc=32):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return QDAB(channel_in, channel_out, init, use_norm=use_norm, device=device, gc=gc)
            else:
                return QDAB(channel_in, channel_out, use_norm=use_norm, device=device, gc=gc)
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

class Db2Wavelet(nn.Module):
    def __init__(self):
        super(Db2Wavelet, self).__init__()
        wavename = 'db2'
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
    # TODO: 注意力个数设置


    attention = None
    cnt = 2
    if cnt == 1:
        attention = GFEB(upscale=upscale, img_size=(height, width),
                         window_size=window_size, img_range=1., depths=[cnt],
                         embed_dim=20, num_heads=[cnt], mlp_ratio=2, upsampler='pixelshuffledirect')
    if cnt == 2:
        attention = GFEB(upscale=upscale, img_size=(height, width),
                           window_size=window_size, img_range=1., depths=[2, 2],
                           embed_dim=20, num_heads=[2, 2], mlp_ratio=2, upsampler='pixelshuffledirect')

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
