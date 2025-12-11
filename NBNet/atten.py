import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import collections.abc
from itertools import repeat

# utils

def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        # warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.", stacklevel=2)
        return None
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

import test_srformer as Transformer2
class GFEB(nn.Module):

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=60, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(GFEB, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        embed_dim = 60
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.window_size = window_size

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.last_conv = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)
        base_win_size = [16, 16]
        self.t = Transformer2.HiT_SRF(upscale=4, img_size=img_size,
                   base_win_size=base_win_size, img_range=1., depths=[3, 3],
                   embed_dim=60, num_heads=[2,2], mlp_ratio=2, upsampler='pixelshuffledirect')
        # self.t = Transformer2.LCFormer()
        self.act2 = nn.GELU()

        # self.ft = FeaTiao_LKA()
        self.ft = SpatialWeighting_Block()
        self.last_conv2 = nn.Conv2d(60, 60, 3, 1, 1)
        self.last_conv3 = nn.Conv2d(120, 60, 1, 1, 1)
        self.last_conv4 = nn.Conv2d(120, 60, 3, 1, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # for lightweight SR
        x = self.conv_first(x)  # BS 3 264 184  ==>  BS 60 264 184
        x_jb = self.act2(self.last_conv2(x))
        x_jb_z = self.ft(x_jb) * x
        x_c = torch.concat([self.t(x), x_jb_z], dim=1)
        # x_c = self.t(x)
        x = self.last_conv4(x_c)
        x = self.act2(x)
        x = self.last_conv(x)
        x = x / self.img_range + self.mean

        return x[:, :, :H, :W]


class FeaTiao_LKA(nn.Module):
    def __init__(self, dim=60):
        super(FeaTiao_LKA, self).__init__()
        # 1. 深度卷积 (5x5) - 捕获局部信息
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        # 2. 深度空洞卷积 (7x7, dilation=3) - 增大感受野
        # 感受野相当于 5 + (7-1)*3 = 23
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        
        # 3. 1x1 卷积 - 融合通道
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):     
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        
        return attn
    
class SpatialWeighting_Block(nn.Module):
    def __init__(self, dim=60):
        super(SpatialWeighting_Block, self).__init__()
        
        # ==========================================
        # 分支 1: 权重生成器 (Weight Generator) - 7x7
        # 目标: 返回权重 (0~1)
        # ==========================================
        self.weight_generator = nn.Sequential(
            # 第一步：大感受野感知上下文
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim), 
            
            # 第二步：通道融合，整合信息
            nn.Conv2d(dim, dim, 1),
            
            # 第三步：生成权重 (关键)
            # Sigmoid 将输出压缩到 0 到 1 之间，使其成为真正的"权重"
            nn.Sigmoid() 
        )
        
        # ==========================================
        # 分支 2: 特征提取器 (Feature Extractor) - 5x5
        # 目标: 提取原始纹理特征
        # ==========================================
        self.feature_extractor = nn.Sequential(
            # 纯局部特征提取
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU() # 这里用 GELU 提取非线性特征
        )

        # 最后的融合层
        self.fusion = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        
        # 1. 生成权重图 (Batch, Dim, H, W)
        # 这里的 weights 里的值都在 0~1 之间
        weights = self.weight_generator(x)
        
        # 2. 生成特征图 (Batch, Dim, H, W)
        features = self.feature_extractor(x)
        
        # 3. 加权操作 (Element-wise Product)
        # 特征 * 权重：保留高权重区域的纹理，抑制低权重区域的噪声
        weighted_features = features * weights
        
        # 4. 融合与残差
        out = self.fusion(weighted_features)
        
        return out