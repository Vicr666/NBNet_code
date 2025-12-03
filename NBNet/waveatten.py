import torch
import torch.nn as nn

# 简单的 DropPath 实现 (如果你的环境没有 timm 库，这个类保证代码能直接跑)
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # 处理不同维度的输入 (B, ...)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock_Stabilizer_Opt(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=5):
        super().__init__()
        
        # --- Part 1: Spatial Mixing (空间平滑) ---
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        
        # [核心修改]: 使用 5x5 卷积，Padding 设为 2
        # 较大的卷积核有助于“修补”小波变换后的局部不连贯
        self.conv2 = nn.Conv2d(
            dw_channel, 
            dw_channel, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, # 5//2 = 2, 保持尺寸不变
            groups=dw_channel         # Depthwise Conv (高效)
        )
        
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)

        # --- Part 2: Channel Mixing (特征整合) ---
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)

        # Norms & LayerScale
        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        # DropPath (随机深度，防止过拟合)
        self.drop_path = DropPath(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, x):
        input_x = x
        
        # --- Block 1: Spatial ---
        # 1. Norm
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)

        # 2. Convolution (1x1 -> 5x5 DW -> Gate -> 1x1)
        x = self.conv1(x)
        x = self.conv2(x) # 5x5 平滑
        x = self.sg(x)
        x = self.conv3(x)
        
        # 3. Residual & DropPath
        x = input_x + self.drop_path(x * self.beta)

        # --- Block 2: Channel ---
        input_x = x # 更新残差基准
        
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)

        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        return input_x + self.drop_path(x * self.gamma)