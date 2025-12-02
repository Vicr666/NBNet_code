import torch
import torch.nn as nn

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock_Stabilizer(nn.Module):
    """
    轻量化稳定器：NAFBlock
    无需 Softmax，纯卷积+门控，极度高效。
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel) # Depthwise 平滑
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)
        
        # SimpleGate 核心
        self.sg = SimpleGate()
        
        # FFN 部分
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)
        
        self.norm1 = nn.LayerNorm(c)
        self.norm2 = nn.LayerNorm(c)
        
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x):
        # x: [B, C, H, W]
        identity = x
        
        # Part 1: Spatial Mixing (平滑)
        # LayerNorm 需要 permute
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        
        x = self.conv1(x)
        x = self.conv2(x) # 深度卷积消除棋盘格效应
        x = self.sg(x)    # 门控去噪
        x = self.conv3(x)
        
        x = identity + x * self.beta
        
        # Part 2: Channel Mixing (特征整合)
        identity = x
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x.permute(0, 3, 1, 2)
        
        x = self.conv4(x)
        x = self.sg(x)    # 再次门控
        x = self.conv5(x)
        
        return identity + x * self.gamma