import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import DownSampleBlock


class Discriminator(nn.Module):
    """
    判别器 (Dis):
    与编码器具有相同的结构，但最后增加一个1x1卷积层用于判别真假
    """
    def __init__(self, in_channels=4, base_channels=320):
        super(Discriminator, self).__init__()
        
        # 初始卷积层
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=1)
        
        # 三层下采样，与编码器结构相同
        self.layer1 = DownSampleBlock(base_channels, base_channels)
        self.layer2 = DownSampleBlock(base_channels, base_channels*2)
        self.layer3 = DownSampleBlock(base_channels*2, base_channels*4)
        
        # 输出层，1x1卷积到2个通道（真/假）
        self.output_conv = nn.Conv2d(base_channels*4, 2, kernel_size=1)
        
    def forward(self, x):
        # 初始卷积
        x = self.init_conv(x)
        
        # 三层下采样
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 最终判别层
        x = self.output_conv(x)
        
        return x
