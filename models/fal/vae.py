import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块：包含两个卷积层和一个跳跃连接"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = F.silu(x)
        return x


class DownSampleBlock(nn.Module):
    """下采样块：对特征图进行2倍下采样"""
    def __init__(self, in_channels, out_channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels, out_channels)
        self.res2 = ResidualBlock(out_channels, out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class VAEEncoder(nn.Module):
    """
    VAE编码器：将3×640×640的图像编码为4×80×80的潜在表示
    基于Rombach等人(2022)的潜在扩散模型
    下采样因子：8 (= 2^3)
    """
    def __init__(self, in_channels=3, z_channels=4, base_channels=64):
        super(VAEEncoder, self).__init__()
        
        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # 下采样路径
        self.down1 = DownSampleBlock(base_channels, base_channels*2)           # 320x320
        self.down2 = DownSampleBlock(base_channels*2, base_channels*4)         # 160x160
        self.down3 = DownSampleBlock(base_channels*4, base_channels*8)         # 80x80
        
        # 残差块
        self.mid_res1 = ResidualBlock(base_channels*8, base_channels*8)
        self.mid_res2 = ResidualBlock(base_channels*8, base_channels*8)
        
        # 输出层 - 直接输出潜在表示而非均值和方差
        self.out_conv = nn.Conv2d(base_channels*8, z_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # 初始卷积
        x = self.init_conv(x)
        
        # 下采样路径
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        
        # 中间残差连接
        x = self.mid_res1(x)
        x = self.mid_res2(x)
        
        # 输出层，直接输出潜在表示
        z = self.out_conv(x)
        
        # 应用缩放因子 (与LDM一致)
        z = z * 0.18215
        
        return z


class VAEDecoder(nn.Module):
    """
    VAE解码器：将4×80×80的潜在表示解码为3×640×640的RGB图像
    """
    def __init__(self, z_channels=4, out_channels=3, base_channels=64):
        super(VAEDecoder, self).__init__()
        
        # 初始卷积
        self.init_conv = nn.Conv2d(z_channels, base_channels*8, kernel_size=3, padding=1)
        
        # 上采样路径
        self.up1 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 4, 2, 1)  # 160x160
        self.res1_1 = ResidualBlock(base_channels*4, base_channels*4)
        self.res1_2 = ResidualBlock(base_channels*4, base_channels*4)
        
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, 2, 1)  # 320x320
        self.res2_1 = ResidualBlock(base_channels*2, base_channels*2)
        self.res2_2 = ResidualBlock(base_channels*2, base_channels*2)
        
        self.up3 = nn.ConvTranspose2d(base_channels*2, base_channels, 4, 2, 1)  # 640x640
        self.res3_1 = ResidualBlock(base_channels, base_channels)
        self.res3_2 = ResidualBlock(base_channels, base_channels)
        
        # 输出层
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, z):
        # 初始卷积
        x = self.init_conv(z)
        
        # 上采样路径
        x = self.up1(x)
        x = self.res1_1(x)
        x = self.res1_2(x)
        
        x = self.up2(x)
        x = self.res2_1(x)
        x = self.res2_2(x)
        
        x = self.up3(x)
        x = self.res3_1(x)
        x = self.res3_2(x)
        
        # 输出层
        x = self.out_conv(x)
        x = torch.tanh(x)  # 归一化到[-1, 1]范围
        
        return x


class VAE(nn.Module):
    """
    完整的VAE：包含编码器和解码器
    在HiFiVFS模型中，既需要将图像编码到潜在空间，也需要将潜在表示解码回RGB空间
    """
    def __init__(self, in_channels=3, out_channels=3, z_channels=4, base_channels=64):
        super(VAE, self).__init__()
        
        self.encoder = VAEEncoder(in_channels, z_channels, base_channels)
        self.decoder = VAEDecoder(z_channels, out_channels, base_channels)
        self.scale_factor = 0.18215  # 添加与LDM一致的缩放因子
    
    def encode(self, x):
        """编码图像到潜在空间"""
        return self.encoder(x)
    
    def decode(self, z):
        """解码潜在表示到图像空间"""
        # 首先除以缩放因子
        z = z / self.scale_factor
        return self.decoder(z)
    
    def forward(self, x, decode=False):
        """前向传播"""
        z = self.encode(x)
        if decode:
            return self.decode(z)
        return z
