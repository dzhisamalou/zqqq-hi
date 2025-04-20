import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import ResidualBlock, CrossAttention


class EnhancedUpSampleBlock(nn.Module):
    """增强上采样块：包含两个交叉注意力和两个ResBlock，并进行上采样"""
    def __init__(self, in_channels, out_channels, id_dim, use_cross_attn=True):
        super(EnhancedUpSampleBlock, self).__init__()
        self.use_cross_attn = use_cross_attn
        
        if use_cross_attn:
            self.cross_attn1 = CrossAttention(in_channels, id_dim)
            self.res_block1 = ResidualBlock(in_channels)
            self.cross_attn2 = CrossAttention(in_channels, id_dim)
            self.res_block2 = ResidualBlock(in_channels)
        
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, id_feature=None):
        if self.use_cross_attn and id_feature is not None:
            # 第一个交叉注意力和ResBlock
            x = self.cross_attn1(x, id_feature)
            x = self.res_block1(x)
            
            # 第二个交叉注意力和ResBlock
            x = self.cross_attn2(x, id_feature)
            x = self.res_block2(x)
        
        # 上采样    
        x = self.up_conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x


class AttributeDecoder(nn.Module):
    """
    属性解码器 (Dec):
    3层网络:
    - 前2层: 每层包含两个交叉注意力（融合ID特征）和两个ResBlock，然后进行转置卷积上采样
    - 第3层: 转置卷积 + Tanh激活，输出潜在表示
    """
    def __init__(self, id_feature_dim=512, base_channels=320, vae_decoder=None):
        super(AttributeDecoder, self).__init__()
        
        # 第一层上采样 (输入: [B, 1280, 20, 20] -> 输出: [B, 640, 40, 40])
        self.layer1 = EnhancedUpSampleBlock(base_channels*4, base_channels*2, id_feature_dim, True)
        
        # 第二层上采样 (输入: [B, 640, 40, 40] -> 输出: [B, 320, 80, 80])
        self.layer2 = EnhancedUpSampleBlock(base_channels*2, base_channels, id_feature_dim, True)
        
        # 第三层：输出层，不使用交叉注意力，最终转换到潜在表示
        self.layer3 = nn.ConvTranspose2d(base_channels, 4, kernel_size=1, stride=1)
        
        # 详细ID特征处理
        self.detailed_id_conv = nn.Conv2d(512, base_channels*4, kernel_size=1)
        
        # 激活函数
        self.tanh = nn.Tanh()
        
        # VAE解码器，用于将潜在表示转换回像素空间
        self.vae_decoder = vae_decoder
        
    def set_vae_decoder(self, vae_decoder):
        """设置VAE解码器"""
        self.vae_decoder = vae_decoder
    
    def forward(self, fattr, id_feature, detailed_id=None, return_pixel_space=False):
        """
        前向传播
        
        Args:
            fattr: 属性特征 [B, 1280, 20, 20]
            id_feature: ID特征
            detailed_id: 详细ID特征 [B, 512, 7, 7]
            return_pixel_space: 是否返回像素空间的结果
            
        Returns:
            如果return_pixel_space=False: 潜在空间的结果 [B, 4, 80, 80]
            如果return_pixel_space=True: 潜在空间的结果和像素空间的结果 tuple([B, 4, 80, 80], [B, 3, 640, 640])
        """
        # 如果提供了详细ID特征，处理并与fattr融合
        if detailed_id is not None:
            # 处理detailed_id以匹配fattr的空间维度
            detailed_id_feat = F.interpolate(detailed_id, size=(20, 20), mode='bilinear', align_corners=True)
            detailed_id_feat = self.detailed_id_conv(detailed_id_feat)
            
            # 将详细ID特征与fattr融合 (可以使用不同的融合策略)
            fattr = fattr + detailed_id_feat * 0.1  # 使用加权融合
        
        # 第一层上采样，融合ID特征
        x = self.layer1(fattr, id_feature)
        
        # 第二层上采样，融合ID特征
        x = self.layer2(x, id_feature)
        
        # 第三层上采样，输出到原始大小和通道数
        x = self.layer3(x)
        
        # 应用Tanh激活，得到潜在表示
        latent_output = self.tanh(x)  # [B, 4, 80, 80]
        
        # 无论return_pixel_space为何值，都生成像素空间结果，确保处理流程完整
        pixel_output = None
        if self.vae_decoder is not None:
            # 使用VAE解码器将潜在表示转换回像素空间
            with torch.no_grad():
                pixel_output = self.vae_decoder(latent_output)  # [B, 3, 640, 640]
        
        if return_pixel_space:
            if pixel_output is None:
                raise ValueError("VAE decoder is not set, cannot return pixel space result")
            return latent_output, pixel_output
        
        return latent_output
