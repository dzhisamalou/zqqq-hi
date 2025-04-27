import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import ResidualBlock, CrossAttention


class AttributeDecoder(nn.Module):
    """
    属性解码器 (Dec):
    3层网络:
    - 第1层: 残差块->交叉注意力->残差块->交叉注意力 (维持尺寸不变)
           输入: [B, 1280, 20, 20] -> 输出: [B, 1280, 20, 20]
    - 第2层: 残差块->交叉注意力->残差块->交叉注意力->转置卷积上采样
           输入: [B, 1280, 20, 20] -> 输出: [B, 640, 40, 40]
    - 第3层: 两个残差块+转置卷积上采样+1x1卷积
           输入: [B, 640, 40, 40] -> 输出: [B, 4, 80, 80]
    """
    def __init__(self, id_feature_dim=512, base_channels=320, in_channels=1280):
        super(AttributeDecoder, self).__init__()
        
        # 第一层：维持尺寸不变，只融合身份特征 [1280,20,20]->[1280,20,20]
        self.layer1_res1 = ResidualBlock(in_channels)
        self.layer1_cross1 = CrossAttention(in_channels, id_feature_dim)
        self.layer1_res2 = ResidualBlock(in_channels)
        self.layer1_cross2 = CrossAttention(in_channels, id_feature_dim)
        
        # 第二层：也按照相同结构，然后上采样至 [1280,20,20]->[640,40,40]
        self.layer2_res1 = ResidualBlock(in_channels)
        self.layer2_cross1 = CrossAttention(in_channels, id_feature_dim)
        self.layer2_res2 = ResidualBlock(in_channels)
        self.layer2_cross2 = CrossAttention(in_channels, id_feature_dim)
        self.layer2_up = nn.ConvTranspose2d(in_channels, base_channels*2, kernel_size=4, stride=2, padding=1)
        self.layer2_norm = nn.BatchNorm2d(base_channels*2)
        
        # 第三层：上采样至 [640,40,40]->[320,80,80]，再通过1x1卷积至 [4,80,80]
        self.layer3_res1 = ResidualBlock(base_channels*2)
        self.layer3_res2 = ResidualBlock(base_channels*2)
        self.layer3_up = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=4, stride=2, padding=1)
        self.layer3_norm = nn.BatchNorm2d(base_channels)
        self.layer3_conv = nn.Conv2d(base_channels, 4, kernel_size=1, stride=1)
    
    def forward(self, fattr, id_feature, detailed_id=None):
        """
        前向传播
        
        Args:
            fattr: 属性特征，每个样本维度 [1280, 20, 20]，实际输入形状 [B, 1280, 20, 20]
            id_feature: ID特征，每个样本维度 [512]，实际输入形状 [B, 512]
            detailed_id: 详细ID特征参数已保留但不再使用
            
        Returns:
            潜在空间的结果，每个样本维度 [4, 80, 80]，实际输出形状 [B, 4, 80, 80]
        """
        # 第一层：维持尺寸不变，融合身份特征 [1280,20,20]->[1280,20,20]
        x = self.layer1_res1(fattr)
        x = self.layer1_cross1(x, id_feature)
        x = self.layer1_res2(x)
        x = self.layer1_cross2(x, id_feature)
        
        # 第二层：按照与第一层相同的结构，然后上采样至 [1280,20,20]->[640,40,40]
        x = self.layer2_res1(x)
        x = self.layer2_cross1(x, id_feature)
        x = self.layer2_res2(x)
        x = self.layer2_cross2(x, id_feature)
        x = self.layer2_up(x)
        x = self.layer2_norm(x)
        
        # 第三层：上采样至 [640,40,40]->[320,80,80]，再通过1x1卷积至 [4,80,80]
        x = self.layer3_res1(x)
        x = self.layer3_res2(x)
        x = self.layer3_up(x)
        x = self.layer3_norm(x)
        x = self.layer3_conv(x)

        return x  # [B, 4, 80, 80]
