import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import UpSampleBlock


class AttributeDecoder(nn.Module):
    """
    属性解码器 (Dec):
    3层网络:
    - 前2层: 交叉注意力（融合ID特征）+ 转置卷积上采样
    - 第3层: 转置卷积 + Tanh激活
    """
    def __init__(self, id_feature_dim=512, base_channels=320):
        super(AttributeDecoder, self).__init__()
        
        # 第一层上采样 (输入: [B, 1280, 20, 20] -> 输出: [B, 640, 40, 40])
        self.layer1 = UpSampleBlock(base_channels*4, base_channels*2, id_feature_dim, True)
        
        # 第二层上采样 (输入: [B, 640, 40, 40] -> 输出: [B, 320, 80, 80])
        self.layer2 = UpSampleBlock(base_channels*2, base_channels, id_feature_dim, True)
        
        # 第三层上采样，不使用交叉注意力，最终恢复到原始大小和通道数
        self.layer3 = nn.ConvTranspose2d(base_channels, 4, kernel_size=1, stride=1)
        
        # 激活函数
        self.tanh = nn.Tanh()
        
    def forward(self, fattr, id_feature):
        # 第一层上采样，融合ID特征
        x = self.layer1(fattr, id_feature)
        
        # 第二层上采样，融合ID特征
        x = self.layer2(x, id_feature)
        
        # 第三层上采样，输出到原始大小和通道数
        x = self.layer3(x)
        
        # 应用Tanh激活
        x = self.tanh(x)
        
        return x
