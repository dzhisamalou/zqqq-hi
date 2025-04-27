import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import DownSampleBlock

class AttributeEncoder(nn.Module):
    """
    属性编码器 (Eattr)：
    3层网络，每层包含卷积下采样、2个残差块和2个自注意力机制
    输出：
    1. fattr: 高级属性特征，用于解码和条件注入
    2. flow: 低级特征，用于与UNet结合
    
    输入: 视频潜在表示 [B, F, C, H, W]，F固定为16帧
    """
    def __init__(self, in_channels=4, base_channels=320, with_temporal=True):
        super(AttributeEncoder, self).__init__()
        
        # 初始卷积层，从输入通道映射到基础通道数
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=1)
        
        # 第一层下采样 (输入: [B, 320, 80, 80] -> 输出: [B, 320, 80, 80])
        self.layer1 = DownSampleBlock(base_channels, base_channels, downsample=False)
        
        # 第二层下采样 (输入: [B, 320, 80, 80] -> 输出: [B, 640, 40, 40])
        self.layer2 = DownSampleBlock(base_channels, base_channels*2)
        
        # 第三层下采样 (输入: [B, 640, 40, 40] -> 输出: [B, 1280, 20, 20])
        self.layer3 = DownSampleBlock(base_channels*2, base_channels*4)
        
        # 标记是否使用隐式的时间自注意力
        self.with_temporal = with_temporal
        
    def forward(self, x):
        """
        处理视频输入
        
        Args:
            x: 视频输入, 形状 [B, F, C, H, W]，F固定为16帧
            
        Returns:
            fattr: 高级属性特征
            flow: 低级特征
        """
        # 输入始终是视频，格式为 [B, F, C, H, W]
        batch_size = x.shape[0]
        frames = x.shape[1]  # 固定为16帧
        
        # 将时间维度展平到批次维度: [B*F, C, H, W]
        x = x.reshape(-1, *x.shape[2:])
        
        # 初始卷积
        x = self.init_conv(x)
        
        # 第一层
        feature1 = self.layer1(x)
        
        # 第二层
        feature2 = self.layer2(feature1)
        
        # 第三层
        feature3 = self.layer3(feature2)

        # 所有特征已经是 [B*F, C, H, W] 形状，直接使用即可
        
        # fattr是最后一层的输出，用于表示高层次属性特征
        fattr = feature3
        
        # flow应该只包含第一层特征
        flow = [feature1]
        
        return fattr, flow
