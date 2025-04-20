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
    
    输入: 视频潜在表示 [B, F, C, H, W] 或单帧 [B, C, H, W]
    """
    def __init__(self, in_channels=4, base_channels=320, with_temporal=True):
        super(AttributeEncoder, self).__init__()
        
        # 初始卷积层，从输入通道映射到基础通道数
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=1)
        
        # 第一层下采样 (输入: [B, 320, 80, 80] -> 输出: [B, 320, 80, 80])
        # 注意：这里保持相同的空间尺寸
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
            x: 视频输入, 形状 [B, F, C, H, W] 或单帧 [B, C, H, W]
            
        Returns:
            fattr: 高级属性特征
            flow: 低级特征
        """
        # 检测输入是否包含时间维度
        has_time_dim = x.dim() == 5
        batch_size = x.shape[0]
        
        if has_time_dim:
            # 视频输入: [B, F, C, H, W]
            frames = x.shape[1]
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
        
        # 预处理输出特征
        if has_time_dim:
            # 重新整理特征，恢复时间维度
            b_frames = batch_size * frames
            
            # 恢复时间维度 [B*F, C, H, W] -> [B, F, C, H, W]
            feature1 = feature1.view(batch_size, frames, *feature1.shape[1:])
            feature2 = feature2.view(batch_size, frames, *feature2.shape[1:])
            feature3 = feature3.view(batch_size, frames, *feature3.shape[1:])
            
            # 再次展平为 [B*F, C, H, W] 以供后续使用
            feature1 = feature1.view(b_frames, *feature1.shape[2:])
            feature2 = feature2.view(b_frames, *feature2.shape[2:])
            feature3 = feature3.view(b_frames, *feature3.shape[2:])
        
        # fattr是最后一层的输出，用于表示高层次属性特征
        fattr = feature3
        
        # flow应该只包含第一层特征
        flow = [feature1]
        
        return fattr, flow
