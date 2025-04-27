import torch
import torch.nn as nn
import torch.nn.functional as F

class DetailedIdentityTokenizer(nn.Module):
    """
    详细身份特征Tokenizer (DIT)
    将详细身份特征(fdid)转换为tokens，以便注入到UNet的cross attention中
    
    输入: 详细身份特征 [B, C, H, W]，例如从CosFace的最后一个Res-Block层提取的特征 [B, 512, 7, 7]
    输出: 身份tokens [B, H*W, D]，其中D是cross attention的维度
    """
    def __init__(self, input_channels=512, token_dim=1024):
        """
        初始化DIT
        
        Args:
            input_channels: 输入特征通道数
            token_dim: token维度，通常与UNet中cross attention的维度相同 (修改默认值为 1024)
        """
        super(DetailedIdentityTokenizer, self).__init__()
        
        # 用卷积层调整通道数，以匹配cross attention的维度
        self.channel_conv = nn.Conv2d(input_channels, token_dim, kernel_size=1)
        
        # Token维度
        self.token_dim = token_dim
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 卷积层初始化，使用线性初始化而非ReLU
        nn.init.kaiming_normal_(self.channel_conv.weight, mode='fan_in', nonlinearity='linear')
        if self.channel_conv.bias is not None:
            nn.init.constant_(self.channel_conv.bias, 0)
    
    def forward(self, fdid):
        """
        将详细身份特征转换为tokens
        
        Args:
            fdid: 详细身份特征 [B, C, H, W]
            
        Returns:
            tokens: 身份tokens [B, H*W, D]
        """
        batch_size = fdid.shape[0]
        
        # 使用卷积调整通道数
        x = self.channel_conv(fdid)  # [B, token_dim, H, W]
        
        # 调整维度 [B, token_dim, H, W] -> [B, token_dim, H*W] -> [B, H*W, token_dim]
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, token_dim]
        
        return x
