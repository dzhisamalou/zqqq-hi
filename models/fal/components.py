import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    """残差块：用于编码器和判别器"""
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制：用于编码器和判别器，可以隐式处理时间关系"""
    def __init__(self, dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "维度必须能被头数整除"
        
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.LayerNorm([dim])
        
    def forward(self, x):
        batch_size, c, h, w = x.shape
        
        # 保存原始输入用于残差连接
        residual = x
        
        # LayerNorm需要调整维度
        x_norm = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_norm = self.norm(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 计算QKV
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(batch_size, 3, self.num_heads, self.head_dim, h * w)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, num_heads, H*W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 每个形状: [B, num_heads, H*W, head_dim]
        
        # 计算注意力 - 这里将有多个帧时，会隐式地在批次维度上处理时间关系
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, num_heads, H*W, H*W]
        attn = attn.softmax(dim=-1)
        
        # 应用注意力权重
        x = (attn @ v)  # [B, num_heads, H*W, head_dim]
        x = x.transpose(1, 2).reshape(batch_size, h*w, self.dim)  # [B, H*W, C]
        x = x.reshape(batch_size, h, w, self.dim).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 投影并添加残差连接
        x = self.proj(x)
        return x + residual


class CrossAttention(nn.Module):
    """交叉注意力机制：用于解码器中融合ID特征"""
    def __init__(self, q_dim, k_dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        assert self.head_dim * num_heads == q_dim, "维度必须能被头数整除"
        
        self.q_proj = nn.Conv2d(q_dim, q_dim, kernel_size=1)
        self.k_proj = nn.Linear(k_dim, q_dim)
        self.v_proj = nn.Linear(k_dim, q_dim)
        self.out_proj = nn.Conv2d(q_dim, q_dim, kernel_size=1)
        self.norm = nn.LayerNorm([q_dim])
        
    def forward(self, x, id_feature):
        batch_size, c, h, w = x.shape
        
        # 保存原始输入用于残差连接
        residual = x
        
        # LayerNorm需要调整维度
        x_norm = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x_norm = self.norm(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 计算Q
        q = self.q_proj(x_norm)
        q = q.reshape(batch_size, self.num_heads, self.head_dim, h * w)
        q = q.permute(0, 1, 3, 2)  # [B, num_heads, H*W, head_dim]
        
        # 计算K和V（来自ID特征）
        k = self.k_proj(id_feature)  # [B, feature_dim] -> [B, q_dim]
        v = self.v_proj(id_feature)  # [B, feature_dim] -> [B, q_dim]
        
        # 调整K和V的形状以匹配注意力操作
        k = k.view(batch_size, self.num_heads, self.head_dim, 1)
        k = k.permute(0, 1, 3, 2)  # [B, num_heads, 1, head_dim]
        
        v = v.view(batch_size, self.num_heads, self.head_dim, 1)
        v = v.permute(0, 1, 3, 2)  # [B, num_heads, 1, head_dim]
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, num_heads, H*W, 1]
        attn = attn.softmax(dim=-1)
        
        # 应用注意力权重
        x = (attn @ v)  # [B, num_heads, H*W, head_dim]
        x = x.transpose(1, 2).reshape(batch_size, h*w, self.q_dim)  # [B, H*W, C]
        x = x.reshape(batch_size, h, w, self.q_dim).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # 投影并添加残差连接
        x = self.out_proj(x)
        return x + residual


class DownSampleBlock(nn.Module):
    """下采样块：卷积降采样 + 残差块 + 自注意力"""
    def __init__(self, in_channels, out_channels, downsample=True):
        super(DownSampleBlock, self).__init__()
        
        # 下采样卷积，只在需要时应用
        if downsample:
            self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            # 当不需要下采样时，使用1x1卷积调整通道数
            self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            
        self.res1 = ResidualBlock(out_channels)
        self.attn1 = MultiHeadSelfAttention(out_channels)
        self.res2 = ResidualBlock(out_channels)
        self.attn2 = MultiHeadSelfAttention(out_channels)
    
    def forward(self, x):
        x = self.down_conv(x)
        x = self.res1(x)
        x = self.attn1(x)
        x = self.res2(x)
        x = self.attn2(x)
        return x


class UpSampleBlock(nn.Module):
    """上采样块：交叉注意力 + 转置卷积上采样"""
    def __init__(self, in_channels, out_channels, id_dim, use_cross_attn=True):
        super(UpSampleBlock, self).__init__()
        self.use_cross_attn = use_cross_attn
        
        if use_cross_attn:
            self.cross_attn = CrossAttention(in_channels, id_dim)
            
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, id_feature=None):
        if self.use_cross_attn and id_feature is not None:
            x = self.cross_attn(x, id_feature)
            
        x = self.up_conv(x)
        x = self.norm(x)
        x = F.relu(x)
        return x
