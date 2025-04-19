import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .encoder import AttributeEncoder
from .decoder import AttributeDecoder
from .discriminator import Discriminator
from .losses import FALLoss, IdentityLoss


class FALModel(nn.Module):
    """
    Fine-grained Attribute Learning (FAL) 模型
    整合编码器、解码器和判别器，实现属性学习和ID注入
    
    支持处理带有时间维度的输入 [B, F, C, H, W]
    """
    def __init__(self, id_feature_dim=512, detailed_id_dim=2048, in_channels=4, base_channels=320):
        super(FALModel, self).__init__()
        
        # 编码器
        self.encoder = AttributeEncoder(in_channels, base_channels, with_temporal=True)
        
        # 解码器
        self.decoder = AttributeDecoder(id_feature_dim, base_channels)
        
        # 判别器
        self.discriminator = Discriminator(in_channels, base_channels)
        
        # 损失函数
        self.fal_loss = FALLoss()
        self.id_loss = IdentityLoss()
        
        # 是否在训练模式
        self.training_mode = True
        
    def generate_frame_mask(self, batch_size, num_frames, height, width, warmup=False, p=0.5):
        """
        生成帧掩码
        
        Args:
            batch_size: 批次大小
            num_frames: 帧数
            height: 高度
            width: 宽度
            warmup: 是否处于热身阶段
            p: 随机掩码的概率
            
        Returns:
            帧掩码：0表示不使用属性特征，1表示使用
        """
        if warmup:
            # 热身阶段：全0掩码（不使用属性特征）
            return torch.zeros((batch_size, num_frames, height, width), dtype=torch.float32, device='cuda')
        else:
            # 主训练阶段：以概率p随机为每个批次生成0或1
            mask = torch.zeros((batch_size, num_frames, height, width), dtype=torch.float32, device='cuda')
            for i in range(batch_size):
                if random.random() < p:
                    mask[i, :, :, :] = 1.0
            return mask
        
    def forward(self, target_video, source_id_feature, id_extractor, warmup=False):
        """
        前向传播
        
        Args:
            target_video: 目标视频 [B, F, C, H, W]
            source_id_feature: 源ID特征 [B, id_dim]
            id_extractor: ID特征提取器，用于提取生成视频的ID
            warmup: 是否处于热身阶段
            
        Returns:
            结果字典，包含各种中间特征和损失
        """
        batch_size, num_frames, channels, height, width = target_video.shape
        
        # 提取原始视频的ID特征 - 使用第一帧
        with torch.no_grad():
            first_frames = target_video[:, 0]  # [B, C, H, W]
            original_id = id_extractor(first_frames)
        
        # 提取目标视频属性特征 - 直接传递整个视频 [B, F, C, H, W]
        fattr, flow = self.encoder(target_video)
        
        # 随机决定是否使用相同的ID（训练技巧）
        use_same_id = (random.random() < 0.5) if self.training else False
        
        # 选择ID特征
        if use_same_id:
            id_feature_to_use = original_id
        else:
            id_feature_to_use = source_id_feature
        
        # 使用解码器生成带有新ID的视频 - 处理每一帧
        b_frames = batch_size * num_frames
        reconstructed_frames = []
        
        # 将fattr分割为每一帧
        fattr_per_frame = fattr.view(batch_size, num_frames, *fattr.shape[1:])
        
        for i in range(num_frames):
            frame_fattr = fattr_per_frame[:, i]  # [B, C, H, W]
            frame_out = self.decoder(frame_fattr, id_feature_to_use)
            reconstructed_frames.append(frame_out)
        
        reconstructed_video = torch.stack(reconstructed_frames, dim=1)  # [B, F, C, H, W]
        reconstructed_video_flat = reconstructed_video.view(b_frames, *reconstructed_video.shape[2:])
        
        # 重新从生成的视频中提取属性特征
        fattr_prime, _ = self.encoder(reconstructed_video)
        
        # 提取生成视频的ID特征（用于损失计算）- 使用第一帧
        with torch.no_grad():
            reconstructed_first_frame = reconstructed_video[:, 0]  # [B, C, H, W]
            reconstructed_id = id_extractor(reconstructed_first_frame)
        
        # 判别器预测
        if self.training:
            # 展平视频维度进行判别
            target_video_flat = target_video.view(b_frames, channels, height, width)
            
            fake_logits = self.discriminator(reconstructed_video_flat)
            real_logits = self.discriminator(target_video_flat)
        else:
            fake_logits = None
            real_logits = None
        
        # 生成帧掩码（用于控制属性特征的注入）
        frame_mask = self.generate_frame_mask(batch_size, num_frames, height, width, warmup)
        
        # 整理结果
        results = {
            'reconstructed_video': reconstructed_video_flat,
            'fattr': fattr,
            'fattr_prime': fattr_prime,
            'flow': flow,
            'frame_mask': frame_mask,
            'original_id': original_id,
            'reconstructed_id': reconstructed_id,
            'source_id': source_id_feature,
            'is_same_id': use_same_id
        }
        
        # 如果在训练模式下，计算损失
        if self.training:
            # 展平原始视频以便计算损失
            target_video_flat = target_video.view(b_frames, channels, height, width)
            
            # 计算FAL损失
            generator_losses = self.fal_loss(
                fattr, fattr_prime, 
                reconstructed_video_flat, target_video_flat,
                reconstructed_id, original_id, id_feature_to_use,
                fake_logits, use_same_id
            )
            
            # 计算判别器损失
            discriminator_losses = self.fal_loss(
                fattr, fattr_prime, 
                reconstructed_video_flat, target_video_flat,
                reconstructed_id, original_id, id_feature_to_use,
                fake_logits, use_same_id, real_logits
            )
            
            # 添加损失到结果
            results['generator_losses'] = generator_losses
            results['discriminator_losses'] = discriminator_losses
        
        return results
    
    def train(self, mode=True):
        """
        设置训练模式
        """
        self.training_mode = mode
        super().train(mode)
        return self
    
    def eval(self):
        """
        设置评估模式
        """
        self.training_mode = False
        return super().eval()
