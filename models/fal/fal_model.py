import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .encoder import AttributeEncoder
from .decoder import AttributeDecoder
from .discriminator import Discriminator
from .losses import FALLoss, IdentityLoss, TripletIdentityLoss, AttributeLoss
from .vae import VAE

class FALModel(nn.Module):
    """
    Fine-grained Attribute Learning (FAL) 模型
    整合编码器、解码器和判别器，实现属性学习和ID注入
    
    支持处理带有时间维度的输入 [B, F, C, H, W]，其中B总是设置为1
    """
    def __init__(self, id_feature_dim=512, detailed_id_dim=2048, in_channels=4, base_channels=320, cosface_path=None):
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
        
        # ID特征提取器 (CosFace)
        self.id_extractor = get_id_extractor(model_path=cosface_path)
        
        # VAE 用于潜在空间和像素空间的转换
        self.vae = None
        
        # 用于存储其他视频的ID特征的缓存池
        self.id_feature_pool = []
        self.max_pool_size = 50  # 缓存池最大容量
    
    def set_vae(self, vae):
        """
        设置VAE，用于潜在空间和像素空间的转换
        
        Args:
            vae: 可以是VAE实例或预训练模型路径
        """
        if isinstance(vae, str):
            # 如果传入字符串，则加载预训练模型
            from .vae import VAE
            self.vae = VAE.from_pretrained(vae)
        else:
            # 否则直接使用传入的VAE实例
            self.vae = vae
        
        # 同时设置解码器中的VAE解码器
        self.decoder.set_vae_decoder(self.vae)
        
        # 将VAE移至相同设备
        if next(self.parameters()).is_cuda:
            self.vae = self.vae.cuda()
        
    def generate_frame_mask(self, batch_size, num_frames, height, width, global_step, warmup=False, p=0.5):
        """
        生成帧掩码，确保批次大小为1
        
        Args:
            batch_size: 批次大小（将被忽略，始终使用1）
            num_frames: 帧数
            height: 高度
            width: 宽度
            global_step: 当前训练步数
            warmup: 是否处于旧的热身阶段（可能被global_step覆盖）
            p: 随机掩码的概率
            
        Returns:
            帧掩码：0表示不使用属性特征，1表示使用
        """
        # 强制将批次大小设为1
        batch_size = 1
        
        # 训练初期（前5000步）固定为零掩码
        if global_step < 5000:
            return torch.zeros((batch_size, num_frames, height, width), dtype=torch.float32, device='cuda')
        # elif warmup: # 根据 global_step < 5000 的逻辑，此处的 warmup 参数可能不再需要或有其他用途
        #     # 热身阶段：全0掩码（不使用属性特征）
        #     return torch.zeros((batch_size, num_frames, height, width), dtype=torch.float32, device='cuda')
        else:
            # 主训练阶段（5000步之后）：以概率p随机为每个批次生成0或1
            mask = torch.zeros((batch_size, num_frames, height, width), dtype=torch.float32, device='cuda')
            if random.random() < p:
                mask[0, :, :, :] = 1.0
            return mask
    
    def get_random_id_feature(self, original_id, original_detailed_id):
        """
        获取随机ID特征，有50%概率返回原始ID特征，50%概率返回其他视频的ID特征
        
        Args:
            original_id: 原始全局ID特征 [B, id_dim]
            original_detailed_id: 原始详细ID特征 [B, detailed_id_dim, H, W]
            
        Returns:
            id_feature_to_use: 要使用的ID特征
            detailed_id_to_use: 要使用的详细ID特征
            is_same_id: 是否使用相同的ID
        """
        # 随机决定是否使用相同的ID（训练技巧）
        use_same_id = (random.random() < 0.5) if self.training else False
        
        if use_same_id or not self.id_feature_pool:
            # 使用原始ID特征
            return original_id, original_detailed_id, True
        else:
            # 从ID特征池中随机选择一个特征
            random_idx = random.randint(0, len(self.id_feature_pool) - 1)
            random_id, random_detailed_id = self.id_feature_pool[random_idx]
            return random_id, random_detailed_id, False
    
    def update_id_feature_pool(self, id_feature, detailed_id_feature):
        """
        更新ID特征缓存池
        
        Args:
            id_feature: 全局ID特征 [B, id_dim]
            detailed_id_feature: 详细ID特征 [B, detailed_id_dim, H, W]
        """
        # 添加新的ID特征到池中
        self.id_feature_pool.append((id_feature.detach(), detailed_id_feature.detach()))
        
        # 如果池的大小超过最大容量，移除最旧的特征
        if len(self.id_feature_pool) > self.max_pool_size:
            self.id_feature_pool.pop(0)
        
    def forward(self, target_video, source_id_feature=None, external_id_extractor=None, warmup=False, global_step=float('inf')):
        """
        前向传播，确保批次大小为1
        
        Args:
            target_video: 目标视频 [B, F, C, H, W] - 已经是VAE编码后的潜在表示，强制B=1
            source_id_feature: 源ID特征 [B, id_dim] 或None（使用内部id_extractor）
            external_id_extractor: 外部ID特征提取器，用于提取生成视频的ID
            warmup: 是否处于旧的热身阶段（可能被global_step覆盖）
            global_step: 当前训练步数，默认为无穷大（推理模式）
            
        Returns:
            结果字典，包含各种中间特征和损失
        """
        # 确保输入的批次大小为1
        if target_video.shape[0] != 1:
            raise ValueError(f"批次大小必须为1，当前为: {target_video.shape[0]}")
        
        batch_size = 1  # 强制设置批次大小为1
        num_frames, channels, height, width = target_video.shape[1:]
        
        # 确保VAE已经设置
        if self.vae is None:
            raise ValueError("VAE has not been set. Please call set_vae() before forward pass.")
        
        # 将潜在表示解码为RGB图像，以便CosFace提取ID特征
        with torch.no_grad():
            # 对每一帧进行解码
            rgb_frames = []
            for i in range(num_frames):
                frame_latent = target_video[:, i]  # [B, C, H, W]
                # 使用VAE将潜在表示解码为RGB图像
                frame_rgb = self.vae.decode(frame_latent)  # [B, 3, H*8, W*8]
                rgb_frames.append(frame_rgb)
            
            # 堆叠所有帧
            rgb_target_video = torch.stack(rgb_frames, dim=1)  # [B, F, 3, H*8, W*8]
            
            # 使用CosFace从第一帧RGB图像中提取ID特征
            first_rgb_frames = rgb_target_video[:, 0]  # [B, 3, H*8, W*8]
            
            # 使用内部或外部ID提取器
            id_extractor = external_id_extractor if external_id_extractor else self.id_extractor
            
            # 提取全局ID特征 (fgid) 和详细ID特征 (fdid)
            original_id, original_detailed_id = id_extractor(first_rgb_frames, return_detailed=True)
            
            # 更新ID特征池
            self.update_id_feature_pool(original_id, original_detailed_id)
        
        # 如果没有提供source_id_feature，则从内部提取或随机选择
        if source_id_feature is None:
            # 获取随机ID特征（可能是原始ID特征或其他视频的ID特征）
            id_feature_to_use, detailed_id_to_use, use_same_id = self.get_random_id_feature(original_id, original_detailed_id)
        else:
            # 使用提供的source_id_feature
            id_feature_to_use = source_id_feature
            detailed_id_to_use = original_detailed_id  # 如果外部提供source_id_feature，通常没有对应的detailed_id_feature
            use_same_id = False
        
        # 提取目标视频属性特征 - 直接传递整个视频 [B, F, C, H, W]
        fattr, flow = self.encoder(target_video)
        
        # 使用解码器生成带有新ID的视频 - 处理每一帧
        b_frames = batch_size * num_frames  # 此处b_frames应该等于num_frames，因为batch_size=1
        reconstructed_frames = []
        reconstructed_rgb_frames = []
        
        # 将fattr分割为每一帧
        fattr_per_frame = fattr.view(batch_size, num_frames, *fattr.shape[1:])
        
        for i in range(num_frames):
            frame_fattr = fattr_per_frame[:, i]  # [B, C, H, W]
            # 传递详细ID特征并要求返回像素空间的结果
            latent_out, rgb_out = self.decoder(frame_fattr, id_feature_to_use, 
                                              detailed_id=detailed_id_to_use, 
                                              return_pixel_space=True)
            reconstructed_frames.append(latent_out)
            reconstructed_rgb_frames.append(rgb_out)
        
        # 堆叠结果
        reconstructed_video = torch.stack(reconstructed_frames, dim=1)  # [B, F, C, H, W]
        reconstructed_video_flat = reconstructed_video.view(b_frames, *reconstructed_video.shape[2:])
        
        reconstructed_rgb_video = torch.stack(reconstructed_rgb_frames, dim=1)  # [B, F, 3, H*8, W*8]
        
        # 重新从生成的视频中提取属性特征
        fattr_prime, _ = self.encoder(reconstructed_video)
        
        # 提取生成视频的ID特征（用于损失计算）- 使用RGB图像的第一帧
        with torch.no_grad():
            reconstructed_first_rgb_frame = reconstructed_rgb_video[:, 0]  # [B, 3, H*8, W*8]
            reconstructed_id = id_extractor(reconstructed_first_rgb_frame)
        
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
        frame_mask = self.generate_frame_mask(batch_size, num_frames, height, width, global_step, warmup)
        
        # 整理结果
        results = {
            'reconstructed_video': reconstructed_video_flat,
            'reconstructed_rgb_video': reconstructed_rgb_video.view(b_frames, *reconstructed_rgb_video.shape[2:]),
            'fattr': fattr,
            'fattr_prime': fattr_prime,
            'flow': flow,
            'frame_mask': frame_mask,
            'original_id': original_id,  # fgid - 从RGB图像中提取
            'detailed_original_id': original_detailed_id,  # fdid
            'reconstructed_id': reconstructed_id,  # fgid' - 从解码后的RGB图像中提取
            'source_id': source_id_feature,  # frid
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
