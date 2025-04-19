import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class VAEEncoder(nn.Module):
    """
    变分自编码器（VAE）编码器
    将原始视频帧(640x640)编码为潜在表示(80x80)
    
    基于论文中使用的LatentDiffusion预处理
    """
    def __init__(self, pretrained_vae_path=None, device=None):
        super(VAEEncoder, self).__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载预训练的VAE模型
        # 注意：这里假设使用的是Stable Diffusion的VAE模型或类似结构
        try:
            from diffusers import AutoencoderKL
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
            
            # 如果提供了自定义权重路径，则加载
            if pretrained_vae_path is not None:
                self.vae.load_state_dict(torch.load(pretrained_vae_path, map_location=self.device))
                
            self.vae = self.vae.to(self.device)
            self.vae.eval()
            self.use_diffusers = True
        except (ImportError, Exception) as e:
            print(f"无法加载预训练VAE模型: {e}")
            # 使用简化版VAE作为备用
            self.use_diffusers = False
            self._init_simplified_vae()
    
    def _init_simplified_vae(self):
        """初始化简化版VAE模型"""
        # 这是一个非常简化的VAE编码器结构
        # 在实际应用中，应该使用完整的预训练VAE
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 640x640 -> 320x320
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 320x320 -> 160x160
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 160x160 -> 80x80
            nn.ReLU(),
            nn.Conv2d(256, A, kernel_size=3, stride=1, padding=1)  # 通道数转换到4通道
        )
    
    def encode(self, x):
        """
        将图像编码为潜在表示
        
        Args:
            x: 输入图像，形状为 [B, 3, H, W] 或 [B, F, 3, H, W]
            
        Returns:
            潜在表示，形状为 [B, 4, H/8, W/8] 或 [B, F, 4, H/8, W/8]
        """
        has_time_dim = x.dim() == 5
        
        if has_time_dim:
            # 视频输入: [B, F, 3, H, W]
            batch_size, frames = x.shape[0], x.shape[1]
            x_flat = x.view(-1, *x.shape[2:])  # [B*F, 3, H, W]
            
            # 编码每一帧
            if self.use_diffusers:
                with torch.no_grad():
                    latents = []
                    # 批量处理以提高效率
                    batch_size_frames = 8  # 每批处理的帧数
                    for i in range(0, x_flat.shape[0], batch_size_frames):
                        batch = x_flat[i:i+batch_size_frames].to(self.device)
                        # 归一化到[-1, 1]
                        if batch.min() >= 0 and batch.max() <= 1:
                            batch = batch * 2 - 1
                        latent_batch = self.vae.encode(batch).latent_dist.sample() * 0.18215
                        latents.append(latent_batch.cpu())
                    latent = torch.cat(latents, dim=0)
                    
                    # 恢复时间维度: [B, F, 4, H/8, W/8]
                    latent = latent.view(batch_size, frames, *latent.shape[1:])
            else:
                # 使用简化版VAE
                latent = self.encoder(x_flat)
                latent = latent.view(batch_size, frames, *latent.shape[1:])
        else:
            # 单帧输入: [B, 3, H, W]
            if self.use_diffusers:
                with torch.no_grad():
                    x_device = x.to(self.device)
                    # 归一化到[-1, 1]
                    if x_device.min() >= 0 and x_device.max() <= 1:
                        x_device = x_device * 2 - 1
                    latent = self.vae.encode(x_device).latent_dist.sample() * 0.18215
            else:
                # 使用简化版VAE
                latent = self.encoder(x)
                
        return latent
    
    def forward(self, x):
        """
        编码图像或视频
        
        Args:
            x: 输入图像或视频，形状为 [B, 3, H, W] 或 [B, F, 3, H, W]
            
        Returns:
            潜在表示
        """
        return self.encode(x)
