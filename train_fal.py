import os
import sys
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import cv2
from tqdm import tqdm
import logging

# 导入自定义模块
from models.fal.fal_model import FALModel
from models.fal.encoder import AttributeEncoder
from models.fal.decoder import AttributeDecoder
from models.fal.discriminator import Discriminator
from models.fal.losses import FALLoss, IdentityLoss
from utils.dataset import FrameSequenceDataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_fal.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ID特征提取器
class IDExtractor:
    def __init__(self, model_path=None, device='cuda'):
        """
        初始化ID特征提取器
        
        Args:
            model_path: 预训练模型路径
            device: 运行设备
        """
        self.device = device
        
        # 这里应该加载预训练的人脸识别模型
        # 为简化示例，我们使用一个随机初始化的模型
        # 实际应用中，应该使用如ArcFace, CosFace等预训练模型
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512)
        ).to(device)
        
        # 设置为评估模式
        self.model.eval()
        
    def __call__(self, images):
        """
        提取ID特征
        
        Args:
            images: 输入图像，形状为[B, 3, H, W]
            
        Returns:
            ID特征，形状为[B, 512]
        """
        with torch.no_grad():
            features = self.model(images)
            # 正则化特征
            features = F.normalize(features, p=2, dim=1)
        return features

# 训练函数
def train(args):
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((80, 80)),  # 调整为80x80的大小，匹配VAE潜在空间
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 创建数据集和数据加载器
    dataset = FrameSequenceDataset(
        data_root=args.data_dir, 
        transform=transform, 
        n_frames=args.n_frames
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    
    # 创建ID特征提取器
    id_extractor = IDExtractor(device=device)
    
    # 创建FAL模型 - 使用4通道输入以匹配VAE潜在空间
    fal_model = FALModel(
        id_feature_dim=512,
        detailed_id_dim=2048,
        in_channels=4,  # 调整为VAE潜在空间的4通道
        base_channels=64
    ).to(device)
    
    # 创建优化器
    optimizer_G = optim.Adam(
        list(fal_model.encoder.parameters()) + list(fal_model.decoder.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999)
    )
    optimizer_D = optim.Adam(
        fal_model.discriminator.parameters(),
        lr=args.lr,
        betas=(0.5, 0.999)
    )
    
    # 创建学习率调度器
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    
    # 创建结果目录
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'samples'), exist_ok=True)
    
    # 训练循环
    total_steps = 0
    warmup = args.warmup
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # 加载数据
        for batch_idx, videos in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            total_steps += 1
            
            # 如果达到预热步数，关闭预热模式
            if total_steps == args.warmup_steps:
                warmup = False
                logger.info("预热阶段结束，进入主训练阶段")
            
            # 将数据移到设备上
            videos = videos.to(device)  # [B, F, C, H, W]
            
            batch_size = videos.shape[0]
            
            # 随机选择一个源ID（从同批次的视频第一帧中选择）
            source_indices = torch.randperm(batch_size)[:batch_size]
            source_videos = videos[source_indices, 0]  # 使用每个视频的第一帧作为源
            
            # 提取源ID特征
            source_id_features = id_extractor(source_videos)
            
            # 训练判别器
            optimizer_D.zero_grad()
            
            # 前向传播 - 直接传递整个视频 [B, F, C, H, W]，不再使用face_masks
            results = fal_model(videos, source_id_features, id_extractor, warmup=warmup)
            
            # 计算判别器损失
            d_loss = results['discriminator_losses']['total']
            
            # 反向传播和优化
            d_loss.backward()
            optimizer_D.step()
            
            # 训练生成器
            optimizer_G.zero_grad()
            
            # 前向传播（重新计算结果，因为判别器已更新）
            results = fal_model(videos, source_id_features, id_extractor, warmup=warmup)
            
            # 计算生成器损失
            g_loss = results['generator_losses']['total']
            
            # 反向传播和优化
            g_loss.backward()
            optimizer_G.step()
            
            # 输出损失信息
            if batch_idx % args.print_freq == 0:
                time_per_step = (time.time() - epoch_start_time) / (batch_idx + 1)
                remaining_time = time_per_step * (len(dataloader) - batch_idx - 1)
                
                logger.info(f"Epoch {epoch+1}/{args.epochs} Batch {batch_idx}/{len(dataloader)} "
                           f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} "
                           f"Attr: {results['generator_losses']['attr'].item():.4f} "
                           f"Tid: {results['generator_losses']['tid'].item():.4f} "
                           f"Rec: {results['generator_losses']['rec'].item():.4f} "
                           f"Time: {time_per_step:.2f}s/step ETA: {remaining_time:.2f}s")
            
            # 保存样本
            if total_steps % args.sample_freq == 0:
                with torch.no_grad():
                    # 获取原始视频和重建视频的第一帧
                    original = videos[0, 0].cpu()
                    reconstructed = results['reconstructed_video'][0].cpu()
                    
                    # 将张量转换为numpy数组
                    original = ((original * 0.5 + 0.5) * 255).permute(1, 2, 0).numpy().astype(np.uint8)
                    reconstructed = ((reconstructed * 0.5 + 0.5) * 255).permute(1, 2, 0).numpy().astype(np.uint8)
                    
                    # 拼接原始图像和重建图像
                    result = np.concatenate([original, reconstructed], axis=1)
                    
                    # 保存图像
                    cv2.imwrite(os.path.join(args.out_dir, 'samples', f'step_{total_steps}.jpg'), 
                               cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        
        # 更新学习率
        scheduler_G.step()
        scheduler_D.step()
        
        # 保存检查点
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.out_dir, 'checkpoints', f'fal_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': fal_model.encoder.state_dict(),
                'decoder_state_dict': fal_model.decoder.state_dict(),
                'discriminator_state_dict': fal_model.discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'scheduler_G_state_dict': scheduler_G.state_dict(),
                'scheduler_D_state_dict': scheduler_D.state_dict(),
            }, checkpoint_path)
            logger.info(f"保存检查点到 {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练FAL模块")
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='datasets', help='包含视频帧序列的数据目录')
    parser.add_argument('--n_frames', type=int, default=16, help='每个视频的帧数')
    
    # 模型参数
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4, help='批量大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='学习率衰减步数')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5, help='学习率衰减系数')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--warmup', action='store_true', help='是否进行预热训练')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='预热步数')
    
    # 输出参数
    parser.add_argument('--out_dir', type=str, default='results/fal', help='输出目录')
    parser.add_argument('--print_freq', type=int, default=10, help='打印频率')
    parser.add_argument('--sample_freq', type=int, default=100, help='采样频率')
    parser.add_argument('--save_freq', type=int, default=5, help='保存频率（轮数）')
    
    args = parser.parse_args()
    
    # 开始训练
    train(args)
