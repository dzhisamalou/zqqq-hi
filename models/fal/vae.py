import torch
import torch.nn as nn
import os

# 使用条件导入，避免在不需要时导致错误
try:
    from diffusers import AutoencoderKL
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("警告: diffusers模块未找到，VAE功能将受限。如需完整功能，请安装: pip install diffusers")

class VAE(nn.Module):
    """
    VAE模型包装器：使用本地保存的Stable Diffusion 2.1的预训练VAE
    为HiFiVFS提供编码和解码能力
    """
    def __init__(self, pretrained_model_path=None):
        super(VAE, self).__init__()
        
        # 设置默认设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 检查是否可以加载diffusers
        if not DIFFUSERS_AVAILABLE:
            self.vae = None
            print("VAE初始化失败: 缺少diffusers模块")
            return
        
        # 如果未指定路径，直接使用项目中的权重目录
        if pretrained_model_path is None:
            pretrained_model_path = "D:\\zju\\code\\hifivfs111\\weights\\vae"
            
        try:
            # 尝试加载本地模型
            if os.path.exists(os.path.join(pretrained_model_path, "config.json")) and \
               os.path.exists(os.path.join(pretrained_model_path, "diffusion_pytorch_model.bin")):
                print(f"正在加载VAE模型: {pretrained_model_path}")
                self.vae = AutoencoderKL.from_pretrained(pretrained_model_path)
                print(f"成功加载VAE模型")
            else:
                # 如果本地文件不存在，抛出错误
                raise FileNotFoundError(f"VAE模型文件未找到，请确保模型文件位于: {pretrained_model_path}")
                
        except Exception as e:
            print(f"加载VAE模型失败: {e}")
            self.vae = None
            
        # 移动模型到指定设备
        if self.vae is not None:
            self.vae = self.vae.to(self.device)
    
    def normalize_input(self, x):
        """归一化输入图像到[-1, 1]范围"""
        if x.max() > 1.0:
            x = x / 255.0
        return 2.0 * x - 1.0
    
    def encode(self, x):
        """编码图像到潜在空间"""
        if self.vae is None:
            raise RuntimeError("VAE模型未正确初始化")
        
        x = self.normalize_input(x)
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.sample()
            return latent
    
    def decode(self, z):
        """解码潜在表示到图像空间"""
        if self.vae is None:
            raise RuntimeError("VAE模型未正确初始化")
            
        with torch.no_grad():
            decoded = self.vae.decode(z).sample
            return decoded
    
    def forward(self, x, decode=False):
        """前向传播"""
        z = self.encode(x)
        if decode:
            return self.decode(z)
        return z
    
    def to(self, device):
        """将模型移动到指定设备"""
        super().to(device)
        if self.vae is not None:
            self.vae = self.vae.to(device)
        self.device = device
        return self
