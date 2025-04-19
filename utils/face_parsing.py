import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import gdown
from pathlib import Path

# 添加对BiSeNet模型的直接导入
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))
from models.bisenet_model import BiSeNet
from torchvision import transforms

class FaceParser:
    def __init__(self, model_path=None, device=None):
        """
        初始化面部解析模型，使用预训练的BiseNet模型
        
        Args:
            model_path: 预训练模型路径，如果为None则使用默认路径
            device: 运行设备，如果为None则根据是否有CUDA自动选择
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型路径
        if model_path is None:
            # 修改为使用用户主目录下的.deepface/weights目录
            home = str(Path.home())
            weights_dir = os.path.join(home, ".deepface", "weights", "face_parsing")
            model_path = os.path.join(weights_dir, "bisenet_model.pt")
            if not os.path.exists(model_path):
                # 尝试查找项目中的模型文件
                project_weight_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                                "weights", "79999_iter.pth")
                if os.path.exists(project_weight_path):
                    model_path = project_weight_path
                else:
                    # 尝试下载预训练权重
                    try:
                        os.makedirs(weights_dir, exist_ok=True)
                        url = "https://github.com/zllrunning/face-parsing.PyTorch/raw/master/res/cp/79999_iter.pth"
                        gdown.download(url, model_path, quiet=True)
                    except Exception:
                        pass
        
        # 加载模型
        try:
            self.net = BiSeNet(n_classes=19)
            self.net.to(self.device)
            
            if os.path.exists(model_path):
                try:
                    state_dict = torch.load(model_path, map_location=self.device)
                    self.net.load_state_dict(state_dict)
                    self.use_model = True
                except Exception:
                    self.use_model = False
            else:
                self.use_model = False
            
            if self.use_model:
                self.net.eval()
        except Exception:
            self.net = None
            self.use_model = False
        
        # 定义预处理变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # 面部区域的类别标签
        self.face_parts = [1, 2, 3, 4, 5, 10, 11, 12, 13]  # 面部区域的类别
    
    def parse(self, image):
        """
        解析人脸，返回面部区域掩码
        
        Args:
            image: 输入图像
            
        Returns:
            mask: 面部区域掩码，255表示面部区域，0表示背景
        """
        if image is None:
            return np.zeros((100, 100), dtype=np.uint8)  # 返回空掩码
            
        if not self.use_model or self.net is None:
            # 如果模型不可用，返回全黑掩码
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)
        
        # 调整图像大小
        h, w = image.shape[:2]
        img = cv2.resize(image, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 转换为tensor
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        try:
            # 前向传播
            with torch.no_grad():
                out = self.net(img_tensor)[0]
                parsing = out.softmax(dim=1).argmax(1)[0].cpu().numpy()
            
            # 创建面部掩码
            mask = np.zeros_like(parsing)
            for part in self.face_parts:
                mask = np.logical_or(mask, parsing == part)
            
            # 将掩码转换为0-255
            mask = mask.astype(np.uint8) * 255
            
            # 形态学操作，填充小洞
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 调整回原始大小
            mask = cv2.resize(mask, (w, h))
            
            return mask
        except Exception:
            # 出错时返回全黑掩码
            return np.zeros((h, w), dtype=np.uint8)