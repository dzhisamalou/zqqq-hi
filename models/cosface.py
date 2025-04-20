import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gdown
from pathlib import Path

class CosFaceBlock(nn.Module):
    """CosFace模型的基本构建块"""
    def __init__(self, in_channel, depth, stride):
        super(CosFaceBlock, self).__init__()
        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        self.residual = nn.Sequential(
            nn.Conv2d(in_channel, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(depth)
        )

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class CosFaceNet(nn.Module):
    """CosFace网络实现"""
    def __init__(self, block_class=CosFaceBlock, num_layers=[2, 2, 2, 2], feature_dim=512, embedding_size=512):
        super(CosFaceNet, self).__init__()
        # 输入层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        
        # Residual块
        self.layer1 = self._make_layer(block_class, 64, 64, num_layers[0], 1)
        self.layer2 = self._make_layer(block_class, 64, 128, num_layers[1], 2)
        self.layer3 = self._make_layer(block_class, 128, 256, num_layers[2], 2)
        self.layer4 = self._make_layer(block_class, 256, 512, num_layers[3], 2)
        
        # 全连接层
        self.fc5 = nn.Sequential(
            nn.Conv2d(512 * 7 * 7, feature_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_dim),
        )
        
        # 输出特征
        self.fc6 = nn.Linear(feature_dim, embedding_size, bias=False)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_channel, depth, num_blocks, stride):
        layers = []
        layers.append(block(in_channel, depth, stride))
        for i in range(num_blocks - 1):
            layers.append(block(depth, depth, 1))
        return nn.Sequential(*layers)

    def forward(self, x, return_detailed_features=False):
        # 输入层
        x = self.conv1(x)
        
        # Residual块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 提取详细特征 (detailed features) - 对应fdid
        detailed_features = x  # [B, 512, 7, 7]
        
        # 全局池化
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # FC层
        features = self.fc5(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)  # [B, 512]
        
        # 归一化特征
        normalized_features = F.normalize(features, p=2, dim=1)
        
        if return_detailed_features:
            return normalized_features, detailed_features
        else:
            return normalized_features


class IDExtractor(nn.Module):
    """ID特征提取器，封装CosFace模型"""
    def __init__(self, model_path=None, device=None):
        super(IDExtractor, self).__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建CosFace模型
        self.model = CosFaceNet()
        self.model.to(self.device)
        
        # 加载预训练权重
        if model_path is None:
            # 使用用户主目录下的默认路径
            home = str(Path.home())
            weights_dir = os.path.join(home, ".deepface", "weights", "cosface")
            model_path = os.path.join(weights_dir, "cosface_pretrained.pth")
            
            if not os.path.exists(model_path):
                os.makedirs(weights_dir, exist_ok=True)
                # 从公共仓库下载预训练权重 (这里使用了示例URL，实际中需替换为真实的URL)
                url = "https://github.com/username/cosface-model/raw/master/cosface_pretrained.pth"
                try:
                    gdown.download(url, model_path, quiet=True)
                except Exception as e:
                    print(f"无法下载预训练权重: {e}")
        
        # 加载权重文件
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"成功加载CosFace预训练权重: {model_path}")
            except Exception as e:
                print(f"加载CosFace预训练权重失败: {e}")
        else:
            print(f"找不到CosFace预训练权重: {model_path}")
        
        # 设置为评估模式
        self.model.eval()
    
    def extract_features(self, face_images, return_detailed=False):
        """
        从人脸图像中提取ID特征
        
        Args:
            face_images: 人脸图像 [B, 3, H, W]
            return_detailed: 是否返回详细特征
            
        Returns:
            如果return_detailed为False: 全局ID特征 [B, 512]
            如果return_detailed为True: (全局ID特征 [B, 512], 详细ID特征 [B, 512, 7, 7])
        """
        with torch.no_grad():
            if return_detailed:
                global_features, detailed_features = self.model(face_images, return_detailed_features=True)
                return global_features, detailed_features
            else:
                global_features = self.model(face_images)
                return global_features
    
    def forward(self, face_images, return_detailed=False):
        """
        前向传播函数
        
        Args:
            face_images: 人脸图像 [B, 3, H, W]
            return_detailed: 是否返回详细特征
            
        Returns:
            如果return_detailed为False: 全局ID特征 [B, 512]
            如果return_detailed为True: (全局ID特征 [B, 512], 详细ID特征 [B, 512, 7, 7])
        """
        return self.extract_features(face_images, return_detailed)


# 创建全局ID提取器实例
def get_id_extractor(model_path=None, device=None):
    """
    获取ID特征提取器
    
    Args:
        model_path: 预训练模型路径
        device: 运行设备
        
    Returns:
        IDExtractor实例
    """
    return IDExtractor(model_path, device)
