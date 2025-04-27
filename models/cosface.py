import torch
import torch.nn.functional as F
import os
from pathlib import Path
import warnings
import numpy as np

# 导入insightface库
try:
    import insightface
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    warnings.warn("insightface未安装，请使用pip install insightface安装")
    INSIGHTFACE_AVAILABLE = False


class IDExtractor(torch.nn.Module):
    """ID特征提取器，使用InsightFace库中的CosFace模型"""
    def __init__(self, model_path=None, device=None):
        super(IDExtractor, self).__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 面部图像大小配置
        self.face_size = (112, 112)  # InsightFace模型默认输入大小
        
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("请先安装insightface: pip install insightface")
        
        # 初始化模型配置 - 直接使用InsightFace自带的CosFace模型
        try:
            # 创建FaceAnalysis应用
            self.app = FaceAnalysis(allowed_modules=['recognition'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.app.prepare(ctx_id=0 if self.device == 'cuda' else -1)
            
            # 获取CosFace模型 - "CosFace: Large Margin Cosine Loss for Deep Face Recognition"论文实现
            self.model = get_model("cosface_r100")
            print("成功加载InsightFace CosFace模型 (Large Margin Cosine Loss)")
        except Exception as e:
            print(f"加载CosFace模型失败: {e}")
            raise
        
        # 确保模型在正确的设备上
        if self.device == 'cuda':
            self.model.to('cuda')
            
        # 初始化钩子相关属性
        self.hook_handles = []
        self.intermediate_features = {}
        # 只设置最后一个Res-Block的输出作为目标
        self.target_layer_name = 'layer4'
        self._register_hooks()
    
    def _register_hooks(self):
        """
        为模型注册钩子，捕获最后一个Res-Block输出
        """
        # 首先移除所有现有钩子
        self._remove_hooks()
        
        # 定义钩子函数
        def hook_fn(module, input, output):
            self.intermediate_features[self.target_layer_name] = output
        
        # 为CosFace模型的最后一个残差块注册钩子
        if hasattr(self.model, 'layer4'):
            handle = self.model.layer4.register_forward_hook(hook_fn)
            self.hook_handles.append(handle)
            print(f"已为CosFace模型的最后残差块 'layer4' 注册钩子")
        elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'layer4'):
            handle = self.model.backbone.layer4.register_forward_hook(hook_fn)
            self.hook_handles.append(handle)
            print(f"已为CosFace模型的backbone中的 'layer4' 注册钩子")
        else:
            raise AttributeError("CosFace模型中没有找到'layer4'层，无法获取中间层特征")
    
    def _remove_hooks(self):
        """移除所有已注册的钩子"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.intermediate_features = {}
    
    def _preprocess_images(self, images):
        """预处理面部图像，调整为模型要求的尺寸并格式化"""
        # 转换为CPU，然后转为numpy
        if images.device.type != 'cpu':
            images = images.cpu()
        
        # 转换为numpy数组
        images_np = images.permute(0, 2, 3, 1).numpy()
        
        # 确保值范围在[0, 255]
        if images_np.max() <= 1.0:
            images_np = images_np * 255.0
        
        # 转换为uint8类型
        images_np = images_np.astype(np.uint8)
        
        # 调整大小
        batch_size = images_np.shape[0]
        processed_images = []
        
        for i in range(batch_size):
            # 使用OpenCV调整大小
            import cv2
            img = cv2.resize(images_np[i], self.face_size)
            processed_images.append(img)
        
        return processed_images
    
    def extract_features(self, face_images, return_detailed=False):
        """
        从人脸图像中提取ID特征
        
        Args:
            face_images: 人脸图像 [B, 3, H, W]
            return_detailed: 是否返回详细特征
            
        Returns:
            如果return_detailed为False: 全局ID特征 [B, 512]
            如果return_detailed为True: (全局ID特征 [B, 512], 详细ID特征 [B, C, H, W])
        """
        # 预处理图像
        processed_images = self._preprocess_images(face_images)
        batch_size = len(processed_images)
        
        # 提取特征
        features = []
        with torch.no_grad():
            # 清空之前的中间特征
            self.intermediate_features = {}
            
            # 获取全局特征
            for img in processed_images:
                # 使用模型提取特征
                feat = self.model.get_feat(img)
                features.append(feat)
            
            # 转换为torch张量
            features = torch.from_numpy(np.stack(features)).to(self.device)
            
            # 如果需要返回详细特征
            if return_detailed:
                # 尝试获取中间层特征
                if self.target_layer_name in self.intermediate_features:
                    # 成功获取到中间层特征
                    detailed_features = self.intermediate_features[self.target_layer_name]
                    
                    # 确保详细特征是正确的形状和批次数量
                    if isinstance(detailed_features, list) and len(detailed_features) == batch_size:
                        detailed_features = torch.stack(detailed_features, dim=0)
                    
                    # 如果详细特征的批次维度不匹配，直接报错
                    if detailed_features.shape[0] != batch_size:
                        raise ValueError(f"详细特征批次大小不匹配 ({detailed_features.shape[0]} vs {batch_size})")
                    
                    # 如果详细特征是张量但需要转移到正确的设备
                    if detailed_features.device != self.device:
                        detailed_features = detailed_features.to(self.device)
                    
                    return features, detailed_features
                else:
                    # 如果没有成功获取到中间层特征，直接报错
                    raise RuntimeError(f"未能获取层 '{self.target_layer_name}' 的详细特征")
            else:
                return features
    
    def forward(self, face_images, return_detailed=False):
        """
        前向传播函数
        
        Args:
            face_images: 人脸图像 [B, 3, H, W]
            return_detailed: 是否返回详细特征
            
        Returns:
            如果return_detailed为False: 全局ID特征 [B, 512]
            如果return_detailed为True: (全局ID特征 [B, 512], 详细ID特征)
        """
        return self.extract_features(face_images, return_detailed)
    
    def __del__(self):
        """在对象销毁时移除所有钩子"""
        self._remove_hooks()


# 创建全局ID提取器实例
def get_id_extractor(model_path=None, device=None):
    """
    获取ID特征提取器
    
    Args:
        model_path: 已废弃参数，保留仅为兼容性
        device: 运行设备
        
    Returns:
        IDExtractor实例
    """
    if model_path is not None:
        print("警告: 自定义模型路径参数已被忽略，现在总是使用InsightFace自带的CosFace模型")
    return IDExtractor(device=device)
