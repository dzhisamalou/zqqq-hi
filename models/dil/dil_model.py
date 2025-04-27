import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F  

# 确保能找到cosface模块
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from models.cosface import IDExtractor, get_id_extractor
from .tokenizer import DetailedIdentityTokenizer

class DIL(nn.Module):
    """
    详细身份学习模块 (DIL)
    
    从源图像提取详细身份特征，并将其转换为tokens用于注入到UNet中
    """
    def __init__(self, id_extractor=None, token_dim=1024, cosface_path=None, device=None):
        """
        初始化DIL模块
        
        Args:
            id_extractor: ID特征提取器，如果为None则自动创建
            token_dim: token维度，应与UNet中cross attention的维度相同 (修改默认值为 1024)
            cosface_path: CosFace模型路径
            device: 运行设备
        """
        super(DIL, self).__init__()
        
        # 初始化设备
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化ID提取器
        if id_extractor is None:
            # 如果没有提供，则根据cosface_path创建
            if cosface_path is None:
                raise ValueError("必须提供 id_extractor 或 cosface_path")
            self.id_extractor = get_id_extractor(model_path=cosface_path)
        else:
            # 使用提供的ID提取器
            self.id_extractor = id_extractor
            
        # 初始化DIT
        self.tokenizer = DetailedIdentityTokenizer(input_channels=512, token_dim=token_dim)
        
        # 将模块移至指定设备
        self.to(self.device)
    
    def extract_features(self, source_image):
        """
        提取全局和详细身份特征
        
        Args:
            source_image: 源图像 [B, 3, H, W]
            
        Returns:
            tuple: (全局身份特征 [B, 512], 详细身份特征 [B, 512, 7, 7])
        """
        # 使用ID提取器获取全局和详细身份特征
        global_features, detailed_features = self.id_extractor(source_image, return_detailed=True)
        
        # 检查详细特征的尺寸是否符合要求，不符合则直接报错
        if detailed_features.shape[2:] != (7, 7):
            raise ValueError(f"详细特征的预期空间尺寸为 (7, 7)，但得到 {detailed_features.shape[2:]}")
        
        return global_features, detailed_features
    
    def tokenize(self, detailed_features):
        """
        将详细身份特征转换为tokens
        
        Args:
            detailed_features: 详细身份特征 [B, 512, 7, 7]
            
        Returns:
            tokens: 身份tokens [B, 49, token_dim]
        """
        return self.tokenizer(detailed_features)
    
    def expand_tokens_to_frames(self, tokens, num_frames=16):
        """
        将tokens扩展到多帧
        
        Args:
            tokens: 原始tokens [B, 49, token_dim]
            num_frames: 帧数，默认为16
            
        Returns:
            expanded_tokens: 扩展后的tokens [B*num_frames, 49, token_dim]
        """
        batch_size = tokens.shape[0]
        # 复制tokens以匹配帧数
        expanded_tokens = tokens.unsqueeze(1).expand(batch_size, num_frames, -1, -1)
        # 重塑为 [B*num_frames, 49, token_dim]
        expanded_tokens = expanded_tokens.reshape(batch_size * num_frames, -1, tokens.shape[-1])
        return expanded_tokens
    
    def forward(self, source_image, return_global=True, num_frames=None):
        """
        DIL模块的前向传播
        
        Args:
            source_image: 源图像 [B, 3, H, W]
            return_global: 是否返回全局身份特征
            num_frames: 如果不为None，则扩展tokens到指定帧数
            
        Returns:
            如果 return_global 为 True: 
                如果 num_frames 为 None: (tokens [B, 49, token_dim], 全局身份特征 [B, 512])
                如果 num_frames 不为 None: (tokens [B*num_frames, 49, token_dim], 全局身份特征 [B, 512])
            如果 return_global 为 False:
                如果 num_frames 为 None: tokens [B, 49, token_dim]
                如果 num_frames 不为 None: tokens [B*num_frames, 49, token_dim]
        """
        global_features, detailed_features = self.extract_features(source_image)
        tokens = self.tokenize(detailed_features)
        
        # 如果指定了num_frames，扩展tokens到多帧
        if num_frames is not None:
            tokens = self.expand_tokens_to_frames(tokens, num_frames)
        
        if return_global:
            return tokens, global_features
        else:
            return tokens
    
    def compute_identity_loss(self, fgid, output_fgid):
        """
        计算身份损失 (例如，使用余弦相似度)
        
        Args:
            fgid: 原始全局身份特征 [B, 512]
            output_fgid: 输出图像的全局身份特征 [B, 512]
            
        Returns:
            identity_loss: 标量损失值
        """
        # 简单的余弦相似度损失 (1 - cos_sim)
        loss = 1 - F.cosine_similarity(fgid, output_fgid, dim=1).mean()
        return loss
