import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeLoss(nn.Module):
    """
    属性损失：确保原始和重构的属性特征一致
    """
    def __init__(self):
        super(AttributeLoss, self).__init__()
        
    def forward(self, fattr, fattr_prime):
        """
        计算属性特征的L2损失
        
        Args:
            fattr: 原始视频的属性特征
            fattr_prime: 重构视频的属性特征
            
        Returns:
            属性损失：原始和重构属性特征之间的MSE
        """
        return 0.5 * F.mse_loss(fattr, fattr_prime)


class ReconstructionLoss(nn.Module):
    """
    重建损失：当使用相同ID时，确保重建的视频与原始视频一致
    """
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        
    def forward(self, reconstructed_video, original_video, is_same_id=False):
        """
        计算重建损失，仅当使用相同ID时
        
        Args:
            reconstructed_video: 重建的视频
            original_video: 原始视频
            is_same_id: 是否使用相同的ID特征
            
        Returns:
            重建损失：当使用相同ID时的MSE
        """
        if is_same_id:
            return 0.5 * F.mse_loss(reconstructed_video, original_video)
        else:
            return torch.tensor(0.0, device=reconstructed_video.device)


class TripletIdentityLoss(nn.Module):
    """
    三元组身份损失：当ID不同时，确保重建视频的ID与目标ID更接近而非原始ID
    """
    def __init__(self, margin=0.4):
        super(TripletIdentityLoss, self).__init__()
        self.margin = margin
        
    def forward(self, f_prime_gid, fgid, frid, is_same_id=False):
        """
        计算三元组ID损失
        
        Args:
            f_prime_gid: 重建视频的ID特征
            fgid: 原始视频的ID特征
            frid: 目标ID特征
            is_same_id: 是否使用相同的ID特征
            
        Returns:
            三元组损失: max(cos(f_prime_gid, fgid) - cos(f_prime_gid, frid) + margin, 0)
        """
        if not is_same_id:
            # 正则化特征
            f_prime_gid = F.normalize(f_prime_gid, p=2, dim=1)
            fgid = F.normalize(fgid, p=2, dim=1)
            frid = F.normalize(frid, p=2, dim=1)
            
            # 计算余弦相似度
            pos_cos = torch.sum(f_prime_gid * frid, dim=1)
            neg_cos = torch.sum(f_prime_gid * fgid, dim=1)
            
            # 计算三元组损失
            loss = torch.clamp(neg_cos - pos_cos + self.margin, min=0.0)
            return loss.mean()
        else:
            return torch.tensor(0.0, device=f_prime_gid.device)


class IdentityLoss(nn.Module):
    """
    身份损失：确保重建视频的ID与目标ID相似
    """
    def __init__(self):
        super(IdentityLoss, self).__init__()
        
    def forward(self, fgid, output_fgid):
        """
        计算身份损失
        
        Args:
            fgid: 目标ID特征
            output_fgid: 输出视频的ID特征
            
        Returns:
            身份损失: 1 - 余弦相似度
        """
        # 正则化特征
        fgid = F.normalize(fgid, p=2, dim=1)
        output_fgid = F.normalize(output_fgid, p=2, dim=1)
        
        # 计算余弦相似度
        cos_sim = torch.sum(fgid * output_fgid, dim=1)
        
        # 返回1-cos作为损失
        return (1 - cos_sim).mean()


class AdvLoss(nn.Module):
    """
    对抗损失：使用hinge loss实现GAN的对抗学习
    """
    def __init__(self):
        super(AdvLoss, self).__init__()
        
    def discriminator_loss(self, real_logits, fake_logits):
        """
        判别器损失
        
        Args:
            real_logits: 真实样本的判别结果
            fake_logits: 生成样本的判别结果
            
        Returns:
            判别器损失
        """
        # 使用hinge loss实现
        real_loss = torch.mean(F.relu(1.0 - real_logits))
        fake_loss = torch.mean(F.relu(1.0 + fake_logits))
        return real_loss + fake_loss
    
    def generator_loss(self, fake_logits):
        """
        生成器损失
        
        Args:
            fake_logits: 生成样本的判别结果
            
        Returns:
            生成器损失
        """
        return -torch.mean(fake_logits)


class FALLoss(nn.Module):
    """
    FAL模块的完整损失函数
    """
    def __init__(self, lambda_attr=10.0, lambda_tid=1.0, lambda_rec=10.0):
        super(FALLoss, self).__init__()
        self.lambda_attr = lambda_attr
        self.lambda_tid = lambda_tid
        self.lambda_rec = lambda_rec
        
        self.adv_loss = AdvLoss()
        self.attr_loss = AttributeLoss()
        self.tid_loss = TripletIdentityLoss()
        self.rec_loss = ReconstructionLoss()
        
    def forward(self, fattr, fattr_prime, reconstructed_video, original_video, 
                f_prime_gid, fgid, frid, fake_logits, is_same_id, real_logits=None):
        """
        计算FAL的综合损失
        
        Args:
            fattr: 原始视频的属性特征
            fattr_prime: 重构视频的属性特征
            reconstructed_video: 重建的视频
            original_video: 原始视频
            f_prime_gid: 重建视频的ID特征
            fgid: 原始视频的ID特征
            frid: 目标ID特征
            fake_logits: 生成样本的判别结果
            is_same_id: 是否使用相同的ID特征
            real_logits: 真实样本的判别结果（仅用于判别器训练）
            
        Returns:
            总损失和各部分损失的字典
        """
        # 计算各部分损失
        l_attr = self.attr_loss(fattr, fattr_prime)
        l_tid = self.tid_loss(f_prime_gid, fgid, frid, is_same_id)
        l_rec = self.rec_loss(reconstructed_video, original_video, is_same_id)
        
        # 计算对抗损失
        if real_logits is not None:
            # 判别器损失
            l_adv = self.adv_loss.discriminator_loss(real_logits, fake_logits)
        else:
            # 生成器损失
            l_adv = self.adv_loss.generator_loss(fake_logits)
        
        # 计算总损失
        total_loss = l_adv + self.lambda_attr * l_attr + self.lambda_tid * l_tid + self.lambda_rec * l_rec
        
        # 返回总损失和各部分损失
        return {
            'total': total_loss,
            'adv': l_adv,
            'attr': l_attr,
            'tid': l_tid,
            'rec': l_rec
        }
