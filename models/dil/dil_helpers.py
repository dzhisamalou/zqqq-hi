import torch
import random

def random_frame_selector(video_frames):
    """
    从视频帧中随机选择一帧作为源帧
    
    Args:
        video_frames: 视频帧 [B, F, C, H, W]
        
    Returns:
        source_frames: 选中的源帧 [B, C, H, W]
    """
    batch_size, num_frames, channels, height, width = video_frames.shape
    
    selected_frames = []
    for b in range(batch_size):
        # 随机选择一帧
        frame_idx = random.randint(0, num_frames-1)
        selected_frame = video_frames[b, frame_idx]  # [C, H, W]
        selected_frames.append(selected_frame)
    
    # 在批次维度上堆叠
    source_frames = torch.stack(selected_frames, dim=0)  # [B, C, H, W]
    
    return source_frames