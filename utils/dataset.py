import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import glob
import re

class FrameSequenceDataset(Dataset):
    """
    用于加载帧序列的数据集
    只从frames目录加载视频帧: datasets/video_name/frames/frame_*.png
    """
    def __init__(self, data_root='datasets', transform=None, n_frames=16, max_videos=None):
        """
        初始化数据集
        
        Args:
            data_root: 数据根目录，包含视频文件夹
            transform: 应用于图像的变换
            n_frames: 每个视频序列的帧数
            max_videos: 最大视频数量限制
        """
        self.data_root = data_root
        self.transform = transform
        self.n_frames = n_frames
        
        # 获取所有视频文件夹
        self.video_folders = []
        for folder in os.listdir(data_root):
            video_path = os.path.join(data_root, folder)
            frames_path = os.path.join(video_path, 'frames')
            
            # 检查该文件夹是否包含frames子文件夹
            if os.path.isdir(frames_path):
                # 检查是否有足够的帧
                frame_files = glob.glob(os.path.join(frames_path, 'frame_*.png'))
                if len(frame_files) >= n_frames:
                    self.video_folders.append(video_path)
        
        # 如果指定了最大视频数，则限制
        if max_videos is not None and max_videos < len(self.video_folders):
            self.video_folders = self.video_folders[:max_videos]
        
        print(f"找到 {len(self.video_folders)} 个有效视频文件夹")
    
    def __len__(self):
        return len(self.video_folders)
    
    def __getitem__(self, idx):
        video_folder = self.video_folders[idx]
        frames_path = os.path.join(video_folder, 'frames')
        
        # 获取所有帧文件并按序号排序
        frame_files = sorted(glob.glob(os.path.join(frames_path, 'frame_*.png')), 
                          key=lambda x: int(re.search(r'frame_(\d+)\.png', x).group(1)))
        
        # 如果帧数过多，则随机选择连续的n_frames帧
        if len(frame_files) > self.n_frames:
            start_idx = np.random.randint(0, len(frame_files) - self.n_frames)
            frame_files = frame_files[start_idx:start_idx + self.n_frames]
        
        # 加载帧
        frames = []
        for frame_file in frame_files:
            # 读取图像
            frame = cv2.imread(frame_file)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 应用变换
            if self.transform:
                frame = self.transform(frame)
            else:
                # 将图像转换为torch张量并归一化
                frame = torch.from_numpy(frame).float() / 255.0
                frame = frame.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            
            frames.append(frame)
        
        # 将帧堆叠为张量
        frames_tensor = torch.stack(frames)  # [n_frames, C, H, W]
        
        return frames_tensor

    def get_source_frame(self, idx, frame_idx=0):
        """获取指定视频的特定帧作为源帧"""
        video_folder = self.video_folders[idx]
        frames_path = os.path.join(video_folder, 'frames')
        
        # 获取所有帧文件并按序号排序
        frame_files = sorted(glob.glob(os.path.join(frames_path, 'frame_*.png')), 
                          key=lambda x: int(re.search(r'frame_(\d+)\.png', x).group(1)))
        
        if frame_idx >= len(frame_files):
            frame_idx = 0
            
        # 读取图像
        frame = cv2.imread(frame_files[frame_idx])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 应用变换
        if self.transform:
            frame = self.transform(frame)
        else:
            # 将图像转换为torch张量并归一化
            frame = torch.from_numpy(frame).float() / 255.0
            frame = frame.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            
        return frame
