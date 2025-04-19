import os
import cv2
import numpy as np
import random
from processors.face_preprocess import FaceProcessor

class VideoProcessor:
    def __init__(self, device='cuda', retinaface_model_path=None):
        """
        初始化视频处理器
        """
        self.device = device
        self.face_processor = FaceProcessor(device=device, retinaface_model_path=retinaface_model_path)
        
        # 设置视频参数
        self.output_fps = 25.0  # 输出视频的帧率
        
        # 设置处理的最大帧数，避免处理过长的视频
        self.max_frames = 64
    
    def process_video(self, video_path, random_clip=True):
        """
        处理视频，提取人脸
        
        Args:
            video_path: 视频文件路径
            random_clip: 是否随机提取片段，默认为True，尝试提取16帧
            
        Returns:
            aligned_frames: 对齐的人脸帧列表 (保证16帧，除非视频总帧数不足或人脸检测失败)
            vis_frames: 可视化的原始帧列表
            face_masks: 面部掩码列表 (将返回None，由后续流程生成)
        """
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频文件 {video_path}")
            return None, None, None
        
        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps}fps, 总帧数: {total_frames}")
        
        # 检测视频旋转信息
        rotation = self._detect_video_rotation(video_path)
        
        # --- 重试逻辑 ---
        max_attempts = 2 if random_clip else 1 # 如果是随机裁剪，最多尝试2次
        attempt = 0
        first_start_frame = -1 # 记录第一次尝试的起始帧
        
        while attempt < max_attempts:
            attempt += 1
            print(f"尝试 {attempt}/{max_attempts} 处理视频: {video_path}")
            
            # 重置帧列表和计数器
            aligned_frames = []
            vis_frames = []
            frame_count = 0
            success_count = 0
            
            # 如果需要随机提取片段 - 确保精确提取16帧
            if random_clip and total_frames >= 16:
                available_range = total_frames - 16
                if available_range <= 0: # 如果总帧数刚好16或更少
                    start_frame = 0
                elif attempt == 1: # 第一次尝试
                    start_frame = random.randint(0, available_range)
                    first_start_frame = start_frame
                else: # 第二次尝试，选择不同的起始帧
                    start_frame = random.randint(0, available_range)
                    # 确保与第一次不同 (如果可能)
                    if available_range > 0 and start_frame == first_start_frame:
                        start_frame = (start_frame + 1) % (available_range + 1)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                max_frames_to_process = 16
                print(f"尝试 {attempt}: 从第 {start_frame} 帧开始提取 {max_frames_to_process} 帧")
            else: # 不随机裁剪或总帧数不足16
                start_frame = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                max_frames_to_process = min(self.max_frames, total_frames) # 处理最多max_frames帧或所有帧
                # 如果不要求随机裁剪（即提取整个视频），则只尝试一次
                if not random_clip:
                    max_attempts = 1 
            
            # 处理视频帧
            while cap.isOpened() and frame_count < max_frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 保存原始帧以便可视化
                original_frame = frame.copy()
                
                # 处理视频旋转
                if rotation != 0:
                    frame = self._correct_rotation(frame, rotation)
                
                # 处理当前帧，传递旋转信息
                aligned_face, vis_frame, _ = self.face_processor.process_video_frame(
                    frame, 
                    original_frame=original_frame if rotation != 0 else None,
                    rotation=rotation
                )
                
                # 保存处理结果
                vis_frames.append(vis_frame)
                
                if aligned_face is not None:
                    # 确保对齐的面部图像尺寸为640x640，这是一个额外检查
                    if aligned_face.shape[0] != 640 or aligned_face.shape[1] != 640:
                        aligned_face = cv2.resize(aligned_face, (640, 640))
                    
                    aligned_frames.append(aligned_face)
                    success_count += 1
                
                # 更新计数
                frame_count += 1
            
            # 检查是否成功提取了16帧 (仅在random_clip模式下)
            if random_clip and success_count >= 16:
                print(f"尝试 {attempt}: 成功提取 {success_count} 帧，满足要求")
                break # 成功，跳出重试循环
            elif random_clip and attempt < max_attempts:
                print(f"尝试 {attempt}: 仅提取到 {success_count} 帧，准备重试")
                # 重置视频捕获到开头，以便下一次尝试设置正确的起始帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
            elif not random_clip: # 非随机裁剪模式，处理完就结束
                 print(f"尝试 {attempt}: 处理完成，提取到 {success_count} 帧")
                 break

        # --- 重试逻辑结束 ---
                
        # 释放视频
        cap.release()
                
        # 如果最终没有成功提取任何人脸，返回空结果
        if success_count == 0:
            print(f"处理失败: 未能从 {video_path} 提取到任何有效人脸帧")
            return None, vis_frames, None # 仍然返回vis_frames用于调试
        
        # 如果是随机裁剪模式，但最终帧数仍不足16，记录警告
        if random_clip and success_count < 16:
             print(f"警告: 经过 {max_attempts} 次尝试，最终只提取到 {success_count} 帧 (少于16帧) 来自 {video_path}")
             # 根据需求，可以选择返回None或者返回不足的帧
             # 这里选择返回不足的帧，让调用者决定如何处理
             # return None, vis_frames, None 
        
        # 返回 None 作为 face_masks，将由后续流程生成
        return aligned_frames, vis_frames, None
    
    def _detect_video_rotation(self, video_path):
        """检测视频旋转角度，需要使用外部工具如ffprobe
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            rotation: 旋转角度 (0, 90, 180, 或 270)
        """
        try:
            import subprocess
            import json
            
            # 使用ffprobe检查旋转元数据
            cmd = [
                'ffprobe', '-v', 'error', 
                '-select_streams', 'v:0', 
                '-show_entries', 'stream_tags=rotate', 
                '-of', 'json', 
                video_path
            ]
            
            try:
                output = subprocess.check_output(cmd).decode('utf-8')
                data = json.loads(output)
                rotation = int(data.get('streams', [{}])[0].get('tags', {}).get('rotate', 0))
                return rotation
            except (subprocess.SubprocessError, json.JSONDecodeError, ValueError, IndexError):
                # 尝试其他方法
                # 通过分析视频的几帧来检测方向 (简单版本)
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                if ret:
                    # 检查是否是竖屏视频 (高度明显大于宽度)
                    h, w = frame.shape[:2]
                    if h > w * 1.5:  # 如果高明显大于宽，可能是90°或270°旋转
                        return 90  # 假设为90度旋转
                cap.release()
                
                return 0  # 默认无旋转
        except Exception as e:
            print(f"检测视频旋转失败: {e}")
            return 0
    
    def _correct_rotation(self, frame, rotation):
        """根据旋转角度纠正帧方向
        
        Args:
            frame: 原始帧
            rotation: 旋转角度
            
        Returns:
            corrected_frame: 纠正后的帧
        """
        if rotation == 0:
            return frame
        elif rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270 or rotation == -90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        else:
            # 对于其他角度，使用更通用的旋转
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -rotation, 1.0)
            return cv2.warpAffine(frame, M, (w, h))
    
    def frames_to_video(self, frames, output_path):
        """
        将帧列表转换为视频文件
        
        Args:
            frames: 帧列表
            output_path: 输出视频路径
        """
        if not frames or len(frames) == 0:
            print(f"警告: 没有帧可以写入 {output_path}")
            return
            
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码
        out = cv2.VideoWriter(output_path, fourcc, self.output_fps, (width, height))
        
        for frame in frames:
            out.write(frame)
            
        out.release()
        print(f"已保存视频: {output_path}")
    
    def create_comparison_video(self, original_frames, aligned_frames, output_path):
        """
        创建原始帧和对齐帧的对比视频
        
        Args:
            original_frames: 原始帧列表
            aligned_frames: 对齐帧列表
            output_path: 输出视频路径
        """
        if not original_frames or not aligned_frames or len(original_frames) == 0 or len(aligned_frames) == 0:
            print(f"警告: 没有足够的帧可以创建对比视频 {output_path}")
            return
        
        # 获取原始帧和对齐帧的尺寸
        orig_h, orig_w = original_frames[0].shape[:2]
        align_h, align_w = aligned_frames[0].shape[:2]
        
        # 创建对比视频的尺寸
        comp_w = orig_w + align_w
        comp_h = max(orig_h, align_h)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.output_fps, (comp_w, comp_h))
        
        # 创建对比帧并写入视频
        for i in range(len(aligned_frames)):
            # 确保有对应的原始帧
            if i >= len(original_frames):
                break
                
            # 调整原始帧的大小，保持纵横比
            scale = min(comp_h / orig_h, (comp_w/2) / orig_w)
            resized_orig = cv2.resize(original_frames[i], None, fx=scale, fy=scale)
            
            # 创建空白背景
            comp_frame = np.zeros((comp_h, comp_w, 3), dtype=np.uint8)
            
            # 将原始帧放在左侧
            h, w = resized_orig.shape[:2]
            y_offset = (comp_h - h) // 2
            comp_frame[y_offset:y_offset+h, :w] = resized_orig
            
            # 将对齐帧放在右侧
            resized_align = cv2.resize(aligned_frames[i], (align_w, align_h))
            x_offset = orig_w
            comp_frame[:align_h, x_offset:x_offset+align_w] = resized_align
            
            # 添加标签
            cv2.putText(comp_frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(comp_frame, "Aligned Face", (orig_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 写入视频
            out.write(comp_frame)
        
        out.release()
        print(f"已保存对比视频: {output_path}")
    
    def create_mask_overlay_video(self, aligned_frames, face_masks, output_path):
        """
        创建面部掩码覆盖视频
        
        Args:
            aligned_frames: 对齐帧列表
            face_masks: 面部掩码列表
            output_path: 输出视频路径
        """
        if not aligned_frames or not face_masks or len(aligned_frames) == 0 or len(face_masks) == 0:
            print(f"警告: 没有足够的帧可以创建掩码视频 {output_path}")
            return
        
        # 获取帧尺寸
        h, w = aligned_frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.output_fps, (w, h))
        
        # 创建掩码覆盖帧并写入视频
        for i in range(min(len(aligned_frames), len(face_masks))):
            # 创建彩色掩码
            colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
            mask = face_masks[i]
            
            # 确保掩码与帧尺寸一致
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h))
                
            colored_mask[mask > 0] = [0, 255, 0]  # 使用绿色表示面部区域
            
            # 将掩码与原图像混合
            alpha = 0.3
            overlay_frame = cv2.addWeighted(aligned_frames[i], 1, colored_mask, alpha, 0)
            
            # 添加标签
            cv2.putText(overlay_frame, "Face Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 写入视频
            out.write(overlay_frame)
        
        out.release()
        print(f"已保存掩码覆盖视频: {output_path}")