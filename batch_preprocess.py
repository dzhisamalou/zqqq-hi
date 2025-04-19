import os
import argparse
import torch
import traceback
import glob
import cv2
import numpy as np
from tqdm import tqdm
import warnings
import sys
import re
import logging
import shutil
from pathlib import Path
import concurrent.futures
from functools import partial

# 忽略特定警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from processors.video_processor import VideoProcessor
from utils.face_parsing import FaceParser

# 直接实现SuppressOutput类，而不是从utils.io_utils导入
class SuppressOutput:
    """
    上下文管理器，用于临时抑制标准输出和错误输出
    """
    def __init__(self):
        self.original_stdout = None
        self.original_stderr = None
        self.null_file = None
    
    def __enter__(self):
        # 保存原始的输出流
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # 创建一个空的输出流
        self.null_file = open(os.devnull, 'w')
        
        # 重定向标准输出和错误输出到空输出流
        sys.stdout = self.null_file
        sys.stderr = self.null_file
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 恢复原始的输出流
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # 关闭空输出流
        if self.null_file:
            self.null_file.close()

# 修改 SuppressOutput 类以同时抑制警告
class SuppressAllOutput(SuppressOutput):
    def __enter__(self):
        # 调用父类的方法
        super().__enter__()
        
        # 保存原始的 showwarning
        self.original_showwarning = warnings.showwarning
        # 替换为无操作函数
        warnings.showwarning = lambda *args, **kwargs: None
        
        return self
    
    def __exit__(self, *args):
        # 调用父类的方法
        super().__exit__(*args)
        
        # 恢复原始的 showwarning
        if hasattr(self, 'original_showwarning'):
            warnings.showwarning = self.original_showwarning

def setup_logger(output_dir):
    """
    设置日志记录器
    
    Args:
        output_dir: 输出目录，日志文件将保存在该目录下
        
    Returns:
        logger: 配置好的日志记录器
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建日志文件路径
    log_file = os.path.join(output_dir, 'processing_errors.log')
    
    # 配置日志记录器
    logger = logging.getLogger('video_processor')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    
    return logger

def process_video_to_frames(video_path, input_root, output_dir, device='cuda', retinaface_model_path=None, 
                            face_parsing_model_path=None, use_parsing=True, random_clip=True, logger=None):
    """
    处理单个视频并保存为帧序列，保持原始目录结构。
    如果最终处理得到的帧数少于16帧，则丢弃该视频。
    
    Args:
        video_path: 视频文件路径
        input_root: 输入根目录，用于计算相对路径
        output_dir: 输出目录
        device: 运行设备
        retinaface_model_path: RetinaFace模型路径
        face_parsing_model_path: 面部解析模型路径
        use_parsing: 是否使用面部解析生成更精确的掩码
        random_clip: 是否随机提取16帧
        logger: 日志记录器
    
    Returns:
        (success, error_message): 处理是否成功及错误信息
    """
    try:
        # 获取相对路径，保持原始目录结构
        rel_path = os.path.relpath(os.path.dirname(video_path), input_root)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 创建输出目录结构
        video_output_dir = os.path.join(output_dir, rel_path, video_name)
        frames_dir = os.path.join(video_output_dir, 'frames')
        masks_dir = os.path.join(video_output_dir, 'masks')
        
        # 检查是否已经处理过，如果目录中已有文件，则跳过
        if os.path.exists(frames_dir) and os.path.exists(masks_dir):
            # 确保已处理的帧数满足要求
            if len(os.listdir(frames_dir)) >= 16 and len(os.listdir(masks_dir)) >= 16:
                return True, "已处理，跳过" # 返回特定消息表示跳过
        
        # 使用临时目录处理，处理成功后再移动到最终目录
        temp_dir = os.path.join(output_dir, f"temp_{os.path.basename(video_name)}_{os.getpid()}")
        temp_frames_dir = os.path.join(temp_dir, 'frames')
        temp_masks_dir = os.path.join(temp_dir, 'masks')
        
        os.makedirs(temp_frames_dir, exist_ok=True)
        os.makedirs(temp_masks_dir, exist_ok=True)
        
        # 使用抑制所有输出上下文，初始化视频处理器
        with SuppressAllOutput():
            processor = VideoProcessor(device=device, retinaface_model_path=retinaface_model_path)
            
            # 处理视频，强制使用随机片段以提取16帧
            aligned_frames, _, _ = processor.process_video(video_path, random_clip=random_clip)
        
        if aligned_frames is None or len(aligned_frames) == 0:
            error_msg = f"未检测到人脸或处理失败"
            if logger:
                logger.error(f"处理失败 (丢弃): {video_path} - {error_msg}")
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False, error_msg
        
        # 检查提取到的对齐帧数是否满足16帧的要求
        if len(aligned_frames) < 16:
            error_msg = f"仅提取到 {len(aligned_frames)} 个对齐帧，少于要求的16帧"
            if logger:
                logger.error(f"处理失败 (丢弃): {video_path} - {error_msg}")
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False, error_msg
        
        # 生成面部掩码
        face_masks = None
        if use_parsing:
            try:
                # 使用抑制所有输出上下文，初始化面部解析器
                with SuppressAllOutput():
                    # 初始化面部解析器
                    face_parser = FaceParser(model_path=face_parsing_model_path, device=device)
                    
                    # 对每一帧进行解析
                    face_masks = []
                    for frame in aligned_frames:
                        # 使用抑制所有输出上下文，进行面部解析
                        with SuppressAllOutput():
                            parsing_mask = face_parser.parse(frame)
                        face_masks.append(parsing_mask)
                
                # 检查生成的掩码数量是否与帧数一致且满足16帧
                if len(face_masks) != len(aligned_frames) or len(face_masks) < 16:
                    error_msg = f"生成的掩码数量 ({len(face_masks)}) 与帧数 ({len(aligned_frames)}) 不匹配或少于16"
                    if logger:
                        logger.error(f"处理失败 (丢弃): {video_path} - {error_msg}")
                    # 清理临时目录
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    return False, error_msg
                    
            except Exception as e:
                error_msg = f"处理视频掩码时出错: {str(e)}"
                if logger:
                    logger.error(f"处理失败 (丢弃): {video_path} - {error_msg}")
                # 清理临时目录
                shutil.rmtree(temp_dir, ignore_errors=True)
                return False, error_msg
        else:
             # 如果不使用解析，需要创建占位符或简单掩码，确保数量匹配
             # 这里假设后续流程不需要掩码，或者VideoProcessor已提供基础掩码
             # 如果需要基础掩码，应在此处生成
             # 为保持一致性，检查对齐帧数是否足够
             if len(aligned_frames) < 16:
                 error_msg = f"未使用解析，但对齐帧数 ({len(aligned_frames)}) 少于16"
                 if logger:
                     logger.error(f"处理失败 (丢弃): {video_path} - {error_msg}")
                 shutil.rmtree(temp_dir, ignore_errors=True)
                 return False, error_msg
             # 创建一个与帧数相同数量的None列表或其他占位符
             face_masks = [None] * len(aligned_frames)


        # 保存帧和掩码 (仅当掩码存在或不需要时)
        saved_frame_count = 0
        saved_mask_count = 0
        for i, frame in enumerate(aligned_frames):
            # 生成四位数字的帧编号
            frame_num = f"{i+1:04d}"
            
            # 保存帧
            frame_path = os.path.join(temp_frames_dir, f"frame_{frame_num}.png")
            cv2.imwrite(frame_path, frame)
            saved_frame_count += 1
            
            # 保存掩码 (如果存在)
            if face_masks and i < len(face_masks) and face_masks[i] is not None:
                mask_path = os.path.join(temp_masks_dir, f"mask_{frame_num}.png")
                cv2.imwrite(mask_path, face_masks[i])
                saved_mask_count += 1
        
        # 再次检查最终保存的文件数量是否满足16帧的要求
        # 如果使用解析，则帧和掩码都需要满足16；否则只检查帧
        min_required_files = 16
        if (saved_frame_count < min_required_files) or \
           (use_parsing and saved_mask_count < min_required_files):
            error_msg = f"最终保存的文件不足16帧 (帧: {saved_frame_count}, 掩码: {saved_mask_count})"
            if logger:
                logger.error(f"处理失败 (丢弃): {video_path} - {error_msg}")
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False, error_msg
        
        # 创建最终的输出目录
        os.makedirs(os.path.dirname(video_output_dir), exist_ok=True)
        
        # 如果最终目录已存在，先删除
        if os.path.exists(video_output_dir):
            shutil.rmtree(video_output_dir, ignore_errors=True)
        
        # 移动临时目录到最终目录
        shutil.move(temp_dir, video_output_dir)
        
        return True, ""
        
    except Exception as e:
        error_msg = f"处理视频时出错: {str(e)}"
        if logger:
            logger.error(f"处理失败: {video_path} - {error_msg}")
            logger.error(traceback.format_exc())
        
        # 清理临时目录（如果存在）
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        return False, error_msg

def process_video_wrapper(args):
    """
    用于并行处理的包装函数
    
    Args:
        args: 包含所有参数的元组
    
    Returns:
        (video_path, success, error_message): 元组包含视频路径、处理结果和错误信息
    """
    video_path, input_root, output_dir, device, retinaface_model_path, face_parsing_model_path, use_parsing, random_clip, logger = args
    
    # 运行处理函数
    success, error_msg = process_video_to_frames(
        video_path, input_root, output_dir, device, 
        retinaface_model_path, face_parsing_model_path, 
        use_parsing, random_clip, logger
    )
    
    # 如果是跳过的情况，也认为是成功的，但不计入错误
    if error_msg == "已处理，跳过":
        return video_path, True, "" # 返回成功，空错误信息

    return video_path, success, error_msg

def process_id_folder(id_folder, input_root, output_dir, device='cuda', retinaface_model_path=None, 
                     face_parsing_model_path=None, use_parsing=True, num_frames=16, num_workers=4, logger=None):
    """
    处理单个ID文件夹中的所有视频
    
    Args:
        id_folder: ID文件夹路径
        input_root: 输入根目录，用于计算相对路径
        output_dir: 输出目录
        device: 运行设备
        retinaface_model_path: RetinaFace模型路径
        face_parsing_model_path: 面部解析模型路径
        use_parsing: 是否使用面部解析生成更精确的掩码
        num_frames: 每个视频提取的帧数
        num_workers: 并行处理的工作进程数
        logger: 日志记录器
    
    Returns:
        success_count: 成功处理的视频数量
        total_count: 总视频数量
        error_dict: 错误信息字典，键为视频路径，值为错误原因
    """
    # 设置是否要精确提取16帧
    random_clip = True if num_frames == 16 else False
    
    # 查找ID文件夹中所有视频文件
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(id_folder, "**", ext), recursive=True))
    
    video_files = sorted(list(set(video_files)))  # 去重并排序
    
    if not video_files:
        if logger:
            logger.warning(f"在 {id_folder} 目录中未找到视频文件")
        print(f"警告: 在 {id_folder} 目录中未找到视频文件")
        return 0, 0, {}
    
    print(f"在 {os.path.basename(id_folder)} 中找到 {len(video_files)} 个视频文件")
    
    # 准备并行处理的参数
    args_list = [
        (video_path, input_root, output_dir, device, retinaface_model_path, 
         face_parsing_model_path, use_parsing, random_clip, logger)
        for video_path in video_files
    ]
    
    # 处理该ID文件夹中的每个视频 (并行)
    success_count = 0
    error_dict = {}
    
    # 确定使用的并行度
    effective_workers = min(num_workers, len(video_files))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers) as executor:
        results = list(tqdm(
            executor.map(process_video_wrapper, args_list), 
            total=len(video_files),
            desc=f"处理 {os.path.basename(id_folder)}"
        ))
        
        # 统计成功处理的视频数量和错误信息
        for video_path, success, error_msg in results:
            if success:
                success_count += 1
            else:
                error_dict[video_path] = error_msg
    
    return success_count, len(video_files), error_dict

def process_id_folder_wrapper(args):
    """
    用于并行处理ID文件夹的包装函数
    
    Args:
        args: 包含所有参数的元组
    
    Returns:
        (id_folder, success_count, total_count, error_dict): 元组包含ID文件夹路径、处理结果和错误信息
    """
    id_folder, input_root, output_dir, device, retinaface_model_path, face_parsing_model_path, use_parsing, num_frames, videos_per_process, logger = args
    success_count, total_count, error_dict = process_id_folder(
        id_folder, input_root, output_dir, device,
        retinaface_model_path, face_parsing_model_path,
        use_parsing, num_frames, videos_per_process, logger
    )
    return id_folder, success_count, total_count, error_dict

def batch_process_vox2(root_dir, output_dir, device='cuda', retinaface_model_path=None, 
                      face_parsing_model_path=None, use_parsing=True, num_frames=16, 
                      batch_size=5, folders_per_process=2, videos_per_process=4):
    """
    批量处理Vox2数据集中的视频，并行处理多个ID文件夹
    
    Args:
        root_dir: Vox2数据集根目录
        output_dir: 输出目录
        device: 运行设备
        retinaface_model_path: RetinaFace模型路径
        face_parsing_model_path: 面部解析模型路径
        use_parsing: 是否使用面部解析生成更精确的掩码
        num_frames: 每个视频提取的帧数
        batch_size: 每批处理的ID文件夹数量
        folders_per_process: 同时处理的ID文件夹数量
        videos_per_process: 每个ID文件夹内同时处理的视频数量
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志记录器
    logger = setup_logger(output_dir)
    logger.info(f"开始批量处理 {root_dir} 中的视频")
    
    # 获取所有ID文件夹
    id_folders = []
    id_pattern = re.compile(r'^id\d{5}$')
    
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and id_pattern.match(item):
            id_folders.append(item_path)
    
    id_folders.sort()  # 确保按字母顺序处理
    
    if not id_folders:
        error_msg = f"在 {root_dir} 目录中未找到符合 'idxxxxx' 格式的文件夹"
        logger.error(error_msg)
        print(f"错误: {error_msg}")
        return
    
    logger.info(f"找到 {len(id_folders)} 个ID文件夹")
    print(f"找到 {len(id_folders)} 个ID文件夹")
    
    # 记录总体进度
    total_success = 0
    total_videos = 0
    all_errors = {}
    
    # 按批次处理ID文件夹
    for i in range(0, len(id_folders), batch_size):
        batch = id_folders[i:i+batch_size]
        batch_msg = f"\n开始处理第 {i//batch_size + 1} 批 (共 {len(id_folders)//batch_size + (1 if len(id_folders)%batch_size > 0 else 0)} 批)"
        logger.info(batch_msg)
        print(batch_msg)
        
        batch_ids = ', '.join([os.path.basename(folder) for folder in batch])
        logger.info(f"本批处理的ID: {batch_ids}")
        print(f"本批处理的ID: {batch_ids}")
        
        # 准备并行处理的参数
        args_list = [
            (id_folder, root_dir, output_dir, device, retinaface_model_path,
             face_parsing_model_path, use_parsing, num_frames, videos_per_process, logger)
            for id_folder in batch
        ]
        
        # 确定使用的并行度
        effective_workers = min(folders_per_process, len(batch))
        
        # 并行处理多个ID文件夹
        with concurrent.futures.ProcessPoolExecutor(max_workers=effective_workers) as executor:
            results = list(executor.map(process_id_folder_wrapper, args_list))
            
            # 打印每个ID文件夹的处理结果并更新总计数
            for id_folder, success_count, total_count, error_dict in results:
                total_success += success_count
                total_videos += total_count
                all_errors.update(error_dict)
                
                result_msg = f"{os.path.basename(id_folder)} 完成: {success_count}/{total_count} 个视频处理成功"
                logger.info(result_msg)
                print(result_msg)
                
                # 记录该ID文件夹的错误信息
                if error_dict:
                    logger.info(f"{os.path.basename(id_folder)} 有 {len(error_dict)} 个视频处理失败")
    
    # 打印总体处理结果
    summary_msg = f"\n批量处理完成. 总计: {total_success}/{total_videos} 个视频处理成功"
    logger.info(summary_msg)
    print(summary_msg)
    
    # 记录所有错误的汇总
    if all_errors:
        logger.info(f"总计 {len(all_errors)} 个视频处理失败，详细信息已记录在日志中")
        print(f"总计 {len(all_errors)} 个视频处理失败，详细信息已记录在日志中")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="批量处理Vox2数据集为帧序列")
    parser.add_argument("--input", "-i", default="D:\\zju\\datasets\\Vox2-mp4\\dev", 
                      help="Vox2数据集根目录 (默认: D:\\zju\\datasets\\Vox2-mp4\\dev)")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument("--device", "-d", choices=["cuda", "cpu"], 
                      default="cuda" if torch.cuda.is_available() else "cpu",
                      help="运行设备 (默认: cuda如果可用，否则cpu)")
    parser.add_argument("--model-path", "-m", 
                      default=None,
                      help="RetinaFace模型权重文件路径，留空使用默认模型")
    parser.add_argument("--parsing-model", "-p", 
                      default=None,
                      help="面部解析模型路径，留空将尝试使用项目中的权重或下载")
    parser.add_argument("--no-parsing", action="store_true", 
                      help="不使用面部解析，只使用简单的几何方法生成掩码")
    parser.add_argument("--num-frames", "-n", type=int, default=16,
                      help="每个视频提取的帧数 (默认: 16)")
    parser.add_argument("--batch-size", "-b", type=int, default=5,
                      help="每批处理的ID文件夹数量 (默认: 5)")
    parser.add_argument("--folders-per-process", "-fp", type=int, default=1,
                      help="同时处理的ID文件夹数量 (默认: 1)")
    parser.add_argument("--videos-per-process", "-vp", type=int, default=3,
                      help="每个ID文件夹内同时处理的视频数量 (默认: 4)")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入目录 {args.input} 不存在")
        return
    
    # 批量处理Vox2数据集
    batch_process_vox2(
        args.input, 
        args.output, 
        device=args.device,
        retinaface_model_path=args.model_path,
        face_parsing_model_path=args.parsing_model,
        use_parsing=not args.no_parsing,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        folders_per_process=args.folders_per_process,
        videos_per_process=args.videos_per_process
    )

if __name__ == "__main__":
    main()
