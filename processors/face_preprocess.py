import cv2
import numpy as np
import torch
import os
import tempfile
import shutil
import importlib
import sys
import traceback
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align

class FaceProcessor:
    def __init__(self, detector_backend='retinaface', device='cuda', retinaface_model_path=None):
        """初始化人脸处理器"""
        self.device = device
        
        # 使用insightface的FaceAnalysis初始化
        try:
            # 创建FaceAnalysis对象
            self.app = FaceAnalysis(providers=['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider'])
            # 准备模型，默认使用retinaface
            self.app.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))
        except Exception as e:
            self.app = None
    
    def detect_face(self, img):
        """使用InsightFace检测人脸"""
        if self.app is None:
            return []
        
        try:
            # InsightFace使用BGR格式，所以不需要转换
            faces = self.app.get(img)
            return faces
        except Exception:
            return []
    
    def process_image(self, image):
        """处理单个图像，检测人脸并提取特征"""
        try:
            faces = self.detect_face(image)
            
            if not faces or len(faces) == 0:
                return None, None, None, None
            
            # 获取第一个人脸
            face = faces[0]
            
            # 获取边界框
            bbox = face.bbox.astype(np.int32)
            box = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            
            # 获取关键点（5个关键点）
            kps = face.kps.astype(np.int32)
            landmarks = kps  # InsightFace默认返回5个关键点
            
            # 使用InsightFace的face_align进行人脸对齐，修改为640x640像素
            aligned_face = face_align.norm_crop(image, landmarks, image_size=640)
            
            # 计算对齐后的关键点坐标
            aligned_landmarks = self._recalculate_landmarks_after_alignment(landmarks)
            
            # 创建面部区域掩码，使用对齐后的关键点
            face_mask = self.create_face_mask(aligned_face, aligned_landmarks)
            
            return aligned_face, landmarks, box, face_mask
            
        except Exception:
            return None, None, None, None
    
    def _recalculate_landmarks_after_alignment(self, landmarks):
        """在人脸对齐后重新计算关键点坐标
        
        Args:
            landmarks: 原始关键点坐标
            
        Returns:
            aligned_landmarks: 对齐后的关键点坐标
        """
        # 标准5点关键点位置（标准化到[0,1]空间内）
        # 这些是InsightFace face_align.norm_crop函数的标准输出位置
        standard_points = np.array([
            [0.31556875000000000, 0.4615741071428571],  # 右眼
            [0.68262291666666670, 0.4615741071428571],  # 左眼
            [0.50026249999999990, 0.6405053571428571],  # 鼻子
            [0.34947187500000004, 0.8246919642857142],  # 右嘴角
            [0.65343645833333330, 0.8246919642857142]   # 左嘴角
        ])
        
        # 映射到640x640的对齐图像上
        aligned_landmarks = (standard_points * 640).astype(np.int32)
        
        return aligned_landmarks
    
    def _estimate_landmarks(self, box):
        """已废弃，由于使用InsightFace直接获取关键点，此方法不再需要"""
        return np.zeros((5, 2), dtype=np.int32)  # 返回空数组以保持兼容性
    
    def create_face_mask(self, face_image, landmarks=None, dilation_kernel_size=11):
        """创建面部区域掩码
        
        Args:
            face_image: 人脸图像
            landmarks: 面部关键点
            dilation_kernel_size: 膨胀核大小
            
        Returns:
            面部区域掩码
        """
        h, w = face_image.shape[:2]
        
        if landmarks is not None and len(landmarks) >= 3:
            # 使用关键点创建多边形掩码
            mask = np.zeros((h, w), dtype=np.uint8)
            
            try:
                # 检查landmarks的类型和形状
                landmark_points = landmarks.copy().astype(np.int32)
                
                # 确保所有坐标点都在图像边界内
                for i in range(len(landmark_points)):
                    landmark_points[i][0] = max(0, min(w-1, landmark_points[i][0]))
                    landmark_points[i][1] = max(0, min(h-1, landmark_points[i][1]))
                
                # 使用凸包计算轮廓
                hull = cv2.convexHull(landmark_points)
                
                # 填充多边形
                cv2.fillConvexPoly(mask, hull, 255)
                
                # 膨胀掩码以覆盖更多的面部区域
                kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
            except Exception as e:
                # 出错时使用椭圆掩码作为后备
                mask = np.zeros((h, w), dtype=np.uint8)
                center = (w // 2, h // 2)
                axes_length = (int(w * 0.4), int(h * 0.5))
                cv2.ellipse(mask, center, axes_length, 0, 0, 360, 255, -1)
        else:
            # 如果没有有效的关键点，创建简单的椭圆掩码
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            axes_length = (int(w * 0.4), int(h * 0.5))
            cv2.ellipse(mask, center, axes_length, 0, 0, 360, 255, -1)
        
        return mask
    
    def visualize_landmarks(self, image, landmarks, box):
        """可视化人脸关键点和边界框"""
        vis_image = image.copy()
        
        # 绘制人脸边界框
        x1, y1, x2, y2 = box
        cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # 绘制关键点
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(vis_image, (int(x), int(y)), 3, (0, 0, 255), -1)
        
        return vis_image
    
    def visualize_mask(self, image, mask):
        """可视化面部区域掩码"""
        color_mask = np.zeros_like(image)
        color_mask[:, :, 2] = mask  # 将掩码应用到红色通道
        
        alpha = 0.5
        mask_vis = cv2.addWeighted(image, 1, color_mask, alpha, 0)
        
        return mask_vis
    
    def process_video_frame(self, frame, original_frame=None, rotation=0):
        """处理视频帧，检测人脸并提取特征"""
        try:
            # 使用 process_image 检测人脸
            aligned_face, landmarks, box, _ = self.process_image(frame)
            
            if landmarks is None:
                # 如果未检测到人脸，则返回原始帧
                return None, frame, None
            
            # 保存检测结果
            result_frame = frame.copy() if original_frame is None else original_frame.copy()
            
            # 如果是旋转帧，需要转换坐标
            if rotation != 0 and original_frame is not None:
                # 转换边界框和关键点坐标到原始帧
                box, orig_landmarks = self._transform_coordinates(box, landmarks, frame, original_frame, rotation)
                # 可视化关键点
                vis_frame = self.visualize_landmarks(result_frame, orig_landmarks, box)
            else:
                # 可视化关键点
                vis_frame = self.visualize_landmarks(result_frame, landmarks, box)
            
            # 返回 None 作为 face_mask，将由后续流程生成
            return aligned_face, vis_frame, None
            
        except Exception:
            return None, frame, None
    
    def _transform_coordinates(self, box, landmarks, rotated_frame, original_frame, rotation):
        """将旋转帧中的坐标转换回原始帧坐标系
        
        Args:
            box: 人脸边界框 [x1, y1, x2, y2]
            landmarks: 面部关键点
            rotated_frame: 旋转后的帧
            original_frame: 原始帧
            rotation: 旋转角度
            
        Returns:
            transformed_box: 转换后的边界框
            transformed_landmarks: 转换后的关键点
        """
        h_rotated, w_rotated = rotated_frame.shape[:2]
        h_orig, w_orig = original_frame.shape[:2]
        
        transformed_box = box.copy()
        transformed_landmarks = landmarks.copy()
        
        # 根据不同旋转角度进行坐标转换
        if rotation == 90:
            # 90度旋转：(x,y) -> (y, width-x)
            # 边界框
            x1, y1, x2, y2 = box
            transformed_box[0] = y1
            transformed_box[1] = w_rotated - x2
            transformed_box[2] = y2
            transformed_box[3] = w_rotated - x1
            
            # 关键点
            for i in range(len(landmarks)):
                x, y = landmarks[i]
                transformed_landmarks[i][0] = y
                transformed_landmarks[i][1] = w_rotated - x
                
        elif rotation == 270 or rotation == -90:
            # 270度旋转：(x,y) -> (height-y, x)
            # 边界框
            x1, y1, x2, y2 = box
            transformed_box[0] = h_rotated - y2
            transformed_box[1] = x1
            transformed_box[2] = h_rotated - y1
            transformed_box[3] = x2
            
            # 关键点
            for i in range(len(landmarks)):
                x, y = landmarks[i]
                transformed_landmarks[i][0] = h_rotated - y
                transformed_landmarks[i][1] = x
                
        elif rotation == 180:
            # 180度旋转：(x,y) -> (width-x, height-y)
            # 边界框
            x1, y1, x2, y2 = box
            transformed_box[0] = w_rotated - x2
            transformed_box[1] = h_rotated - y2
            transformed_box[2] = w_rotated - x1
            transformed_box[3] = h_rotated - y1
            
            # 关键点
            for i in range(len(landmarks)):
                x, y = landmarks[i]
                transformed_landmarks[i][0] = w_rotated - x
                transformed_landmarks[i][1] = h_rotated - y
        
        return transformed_box, transformed_landmarks