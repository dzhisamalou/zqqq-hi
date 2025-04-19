# HiFi-VFS 人脸视频预处理工具

本项目实现了用于HiFi-VFS（高保真视频换脸系统）数据准备的预处理工具，包含人脸检测、关键点提取、对齐、裁剪和生成面部区域掩码的功能。

## 功能特点

- 从视频中随机提取16帧序列
- 使用RetinaFace进行人脸检测和关键点提取
  - **支持多角度人脸检测(0°、90°、180°、270°)**，确保旋转的人脸也能被正确识别
- 基于关键点的人脸对齐
- 将人脸裁剪为640x640像素
- 使用预训练的BiSeNet模型生成人脸区域掩码
- 提供可视化界面展示处理结果和关键点标注

## 环境配置

### 使用Conda创建虚拟环境

```bash
# 创建conda环境
conda create -n hifivfs python=3.8
conda activate hifivfs

# 安装PyTorch (CUDA 11.3版本，根据您的CUDA版本调整)
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# 安装其他依赖
pip install -r requirements.txt
```

## 模型准备

### 下载BiSeNet模型

您需要下载预训练的BiSeNet人脸解析模型：

```bash
# 克隆BiSeNet代码库
git clone https://github.com/zllrunning/face-parsing.PyTorch.git

# 下载预训练模型
cd face-parsing.PyTorch
wget -O model/79999_iter.pth https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view?usp=sharing
```

或者访问以下链接手动下载模型：
- [BiSeNet预训练模型](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view?usp=sharing)

## 使用方法

### 命令行使用

```bash
# 基本用法
python preprocess_videos.py --input_dir /path/to/videos --output_dir /path/to/output --model_path /path/to/79999_iter.pth

# 保存带关键点标注的帧和视频
python preprocess_videos.py --input_dir /path/to/videos --output_dir /path/to/output --model_path /path/to/79999_iter.pth --save_landmarks --temp_dir D:/zju/code/hifivfs/test
```

参数说明：
- `--input_dir`: 输入视频目录
- `--output_dir`: 输出处理结果目录
- `--model_path`: BiSeNet人脸解析模型路径
- `--device`: 使用的设备 (默认: cuda:0)
- `--size`: 输出图像大小 (默认: 640)
- `--ext`: 要处理的视频扩展名 (默认: mp4,avi,mov)
- `--save_landmarks`: 是否保存带关键点标注的图像
- `--temp_dir`: 临时文件存储路径

### 可视化界面

我们提供了一个基于Gradio的可视化界面，方便查看处理结果：

```bash
python batch_preprocess.py --input "D:\zju\datasets\Vox2-mp4\dev" --output "D:\zju\code\hifivfs\dataset"
```

界面功能：
- **处理新视频**: 上传视频并设置处理参数
- **浏览已处理结果**: 查看处理过的视频、关键点标注和对齐后的帧

## 输出说明

处理后的数据组织结构如下：

```
output_dir/
├── video_name1/
│   ├── video_name1_frame_00.png  # 对齐后的人脸帧
│   ├── video_name1_frame_01.png
│   ├── ...
│   ├── video_name1_mask_00.png   # 面部区域掩码
│   ├── video_name1_mask_01.png
│   └── ...
├── video_name2/
│   └── ...
└── preprocessing_results.json    # 处理结果信息
```

临时文件目录：

```
temp_dir/
├── video_name1_landmarks.mp4     # 带关键点标注的视频
├── video_name1_raw.mp4          # 原始视频片段
├── video_name1_landmarks_00.png  # 带关键点标注的单帧
├── video_name1_landmarks_01.png
└── ...
```

## 项目结构

```
d:\zju\code\hifivfs\
├── preprocessing/
│   ├── __init__.py
│   ├── face_processor.py       # 人脸处理基类
│   └── face_parser.py          # 人脸解析模型实现
├── preprocess_videos.py        # 主预处理脚本
├── gradio_interface.py         # 可视化界面
├── requirements.txt            # 依赖项
└── README.md                   # 本文档
```

## 注意事项

1. 确保已安装正确版本的CUDA和cuDNN，与PyTorch兼容。
2. BiSeNet模型需要单独下载，不包含在本代码库中。
3. 处理高清视频可能需要较大的内存和显存资源。

## 引用

如果您使用了本工具，请考虑引用以下论文：

```
@inproceedings{HiFiVFS,
  title={High-Fidelity Video Face Swapping},
  author={...},
  booktitle={...},
  year={...}
}
```
