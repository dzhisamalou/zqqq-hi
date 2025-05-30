# HiFiVFS 

目前实现了HiFiVFS数据预处理部分、训练部分的基本框架，但还有待完善。


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

# 下载预训练模型
wget -O model/79999_iter.pth https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view?usp=sharing
```

## 使用方法

### 命令行使用

```bash
python batch_preprocess.py --input "D:\zju\datasets\Vox2-mp4\dev" --output "D:\zju\code\hifivfs\dataset"
```

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
