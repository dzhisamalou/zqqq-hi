# HiFiVFS 

本项目实现了HiFiVFS数据预处理部分的人脸检测、关键点提取、对齐、裁剪和生成面部区域掩码的功能，目前正在完成FAL模块。


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
