# DDPM病理学图像生成项目

基于Denoising Diffusion Probabilistic Models (DDPM)的病理学癌症检测图像生成项目

## 项目简介

本项目使用DDPM模型生成病理学组织切片图像，用于癌症检测研究。项目训练两个独立的模型：
- **阳性模型**: 生成癌症阳性病例的病理图像 (label=0)
- **阴性模型**: 生成正常组织的病理图像 (label=1)

## 项目结构

```
DDPM-Cancer-Detection/
├── data/                   # 数据目录
│   ├── positive/          # 阳性样本 (癌症)
│   └── negative/          # 阴性样本 (正常)
├── models/                # 模型保存目录
├── results/               # 训练结果目录
│   ├── positive/         # 阳性模型结果
│   └── negative/         # 阴性模型结果
├── scripts/              # 脚本目录
│   ├── preprocess_data.py      # 数据预处理
│   ├── train_ddpm.py          # 训练单个模型
│   ├── train_both_models.py   # 训练两个模型
│   ├── generate_images.py     # 生成图像
│   └── evaluate_results.py    # 评估结果
└── README.md
```

## 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA支持的GPU (推荐RTX系列)
- 已安装的conda环境

## 安装依赖

```bash
pip install denoising_diffusion_pytorch
pip install pillow pandas matplotlib tqdm
```

## 使用步骤

### 1. 数据预处理

首先需要将原始数据预处理并分类：

```bash
python scripts/preprocess_data.py \
    --data_dir "..\..\DataKaggle\histopathologic-cancer-detection-gpu" \
    --output_dir "data" \
    --image_size 256
```

参数说明：
- `--data_dir`: 原始数据集目录（相对路径）
- `--output_dir`: 预处理后数据保存目录（相对路径）
- `--image_size`: 目标图像尺寸 (默认256x256)

### 2. 训练模型

#### 方式1: 训练两个模型（推荐）

```bash
python scripts/train_both_models.py \
    --data_dir "data" \
    --results_dir "results" \
    --image_size 256 \
    --train_steps 10000
```

#### 方式2: 单独训练模型

训练阳性模型：
```bash
python scripts/train_ddpm.py \
    --data_path "data\positive" \
    --results_folder "results\positive" \
    --image_size 256 \
    --train_steps 10000
```

训练阴性模型：
```bash
python scripts/train_ddpm.py \
    --data_path "data\negative" \
    --results_folder "results\negative" \
    --image_size 256 \
    --train_steps 10000
```

### 3. 生成图像

使用训练好的模型生成新图像：

```bash
python scripts/generate_images.py \
    --model_path "results\positive\model.pt" \
    --output_dir "generated\positive" \
    --num_images 16 \
    --image_size 256 \
    --batch_size 4
```

### 4. 评估结果

可视化真实样本和生成样本的对比：

```bash
python scripts/evaluate_results.py \
    --mode both \
    --real_dir "data\positive" \
    --generated_dir "results\positive\samples" \
    --output_path "comparison.png" \
    --data_dir "data"
```

## 训练参数说明

### 模型参数
- `dim`: U-Net基础维度 (默认64)
- `dim_mults`: 维度乘数 (默认(1, 2, 4, 8))
- `channels`: 输入通道数 (默认3, RGB图像)
- `flash_attn`: 是否使用Flash Attention (默认True)

### 训练参数
- `train_batch_size`: 批处理大小 (默认16)
- `train_lr`: 学习率 (默认8e-5)
- `train_num_steps`: 训练步数 (默认10000)
- `gradient_accumulate_every`: 梯度累积步数 (默认2)
- `ema_decay`: EMA衰减率 (默认0.995)
- `amp`: 是否使用混合精度训练 (默认True)

### 扩散参数
- `timesteps`: 扩散步数 (默认1000)
- `sampling_timesteps`: 采样步数 (默认250)

## 训练规模建议

### 快速实验 (10K步)
- 训练时间: 1-2小时
- 用途: 验证模型是否正常工作
- 生成质量: 初步可辨识

### 中等规模 (100K步)
- 训练时间: 10-15小时
- 用途: 获得初步可用的生成结果
- 生成质量: 较好

### 完整训练 (700K步)
- 训练时间: 数天
- 用途: 获得高质量生成结果
- 生成质量: 最佳

## 数据集说明

本项目使用Kaggle Histopathologic Cancer Detection数据集：
- **阳性样本 (label=0)**: 癌症阳性病例的病理图像
- **阴性样本 (label=1)**: 正常组织的病理图像
- **原始图像尺寸**: 96x96像素
- **预处理后尺寸**: 256x256像素

## 注意事项

1. **GPU内存**: 确保GPU内存足够，建议至少8GB显存
2. **训练时间**: 根据训练规模，可能需要数小时到数天
3. **数据平衡**: 数据集中阳性和阴性样本数量可能不平衡
4. **模型保存**: 模型会定期保存在results目录下
5. **生成质量**: 训练步数越多，生成质量越好

## 故障排除

### CUDA内存不足
- 减小`train_batch_size`
- 减小`image_size`
- 使用梯度累积

### 训练不稳定
- 降低学习率`train_lr`
- 增加`gradient_accumulate_every`
- 检查数据预处理是否正确

### 生成质量差
- 增加训练步数
- 检查数据集质量
- 调整模型参数

## 参考文献

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion PyTorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
- [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection)

## 作者

DDPM病理学图像生成项目

## 许可证

本项目仅供学术研究使用