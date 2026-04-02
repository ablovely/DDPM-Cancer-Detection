# Kaggle 使用指南

本指南说明如何在 Kaggle 上训练 DDPM 癌症检测图像生成模型。

## 快速开始

### 方法 1: 使用提供的 Notebook (推荐)

1. 访问 [Kaggle](https://www.kaggle.com) 并登录
2. 点击 **"Notebooks"** → **"New Notebook"**
3. 选择 **"File"** → **"Import Notebook"**
4. 上传项目中的 `kaggle_notebook.ipynb` 文件
5. 在 Notebook 设置中：
   - **Accelerator**: 选择 `GPU T4 x2` 或 `GPU P100`
   - **Internet**: 开启 (需要安装依赖)
6. 点击 **"Save & Run All"**

### 方法 2: 手动创建 Notebook

#### 步骤 1: 创建新 Notebook

1. 访问 Kaggle 竞赛页面: https://www.kaggle.com/competitions/histopathologic-cancer-detection
2. 点击 **"Notebooks"** → **"New Notebook"**
3. 选择 **"Python"** 作为语言

#### 步骤 2: 配置环境

在 Notebook 设置中：
- **Accelerator**: 选择 `GPU T4 x2` (推荐) 或 `GPU P100`
- **Internet**: 开启 (On)
- **Dataset**: 添加 `histopathologic-cancer-detection` 数据集

#### 步骤 3: 复制代码

将以下代码分段复制到 Notebook 中运行：

```python
# 1. 检查GPU
!nvidia-smi

# 2. 安装依赖
!pip install denoising-diffusion-pytorch einops ema-pytorch accelerate

# 3. 准备数据集
import os
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm

KAGGLE_DATA_DIR = '/kaggle/input/histopathologic-cancer-detection'
WORKING_DIR = '/kaggle/working'
DATA_DIR = os.path.join(WORKING_DIR, 'data')

os.makedirs(os.path.join(DATA_DIR, 'positive'), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, 'negative'), exist_ok=True)

def prepare_dataset(num_samples=5000, image_size=96):
    df = pd.read_csv(os.path.join(KAGGLE_DATA_DIR, 'train_labels.csv'))
    df = df.sample(n=num_samples, random_state=42)
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_id = row['id']
        label = row['label']
        
        src_path = os.path.join(KAGGLE_DATA_DIR, 'train', f"{image_id}.tif")
        
        if label == 0:
            dst_path = os.path.join(DATA_DIR, 'positive', f"{image_id}.png")
        else:
            dst_path = os.path.join(DATA_DIR, 'negative', f"{image_id}.png")
        
        img = Image.open(src_path)
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.BILINEAR)
        img.save(dst_path, 'PNG')

prepare_dataset(num_samples=5000)

# 4. 训练模型
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    channels = 3,
    flash_attn = False
)

diffusion = GaussianDiffusion(
    model,
    image_size = 96,
    timesteps = 1000,
    sampling_timesteps = 250
)

trainer = Trainer(
    diffusion,
    os.path.join(DATA_DIR, 'positive'),
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 10000,
    gradient_accumulate_every = 2,
    ema_decay = 0.995,
    amp = True,
    results_folder = os.path.join(WORKING_DIR, 'results'),
    save_and_sample_every = 1000
)

trainer.train()
```

## Kaggle 免费额度

- **GPU 时间**: 每周 30 小时
- **GPU 类型**: T4 或 P100
- **存储空间**: 足够存储模型和结果

## 参数调整建议

### 对于 Kaggle T4 GPU (16GB 显存):

```python
train_batch_size = 32
gradient_accumulate_every = 2
image_size = 96
train_num_steps = 10000  # 可以增加到 50000+
```

### 快速测试:

```python
train_batch_size = 16
train_num_steps = 1000
save_and_sample_every = 100
```

## 下载结果

训练完成后：

1. 在 Notebook 右侧的 **"Data"** 面板中
2. 找到 `/kaggle/working/results/` 目录
3. 右键点击模型文件 (如 `model-10.pt`)
4. 选择 **"Download"**

或者使用代码：

```python
from IPython.display import FileLink
FileLink('/kaggle/working/results/model-10.pt')
```

## 常见问题

**Q: Notebook 超时怎么办？**
A: Kaggle Notebook 单次运行最长 9 小时。可以保存检查点，下次继续训练。

**Q: 如何保存进度？**
A: Trainer 会自动保存检查点到 `/kaggle/working/results/`。

**Q: 可以训练阴性模型吗？**
A: 可以！将 `os.path.join(DATA_DIR, 'positive')` 改为 `os.path.join(DATA_DIR, 'negative')` 即可。

## 进阶技巧

1. **使用完整数据集**: 移除 `num_samples` 参数，使用所有数据
2. **调整模型大小**: 修改 `dim` 和 `dim_mults`
3. **更长的训练**: 增加 `train_num_steps` 到 50000 或更多
4. **生成更多图像**: 修改采样时的 `batch_size`

## 参考链接

- Kaggle 竞赛: https://www.kaggle.com/competitions/histopathologic-cancer-detection
- 项目 GitHub: https://github.com/ablovely/DDPM-Cancer-Detection
- Denoising Diffusion Pytorch: https://github.com/lucidrains/denoising-diffusion-pytorch
