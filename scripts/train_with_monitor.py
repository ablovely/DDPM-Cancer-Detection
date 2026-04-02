import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import psutil
import time
import os
import multiprocessing
import threading

def print_resources():
    """打印当前资源使用情况"""
    if torch.cuda.is_available():
        gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_percent = (gpu_reserved / gpu_total) * 100
    else:
        gpu_percent = 0
        gpu_reserved = 0
    
    mem = psutil.virtual_memory()
    mem_percent = mem.percent
    mem_used = mem.used / 1024**3
    
    return f"GPU: {gpu_percent:.1f}% ({gpu_reserved:.2f}GB) | 内存: {mem_percent:.1f}% ({mem_used:.2f}GB)"

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    print("=" * 60)
    print("阳性模型快速训练 (100步) - 带资源监控")
    print("=" * 60)
    
    data_path = r"D:\Code\KeShe\Mycode\DDPM-Cancer-Detection\data\positive"
    results_folder = r"D:\Code\KeShe\Mycode\DDPM-Cancer-Detection\results\positive"
    
    print(f"\n训练配置:")
    print(f"  数据路径: {data_path}")
    print(f"  结果保存路径: {results_folder}")
    print(f"  图像尺寸: 96x96")
    print(f"  训练步数: 100")
    print(f"  批次大小: 8")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"  GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"\n当前资源: {print_resources()}")
    
    os.makedirs(results_folder, exist_ok=True)
    
    print(f"\n创建模型...")
    print(f"当前资源: {print_resources()}")
    
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        flash_attn = False
    )
    
    print(f"模型创建完成")
    print(f"当前资源: {print_resources()}")
    
    diffusion = GaussianDiffusion(
        model,
        image_size = 96,
        timesteps = 1000,
        sampling_timesteps = 250
    )
    
    print(f"\n创建训练器...")
    print(f"当前资源: {print_resources()}")
    
    trainer = Trainer(
        diffusion,
        data_path,
        train_batch_size = 8,
        train_lr = 8e-5,
        train_num_steps = 100,
        gradient_accumulate_every = 4,
        ema_decay = 0.995,
        amp = True,
        results_folder = results_folder,
        save_and_sample_every = 50
    )
    
    print(f"训练器创建完成")
    print(f"当前资源: {print_resources()}")
    
    print(f"\n开始训练...")
    print("=" * 60)
    
    trainer.train()
    
    print(f"\n" + "=" * 60)
    print("训练完成！模型保存在:", results_folder)
    print(f"最终资源: {print_resources()}")
    print("=" * 60)
