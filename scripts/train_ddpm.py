import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import argparse
import os
import multiprocessing

def train_ddpm(data_path, results_folder, image_size=96, train_steps=10000, batch_size=8, num_workers=0):
    """
    训练DDPM模型
    
    Args:
        data_path: 训练数据路径
        results_folder: 结果保存路径
        image_size: 图像尺寸
        train_steps: 训练步数
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
    """
    
    print(f"训练配置:")
    print(f"  数据路径: {data_path}")
    print(f"  结果保存路径: {results_folder}")
    print(f"  图像尺寸: {image_size}x{image_size}")
    print(f"  训练步数: {train_steps}")
    print(f"  批次大小: {batch_size}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"  GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        flash_attn = False
    )
    
    diffusion = GaussianDiffusion(
        model,
        image_size = image_size,
        timesteps = 1000,
        sampling_timesteps = 250
    )
    
    trainer = Trainer(
        diffusion,
        data_path,
        train_batch_size = batch_size,
        train_lr = 8e-5,
        train_num_steps = train_steps,
        gradient_accumulate_every = 4,
        ema_decay = 0.995,
        amp = True,
        results_folder = results_folder,
        save_and_sample_every = 1000
    )
    
    print(f"\n开始训练...")
    trainer.train()
    
    print(f"\n训练完成！模型保存在: {results_folder}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description='训练DDPM模型')
    parser.add_argument('--data_path', type=str, required=True,
                       help='训练数据路径')
    parser.add_argument('--results_folder', type=str, required=True,
                       help='结果保存路径')
    parser.add_argument('--image_size', type=int, default=96,
                       help='图像尺寸')
    parser.add_argument('--train_steps', type=int, default=10000,
                       help='训练步数')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小 (默认: 8, 4GB显存推荐)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='数据加载工作进程数 (默认: 0, Windows推荐)')
    
    args = parser.parse_args()
    
    os.makedirs(args.results_folder, exist_ok=True)
    
    train_ddpm(args.data_path, args.results_folder, args.image_size, args.train_steps, 
               args.batch_size, args.num_workers)