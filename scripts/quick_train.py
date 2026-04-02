import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    print("=" * 60)
    print("阳性模型快速训练 (100步)")
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
    
    os.makedirs(results_folder, exist_ok=True)
    
    print(f"\n创建模型...")
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
    
    print(f"\n创建训练器...")
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
    
    print(f"\n开始训练...")
    trainer.train()
    
    print(f"\n" + "=" * 60)
    print("训练完成！模型保存在:", results_folder)
    print("=" * 60)
