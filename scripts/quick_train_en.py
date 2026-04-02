import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    print("=" * 60)
    print("Quick Training - Positive Model (100 steps)")
    print("=" * 60)
    
    data_path = r"D:\Code\KeShe\Mycode\DDPM-Cancer-Detection\data\positive"
    results_folder = r"D:\Code\KeShe\Mycode\DDPM-Cancer-Detection\results\positive"
    
    print(f"\nTraining Configuration:")
    print(f"  Data path: {data_path}")
    print(f"  Results path: {results_folder}")
    print(f"  Image size: 96x96")
    print(f"  Training steps: 100")
    print(f"  Batch size: 8")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU device: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    os.makedirs(results_folder, exist_ok=True)
    
    print(f"\nCreating model...")
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
    
    print(f"\nCreating trainer...")
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
    
    print(f"\nStarting training...")
    trainer.train()
    
    print(f"\n" + "=" * 60)
    print(f"Training complete! Model saved in: {results_folder}")
    print("=" * 60)
