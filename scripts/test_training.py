import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os

print("=" * 60)
print("DDPM训练诊断测试")
print("=" * 60)

print(f"\n1. 检查GPU状态:")
print(f"   CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"   GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print(f"\n2. 检查数据目录:")
data_path = r"D:\Code\KeShe\Mycode\DDPM-Cancer-Detection\data\positive"
print(f"   数据路径: {data_path}")
print(f"   目录存在: {os.path.exists(data_path)}")

if os.path.exists(data_path):
    files = [f for f in os.listdir(data_path) if f.endswith('.png')]
    print(f"   图像数量: {len(files)}")
    if len(files) > 0:
        print(f"   示例文件: {files[0]}")

print(f"\n3. 创建模型:")
try:
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        flash_attn = False
    )
    print("   ✓ Unet模型创建成功")
except Exception as e:
    print(f"   ✗ Unet模型创建失败: {e}")

print(f"\n4. 创建扩散模型:")
try:
    diffusion = GaussianDiffusion(
        model,
        image_size = 96,
        timesteps = 1000,
        sampling_timesteps = 250
    )
    print("   ✓ GaussianDiffusion创建成功")
except Exception as e:
    print(f"   ✗ GaussianDiffusion创建失败: {e}")

print(f"\n5. 创建训练器 (这可能需要一些时间):")
try:
    trainer = Trainer(
        diffusion,
        data_path,
        train_batch_size = 8,
        train_lr = 8e-5,
        train_num_steps = 10,
        gradient_accumulate_every = 4,
        ema_decay = 0.995,
        amp = True,
        results_folder = r"D:\Code\KeShe\Mycode\DDPM-Cancer-Detection\results\test",
        save_and_sample_every = 5
    )
    print("   ✓ Trainer创建成功")
except Exception as e:
    print(f"   ✗ Trainer创建失败: {e}")
    import traceback
    traceback.print_exc()

print(f"\n6. 测试训练一步:")
try:
    print("   开始训练...")
    trainer.train()
    print("   ✓ 训练成功完成")
except Exception as e:
    print(f"   ✗ 训练失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
