import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import argparse
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def generate_images(model_path, output_dir, num_images=16, image_size=256, batch_size=4):
    """
    使用训练好的DDPM模型生成图像
    
    Args:
        model_path: 模型路径
        output_dir: 输出目录
        num_images: 要生成的图像数量
        image_size: 图像尺寸
        batch_size: 批处理大小
    """
    
    print(f"生成配置:")
    print(f"  模型路径: {model_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  生成数量: {num_images}")
    print(f"  图像尺寸: {image_size}x{image_size}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        flash_attn = True
    ).to(device)
    
    diffusion = GaussianDiffusion(
        model,
        image_size = image_size,
        timesteps = 1000,
        sampling_timesteps = 250
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"\n开始生成图像...")
    
    generated_count = 0
    
    with torch.no_grad():
        while generated_count < num_images:
            current_batch_size = min(batch_size, num_images - generated_count)
            
            sampled_images = diffusion.sample(batch_size=current_batch_size)
            
            for i in range(current_batch_size):
                img = sampled_images[i]
                img = (img + 1) * 0.5
                img = img.clamp(0, 1)
                img = (img * 255).byte()
                img = img.permute(1, 2, 0).cpu().numpy()
                
                img_pil = Image.fromarray(img, mode='RGB')
                img_path = os.path.join(output_dir, f'generated_{generated_count + i:04d}.png')
                img_pil.save(img_path)
            
            generated_count += current_batch_size
            print(f"已生成 {generated_count}/{num_images} 张图像")
    
    print(f"\n生成完成！图像保存在: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='使用DDPM模型生成图像')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型路径')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    parser.add_argument('--num_images', type=int, default=16,
                       help='要生成的图像数量')
    parser.add_argument('--image_size', type=int, default=256,
                       help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批处理大小')
    
    args = parser.parse_args()
    
    generate_images(args.model_path, args.output_dir, args.num_images, args.image_size, args.batch_size)