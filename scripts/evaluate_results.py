import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from pathlib import Path

def visualize_samples(real_dir, generated_dir, output_path, num_samples=8):
    """
    可视化真实样本和生成样本的对比
    
    Args:
        real_dir: 真实样本目录
        generated_dir: 生成样本目录
        output_path: 输出图像路径
        num_samples: 每类样本数量
    """
    
    real_images = [f for f in os.listdir(real_dir) if f.endswith(('.png', '.tif', '.jpg'))]
    generated_images = [f for f in os.listdir(generated_dir) if f.endswith(('.png', '.tif', '.jpg'))]
    
    real_images = real_images[:num_samples]
    generated_images = generated_images[:num_samples]
    
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 5))
    fig.suptitle('真实样本 vs 生成样本对比', fontsize=16)
    
    for i, img_name in enumerate(real_images):
        img_path = os.path.join(real_dir, img_name)
        img = Image.open(img_path)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'真实 {i+1}')
        axes[0, i].axis('off')
    
    for i, img_name in enumerate(generated_images):
        img_path = os.path.join(generated_dir, img_name)
        img = Image.open(img_path)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f'生成 {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"可视化结果已保存到: {output_path}")
    plt.close()

def analyze_dataset(data_dir):
    """
    分析数据集统计信息
    
    Args:
        data_dir: 数据目录
    """
    
    positive_dir = os.path.join(data_dir, 'positive')
    negative_dir = os.path.join(data_dir, 'negative')
    
    positive_count = len([f for f in os.listdir(positive_dir) if f.endswith('.png')])
    negative_count = len([f for f in os.listdir(negative_dir) if f.endswith('.png')])
    
    print(f"\n数据集统计:")
    print(f"  阳性样本 (label=0): {positive_count}")
    print(f"  阴性样本 (label=1): {negative_count}")
    print(f"  总样本数: {positive_count + negative_count}")
    
    labels = ['阳性 (癌症)', '阴性 (正常)']
    sizes = [positive_count, negative_count]
    colors = ['#ff9999', '#66b3ff']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title('数据集分布')
    
    bars = ax2.bar(labels, sizes, color=colors)
    ax2.set_title('样本数量')
    ax2.set_ylabel('数量')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(data_dir), 'dataset_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"数据集分析图表已保存到: {output_path}")
    plt.close()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(description='评估和可视化DDPM生成结果')
    parser.add_argument('--mode', type=str, choices=['visualize', 'analyze', 'both'],
                       default='both', help='运行模式')
    parser.add_argument('--real_dir', type=str,
                       default=os.path.join(project_root, 'data', 'positive'),
                       help='真实样本目录')
    parser.add_argument('--generated_dir', type=str,
                       default=os.path.join(project_root, 'results', 'positive', 'samples'),
                       help='生成样本目录')
    parser.add_argument('--output_path', type=str,
                       default=os.path.join(project_root, 'comparison.png'),
                       help='输出图像路径')
    parser.add_argument('--data_dir', type=str,
                       default=os.path.join(project_root, 'data'),
                       help='数据目录（用于分析）')
    parser.add_argument('--num_samples', type=int, default=8,
                       help='可视化样本数量')
    
    args = parser.parse_args()
    
    if args.mode in ['visualize', 'both']:
        if os.path.exists(args.real_dir) and os.path.exists(args.generated_dir):
            visualize_samples(args.real_dir, args.generated_dir, args.output_path, args.num_samples)
        else:
            print("警告: 真实样本或生成样本目录不存在，跳过可视化")
    
    if args.mode in ['analyze', 'both']:
        if os.path.exists(args.data_dir):
            analyze_dataset(args.data_dir)
        else:
            print("警告: 数据目录不存在，跳过分析")