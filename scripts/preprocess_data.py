import os
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse

def preprocess_data(data_dir, output_dir, image_size=256):
    """
    预处理病理学图像数据，将其分类为阳性和阴性样本
    
    Args:
        data_dir: 数据集根目录
        output_dir: 输出目录
        image_size: 目标图像尺寸
    """
    
    train_labels_path = os.path.join(data_dir, 'train_labels.csv')
    train_images_dir = os.path.join(data_dir, 'train')
    
    if not os.path.exists(train_labels_path):
        raise FileNotFoundError(f"找不到标签文件: {train_labels_path}")
    
    if not os.path.exists(train_images_dir):
        raise FileNotFoundError(f"找不到训练图像目录: {train_images_dir}")
    
    df = pd.read_csv(train_labels_path)
    
    print(f"总样本数: {len(df)}")
    print(f"阳性样本 (label=0): {len(df[df['label'] == 0])}")
    print(f"阴性样本 (label=1): {len(df[df['label'] == 1])}")
    
    positive_dir = os.path.join(output_dir, 'positive')
    negative_dir = os.path.join(output_dir, 'negative')
    
    os.makedirs(positive_dir, exist_ok=True)
    os.makedirs(negative_dir, exist_ok=True)
    
    print(f"\n开始预处理图像，目标尺寸: {image_size}x{image_size}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理图像"):
        image_id = row['id']
        label = row['label']
        
        image_path = os.path.join(train_images_dir, f"{image_id}.tif")
        
        if not os.path.exists(image_path):
            print(f"警告: 找不到图像 {image_path}")
            continue
        
        try:
            img = Image.open(image_path)
            
            if img.size != (image_size, image_size):
                img = img.resize((image_size, image_size), Image.BILINEAR)
            
            if label == 0:
                output_path = os.path.join(positive_dir, f"{image_id}.png")
            else:
                output_path = os.path.join(negative_dir, f"{image_id}.png")
            
            img.save(output_path, 'PNG')
            
        except Exception as e:
            print(f"处理图像 {image_id} 时出错: {e}")
            continue
    
    positive_count = len([f for f in os.listdir(positive_dir) if f.endswith('.png')])
    negative_count = len([f for f in os.listdir(negative_dir) if f.endswith('.png')])
    
    print(f"\n预处理完成!")
    print(f"阳性样本保存至: {positive_dir} ({positive_count} 张)")
    print(f"阴性样本保存至: {negative_dir} ({negative_count} 张)")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(description='预处理病理学图像数据')
    parser.add_argument('--data_dir', type=str, 
                       default=os.path.join(project_root, '..', '..', 'DataKaggle', 'histopathologic-cancer-detection-gpu'),
                       help='数据集根目录')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(project_root, 'data'),
                       help='输出目录')
    parser.add_argument('--image_size', type=int, default=256,
                       help='目标图像尺寸')
    
    args = parser.parse_args()
    
    preprocess_data(args.data_dir, args.output_dir, args.image_size)