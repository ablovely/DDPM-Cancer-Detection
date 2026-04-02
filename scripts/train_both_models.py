import os
import sys
import subprocess
import argparse

def train_both_models(data_dir, results_dir, image_size=96, train_steps=10000):
    """
    训练阳性和阴性两个DDPM模型
    
    Args:
        data_dir: 数据目录
        results_dir: 结果保存目录
        image_size: 图像尺寸
        train_steps: 训练步数
    """
    
    positive_data = os.path.join(data_dir, 'positive')
    negative_data = os.path.join(data_dir, 'negative')
    
    positive_results = os.path.join(results_dir, 'positive')
    negative_results = os.path.join(results_dir, 'negative')
    
    if not os.path.exists(positive_data):
        raise FileNotFoundError(f"阳性数据目录不存在: {positive_data}")
    
    if not os.path.exists(negative_data):
        raise FileNotFoundError(f"阴性数据目录不存在: {negative_data}")
    
    print("=" * 60)
    print("开始训练阳性模型 (label=0, 癌症阳性)")
    print("=" * 60)
    
    cmd_positive = [
        sys.executable,
        'train_ddpm.py',
        '--data_path', positive_data,
        '--results_folder', positive_results,
        '--image_size', str(image_size),
        '--train_steps', str(train_steps)
    ]
    
    subprocess.run(cmd_positive, cwd=os.path.dirname(__file__))
    
    print("\n" + "=" * 60)
    print("开始训练阴性模型 (label=1, 正常组织)")
    print("=" * 60)
    
    cmd_negative = [
        sys.executable,
        'train_ddpm.py',
        '--data_path', negative_data,
        '--results_folder', negative_results,
        '--image_size', str(image_size),
        '--train_steps', str(train_steps)
    ]
    
    subprocess.run(cmd_negative, cwd=os.path.dirname(__file__))
    
    print("\n" + "=" * 60)
    print("所有模型训练完成！")
    print("=" * 60)
    print(f"阳性模型保存在: {positive_results}")
    print(f"阴性模型保存在: {negative_results}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    parser = argparse.ArgumentParser(description='训练阳性和阴性两个DDPM模型')
    parser.add_argument('--data_dir', type=str,
                       default=os.path.join(project_root, 'data'),
                       help='数据目录')
    parser.add_argument('--results_dir', type=str,
                       default=os.path.join(project_root, 'results'),
                       help='结果保存目录')
    parser.add_argument('--image_size', type=int, default=96,
                       help='图像尺寸')
    parser.add_argument('--train_steps', type=int, default=10000,
                       help='训练步数')
    
    args = parser.parse_args()
    
    train_both_models(args.data_dir, args.results_dir, args.image_size, args.train_steps)