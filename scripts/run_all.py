import os
import sys
import subprocess

def run_command(cmd, cwd=None):
    """运行命令并实时显示输出"""
    print(f"\n执行命令: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"命令执行失败，返回码: {result.returncode}")
        return False
    return True

def main():
    """一键运行整个DDPM训练流程"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    scripts_dir = os.path.join(project_root, 'scripts')
    
    print("=" * 60)
    print("DDPM病理学图像生成项目 - 一键启动")
    print("=" * 60)
    
    print("\n步骤1: 数据预处理")
    print("-" * 60)
    cmd = [
        sys.executable,
        'preprocess_data.py',
        '--data_dir', os.path.join(project_root, '..', '..', 'DataKaggle', 'histopathologic-cancer-detection-gpu'),
        '--output_dir', os.path.join(project_root, 'data'),
        '--image_size', '256'
    ]
    if not run_command(cmd, cwd=scripts_dir):
        print("数据预处理失败！")
        return
    
    print("\n步骤2: 训练模型")
    print("-" * 60)
    cmd = [
        sys.executable,
        'train_both_models.py',
        '--data_dir', os.path.join(project_root, 'data'),
        '--results_dir', os.path.join(project_root, 'results'),
        '--image_size', '256',
        '--train_steps', '10000'
    ]
    if not run_command(cmd, cwd=scripts_dir):
        print("模型训练失败！")
        return
    
    print("\n步骤3: 生成图像")
    print("-" * 60)
    
    print("\n生成阳性样本图像...")
    cmd = [
        sys.executable,
        'generate_images.py',
        '--model_path', os.path.join(project_root, 'results', 'positive', 'model.pt'),
        '--output_dir', os.path.join(project_root, 'generated', 'positive'),
        '--num_images', '16',
        '--image_size', '256',
        '--batch_size', '4'
    ]
    if not run_command(cmd, cwd=scripts_dir):
        print("阳性样本生成失败！")
    
    print("\n生成阴性样本图像...")
    cmd = [
        sys.executable,
        'generate_images.py',
        '--model_path', os.path.join(project_root, 'results', 'negative', 'model.pt'),
        '--output_dir', os.path.join(project_root, 'generated', 'negative'),
        '--num_images', '16',
        '--image_size', '256',
        '--batch_size', '4'
    ]
    if not run_command(cmd, cwd=scripts_dir):
        print("阴性样本生成失败！")
    
    print("\n步骤4: 评估结果")
    print("-" * 60)
    cmd = [
        sys.executable,
        'evaluate_results.py',
        '--mode', 'both',
        '--real_dir', os.path.join(project_root, 'data', 'positive'),
        '--generated_dir', os.path.join(project_root, 'results', 'positive', 'samples'),
        '--output_path', os.path.join(project_root, 'comparison.png'),
        '--data_dir', os.path.join(project_root, 'data')
    ]
    if not run_command(cmd, cwd=scripts_dir):
        print("结果评估失败！")
    
    print("\n" + "=" * 60)
    print("所有步骤完成！")
    print("=" * 60)
    print(f"项目目录: {project_root}")
    print(f"生成图像目录: {os.path.join(project_root, 'generated')}")
    print(f"模型保存目录: {os.path.join(project_root, 'results')}")

if __name__ == "__main__":
    main()