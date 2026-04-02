import torch
import json
import os
import sys

def get_gpu_info():
    """获取GPU信息"""
    if not torch.cuda.is_available():
        return None
    
    gpu_info = {
        'name': torch.cuda.get_device_name(0),
        'memory_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
        'memory_available_gb': torch.cuda.memory_reserved(0) / (1024**3),
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__,
        'device_count': torch.cuda.device_count()
    }
    
    return gpu_info

def recommend_training_params(gpu_memory_gb):
    """根据GPU显存推荐训练参数"""
    if gpu_memory_gb >= 24:
        return {
            'batch_size': 32,
            'gradient_accumulate_every': 1,
            'image_size': 256,
            'num_workers': 8,
            'description': '高端GPU (24GB+) - 最大性能配置'
        }
    elif gpu_memory_gb >= 12:
        return {
            'batch_size': 16,
            'gradient_accumulate_every': 2,
            'image_size': 256,
            'num_workers': 6,
            'description': '中高端GPU (12-24GB) - 平衡配置'
        }
    elif gpu_memory_gb >= 8:
        return {
            'batch_size': 12,
            'gradient_accumulate_every': 2,
            'image_size': 256,
            'num_workers': 4,
            'description': '中等GPU (8-12GB) - 优化配置'
        }
    elif gpu_memory_gb >= 6:
        return {
            'batch_size': 8,
            'gradient_accumulate_every': 4,
            'image_size': 128,
            'num_workers': 4,
            'description': '入门GPU (6-8GB) - 轻量配置'
        }
    elif gpu_memory_gb >= 3.5:
        return {
            'batch_size': 8,
            'gradient_accumulate_every': 4,
            'image_size': 96,
            'num_workers': 2,
            'description': 'RTX 3050 Ti优化配置 (4GB) - 推荐配置'
        }
    else:
        return {
            'batch_size': 4,
            'gradient_accumulate_every': 8,
            'image_size': 64,
            'num_workers': 2,
            'description': '极低显存 (<4GB) - 极简配置'
        }

def update_config_file(config_path, gpu_info, recommended_params):
    """更新配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {}
    
    config['environment'] = {
        'conda_env': os.path.basename(sys.prefix),
        'pytorch_version': gpu_info['pytorch_version'],
        'cuda_version': gpu_info['cuda_version'],
        'gpu_name': gpu_info['name'],
        'gpu_memory_gb': round(gpu_info['memory_total_gb'], 1),
        'cuda_available': True,
        'recommended_batch_size': recommended_params['batch_size']
    }
    
    if 'training' not in config:
        config['training'] = {}
    
    config['training'].update({
        'batch_size': recommended_params['batch_size'],
        'gradient_accumulate_every': recommended_params['gradient_accumulate_every'],
        'num_workers': recommended_params['num_workers']
    })
    
    if 'data' not in config:
        config['data'] = {}
    config['data']['image_size'] = recommended_params['image_size']
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return config

def main():
    print("=" * 60)
    print("GPU自动配置工具")
    print("=" * 60)
    
    gpu_info = get_gpu_info()
    
    if gpu_info is None:
        print("\n⚠ 未检测到GPU，将使用CPU配置")
        recommended_params = {
            'batch_size': 4,
            'gradient_accumulate_every': 8,
            'image_size': 64,
            'num_workers': 2,
            'description': 'CPU模式 - 极简配置'
        }
    else:
        print(f"\n✓ 检测到GPU:")
        print(f"  GPU名称: {gpu_info['name']}")
        print(f"  显存大小: {gpu_info['memory_total_gb']:.1f} GB")
        print(f"  CUDA版本: {gpu_info['cuda_version']}")
        print(f"  PyTorch版本: {gpu_info['pytorch_version']}")
        
        recommended_params = recommend_training_params(gpu_info['memory_total_gb'])
    
    print(f"\n📋 推荐配置:")
    print(f"  配置类型: {recommended_params['description']}")
    print(f"  批次大小: {recommended_params['batch_size']}")
    print(f"  梯度累积: {recommended_params['gradient_accumulate_every']}")
    print(f"  图像尺寸: {recommended_params['image_size']}x{recommended_params['image_size']}")
    print(f"  工作进程: {recommended_params['num_workers']}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'config.json')
    
    print(f"\n💾 更新配置文件: {config_path}")
    
    if gpu_info:
        config = update_config_file(config_path, gpu_info, recommended_params)
    else:
        config = update_config_file(config_path, {'pytorch_version': 'cpu', 'cuda_version': 'N/A', 
                                                   'name': 'CPU', 'memory_total_gb': 0}, 
                                   recommended_params)
    
    print("✓ 配置文件已更新")
    
    print("\n" + "=" * 60)
    print("配置完成！")
    print("=" * 60)
    
    print("\n💡 使用建议:")
    if gpu_info and gpu_info['memory_total_gb'] < 6:
        print("  - 显存较小，建议使用较小的图像尺寸")
        print("  - 如遇显存不足，可降低batch_size或增大gradient_accumulate_every")
        print("  - 训练时间可能较长，请耐心等待")
    elif gpu_info:
        print("  - GPU配置良好，可以正常训练")
        print("  - 如需加速，可适当增加batch_size")
    else:
        print("  - CPU训练速度较慢，建议使用GPU")
    
    return config

if __name__ == "__main__":
    main()
