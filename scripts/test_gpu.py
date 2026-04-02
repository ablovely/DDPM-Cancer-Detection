import torch

print("="*60)
print("GPU检测测试")
print("="*60)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    
    device = torch.device('cuda')
    x = torch.rand(3, 3).to(device)
    print(f"\n✓ 成功在GPU上创建张量")
    print(f"张量设备: {x.device}")
else:
    print("\n⚠ 未检测到GPU，将使用CPU")
print("="*60)
