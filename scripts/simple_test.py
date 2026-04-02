import torch
import os

print("=" * 60)
print("系统环境检测")
print("=" * 60)

print(f"\n1. PyTorch版本: {torch.__version__}")
print(f"2. CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"3. GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"4. GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"5. CUDA版本: {torch.version.cuda}")

data_path = r"D:\Code\KeShe\Mycode\DDPM-Cancer-Detection\data\positive"
print(f"\n6. 数据目录: {data_path}")
print(f"7. 目录存在: {os.path.exists(data_path)}")

if os.path.exists(data_path):
    files = [f for f in os.listdir(data_path) if f.endswith('.png')]
    print(f"8. 图像数量: {len(files)}")
    if len(files) > 0:
        print(f"9. 第一个文件: {files[0]}")
        file_path = os.path.join(data_path, files[0])
        print(f"10. 文件大小: {os.path.getsize(file_path)} 字节")

print("\n" + "=" * 60)
print("检测完成")
print("=" * 60)
