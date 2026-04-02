import torch
import psutil
import time
import os
import multiprocessing

def monitor_resources(interval=2):
    """监控GPU和系统资源"""
    print("=" * 60)
    print("资源监控")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print(f"总内存: {psutil.virtual_memory().total / 1024**3:.2f} GB")
    print("=" * 60)
    
    print("\n开始监控 (每{}秒更新一次)...\n".format(interval))
    print("{:<10} {:<15} {:<15} {:<15} {:<15}".format(
        "时间", "GPU显存(%)", "GPU显存(GB)", "系统内存(%)", "系统内存(GB)"
    ))
    print("-" * 70)
    
    start_time = time.time()
    
    try:
        while True:
            elapsed = time.time() - start_time
            
            if torch.cuda.is_available():
                gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
                gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
                gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_percent = (gpu_reserved / gpu_total) * 100
            else:
                gpu_percent = 0
                gpu_allocated = 0
            
            mem = psutil.virtual_memory()
            mem_percent = mem.percent
            mem_used = mem.used / 1024**3
            
            print("{:<10.1f} {:<15.1f} {:<15.2f} {:<15.1f} {:<15.2f}".format(
                elapsed, gpu_percent, gpu_reserved if torch.cuda.is_available() else 0, 
                mem_percent, mem_used
            ))
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    monitor_resources()
