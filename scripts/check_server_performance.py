#!/usr/bin/env python3
import psutil
import platform
import cpuinfo
import torch
import sys
from datetime import datetime


def get_size(bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f} {unit}{suffix}"
        bytes /= factor


def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def main():
    print_section("服务器性能检查报告")
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print_section("系统信息")
    uname = platform.uname()
    print(f"操作系统: {uname.system} {uname.release}")
    print(f"主机名: {uname.node}")
    print(f"架构: {uname.machine}")
    print(f"Python版本: {sys.version}")

    print_section("CPU信息")
    try:
        cpu_info = cpuinfo.get_cpu_info()
        print(f"CPU型号: {cpu_info.get('brand_raw', '未知')}")
    except:
        pass
    print(f"物理核心数: {psutil.cpu_count(logical=False)}")
    print(f"逻辑核心数: {psutil.cpu_count(logical=True)}")
    print(f"CPU使用率: {psutil.cpu_percent(interval=1)}%")
    print(f"CPU频率: {psutil.cpu_freq().current:.2f} MHz")

    print_section("内存信息")
    mem = psutil.virtual_memory()
    print(f"总内存: {get_size(mem.total)}")
    print(f"已用内存: {get_size(mem.used)}")
    print(f"可用内存: {get_size(mem.available)}")
    print(f"内存使用率: {mem.percent}%")

    print_section("磁盘信息")
    print("磁盘分区:")
    partitions = psutil.disk_partitions()
    for partition in partitions:
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
        except PermissionError:
            continue
        print(f"  {partition.device}")
        print(f"    挂载点: {partition.mountpoint}")
        print(f"    文件系统: {partition.fstype}")
        print(f"    总大小: {get_size(partition_usage.total)}")
        print(f"    已用: {get_size(partition_usage.used)}")
        print(f"    可用: {get_size(partition_usage.free)}")
        print(f"    使用率: {partition_usage.percent}%")

    print_section("GPU信息")
    if torch.cuda.is_available():
        print(f"CUDA可用: 是")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"PyTorch版本: {torch.__version__}")
        gpu_count = torch.cuda.device_count()
        print(f"GPU数量: {gpu_count}")
        for i in range(gpu_count):
            print(f"\n  GPU {i}:")
            print(f"    设备名: {torch.cuda.get_device_name(i)}")
            print(f"    总显存: {get_size(torch.cuda.get_device_properties(i).total_memory)}")
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            print(f"    已分配显存: {get_size(allocated)}")
            print(f"    已保留显存: {get_size(reserved)}")
    else:
        print("CUDA可用: 否 (未检测到NVIDIA GPU或未安装CUDA)")
        print("提示: 如果有GPU，请确保安装了正确的PyTorch版本和CUDA驱动")

    print_section("网络信息")
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            if str(address.family) == 'AddressFamily.AF_INET':
                print(f"  {interface_name}: {address.address}")

    print_section("检查完成")
    print("如需持续监控，可以运行: python scripts/monitor_resources.py")


if __name__ == "__main__":
    main()
