import os
import sys

def check_empty_files(directory):
    empty_files = []
    total_files = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                total_files += 1
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                
                if file_size == 0:
                    empty_files.append(file_path)
                    print(f"⚠ 发现空文件: {file_path}")
                
                if total_files % 1000 == 0:
                    print(f"已检查 {total_files} 个文件...", end='\r')
    
    return empty_files, total_files

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    
    print("=" * 60)
    print("开始检查所有图像文件")
    print("=" * 60)
    
    all_empty_files = []
    total_checked = 0
    
    for folder in ['positive', 'negative']:
        folder_path = os.path.join(data_dir, folder)
        if os.path.exists(folder_path):
            print(f"\n检查 {folder} 文件夹...")
            empty_files, file_count = check_empty_files(folder_path)
            all_empty_files.extend(empty_files)
            total_checked += file_count
            print(f"\n{folder} 文件夹检查完成: {file_count} 个文件")
            if empty_files:
                print(f"⚠ 发现 {len(empty_files)} 个空文件")
            else:
                print(f"✓ 未发现空文件")
    
    print("\n" + "=" * 60)
    print("检查完成！")
    print("=" * 60)
    print(f"总共检查: {total_checked} 个文件")
    print(f"发现空文件: {len(all_empty_files)} 个")
    
    if all_empty_files:
        print("\n空文件列表:")
        for f in all_empty_files:
            print(f"  - {f}")
    else:
        print("\n✓ 所有文件都正常！")
