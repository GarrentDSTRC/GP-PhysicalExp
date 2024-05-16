import os
import shutil
# 获取当前脚本所在的目录
current_directory = os.path.dirname(os.path.abspath(__file__))
# 源文件夹路径
source_folder = os.path.join(current_directory, 'MMGP_OL0')
# 目标文件夹路径
target_folder = current_directory
n=8
# 循环复制和重命名文件夹
for i in range(1, n):
    # 构造目标文件夹路径
    target_path = os.path.join(target_folder, f'MMGP_OL{i}')
    # 复制文件夹
    shutil.copytree(source_folder, target_path)
    # 循环重命名文件夹内的文件
    for file_name in os.listdir(target_path):
        # 构造源文件路径和目标文件路径
        source_file = os.path.join(target_path, file_name)
        target_file = os.path.join(target_path, file_name.replace('1', str(i)))
        # 重命名文件
        os.rename(source_file, target_file)