import os
# os.chdir('..')
# 创建文件夹名列表
folder_names = [f"MMGP_OL{i}" for i in  range(2, 8)]

# 遍历每个文件夹
for folder in folder_names:
    # 确保文件夹存在，如果不存在则创建
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # 文件路径
    file_path = os.path.join(folder, "flag.txt")
    
    # 写入 '0' 到 flag.txt
    with open(file_path, 'w') as file:
        file.write('1')
