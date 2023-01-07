import os

folder_path = "D:\workspace\Python\datadealbypython" # 替换为实际文件夹路径
new_name_prefix = "add_noise" # 替换为新文件名前缀

# 遍历指定文件夹及其子文件夹内的所有文件
for root, dirs, files in os.walk(folder_path):
    for file_name in files:
        # 构造新文件名，新文件名由前缀 + 原文件名组成
        new_file_name = new_name_prefix + file_name
        # 构造原文件和新文件的完整路径
        old_file_path = os.path.join(root, file_name)
        new_file_path = os.path.join(root, new_file_name)
        # 重命名文件
        os.rename(old_file_path, new_file_path)