import os

folder_path = "D:\workspace\Python\数据增强\原数据\承插不到位"  # 替换为实际文件夹路径
file_name = "cc"
i = 1
# 遍历文件夹
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    new_file_name = file_name+str(i)+".jpg"
    i = i+1
    if os.path.isfile(file_path):
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(file_path, new_file_path)

folder_path = os.path.join(folder_path, "SegmentationClassPNG")

i = 1
# 遍历文件夹
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    new_file_name = file_name+str(i)+".png"
    i = i+1
    if os.path.isfile(file_path):
        new_file_path = os.path.join(folder_path, new_file_name)
        os.rename(file_path, new_file_path)
