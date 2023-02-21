from noise import *
import os
from PIL import Image
import cv2
import shutil

# folder_path = "D:\workspace\Python\datadealbypython\\chengcha"
# path = "D:\workspace\Python\deal"

# src_path = os.path.join(folder_path, "SegmentationClassPNG")
# des_path = os.path.join(path, "SegmentationClassPNG")
# shutil.copytree(src_path, des_path)



# # 遍历文件夹
# for filename in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, filename)
#     if os.path.isfile(file_path):
#         image = cv2.imread(file_path)
#         save_path = os.path.join(path, filename)
#         noisy_image = add_salt_and_pepper_noise(image,0.0005)
#         cv2.imwrite(save_path, noisy_image)


folder_path = "D:\workspace\Python\datadealbypython"
new_name_prefix = "fz"
# 遍历指定文件夹及其子文件夹内的所有文件
for root, dirs, files in os.walk(folder_path):
    for file_name in files:
        # 构造新文件名，新文件名由前缀 + 原文件名组成
        new_file_name = new_name_prefix + file_name
        # 构造原文件和新文件的完整路径
        old_file_path = os.path.join(root, file_name)
        #读取图像
        im = Image.open(old_file_path)
        out = im.transpose(Image.FLIP_LEFT_RIGHT)
        out.save(old_file_path)
        # 重命名文件
        new_file_path = os.path.join(root, new_file_name)
        os.rename(old_file_path, new_file_path)