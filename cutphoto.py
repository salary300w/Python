from PIL import Image
import os


def cutphoto(path, x, y, w, h, save_path=""):
    """
    path:图片路径
    x,y:像素起点
    w,h:裁剪宽度与高度
    """
    # 打开原始图像
    image = Image.open(path)

    # 裁剪图像
    cropped_image = image.crop((x, y, x + w, y + h))

    # 保存修改后的图像
    if save_path == "":
        save_path = path
    cropped_image.save(save_path)


def main():
    folder_path = "/mnt/d/workspace/论文/datadealbypython"
    for child_folder in os.listdir(folder_path):
        child_folder = os.path.join(folder_path, child_folder)
        for file_name in os.listdir(child_folder):
            file_path = os.path.join(child_folder, file_name)
            cutphoto(file_path, 250, 100, 400, 260)
        print(child_folder+"***裁剪完成***")
if __name__ == "__main__":
    main()
