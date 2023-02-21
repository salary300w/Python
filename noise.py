import numpy as np


def add_gaussian_noise(image, mean=0.5, std=10):
    """
    对图像添加高斯噪声

    Args:
        image: 待添加噪声的图像
        mean: 高斯分布的均值，默认为0
        std: 高斯分布的方差，默认为10

    Returns:
        添加高斯噪声后的图像
    """
    # 生成与原图像形状相同的高斯噪声
    noise = np.random.normal(loc=mean, scale=std, size=image.shape)
    # 将噪声添加到原图像中
    noisy_image = image + noise
    # 将像素值限制在0-255之间
    noisy_image = np.clip(noisy_image, 0, 255)
    # 将图像数据类型转换为整型
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image


def add_salt_and_pepper_noise(image, prob=0.01):
    """
    对图像添加椒盐噪声

    Args:
        image: 待添加噪声的图像
        prob: 像素被替换为噪声的概率，默认为0.01

    Returns:
        添加椒盐噪声后的图像
    """
    # 复制原图像
    noisy_image = np.copy(image)
    # 生成随机的行列索引
    height, width = image.shape[:2]
    random_y = np.random.randint(0, height, int(height * width * prob))
    random_x = np.random.randint(0, width, int(height * width * prob))
    # 将随机的像素替换为0或255
    noisy_image[random_y, random_x] = [0, 0, 0]
    noisy_image[random_y, random_x] = [255, 255, 255]
    return noisy_image
