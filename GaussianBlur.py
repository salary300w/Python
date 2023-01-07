import cv2

# 读取图像
img = cv2.imread('photo/head.png')

# 高斯滤波
gaussian = cv2.GaussianBlur(img, (5, 5), 0)

# 存储原图和处理后的图像
cv2.imwrite('Original_Image.png', img)
cv2.imwrite('Gaussian_Filtered_Image.png', gaussian)