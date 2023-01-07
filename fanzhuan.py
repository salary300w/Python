
from PIL import Image
 
#读取图像
im = Image.open("D:\workspace\Python\cc000.png")
out = im.transpose(Image.FLIP_LEFT_RIGHT)
out.save("fanzhuan.png")