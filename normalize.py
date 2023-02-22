import torch
import torchvision.transforms as transforms
import torchvision.io
from PIL import Image

# 定义均值和标准差
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

tensor=torchvision.io.read_image('data/val/images/cc7.jpg').to(dtype=torch.float32)
print(type(tensor))

# 定义归一化和反归一化的 transform
normalize = transforms.Normalize(mean=mean, std=std)
denormalize = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)],
                                   std=[1/s for s in std])
Image.fromarray(tensor.permute(1, 2, 0).numpy().astype('uint8')).save('0.png')
# 对 tensor 进行归一化处理
tensor_normalized = normalize(tensor)
Image.fromarray(tensor_normalized.permute(1, 2, 0).numpy().astype('uint8')).save('1.png')

# 对 tensor_normalized 进行反归一化处理
tensor_denormalized = denormalize(tensor_normalized)
Image.fromarray(tensor_denormalized.permute(1, 2, 0).numpy().astype('uint8')).save('2.png')