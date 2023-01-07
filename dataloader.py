import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os
import shutil

dataset_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_trans, download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_trans, download=True)

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

if os.path.exists("logs"):
    shutil.rmtree("logs")

writer = SummaryWriter("logs")
step = 0
for data in test_loader:
    imgs, labels = data
    writer.add_images("net", imgs, step)
    step += 1
writer.close()
