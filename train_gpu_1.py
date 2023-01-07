from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
import time
import os
import shutil
from module import *

# 使用GPU加速进行训练
# 定义训练的设备
dev = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")

# 训练的轮数
epoch = 150

# 当训练集大于此数值会进行测试
accuracy_level = 0.95

# 数据集准备
train_data = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
test_data = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

# 数据集加载
train_loader = DataLoader(dataset=train_data, batch_size=64)
test_loader = DataLoader(dataset=test_data, batch_size=64)

# 创建网络模型,转移至训练设备
module = Mymodule().to(device=dev)

# 损失函数,转移至训练设备
loss_fn = nn.CrossEntropyLoss().to(device=dev)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(params=module.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数

train_step = 0

# 记录测试的次数
test_step = 0

# 使用tensorboard画出训练曲线
writer = SummaryWriter("train_logs")

# 设置模型存储目录
save_path = "model_file"
save_path = os.path.join(save_path, str(time.time()))
os.makedirs(save_path)

# -----开始训练-----
start_time = time.time()
for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i + 1))
    total_train_accuracy = 0  # 记录每次迭代的训练集正确次数
    total_test_accuracy = 0  # 记录每次迭代的测试集正确次数
    # 训练步骤
    # module.train() # 设定为训练模式，仅对某些特殊层生效，具体看说明文档
    for data in train_loader:
        imgs, labels = data
        # 转移至训练设备
        imgs = imgs.to(dev)
        labels = labels.to(dev)
        outputs = module(imgs)

        # 统计正确预测的数量
        total_train_accuracy += (outputs.argmax(1) == labels).sum()
        loss = loss_fn(outputs, labels)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_step += 1
    # 计算本轮训练集的正确率
    total_train_accuracy = total_train_accuracy / len(train_data)
    print("-----训练次数:{}-----,loss:{}".format(train_step, loss))
    print("-----训练集正确率:{}-----".format(total_train_accuracy))

    writer.add_scalar("train_loss", loss.item(), train_step)

    # 训练集准确度达标则进行测试
    if total_train_accuracy >= accuracy_level:
        # 测试步骤开始
        total_test_loss = 0
        # module.eval() # 设定为验证模式，仅对某些特殊层生效，具体看说明文档
        with torch.no_grad():  # 去掉梯度，保证测试过程不会对网络模型的参数调优
            for data in test_loader:
                imgs, labels = data
                # 转移至训练设备
                imgs = imgs.to(dev)
                labels = labels.to(dev)

                outputs = module(imgs)
                total_test_accuracy += (outputs.argmax(1) == labels).sum()
                loss = loss_fn(outputs, labels)
                total_test_loss += loss
        total_test_accuracy = total_test_accuracy / len(test_data)
        print("-----测试集Loss:{}-----".format(total_test_loss))
        print("-----测试集正确率:{}-----".format(total_test_accuracy))
        writer.add_scalar("test_loss", loss.item(), test_step)
        test_step += 1
        # 测试集准确度达标则进行保存，并且退出迭代训练
        if total_test_accuracy >= accuracy_level:
            torch.save(
                module, "{}/module_epoch={}_accuracy=".format(save_path, i, total_test_accuracy))
            break
# -----迭代结束-----

# 如果没有模型保存，则进行模型保存
if not os.listdir(save_path):
    torch.save(module, "{}/module_epoch={}_accuracy=".format(save_path,
               epoch, total_test_accuracy))
writer.close()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")