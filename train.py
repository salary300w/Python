from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
import time
import os
import shutil
from module import *
from emailtool import *


def train(epoch=200, dev="cuda", email=True, email_addr="Atm991014@163.com", accuracy_level=1, tensorboard=True):

    # epoch:迭代次数
    # dev:训练设备
    # email:训练完成是否发送邮件通知
    # email_addr:接收通知的邮箱地址
    # accuracy_level:当训练集准确率大于accuracy_level,会进行测试。测试集准确率大于accuracy_level会进行模型保存并结束训练
    # tensorboard:是否使用tensorboard绘制训练曲线

    # 定义训练的设备
    dev = torch.device(device=dev if torch.cuda.is_available() else "cpu")

    # 数据集准备
    train_data = torchvision.datasets.CIFAR10(
        root="./dataset",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    test_data = torchvision.datasets.CIFAR10(
        root="./dataset",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    # 数据集大小
    print("-----训练集大小= {} -----".format(len(train_data)))
    print("-----测试集大小= {} -----".format(len(test_data)))

    # 数据集加载
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=4)

    # 创建网络模型,转移至训练设备
    module = Mymodule().to(device=dev)

    # 损失函数,转移至训练设备
    loss_fn = nn.CrossEntropyLoss().to(device=dev)

    # 优化器
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(params=module.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(params=module.parameters(), lr=learning_rate)

    # 设置训练网络的一些参数
    # 记录训练的次数

    train_step = 0

    # 记录测试的次数
    test_step = 0

    if tensorboard:
        # 使用tensorboard画出训练曲线
        if os.path.exists("train_logs"):
            shutil.rmtree("train_logs")
        writer = SummaryWriter("train_logs")

    # 设置模型存储目录
    save_path = "module_file"
    save_path = os.path.join(save_path, str(time.time()))
    os.makedirs(save_path)

    # -----开始训练-----
    print("-----开始训练-----")
    start_time = time.time()
    for i in range(1, epoch+1):
        print("-----第 {} 轮训练开始-----".format(i))
        total_train_accuracy = 0  # 记录每次迭代的训练集准确率
        total_train_loss = 0  # 记录每次迭代的总误差值
        total_test_accuracy = 0  # 记录每次迭代的测试集正确率
        total_test_loss = 0  # 记录每次迭代的总误差值
        # 训练步骤
        module.train() # 设定为训练模式，仅对某些特殊层生效，具体看说明文档
        for data in train_loader:
            imgs, labels = data
            # 转移至训练设备
            imgs = imgs.to(dev)
            labels = labels.to(dev)

            # 将数据输入模型
            outputs = module(imgs)

            # 统计训练集正确预测的数量
            total_train_accuracy += (outputs.argmax(1) == labels).sum()

            # 累加训练集的损失值
            loss = loss_fn(outputs, labels)
            total_train_loss += loss

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_step += 1
        # 计算本轮训练集的正确率
        total_train_accuracy = total_train_accuracy / len(train_data)
        print("-----训练次数: {} loss: {} -----".format(train_step, loss))
        print("-----训练集正确率: {} -----".format(total_train_accuracy))
        print(f"-----总用时: {time.time()-start_time:.2f} 秒-----")

        # 绘制训练曲线图
        if tensorboard:
            # writer.add_scalar(tag="train_loss", scalar_value=total_train_loss, global_step=i)
            writer.add_scalar(tag="train_accuracy", scalar_value=total_train_accuracy, global_step=i)

        # 训练集准确度达标则进行测试
        if total_train_accuracy >= accuracy_level or i == epoch:
            # 测试步骤开始
            module.eval() # 设定为验证模式，仅对某些特殊层生效，具体看说明文档
            with torch.no_grad():  # 去掉梯度，保证测试过程不会对网络模型的参数调优
                for data in test_loader:
                    imgs, labels = data
                    # 转移至训练设备
                    imgs = imgs.to(dev)
                    labels = labels.to(dev)
                    # 将数据输入模型
                    outputs = module(imgs)

                    # 统计测试集正确预测的数量
                    total_test_accuracy += (outputs.argmax(1) == labels).sum()

                    # 累加测试集的损失值
                    loss = loss_fn(outputs, labels)
                    total_test_loss += loss
            test_step += 1

            # 计算测试集正确率
            total_test_accuracy = total_test_accuracy / len(test_data)

            # 绘制测试曲线图
            if tensorboard:
                # writer.add_scalar(tag="test_loss", scalar_value=total_test_loss, global_step=test_step)
                writer.add_scalar(tag="test_accuracy", scalar_value=total_test_accuracy, global_step=test_step)
            print("-----测试集Loss: {} -----".format(total_test_loss))
            print("-----测试集正确率: {} -----".format(total_test_accuracy))
            print(f"-----总用时: {time.time()-start_time:.2f} 秒-----")

            # 测试集准确度达标则进行保存，并且退出迭代训练
            if total_test_accuracy >= accuracy_level:
                savemodule(MODULE=module, PATH=save_path, ACCURACY=total_test_accuracy)
                break
    # -----迭代结束-----
    print("-----训练完成-----")

    # 如果没有模型保存，则进行模型保存
    if not os.listdir(save_path):
        savemodule(MODULE=module, PATH=save_path, ACCURACY=total_test_accuracy)
    writer.close()

    # 计算训练用时并输出
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"-----训练总用时: {elapsed_time:.2f} 秒-----")

    # 发送邮件
    if email:
        print("-----发送邮件-----")
        sendemail = Email(email_addr)
        sendemail.send(
            "训练完成<br/>测试集Loss:{}<br/>测试集正确率:{}<br/>迭代次数:{}<br/>训练总次数:{}<br/>用时:{}秒".format(
                total_test_loss, total_test_accuracy, i, train_step, elapsed_time
            )
        )


def savemodule(MODULE, PATH, ACCURACY):
    print("-----保存模型参数-----")
    MODULE.to(torch.device(device="cpu"))  # 将模型转移至cpu保存
    torch.save(MODULE, os.path.join(PATH, "module_accuracy={}".format(round(ACCURACY.item(), 5))))


if __name__ == "__main__":
    train(epoch=100, email=True)