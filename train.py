import torch
import os
import time
import datetime
from data.Unet.result.unet import UNet
from my_dataset import Data_Loader
from torch.utils.data import Dataset
from torch import optim
from torch import nn
def train_net(net, device, data_path, epochs=40, batch_size=2, lr=0.00001):
    train_dataset = Data_Loader(data_path, True)
    # 测试数据读取是否正常
    # print(train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    # 定义优化器
    optimiser = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    # 定义损失函数
    criterion = nn.BCEWithLogitsLoss()

    # best_loss统计，初始化为正无穷
    best_loss = float('inf')

    # 训练epochs次
    for epoch in range(epochs):

        # 训练模式
        net.train()

        # 按照batch_size训练
        for img, lbl in train_loader:

            # 梯度清零
            optimiser.zero_grad()

            # 数据拷贝到device上运行
            img = img.to(device=device, dtype=torch.float32)
            lbl = lbl.to(device=device, dtype=torch.float32)

            # 使用网络参数，输出预测结果
            pred = net(img)
            # 计算loss, loss是评估指标
            loss = criterion(pred, lbl)
            # loss的数值, loss.item()获取对应py类型，只是数值运算下使用，能不消耗内存节省开销
            print('Loss_train', loss.item())

            # 保存最小loss对应的参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')

            # 更新参数
            loss.backward()
            optimiser.step()

if __name__ == '__main__':

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载网络，图片单通道1，分类为1 + 1。
    net = UNet(n_channels=1, n_classes=1)

    # 将网络拷贝到deivce中
    net.to(device=device)

    # 指定训练集地址，开始训练
    data_path = 'E:\dataset\wk1_unet\data'
    train_net(net, device, data_path)