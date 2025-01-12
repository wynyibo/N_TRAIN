#实现一个简单的神经网络训练和测试，用于分类任务
#训练过程使用 Adam优化器 和 交叉熵损失函数，通过反向传播来优化模型参数。
#测试过程中，通过计算 TP、FP、TN、FN 来评估模型的性能，输出准确率、精确率、召回率和 F2 分数等指标。

import torch.nn.functional as F
import torch.nn as nn
import torch

# 定义一个简单的前馈神经网络
class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        # 调用父类初始化函数
        super().__init__()
        # 使用nn.Sequential构建一个简单的三层神经网络
        self.model = nn.Sequential(
            # 第一层：全连接层，输入维度in_dim，输出32，bias=False表示没有偏置项
            nn.Linear(in_dim, 32, bias=False),
            # 激活函数ReLU
            nn.ReLU(),
            # 第二层：全连接层，输入32，输出32，bias=False
            nn.Linear(32, 32, bias=False),
            nn.ReLU(),
            # 第三层：全连接层，输入32，输出out_dim（类别数），bias=False
            nn.Linear(32, out_dim, bias=False),
        )
    
    # 定义前向传播函数
    def forward(self, X):
        return self.model(X)
    

# 训练函数
def train(num_epoch, train_loader, model, device, lr, model_path = None):
    """
    训练神经网络
    :param num_epoch: 训练的轮数
    :param train_loader: 训练数据加载器
    :param model: 训练的模型
    :param device: 训练设备（'cuda' 或 'cpu'）
    :param lr: 学习率
    :param model_path: 模型保存路径（可选）
    :return: 训练完成后的模型
    """
    criterion = nn.CrossEntropyLoss()  # 损失函数，交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器，Adam优化算法
    model.to(device)  # 将模型移到指定设备（GPU或CPU）

    for epoch in range(num_epoch):
        total_loss = 0  # 每一轮的总损失
        for X, y in train_loader:  # 遍历训练数据集
            X, y = X.to(device), y.to(device)  # 将输入和标签数据移到指定设备
            
            output = model(X)  # 前向传播，获取模型输出
            loss = criterion(output, y)  # 计算损失
            total_loss += loss.item()  # 累加损失
            
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

        print(f'{epoch}: {total_loss / len(train_loader)}')  # 打印每一轮的平均损失
        
        # 如果指定了模型路径，则保存模型
        if model_path is not None:
            torch.save(model, model_path)

    return model  # 返回训练好的模型


# 测试函数
def test(test_loader, model, device):
    """
    测试神经网络
    :param test_loader: 测试数据加载器
    :param model: 测试的模型
    :param device: 测试设备（'cuda' 或 'cpu'）
    :return: 损失、精确率、召回率、F2分数
    """
    model.to(device)  # 将模型移到指定设备
    criterion = nn.CrossEntropyLoss()  # 损失函数，交叉熵损失
    with torch.no_grad():  # 在测试时不计算梯度，提高效率
        TP = 0  # 真阳性（True Positives）
        FP = 0  # 假阳性（False Positives）
        TN = 0  # 真阴性（True Negatives）
        FN = 0  # 假阴性（False Negatives）
        loss = 0  # 累计损失
        
        # 遍历测试数据集
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)  # 将输入和标签数据移到指定设备
            
            output = model(X)  # 前向传播，获取模型输出
            loss += criterion(output, y).item()  # 累加损失
            
            # 计算预测类别，使用softmax获取概率分布，然后使用argmax选择最大概率的类别
            y_pred = F.softmax(output, dim=1).argmax(dim=1)
            
            # 计算TP, FP, TN, FN
            TP += torch.logical_and(y, y_pred).sum().item()  # 真阳性：预测为1且实际为1
            FP += torch.logical_and(torch.logical_not(y), y_pred).sum().item()  # 假阳性：预测为1但实际为0
            TN += torch.logical_and(torch.logical_not(y), torch.logical_not(y_pred)).sum().item()  # 真阴性：预测为0且实际为0
            FN += torch.logical_and(y, torch.logical_not(y_pred)).sum().item()  # 假阴性：预测为0但实际为1
        
        # 计算指标
        A = 0.0  # 精确度（Accuracy）
        P = 0.0  # 精确率（Precision）
        R = 0.0  # 召回率（Recall）
        F2 = 0.0  # F2分数（强调召回率）
        
        # 计算准确率、精确率、召回率和F2分数
        try:
            A = (TP + TN) / (TP + FP + TN + FN)  # 准确率
            P = TP / (TP + FP)  # 精确率
            R = TP / (TP + FN)  # 召回率
            F2 = 2 * P * R / (P + R)  # F2分数
            print(f'A:{A:.3f} P:{P:.3f} R:{R:.3f} F:{F2:.3f}')  # 打印指标
        except ZeroDivisionError:  # 防止除0错误
            print(TP, FP, FN)

        # 返回测试结果，包括平均损失、精确率、召回率和F2分数
        return (loss / len(test_loader), P, R, F2)
