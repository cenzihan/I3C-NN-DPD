import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random


# 定义DPD神经网络模型
class DPDNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation):
        super(DPDNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.activation = activation

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        self.layers.append(nn.Linear(hidden_layers[-1], output_size))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.activation(self.layers[i](x))
        return x


# 定义搜索空间和评价函数
search_space = {
    'input_size': [32, 64, 128],
    'output_size': [32, 64, 128],
    'hidden_layers': [[64, 32], [128, 64], [256, 128]],
    'activation': [F.relu, F.tanh, F.sigmoid]
}


def evaluate(model, eval_data):
    criterion = nn.MSELoss()
    loss = 0.0
    for x, y in eval_data:
        x, y = Variable(x), Variable(y)
        output = model(x)
        loss += criterion(output, y)
    return loss.item()


# 定义搜索算法
def search(eval_data, search_space, num_trials):
    best_model = None
    best_loss = float('inf')
    for i in range(num_trials):
        # 从搜索空间中随机采样一个模型
        input_size = random.choice(search_space['input_size'])
        output_size = random.choice(search_space['output_size'])
        hidden_layers = random.choice(search_space['hidden_layers'])
        activation = random.choice(search_space['activation'])
        model = DPDNet(input_size, output_size, hidden_layers, activation)

        # 训练模型并评估性能
        optimizer = optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        for epoch in range(100):
            for x, y in eval_data:
                x, y = Variable(x), Variable(y)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
        loss = evaluate(model, eval_data)

        # 选择最优模型
        if loss < best_loss:
            best_model = model
            best_loss = loss

    return best_model


# 示例代码：加载数据、划分训练集和测试集、执行搜索算法
# data =  # 加载数据
# train_data, eval_data =  # 划分训练集和测试集
# model = search(eval_data, search_space, num_trials=10)