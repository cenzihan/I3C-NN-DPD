import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as scio
from thop import profile
from ptflops import get_model_complexity_info
# import numpy as np
from torch.utils.data import DataLoader
from enable_wandb import WANDB_MODE
if WANDB_MODE != 'disabled':
    import wandb
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# 检查传递的参数个数是否足够
if len(sys.argv) < 2:
    label = 980480
else:
    label = sys.argv[1]
    print("Argument 1:", label)

# 1. Start a W&B Run,
config = dict(
    learning_rate=0.001,
    epochs=100,
    M=5,
    neurons =24,
    isDPD=True,
    activateF="nn.Tanh()",
    DatasetName = '_DPDtrain_220k_M5_Len1600_MCS9_CBW80_fs480m_'
    # PAname='5_M5_L215k_PAtrain'
)

# 1. Start a W&B Run,
if WANDB_MODE != 'disabled':
    run = wandb.init(
        project="ARVTDNN0416",
        config=config,
        mode=WANDB_MODE
    )
    learning_rate = wandb.config.learning_rate
    epochs = wandb.config.epochs
    M = wandb.config.M
    isDPD = wandb.config.isDPD
    DatasetName = wandb.config.DatasetName
    neurons = wandb.config.neurons
else:
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    M = config['M']
    isDPD = config['isDPD']
    DatasetName = config['DatasetName']
    neurons = config['neurons']


# 定义网络结构
class ARVTDNN(nn.Module):
    def __init__(self):
        super(ARVTDNN, self).__init__()
        feature_num = neurons
        self.fc1 = nn.Linear(in_features=5 * (M + 1), out_features=feature_num)
        self.activateF = eval(config['activateF'])
        self.fc2 = nn.Linear(in_features=feature_num, out_features=2)

    def forward(self, x):

        x = self.fc1(x)
        x = self.activateF(x)
        x = self.fc2(x)
        return x


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        input_path = f'./data_xzr/input_data_ARVTDNN{DatasetName}.mat'
        label_path = f'./data_xzr/label_data_NN{DatasetName}.mat'
        input_data = scio.loadmat(input_path)
        labels_data = scio.loadmat(label_path)
        input_data_array = input_data['input_data_ARVTDNN']
        labels_data_array = labels_data['label_data_NN']
        self.data = torch.FloatTensor(input_data_array).permute(1, 0).to(device)
        self.label = torch.FloatTensor(labels_data_array).to(device)
        self.length = len(self.label)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx, :].unsqueeze(0), self.label[idx, :]

if __name__ == "__main__":
    # 检查是否有可用GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 加载数据集
    mydataset = MyDataset()
    batchsize = 256
    if isDPD:
        trainset_ratio = 0.9
    else:
        trainset_ratio = 0.9
    trainsize = int(len(mydataset) * trainset_ratio)
    testsize = int(len(mydataset) - trainsize)
    trainset, testset = torch.utils.data.random_split(mydataset, [trainsize, testsize])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False)

    net = ARVTDNN().to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss(reduction='mean')
    # loss = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
    #输出参数数量
    params = sum(p.numel() for p in net.parameters())
    print("Total number of parameters: ", params)
    # 输出参数数量和FLOPs
    inputdim = torch.randn(1, (M + 1)*5).to(device)
    flops1, params1 = profile(net, (inputdim,))
    print('sub:\n  macs: %0.2f , params: %0.2f ' % (flops1, params1))

    macs, params = get_model_complexity_info(net, (1, (M + 1)*5), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # 训练网络
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            net.train()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs= outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if i % 100 == 99:
        print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss))
        if WANDB_MODE != 'disabled':
            wandb.log({"loss": running_loss})

        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                net.eval()
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                outputs= outputs.squeeze()
                MSE = criterion(outputs, labels)
                SQUARES = labels ** 2
                NMSE = 10 * torch.log10(MSE / SQUARES.mean()).cpu().numpy()
            if WANDB_MODE != 'disabled':
                wandb.log({"NMSEvalid": NMSE})
    print('Finished Training')


    if isDPD:
        torch.save(net.state_dict(), './coefs/ARVTDNN_DPDmodel%s.pth' % (label))
    else:
        torch.save(net.state_dict(), './coefs/ARVTDNN_PAmodel%s.pth' % (label))


    print(testloader)
    # 测试网络
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            net.eval()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            outputs= outputs.squeeze()
            MSE = criterion(outputs, labels)
            SQUARES = labels ** 2
            NMSE = 10 * torch.log10(MSE / SQUARES.mean()).cpu().numpy()
        if WANDB_MODE != 'disabled':
            wandb.log({"NMSEtest": NMSE})

    print('NMSE of the network on the %d test set : %f ' % (testsize, NMSE))
