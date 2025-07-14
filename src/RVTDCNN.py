import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as scio 
# import numpy as np
from torch.utils.data import DataLoader
from thop import profile
from ptflops import get_model_complexity_info
from enable_wandb import WANDB_MODE
if WANDB_MODE != 'disabled':
    import wandb
import os
import sys

# learning_rate = 0.001
# epochs = 100
# M = 5
# neurons = 10
# isDPD = False
# activateF = "nn.Tanh()"
# DatasetName = '0dB_Len200_MCS3_CBW20_fs100_M5_L208k_PAtrain'
#
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
    neurons=12,
    isDPD = True,
    activateF="nn.Tanh()",
    DatasetName = '_test_29k_M5_Len1600_MCS9_CBW40_fs246m_'
    # DatasetName= '0dB_Len200_MCS3_CBW20_fs100_M5_L208k_PAtrain'
 
)

if WANDB_MODE != 'disabled':
    run = wandb.init(
        project="RVTDCNN0416",
        # notes="My first experiment",
        # tags=["FC2"],
        config=config,
        mode=WANDB_MODE
    )
    learning_rate = wandb.config.learning_rate
    epochs = wandb.config.epochs
    M = wandb.config.M
    neurons = wandb.config.neurons
    isDPD = wandb.config.isDPD
    activateF = wandb.config.activateF
    DatasetName = wandb.config.DatasetName
else:
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    M = config['M']
    neurons = config['neurons']
    isDPD = config['isDPD']
    activateF = config['activateF']
    DatasetName = config['DatasetName']


# 定义网络结构
class RVTDCNN(nn.Module):
    def __init__(self):
        super(RVTDCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0)
        self.activateF = nn.Tanh()
        feature_num = neurons
        self.fc1 = nn.Linear(in_features=3*3*(M-1), out_features=feature_num)
        self.fc2 = nn.Linear(in_features=feature_num, out_features=2)

    def forward(self, x):
        x = self.activateF(self.conv1(x))
        x = x.view(-1, 3*3*(M-1))
        x = self.fc1(x)
        x = self.activateF(x)
        x = self.fc2(x)
        return x


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        input_path = f'./data_xzr/input_data_RVTDCNN{DatasetName}.mat'
        label_path = f'./data_xzr/label_data_NN{DatasetName}.mat'
        input_data = scio.loadmat(input_path)
        labels_data = scio.loadmat(label_path)
        input_data_array = input_data['input_data_RVTDCNN']
        labels_data_array = labels_data['label_data_NN']
        self.data = torch.FloatTensor(input_data_array).permute(2, 0, 1).to(device)
        self.label = torch.FloatTensor(labels_data_array).to(device)
        self.length = len(self.label)
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx, :, :].unsqueeze(0), self.label[idx, :]

if __name__ == "__main__":
    istraning = 1  # 是否训练网络，否直接读取系数
    # 检查是否有可用GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    net = RVTDCNN().to(device)
    # 输出参数数量和FLOPs
    inputdim = torch.randn(1, 5, M + 1).to(device)
    macs1, params1 = profile(net, inputs=(inputdim,))
    print('sub:\n  macs: %0.2f , params: %0.2f ' % (macs1, params1))
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



    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
    #输出参数数量
    # params = sum(p.numel() for p in net.parameters())
    # print("Total number of parameters: ", params)
    # 输出参数数量和FLOPs
    inputdim = torch.randn(1, 5, M+1).to(device)
    macs1, params1 = profile(net, inputs=(inputdim,))
    print('sub:\n  macs: %0.2f , params: %0.2f ' % (macs1 , params1))

    macs, params = get_model_complexity_info(net, (5, M+1), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    # 训练网络
    if istraning ==1:
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                net.train()
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # if i % 100 == 99:
            print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss))
            if WANDB_MODE != 'disabled':
                wandb.log({ "loss": running_loss})

        print('Finished Training')
        if isDPD:
            torch.save(net.state_dict(), './coefs/RVTDCNN_DPDmodel%s.pth'% (label))
        else:
            torch.save(net.state_dict(), './coefs/RVTDCNN_PAmodel%s.pth'% (label))

    # if istraning==0:
    #     net.load_state_dict(torch.load('/coefs/RVTDCNN.pth'))

    # 测试网络
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            net.eval()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            MSE = criterion(outputs, labels)
            SQUARES = labels**2
            NMSE = 10 * torch.log10(MSE/SQUARES.mean()).cpu().numpy()
        if WANDB_MODE != 'disabled':
            wandb.log({"NMSE": NMSE})

    # print(labels)
    # print(outputs)

    # print('NMSE of the network on the 1000 test set : %f ' % (10000000*MSE))
    print('NMSE of the network on the %d test set : %f ' % (testsize, NMSE))


# 4. Log an artifact to W&B
# wandb.log_artifact(net)
# Optional: save model at the end
# net.to_onnx()
# wandb.save("model.onnx")