import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as scio
import numpy as np
from torch.utils.data import DataLoader
from thop import profile
from ptflops import get_model_complexity_info
from SelfAttention_X import SelfAttention
import wandb
import time
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# 检查传递的参数个数是否足够，用于脚本调参实验
if len(sys.argv) < 3:
    label = 980480
    activateF = "nn.Tanh()"
    neuronsN = 10
else:
    label = sys.argv[1]
    activateF = sys.argv[2]
    neuronsN = int(sys.argv[3])
    # print("Argument 1:", label)
    # print("Argument 2:", activateF)


# 1. Start a W&B Run,
#wandb 调整参数
config=dict(
  learning_rate=0.001,
  epochs= 100,
  M=5,
  dim_kq = 5,
  neurons=neuronsN,
  isDPD = True,
  # activateF1="relu",
  activate = activateF,
  DatasetName = '_DPDtrain_220k_M5_Len1600_MCS9_CBW80_fs480m_'
)

run = wandb.init(
  project="RVTDSAN0416",
  # tags=["FC2"],
  config=config,
  # notes="My first experiment",
)

learning_rate = wandb.config.learning_rate
epochs = wandb.config.epochs
M = wandb.config.M
dim_kq = wandb.config.dim_kq
neurons = wandb.config.neurons
activate = wandb.config.activate
isDPD = wandb.config.isDPD
DatasetName = wandb.config.DatasetName
print(wandb.config)

# 定义网络结构
class RVTDSAN(nn.Module):
    def __init__(self):
        super(RVTDSAN,self).__init__()
        feature_num = neurons
        self.SelfAttention = SelfAttention(dim_q = dim_kq, dim_k = dim_kq, dim_v = 5, dim = 5 )
        self.activate = nn.LeakyReLU()
        self.fc1 = nn.Linear(in_features=5 * (M+1), out_features=feature_num)
        self.fc2 = nn.Linear(in_features=feature_num, out_features=2)
        # self.saved_x = None  # 添加一个成员变量来保存 x 的值

    def forward(self, x):
        # 输入b * 4 * 5
        # b = x.shape[0]
        # x = self.embedding(x)
        x = self.SelfAttention(x)
        # x = self.activate(x)
        # self.saved_x = x.detach()  # 在 forward 函数中更新 saved_x 的值
        # x = x.reshape(b, -1)
        # x = self.activate(x)
        x = x.view(-1 ,5 * (M+1))
        x = self.fc1(x)
        x = self.activate(x)
        x = self.fc2(x)
        return x


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        input_data = scio.loadmat('./data_xzr/%s/input_data_RVTDSAN%s.mat'%(DatasetName,DatasetName))
        labels_data = scio.loadmat('./data_xzr/%s/label_data_NN%s.mat'%(DatasetName,DatasetName))
        input_data_array = input_data['input_data_RVTDSAN']
        labels_data_array = labels_data['label_data_NN']
        self.data = torch.FloatTensor(input_data_array).permute(2, 0, 1).to(device)
        self.label = torch.FloatTensor(labels_data_array).to(device)
        self.length = len(self.label)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx, :, :].unsqueeze(0), self.label[idx, :]

if __name__ == "__main__":
    start = time.time()
    # 检查是否有可用GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"wandb config: {wandb.config}")
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
    testloader = torch.utils.data.DataLoader(testset, batch_size=2048, shuffle=False)

    net = RVTDSAN().to(device)
    print('activate function:%s'%(net.activate))
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
    # 输出参数数量
    # params = sum(p.numel() for p in net.parameters())
    # print("Total number of parameters: ", params)
    # 输出参数数量和FLOPs
    inputdim = torch.randn(1, M+1, 5).to(device)
    macs1, params1 = profile(net, inputs=(inputdim,))
    print('sub:\n  macs: %0.2f , params: %0.2f ' % (macs1 , params1))

    macs, params = get_model_complexity_info(net, (M+1, 5), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # 训练网络
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            net.train()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.squeeze()
            # labels= labels.squeeze()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if i % 100 == 99:
        print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss))
        wandb.log({"loss": running_loss})
        # 测试网络
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                net.eval()
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.squeeze()
                outputs = net(inputs)
                MSE = criterion(outputs, labels)
                SQUARES = labels ** 2
                NMSE = 10 * torch.log10(MSE/SQUARES.mean()).cpu().numpy()
            wandb.log({"NMSEvalid": NMSE})
    print('Finished Training')

    # for param_tensor in net.state_dict():
    #     print(param_tensor, "\t", net.state_dict()[param_tensor])
    # 打印自注意矩阵
    # print(net.saved_x)

    if isDPD:
        torch.save(net.state_dict(), './coefs/RVTDSAN_DPDmodel%s.pth'% (label))
    else:
        torch.save(net.state_dict(), './coefs/RVTDSAN_PAmodel%s.pth'% (label))

    # 测试网络
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            net.eval()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.squeeze()
            outputs = net(inputs)
            MSE = criterion(outputs, labels)
            SQUARES = labels ** 2
            NMSE = 10 * torch.log10(MSE/SQUARES.mean()).cpu().numpy()
        wandb.log({"NMSEtest": NMSE})


    print('NMSE of the network on the %d test set : %f ' % (testsize, NMSE))

    end = time.time()
    print ('Run time: %f s' %(end-start))
    # 4. Log an artifact to W&B
    # wandb.log_artifact(net)
    # Optional: save model at the end
    # net.to_onnx()
    # wandb.save("model.onnx")