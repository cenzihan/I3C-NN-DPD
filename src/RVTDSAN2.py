import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as scio
import numpy as np
from torch.utils.data import DataLoader
from thop import profile
from vit import Transformer
from enable_wandb import WANDB_MODE
if WANDB_MODE != 'disabled':
    import wandb
import time
import os
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# 检查传递的参数个数是否足够
if len(sys.argv) < 4:
    label = 1
    activateF ='Leaky ReLU'
    neuronsN = 6
else:
    label = sys.argv[1]
    activateF = sys.argv[2]
    neuronsN = int(sys.argv[3])

# print("Argument 1:", label)
# print("Argument 2:", activateF)

# 1. Start a W&B Run,

config=dict(
  learning_rate=0.001,
  epochs= 200,
  M=5,
  isDPD = False,
  # activateF1="relu",
  activate = activateF,
  neurons=neuronsN,
  # PAname = '0dB_Len200_MCS3_CBW20_fs100_M5_L208k_DPDtrain'
  PAname = '5_M5_L215k_PAtrain'
)

if WANDB_MODE != 'disabled':
    run = wandb.init(
      project="RVTDSAN24",
      config=config,
      mode=WANDB_MODE
    )

# learning_rate=0.0001
# epochs=100
# M=5
# activateF1="relu"
# activateF2="nn.Tanh()"
# PAname = '0dB_Len200_MCS3_CBW20_fs100_M5_L208k_DPDtrain'

if WANDB_MODE != 'disabled':
    learning_rate = wandb.config.learning_rate
    epochs = wandb.config.epochs
    M = wandb.config.M
    # print('1234567:%s'%(wandb.config.activate))
    # print(type(wandb.config.activate))
    activate = wandb.config.activate
    neurons = wandb.config.neurons
    isDPD = wandb.config.isDPD
    PAname = wandb.config.PAname
else:
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    M = config['M']
    activate = config['activate']
    neurons = config['neurons']
    isDPD = config['isDPD']
    PAname = config['PAname']


# 定义网络结构
class RVTDSAN(nn.Module):
    def __init__(self):
        super(RVTDSAN,self).__init__()
        # embed_size = 16
        # num_layers = 3
        # self.embed_size = embed_size
        # self.embedding = nn.Linear(5, embed_size)
        # embed_size = 5
        # num_layers = 1
        self.feature_num = neurons
        self.transformer_encoder = Transformer(dim =5, depth=1, heads=1, dim_head=5, mlp_dim=5 )
        if activate == 'Sigmoid':
            self.activate = nn.Sigmoid()
        if activate == 'Tanh':
            self.activate = nn.Tanh()
        if activate == 'ReLU':
            self.activate = nn.ReLU()
        if activate == 'Leaky ReLU':
            self.activate = nn.LeakyReLU()
        if activate == 'GELU':
            self.activate = nn.GELU()
        if activate == 'Swish':
            self.activate = nn.Hardswish()
        # self.activate = nn.LeakyReLU()
        # print(self.feature_num)
        self.fc1 = nn.Linear(in_features=5 * (M+1), out_features=self.feature_num)
        self.fc2 = nn.Linear(in_features=self.feature_num, out_features=2)
        # self.saved_x = None  # 添加一个成员变量来保存 x 的值

    def forward(self, x):
        # 输入b * 4 * 5
        # b = x.shape[0]
        # x = self.embedding(x)
        x = self.transformer_encoder(x)
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
        input_path = f'./data_xzr/input_data_Transformer{PAname}.mat'
        label_path = f'./data_xzr/label_data_NN{PAname}.mat'
        input_data = scio.loadmat(input_path)
        labels_data = scio.loadmat(label_path)
        input_data_array = input_data['input_data_Transformer']
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
    if WANDB_MODE != 'disabled':
        print(f"wandb config: {wandb.config}")
    else:
        print(f"config: {config}")
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
    print('neurons num:%s'%(net.feature_num))
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
        if WANDB_MODE != 'disabled':
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
            if WANDB_MODE != 'disabled':
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
        if WANDB_MODE != 'disabled':
            wandb.log({"NMSEtest": NMSE})


    print('NMSE of the network on the %d test set : %f ' % (testsize, NMSE))

    end = time.time()
    print ('Run time: %f s' %(end-start))
    # 4. Log an artifact to W&B
    # wandb.log_artifact(net)
    # Optional: save model at the end
    # net.to_onnx()
    # wandb.save("model.onnx")