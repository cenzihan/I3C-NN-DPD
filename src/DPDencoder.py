import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as scio
import numpy as np
from torch.utils.data import DataLoader
from thop import profile
from vit import Transformer
import wandb
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


# 1. Start a W&B Run,
config=dict(
  learning_rate=0.0001,
  epochs=100,
  M=5,
  activateF1="relu",
  activateF2="nn.Tanh()",
  # PAname = '_GMP_M5_L200k'
  PAname = '0dB_Len200_MCS3_CBW20_fs100_M5_L208k_DPDtrain'
)

run = wandb.init(
  project="Encoder",
  # tags=["FC2"],
  config=config,
  # notes="My first experiment",
)

M = wandb.config.M
# 定义网络结构
class DPDencoder(nn.Module):
    def __init__(self):
        super(DPDencoder, self).__init__()
        # embed_size = 16
        # num_layers = 3
        # self.embed_size = embed_size
        # self.embedding = nn.Linear(5, embed_size)
        # embed_size = 5
        # num_layers = 1
        feature_num = 10
        self.transformer_encoder = Transformer(dim =5, depth=1, heads=1, dim_head=5, mlp_dim=5 )
        self.activate = nn.Tanh()
        self.fc1 = nn.Linear(in_features=5 * (M+1), out_features=feature_num)
        self.fc2 = nn.Linear(in_features=feature_num, out_features=2)

    def forward(self, x):
        # 输入b * 4 * 5
        # b = x.shape[0]
        # x = self.embedding(x)
        x = self.transformer_encoder(x)
        # x = x.reshape(b, -1)
        x = self.activate(x)
        x = x.view(-1 ,5 * (M+1))
        x = self.fc1(x)
        x = self.activate(x)
        x = self.fc2(x)
        return x


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        input_data = scio.loadmat('./data_xzr/input_data_Transformer%s.mat'%(wandb.config.PAname))
        labels_data = scio.loadmat('./data_xzr/label_data_NN%s.mat'%(wandb.config.PAname))
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
    print(f"wandb config: {wandb.config}")
    # 加载数据集
    mydataset = MyDataset()
    batchsize = 256
    trainset_ratio = 0.9
    trainsize = int(len(mydataset) * trainset_ratio)
    testsize = int(len(mydataset) - trainsize)
    trainset, testset = torch.utils.data.random_split(mydataset, [trainsize, testsize])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2048, shuffle=False)

    net = DPDencoder().to(device)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
    # 输出参数数量
    params = sum(p.numel() for p in net.parameters())
    print("Total number of parameters: ", params)
    # 输出参数数量和FLOPs
    inputdim = torch.randn(1, M+1, 5).to(device)
    macs1, params1 = profile(net, inputs=(inputdim,))
    print('sub:\n  macs: %0.2f , params: %0.2f ' % (macs1 , params1))
    # print("children")
    # for i, module in enumerate( net.children()):
    #     print(i, module)
    # print("modules")
    # for i, module in enumerate( net.modules()):
    #     print(i, module)

    # 训练网络
    for epoch in range(wandb.config.epochs):
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

    print('Finished Training')
    torch.save(net.state_dict(), './coefs/DPDencoder.pth')

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
        wandb.log({"NMSE": NMSE})


    print('NMSE of the network on the %d test set : %f ' % (testsize, NMSE))

    end = time.time()
    print ('Run time: %f s' %(end-start))
    # 4. Log an artifact to W&B
    # wandb.log_artifact(net)
    # Optional: save model at the end
    # net.to_onnx()
    # wandb.save("model.onnx")