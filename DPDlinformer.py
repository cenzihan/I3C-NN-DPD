import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as scio
import numpy as np
from linformer_pytorch1 import Linformer
from torch.utils.data import DataLoader
from thop import profile
import wandb
import time


# 1. Start a W&B Run,
config=dict(
  learning_rate=0.01,
  epochs=100,
  M=5,
  activateF="relu",
  PAname = '5_M5_L215k_DPDtrain'
)

run = wandb.init(
  project="Linformer",
  # tags=["FC1"],
  config=config,
  # notes="My first experiment",
)


M = wandb.config.M
# 定义网络结构
class DPDlinformer(nn.Module):
    def __init__(self):
        super(DPDlinformer, self).__init__()
        feature_num = 10
        # encoder_layer = nn.TransformerEncoderLayer(d_model=5, nhead=1, dim_feedforward=5 * (M+1), activation='relu', batch_first=True, norm_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1 )
        self.linformer = Linformer(
        input_size= M+1, # Dimension 1 of the input
        channels=5, # Dimension 2 of the input
        dim_d=None, # Overwrites the inner dim of the attention heads. If None, sticks with the recommended channels // nhead, as in the "Attention is all you need" paper
        dim_k=4, # The second dimension of the P_bar matrix from the paper
        dim_ff= 5, # Dimension in the feed forward network
        dropout_ff=0, # Dropout for feed forward network
        nhead=1, # Number of attention heads
        depth=1, # How many times to run the model
        dropout=0, # How much dropout to apply to P_bar after softmax
        activation= "relu", # What activation to use. Currently, only gelu and relu supported, and only on ff network.
        checkpoint_level="C0", # What checkpoint level to use. For more information, see below.
        parameter_sharing="layerwise", # What level of parameter sharing to use. For more information, see below.
        k_reduce_by_layer=0, # Going down `depth`, how much to reduce `dim_k` by, for the `E` and `F` matrices. Will have a minimum value of 1.
        full_attention=False, # Use full attention instead, for O(n^2) time and space complexity. Included here just for comparison
        include_ff=False, # Whether or not to include the Feed Forward layer
        w_o_intermediate_dim=None, # If not None, have 2 w_o matrices, such that instead of `dim*nead,channels`, you have `dim*nhead,w_o_int`, and `w_o_int,channels`
        )
        self.fc1 = nn.Linear(in_features=5 * (M+1), out_features=feature_num)
        self.activate = nn.Tanh()
        self.fc2 = nn.Linear(in_features=feature_num, out_features=2)

    def forward(self, x):
        # 输入b * 4 * 5
        # b = x.shape[0]
        # x = self.embedding(x)
        x = self.linformer(x)
        # x = x.reshape(b, -1)
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
    #检查是否有可用GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"wandb config: {wandb.config}")
    # 加载数据集
    mydataset = MyDataset()
    batchsize = 256
    trainset_ratio = 0.98
    trainsize = int(len(mydataset) * trainset_ratio)
    testsize = int(len(mydataset) - trainsize)
    trainset, testset = torch.utils.data.random_split(mydataset, [trainsize, testsize])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2048, shuffle=False)

    net = DPDlinformer().to(device)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
    #输出参数数量
    params = sum(p.numel() for p in net.parameters())
    print("Total number of parameters: ", params)
    # 输出参数数量和FLOPs
    inputdim = torch.randn(1, M+1, 5).to(device)
    macs1, params1 = profile(net, inputs=(inputdim,))
    print('sub:\n  macs: %0.2f , params: %0.2f ' % (macs1 , params1))

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
    torch.save(net.state_dict(), './coefs/DPDlinformer.pth')

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
