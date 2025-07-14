import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.io as scio
import numpy as np
from torch.utils.data import DataLoader
import wandb
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.attn import FullAttention, ProbAttention, AttentionLayer
import time

start = time.time()


# 1. Start a W&B Run,
config=dict(
  learning_rate=0.01,
  epochs=10,
  M=6,
  activateF1="relu",
  activateF2="nn.Tanh()",
)

run = wandb.init(
  project="Informer",
  tags=["FC2"],
  config=config,
  # notes="My first experiment",
)

M = wandb.config.M
#检查是否有可用GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"wandb config: {wandb.config}")


# 定义网络结构
class DPDinformer(nn.Module):
    def __init__(self,factor=8,d_model =5,n_heads=1,e_layers = 1,attn='prob',d_ff =5 * (M+1),dropout = 0.0):
        super(DPDinformer, self).__init__()
        feature_num = 6
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        encoder_layer = EncoderLayer(
                            AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=False),
                                           d_model, n_heads, mix=False),
                            d_model ,
                            d_ff ,
                            dropout=dropout,
                            activation='relu')
        self.informer_encoder = Encoder([encoder_layer for l in range(e_layers)])
        self.fc1 = nn.Linear(in_features=5 * (M+1), out_features=feature_num)
        self.activate = nn.Tanh()
        self.fc2 = nn.Linear(in_features=feature_num, out_features=2)

    def forward(self, x):
        # 输入b * 4 * 5
        # b = x.shape[0]
        # x = self.embedding(x)
        x = self.informer_encoder(x)
        # x = x.reshape(b, -1)
        x = x[0]
        x = x.view(-1 ,5 * (M+1))
        x = self.fc1(x)
        x = self.activate(x)
        x = self.fc2(x)
        return x


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        input_data = scio.loadmat('./data_xzr/input_data_Transformer2_M6_L200k.mat')
        labels_data = scio.loadmat('./data_xzr/label_data_NN2_M6_L200k.mat')
        input_data_array = input_data['input_data_Transformer']
        labels_data_array = labels_data['label_data_NN']
        self.data = torch.FloatTensor(input_data_array).permute(2, 0, 1).to(device)
        self.label = torch.FloatTensor(labels_data_array).to(device)
        self.length = len(self.label)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx, :, :].unsqueeze(0), self.label[idx, :]


# 加载数据集
mydataset = MyDataset()
batchsize = 256
trainset_ratio = 0.95
trainsize = int(len(mydataset) * trainset_ratio)
testsize = int(len(mydataset) - trainsize)
trainset, testset = torch.utils.data.random_split(mydataset, [trainsize, testsize])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=2048, shuffle=False)

net = DPDinformer().to(device)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)

params = sum(p.numel() for p in net.parameters())
print("Total number of parameters: ", params)
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

# 测试网络
loss = nn.MSELoss(reduction='mean')

with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        net.eval()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.squeeze()
        outputs = net(inputs)
        MSE = loss(outputs, labels)
        SQUARES = labels ** 2
        NMSE = 10 * torch.log10(MSE/SQUARES.mean()).cpu().numpy()
    wandb.log({"NMSE": NMSE})
# print(labels)
# print(outputs)

print('NMSE of the network on the %d test set : %f ' % (testsize, NMSE))

end = time.time()
print ('Run time: %f s' %(end-start))
# 4. Log an artifact to W&B
# wandb.log_artifact(net)
# Optional: save model at the end
# net.to_onnx()
# wandb.save("model.onnx")