import torch
import torch.nn as nn
import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
import scipy.io as scio 
import numpy as np
from torch.utils.data import DataLoader
# import theseus-ai as th

# 定义网络结构
class RVTDCNN(nn.Module):
    def __init__(self):
        super(RVTDCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.Tanh()
        self.fc1 = nn.Linear(in_features=18, out_features=10)
        # self.relu2 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=10, out_features=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = x.view(-1, 18)
        x = self.fc1(x)     
        # x = self.relu2(self.fc1(x))
        x = self.fc2(x)
        return x


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        input_data = scio.loadmat('input_data_NN.mat')
        labels_data = scio.loadmat('label_data_NN.mat')
        input_data_array = input_data['input_data_NN']
        labels_data_array = labels_data['label_data_NN']
        self.data = torch.FloatTensor(input_data_array).permute(2, 0, 1)
        self.label = torch.FloatTensor(labels_data_array)
        self.length = len(self.label)
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx, :, :].unsqueeze(0), self.label[idx, :]


# 加载数据集
mydataset = MyDataset()
batchsize = 256
trainset_ratio = 0.99
trainsize = int(len(mydataset) * trainset_ratio)
testsize = int(len(mydataset) - trainsize)
trainset, testset = torch.utils.data.random_split(mydataset, [trainsize, testsize])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=2048, shuffle=False)

net = RVTDCNN()
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
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        net.train()
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # if i % 100 == 99:
        print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss))
        # running_loss = 0.0

print('Finished Training')

# 测试网络
loss = nn.MSELoss(reduction='mean')

with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        net.eval()
        inputs, labels = data
        outputs = net(inputs)
        MSE = loss(outputs, labels)
        SQUARES = labels**2
        NMSE = 10*np.log10(MSE/SQUARES.mean()) 

print(labels)
print(outputs)

# print('NMSE of the network on the 1000 test set : %f ' % (10000000*MSE))
print('NMSE of the network on the 2000 test set : %f ' % (NMSE))

