import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as scio
from torch.utils.data import DataLoader
from thop import profile
from ptflops import get_model_complexity_info
import wandb
import time
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# 检查传递的参数个数是否足够，用于脚本调参实验
if len(sys.argv) < 2:
    label = 1
    neuronsN = 10
else:
    label = sys.argv[1]
    neuronsN = int(sys.argv[2])

# 1. Start a W&B Run,
#wandb 调整参数
config=dict(
  learning_rate=0.001,
  epochs= 100,
  M= 5,
  neurons= neuronsN,
  isDPD = False,
  # PAname = '0dB2_Len200_MCS3_CBW20_fs100_M5_L208k_DPDtrain'
  PAname = '5_M5_L215k_PAtrain'
)

run = wandb.init(
  project="LSTMAttention",
  # tags=["FC2"],
  config=config,
  # notes="My first experiment",
)

learning_rate = wandb.config.learning_rate
epochs = wandb.config.epochs
M = wandb.config.M
neurons = wandb.config.neurons
isDPD = wandb.config.isDPD
PAname = wandb.config.PAname


class LSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, memory_depth):
        super(LSTMAttention, self).__init__()

        self.memory_depth = memory_depth

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM output
        lstm_out, (hn, cn) = self.lstm(x)

        # Attention Mechanism
        # y_lstm = lstm_out[:, -self.memory_depth:, :]
        y_lstm = lstm_out

        y_lstm_T = y_lstm.transpose(-2, -1) #调换倒数第一个维度和倒数第二个维度
        gamma = torch.bmm(y_lstm, y_lstm_T) #p * m * s乘以p * s * n-> p * m * n

        alpha = F.softmax(gamma, dim=-1)

        beta = torch.bmm(alpha, y_lstm)

        # Fully Connected Layer
        output = self.fc(beta)

        return output


class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        input_data = scio.loadmat('./data_xzr/input_data_Transformer%s.mat'%(PAname))
        labels_data = scio.loadmat('./data_xzr/label_data_NN%s.mat'%(PAname))
        input_data_array = input_data['input_data_Transformer']
        labels_data_array = labels_data['label_data_NN']
        self.data = torch.FloatTensor(input_data_array).permute(2, 0, 1).to(device)
        self.label = torch.FloatTensor(labels_data_array).to(device)
        self.length = len(self.label)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx, :, 0:2].unsqueeze(0), self.label[idx, :]


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

    # Instantiate the model with necessary parameters
    # input_size = 2  # I_in and Q_in as inputs
    # hidden_size = 64  # You might want to search for the best hyperparameter
    # output_size = 2  # I_out and Q_out as outputs
    # memory_depth = 80  # As mentioned in the paper
    input_size = 2  # I_in and Q_in as inputs
    output_size = 2  # I_out and Q_out as outputs
    hidden_size = 5  # You might want to search for the best hyperparameter
    memory_depth = M + 1  # As mentioned in the paper
    net = LSTMAttention(input_size, hidden_size, output_size, memory_depth).to(device)


    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
    # 输出参数数量和FLOPs
    inputdim = torch.randn(1, M+1, 2).to(device)
    macs1, params1 = profile(net, inputs=(inputdim,))
    print('sub:\n  macs: %0.2f , params: %0.2f ' % (macs1 , params1))

    print("inputdim shape: ", inputdim.shape)
    macs, params = get_model_complexity_info(net, (M+1, 2), as_strings=True, print_per_layer_stat=True, verbose=True)
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
            outputs = outputs[ :,0:1, :].squeeze()
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
                outputs = outputs[:, 0:1, :].squeeze()
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
        torch.save(net.state_dict(), './coefs/LSTMAttention_DPDmodel%s.pth'% (label))
    else:
        torch.save(net.state_dict(), './coefs/LSTMAttention_PAmodel%s.pth'% (label))

    # 测试网络
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            net.eval()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.squeeze()
            outputs = net(inputs)
            outputs = outputs[:, 0:1, :].squeeze()
            MSE = criterion(outputs, labels)
            SQUARES = labels ** 2
            NMSE = 10 * torch.log10(MSE/SQUARES.mean()).cpu().numpy()
        wandb.log({"NMSEtest": NMSE})


    print('NMSE of the network on the %d test set : %f ' % (testsize, NMSE))

    end = time.time()
    print ('Run time: %f s' %(end-start))




# # Generate some random input data
# batch_size = 1
# input_data = torch.randn(batch_size, memory_depth, input_size)
#
# # Run the model
# output = model(input_data)

# # Print the input and output dimensions
# print("Input data shape: ", input_data.shape)
# print("Output data shape: ", output.shape)
# print("Hidden state shape: ", hn.shape)
# print("Cell state shape: ", cn.shape)

