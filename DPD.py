import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as scio
import numpy as np
from torch.utils.data import DataLoader
from RVTDCNN import RVTDCNN
from ARVTDNN import ARVTDNN
from RVTDSAN import RVTDSAN

# Modelname = 'DPDtransformer'
# Dataname = 'Transformer'
# Modelname = 'DPDlinformer'
# Dataname = 'Transformer'
Modelname = 'RVTDSAN'
Dataname = 'Transformer'
# Modelname = 'RVTDCNN'
# Dataname = 'CNN'
# Modelname = 'ARVTDNN'
# Dataname = 'DNN'

PAname= '5_M5_L215k_test'
# 检查是否有可用GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class NewDataset(torch.utils.data.Dataset):
    def __init__(self):
        input_data = scio.loadmat('./data_xzr/input_data_%s%s.mat' % (Dataname,PAname))
        labels_data = scio.loadmat('./data_xzr/label_data_NN%s.mat' % (PAname))
        input_data_array = input_data['input_data_%s2'% (Dataname)]
        labels_data_array = labels_data['label_data_NN2']

        if Modelname == 'ARVTDNN':
            self.data = torch.FloatTensor(input_data_array).permute(1, 0).to(device)
        else:
            self.data = torch.FloatTensor(input_data_array).permute(2, 0, 1).to(device)

        self.label = torch.FloatTensor(labels_data_array).to(device)
        self.length = len(self.label)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if Modelname == 'ARVTDNN':
            return self.data[idx, :].unsqueeze(0), self.label[idx, :]
        else:
            return self.data[idx, :, :].unsqueeze(0), self.label[idx, :]


# 加载数据集
dataset = NewDataset()
batchsize = 256
testsize = int(len(dataset))
testloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False)


net = globals()[Modelname]()
net =net.to(device)
net.load_state_dict(torch.load('./coefs/%s_DPDmodel.pth'% (Modelname)),strict=False)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)

predictions = []

with torch.no_grad():
    for i,data in enumerate(testloader,0):
        net.eval()
        inputs, labels = data
        inputs = inputs.to(device)
        if Dataname == 'Transformer':
            inputs = inputs.squeeze()

        outputs = net(inputs)
        if Modelname == 'ARVTDNN':
            outputs = outputs.squeeze()
        predictions.append(outputs.cpu().numpy())  # 将输出从 GPU 转到 CPU，并转换为 numpy 数组


predictions = np.concatenate(predictions, axis=0)  # 将所有批次的输出拼接在一起
scio.savemat('./Predictions/%s_pred%s.mat'% (Modelname,PAname), {'predictions': predictions})

