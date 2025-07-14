import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as scio
import os as os
import numpy as np
from torch.utils.data import DataLoader

# from ARVTDNN import ARVTDNN
# Modelname = 'ARVTDNN'

# from RVTDCNN import RVTDCNN
# Modelname = 'RVTDCNN'

from RVTDSAN_X import RVTDSAN
Modelname = 'RVTDSAN'

isDPD = True
DatasetName= '_test_43k_M5_Len1600_MCS9_CBW80_fs480m_'
label = 980480

# 检查是否有可用GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class NewDataset(torch.utils.data.Dataset):
    def __init__(self):
        input_data = scio.loadmat('./data_xzr/%s/input_data_%s%s.mat' % (DatasetName,Modelname,DatasetName))
        labels_data = scio.loadmat('./data_xzr/%s/label_data_NN%s.mat' % (DatasetName,DatasetName))
        input_data_array = input_data['input_data_%s'% (Modelname)]
        labels_data_array = labels_data['label_data_NN']

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
if isDPD:
    net.load_state_dict(torch.load('./coefs/%s_DPDmodel%s.pth'% (Modelname,label)),strict=False)
else:
    net.load_state_dict(torch.load('./coefs/%s_PAmodel%s.pth' % (Modelname,label)), strict=False)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)

predictions = []

with torch.no_grad():
    for i,data in enumerate(testloader,0):
        net.eval()
        inputs, labels = data
        inputs = inputs.to(device)
        if Modelname == 'RVTDSAN':
            inputs = inputs.squeeze()

        outputs = net(inputs)
        if Modelname == 'ARVTDNN':
            outputs = outputs.squeeze()
        predictions.append(outputs.cpu().numpy())  # 将输出从 GPU 转到 CPU，并转换为 numpy 数组


dirs = './Predictions/%s'% (DatasetName)
if not os.path.exists(dirs):
    os.makedirs(dirs)

predictions = np.concatenate(predictions, axis=0)  # 将所有批次的输出拼接在一起
if isDPD:
    scio.savemat('./Predictions/%s/%s_DPDpred.mat'% (DatasetName,Modelname), {'predictions': predictions})
else:
    scio.savemat('./Predictions/%s/%s_PApred.mat' % (DatasetName,Modelname ), {'predictions': predictions})
