import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import nni
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import model_wrapper
from nni.retiarii.evaluator import FunctionalEvaluator
import nni.retiarii.strategy as strategy
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
import scipy.io as scio
import numpy as np


@model_wrapper      # this decorator should be put on the out most
class ModelSpace(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=0)
        # LayerChoice is used to select a layer .
        self.relu1 = nn.LayerChoice([nn.Tanh(),nn.ReLU(),nn.LeakyReLU(),nn.PReLU(),nn.Softplus(),nn.Sigmoid()])
        feature = nn.ValueChoice([5,6,7,8,9,10])
        self.fc1 = nn.Linear(in_features=18, out_features=feature)
        self.fc2 = nn.Linear(in_features=feature, out_features=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = x.view(-1, 18)
        x = self.fc1(x)
        # x = self.relu2(self.fc1(x))
        output = self.fc2(x)
        return output


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


def train_epoch(model, device, train_loader, optimizer, epoch):
    loss_fn = torch.nn.MSELoss()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader,0):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def tst_epoch(model, device, test_loader):
    model.eval()
    loss = nn.MSELoss(reduction='mean')
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            input, label = data.to(device)
            output = model(input)
            MSE = loss(output, label)
            SQUARES = label ** 2

        NMSE = 10 * np.log10(MSE / SQUARES.mean())
    print('\nNMSE of the network on the 2000 test set : %f ' % (NMSE))
    return NMSE


def evaluate_model(model_cls):
    # "model_cls" is a class, need to instantiate
    model = model_cls()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
    # transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # Loading Dataset
    mydataset = MyDataset()
    batchsize = 256
    trainset_ratio = 0.99
    trainsize = int(len(mydataset) * trainset_ratio)
    testsize = int(len(mydataset) - trainsize)
    trainset, testset = torch.utils.data.random_split(mydataset, [trainsize, testsize])
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=2048, shuffle=False)

    for epoch in range(3):
        # train the model for one epoch
        train_epoch(model, device, train_loader, optimizer, epoch)
        # test the model for one epoch
        NMSE = tst_epoch(model, device, test_loader)
        # call report intermediate result. Result can be float or dict
        nni.report_intermediate_result(NMSE)

    # report final test result
    nni.report_final_result(NMSE)
    # return NMSE

model_space = ModelSpace()
# print(model_space)
# dedup=False if deduplication is not wanted
search_strategy = strategy.Random(dedup=True)
evaluator = FunctionalEvaluator(evaluate_model)
exp = RetiariiExperiment(model_space, evaluator, [], search_strategy)
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'NNDPD_search'

exp_config.max_trial_number = 4  # 最多运行 4 个实验
exp_config.trial_concurrency = 2  # 最多同时运行 2 个试验

# exp_config.execution_engine = 'base'
# export_formatter = 'code'
exp.run(exp_config, 8081)
# exp.run

#export the top model
for model_dict in exp.export_top_models(formatter='dict'):
    print(model_dict)

