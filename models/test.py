import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

word_to_idx = {'girl': 0, 'boy': 1, 'woman': 2, 'man': 3}  # 每个单词用一个数字表示
embeds = nn.Embedding(8, 2)  # 单词的个数8，2位embedding的维度

inputs = torch.LongTensor([[word_to_idx[key] for key in word_to_idx.keys()]])
inputs = torch.autograd.Variable(inputs)

# 获取Variable对应的embedding，并打印出来
outputs = embeds(inputs)
print(outputs)
print(embeds.weight)