from math import sqrt
import torch
import torch.nn as nn

# class LayerNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-5):
#         """Construct a layernorm module in the TF style (epsilon inside the square root).
#         """
#         super(LayerNorm, self).__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.bias = nn.Parameter(torch.zeros(hidden_size))
#         self.variance_epsilon = eps
#
#     def forward(self, x):
#         u = x.mean(-1, keepdim=True)
#         s = (x - u).pow(2).mean(-1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.variance_epsilon)
#         y = self.weight * x + self.bias
#         return y

class PerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.norm(x, **kwargs)


class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, dim):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim = dim
        # 定义线性变换函数
        self.linear_q = nn.Linear(dim, dim_q, bias=False)
        self.linear_k = nn.Linear(dim, dim_k, bias=False)
        self.linear_v = nn.Linear(dim, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)
        self.layernorm = PerNorm(dim)

    def forward(self, x):
        # x: batch, n, dim_q
        # 根据文本获得相应的维度

        batch, n, dim = x.shape
        assert dim == self.dim

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        # q*k的转置 并*开根号后的dk
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        # 归一化获得attention的相关系数
        # dist = torch.softmax(dist, dim=-1)  # batch, n, n
        # attention系数和v相乘，获得最终的得分
        att = torch.bmm(dist, v)

        #add&norm
        x =  self.layernorm(att) + x
        # x = self.layernorm(att)
        return x

