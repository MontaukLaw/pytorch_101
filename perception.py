import torch
from torch import nn


# 全连接的Module, 继承自nn.Module
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        # 调用nn.Module的构造
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.randn(out_features))

    # 前向传播
    def forward(self, x):
        # 先求x*w
        x = x.matmul(self.w)  # 矩阵相乘
        y = x + self.b.expand_as(x)  # 保证矩阵形状一致
        return y


class Perception(nn.Module):
    def __init__(self, in_features, out_features, hid_features):
        super(Perception, self).__init__()
        self.layer1 = Linear(in_features, hid_features)
        self.layer2 = Linear(hid_features, out_features)

    def forward(self, x):
        # 第一层
        x = self.layer1(x)
        # 激活
        y = torch.sigmoid(x)
        # 第二层
        y = self.layer2(y)
        # 激活
        y = torch.sigmoid(y)
        return y


perception = Perception(2, 3, 2)
print(perception)

for name, parameter in perception.named_parameters():
    print(name, parameter)

input_data = torch.randn(4, 2)
print("input_data:")
print(input_data)

# 直接调用了forward
output_data = perception(input_data)
print("output_data:")
print(output_data)
