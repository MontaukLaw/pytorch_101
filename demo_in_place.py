import torch

# inplace计算, 相当于+=
a = torch.tensor(1)
b = torch.tensor(2)
a.add_(b)
print(a)

# 广播机制即但不同shape的矩阵相加的时候, 会自动进行扩展, 成为同一shape
# 但是需要满足两个条件, 一个是每个张量最少有一个维度, 然后是向右对齐.

a = torch.rand(2, 1, 1)
b = torch.rand(3)

print("a:")
print(a)

print("b:")
print(b)

print("a+b:")
print(a + b)

# 随机值看着头晕, 看下面的
a = torch.Tensor([1, 2, 5])
b = torch.Tensor([[10], [11]])

# a是一维, 加一个二维, 最后变成了二维
print("a+b:")
c = a + b
print(c)
print(c.size())
