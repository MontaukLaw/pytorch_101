import torch

# inplace计算, 相当于+=
a = torch.tensor(1)
b = torch.tensor(2)
a.add_(b)
print(a)


