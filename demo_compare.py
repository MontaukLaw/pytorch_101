import torch

a = torch.Tensor([1, 2])
b = torch.Tensor([1, 2])
print(torch.eq(a, b))
print(torch.equal(a, b))

c = torch.Tensor([3, 2])
print(torch.eq(a, c))
print(torch.equal(a, c))

# 输出
# tensor([True, True])
# True
# tensor([False,  True])
# False
