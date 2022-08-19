import numpy
import torch

tensor_a = torch.Tensor([[1, 2], [3, 4]])
print(tensor_a)
print(tensor_a.type())

tensor_b = torch.Tensor(2, 3)
print(tensor_b.type())

tensor_c = torch.zeros(2, 2)
print(tensor_c)
print(tensor_c.type())

tensor_d = torch.zeros_like(tensor_c)
print(tensor_d)
print(tensor_d.type())

# 每组标准差为0的5个值
tensor_e = torch.normal(mean=0.0, std=torch.rand(5))
print(tensor_e)
print(tensor_e.type())

# 标准值为随机值
tensor_some = torch.normal(mean=torch.rand(5), std=torch.rand(5))
print(tensor_some)
print(tensor_some.type())

# 从0到9的10个值, 注意不包括10
tensor_some = torch.arange(0, 10, 1)
print(tensor_some)
print(tensor_some.type())

# 步长为2, 仅仅为5个值, 即1x5的张量
tensor_some = torch.arange(0, 10, 2)
print(tensor_some)
print(tensor_some.type())

# 定义数据存放位置为cpu的float32类型tensor
tensor_detail = torch.tensor([1, 2, 3, 4], dtype=torch.float32, device=torch.device('cpu'))
print(tensor_detail)
print(tensor_detail.type())

# new一个在内存中稀疏方式存放的tensor
tensor_indices = torch.tensor([[0, 0, 0], [0, 1, 2]])
tensor_values = torch.tensor([1, 2, 3], dtype=torch.float32)
x = torch.sparse_coo_tensor(tensor_indices, tensor_values, [4, 4])
print(x)
y = x.to_dense()
print(y)

# 输出
# tensor([[1., 0., 0., 0.],
#         [0., 2., 0., 0.],
#         [0., 0., 3., 0.],
#         [0., 0., 0., 0.]])

print("rand: ")
# rand(*size, random scope)
tensor_ran = torch.rand(2, 3)
print(tensor_ran)
print(tensor_ran.type())
