import torch

init_tensor = torch.arange(1, 5)
print("init_tensor before:")
print(init_tensor)
print(init_tensor.shape)

init_tensor.unsqueeze_(0)
print("after unsqueeze_:")
print(init_tensor)
print(init_tensor.shape)

# 去掉size为1的维度
init_tensor2 = torch.Tensor([[[1], [2], [3]], [[4], [5], [6]]])
print("init_tensor2 before:")
print(init_tensor2)
print(init_tensor2.shape)
init_tensor2.squeeze_(2)
print("after squeeze_:")
print(init_tensor2)
print(init_tensor2.shape)

# 2x3的矩阵的unsqueez
init_tensor3 = torch.Tensor([[1., 2., 3.], [4., 5., 6.]])
print("init_tensor3 before:")
print(init_tensor3)
print(init_tensor3.shape)

# 这个0即要插入的size为1的维度的索引
init_tensor3.unsqueeze_(0)
print("after unsqueeze_:")
print(init_tensor3)
print(init_tensor3.shape)
