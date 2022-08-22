import torch

init_tensor = torch.randn(3, 5)
print("init_tensor:")
print(init_tensor)
# 原始数据
# init_tensor:
# tensor([[ 0.1498, -0.0537,  0.0280,  0.6599, -0.1858],
#         [ 0.6758,  0.8378,  0.9563,  1.4581,  0.1337],
#         [-1.2814,  1.1628, -0.4757, -0.5988,  0.7760]])

# 对每一列数据进行排序
print(init_tensor.sort(0, True))

# 返回结果:
# torch.return_types.sort(
# values=tensor([[ 0.6758,  1.1628,  0.9563,  1.4581,  0.7760],
#         [ 0.1498,  0.8378,  0.0280,  0.6599,  0.1337],
#         [-1.2814, -0.0537, -0.4757, -0.5988, -0.1858]]),
# 索引
# indices=tensor([[1, 2, 1, 1, 2],
#         [0, 1, 0, 0, 1],
#         [2, 0, 2, 2, 0]]))
# 数据被完全打乱了
# 对每一行数据进行排序
print(init_tensor.sort(1, True))

# 找到这一维中的最大值以及索引
print(init_tensor.max(0))
