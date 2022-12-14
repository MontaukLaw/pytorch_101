import torch

a = torch.rand(5, 5) * 10
print(a)
a.clamp_(1, 6)
print(a)

# 结果
# tensor([[7.2545, 9.8149, 6.6349, 7.2170, 3.6641],
#         [7.3733, 4.7061, 7.6599, 5.1583, 4.6145],
#         [9.1520, 1.3643, 7.2149, 4.7540, 5.0118],
#         [5.3468, 4.1881, 5.5918, 5.4849, 9.9743],
#         [9.0033, 0.3649, 8.2877, 6.4846, 0.8491]])
# tensor([[7.2545, 9.8149, 6.6349, 7.2170, 5.0000],
#         [7.3733, 5.0000, 7.6599, 5.1583, 5.0000],
#         [9.1520, 5.0000, 7.2149, 5.0000, 5.0118],
#         [5.3468, 5.0000, 5.5918, 5.4849, 9.9743],
#         [9.0033, 5.0000, 8.2877, 6.4846, 5.0000]])
