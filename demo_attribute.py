import torch

# 将tensor的device属性指定为cpu
dev = torch.device("cpu")
tensor_dev = torch.tensor([2, 2], device=dev)
print(tensor_dev)
print(tensor_dev.__dlpack_device__())

