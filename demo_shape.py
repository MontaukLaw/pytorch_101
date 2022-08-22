import torch

init_tensor = torch.arange(1, 5)
print("init_tensor:")
print(init_tensor)

# 用view来改变形状
view_tensor = init_tensor.view(2, 2)
print("view_tensor:")
print(view_tensor)

# 用reshape来改变形状
reshape_tensor = init_tensor.reshape(4, 1)
print("reshape_tensor:")

# 说明在共享内存
init_tensor[0] = 0
print(reshape_tensor)


