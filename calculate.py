import torch

# input 是 1x3
tensor_input_a = torch.tensor([1, 2, 3])
# output是 3x2
tensor_input_b = torch.tensor([[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                               [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]])
# 结果是 1x2
tensor_result = tensor_input_a.matmul(tensor_input_b)

print(tensor_result)
print(tensor_result.size())
