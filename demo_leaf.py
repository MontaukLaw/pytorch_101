import torch

x = torch.randn(1)
w = torch.ones(1, requires_grad=True)
b = torch.ones(1, requires_grad=True)

print(x)
print(w)
print(b)

print("is_leaf:")
print(x.is_leaf)
print(w.is_leaf)
print(b.is_leaf)

print("requires_grad:")
print(x.requires_grad)
print(w.requires_grad)
print(b.requires_grad)

y = w * x
# y也不是leaf
print("y.is_leaf:")
print(y.is_leaf)
# y也需要求导
print("y.requires_grad:")
print(y.requires_grad)

z = y + b
# z不是leaf
print("z.is_leaf:")
print(z.is_leaf)
# z仍然需要求导
print("z.requires_grad:")
print(z.requires_grad)

print(y.grad_fn)
print(z.grad_fn)

z.backward(retain_graph=True)
# 对w*x+b求w的偏导数, 即x
print(w.grad)
# 求b的偏导数, 即1
print(b.grad)
