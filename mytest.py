import torch

a = torch.randn(1, 10)
print(a)
b = a.sort()
print(b)
print(torch.quantile(a, 0.8))
