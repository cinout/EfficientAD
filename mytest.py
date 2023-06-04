import torch
import torch.nn as nn

relu = nn.ReLU(inplace=False)
x1 = torch.randn((5, 3))


x2 = torch.mean(x1)
print(x1)
print(x2)

# a = torch.tensor([torch.inf, -torch.inf])
# print(a)
# a[a == float("Inf")] = 0
# a[a == -float("Inf")] = 0
# print(a)
