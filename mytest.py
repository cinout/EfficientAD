import torch
import torch.nn as nn

a = torch.tensor([torch.inf, -torch.inf])
b = nn.ReLU(inplace=True)(a)
print(a)
print(b)
