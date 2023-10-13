import torch


a = torch.tensor([0, 1, 2, 6, 8, 10, 11], dtype=torch.float)
start = torch.quantile(a, q=0.2)
print(start)
