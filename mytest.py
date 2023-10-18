import torch


a = torch.tensor([0, 1, 2, 6, 8, 10, 11], dtype=torch.float)
b = torch.tensor([-20, 1, 2, 6, 8, 10, 101], dtype=torch.float)
start_a = torch.quantile(a, q=0.2)
start_b = torch.quantile(b, q=0.2)
print(start_a)
print(start_b)
