import torch
import torch.nn as nn

a = torch.randint(0, 100, (3, 4, 5))
print(a)
sorted, _ = torch.sort(torch.flatten(a))
print(sorted)

# a = torch.tensor([torch.inf, -torch.inf])
# print(a)
# a[a == float("Inf")] = 0
# a[a == -float("Inf")] = 0
# print(a)
