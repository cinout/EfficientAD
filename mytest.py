import torch


encoding_indices = torch.randint(0, 10, (5, 5)) == 2  # shape: (BHW, 1)
print(encoding_indices)
a = ~encoding_indices.nonzero().squeeze()
b = ~encoding_indices.nonzero().squeeze()
