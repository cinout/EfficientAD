import torch


_num_embeddings = 10
bhw = 8

encoding_indices = torch.randint(0, 10, (bhw, 1))  # shape: (BHW, 1)
encodings = torch.zeros(encoding_indices.shape[0], _num_embeddings)  # shape: (BHW, N)
print(encodings)
encodings.scatter_(1, encoding_indices, 1)
print("=======")
print(encodings)
