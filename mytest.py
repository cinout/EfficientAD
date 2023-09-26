import torch

features = torch.randn((1, 3, 4, 4))
print(features)
features = torch.nn.functional.interpolate(features, size=(2, 2), mode="bilinear")

print(features)
