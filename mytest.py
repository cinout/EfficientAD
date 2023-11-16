import torch
from functools import reduce
import math

image_ae_features = torch.randint(0, 10, size=(1, 3, 2, 2), dtype=torch.float)
closest_ref_features = torch.randint(0, 10, size=(1, 3, 2, 2), dtype=torch.float)

print("\n------image_ae_features------")
print(image_ae_features)
print("\n------closest_ref_features------")
print(closest_ref_features)

B, C, H, W = image_ae_features.shape
image_ae_features = image_ae_features.squeeze(0).reshape(C, -1)  # shape: [C, H*W]
closest_ref_features = closest_ref_features.squeeze(0).reshape(C, -1)  # shape: [C, H*W]
similarity_matrix = torch.mm(
    image_ae_features.T, closest_ref_features
)  # shape: [H*W, H*W]

print("\n------similarity_matrix------")
print(similarity_matrix)
similarity_matrix = similarity_matrix.flatten()

descending_similarity = torch.argsort(similarity_matrix, descending=True).numpy()
print("\n------descending_similarity------")
print(descending_similarity)

similarity_desc_index = [
    (math.floor(index / (H * W)), index % (H * W)) for index in descending_similarity
]
swap_guide = []  # each element will be (index_ae, index_ref)
for t in similarity_desc_index:
    if len(swap_guide) == 0 or all([t[0] != k[0] and t[1] != k[1] for k in swap_guide]):
        swap_guide.append(t)

# print("\n------swap_guide------")
# print(swap_guide)

for index_ae, index_ref in swap_guide:
    image_ae_features[:, index_ae] = closest_ref_features[:, index_ref]


image_ae_features = image_ae_features.reshape(C, H, W).unsqueeze(0)
print("\n------image_ae_features [after swapping]------")
print(image_ae_features)
