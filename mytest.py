import torch

image_size = 20
map_st = torch.randn((4, 1, 8, 8))
print(map_st.shape)
map_st = torch.nn.functional.interpolate(
    map_st, (image_size, image_size), mode="bilinear"
)
print(map_st.shape)
