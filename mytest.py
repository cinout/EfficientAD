import torch
import cv2
import numpy as np
from torchvision import transforms

image_size = 256
pop = torch.randint(0, 1, (1, 1024, 1024), dtype=torch.float)
resize_operation = transforms.Resize(size=(image_size, image_size), antialias=True)

pop = resize_operation(pop)
print(pop.shape)
