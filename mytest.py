import json
import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import random

saturation_area = 3
shape = 10
heyhey = torch.randn(size=(10,))
print(heyhey)
# k = 4
# topk, _ = torch.topk(heyhey, k=k, largest=True)
# print(topk)
# low_k, _ = torch.topk(heyhey, k=len(heyhey) - k, largest=False)
# print(low_k)
heyhey, _ = torch.sort(heyhey)
print(heyhey[: shape - saturation_area])
print(heyhey[shape - saturation_area :])
