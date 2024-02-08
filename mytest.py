import torch
import cv2
import numpy as np
from torchvision import transforms

image_size_before_geoaug = 512
image_size = 256

geoaug_transform = transforms.Compose(
    [
        transforms.Resize((image_size_before_geoaug, image_size_before_geoaug)),
        transforms.RandomApply(
            [
                transforms.RandomResizedCrop(
                    size=(image_size, image_size), scale=(0.85, 1)
                )
            ],
            p=0.5,
        ),
    ]
)

iteration = 0
print(iteration % 2 == 0)
