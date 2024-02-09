import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import random

choices = [0, 1, 2, 3]
gt = torch.randint(0, 10, size=(1, 4, 4), dtype=torch.float32)

print(gt)
gt = torch.rot90(gt, k=1, dims=(1, 2))
print(gt)

exit()

image_size_before_geoaug = 512
image_size = 256
img_path = "datasets/loco/breakfast_box/train/good/013.png"
img = Image.open(img_path)
img = img.convert("RGB")

hey = torch.rand(size=(1, 986, 711))
hey = transforms.Resize(
    (image_size_before_geoaug, image_size_before_geoaug), antialias=True
)(hey)


geoaug_transform = transforms.Compose(
    [
        transforms.RandomApply(
            [
                transforms.RandomResizedCrop(
                    size=(image_size, image_size), scale=(3 / 4, 4 / 3)
                )
            ],
            p=0.5,
        ),
    ]
)


i, j, h, w = transforms.RandomResizedCrop.get_params(
    torch.rand(size=(1, image_size_before_geoaug, image_size_before_geoaug)),
    scale=(0.8, 1),
    ratio=(3 / 4, 4 / 3),
)
img1 = transforms.Resize((image_size_before_geoaug, image_size_before_geoaug))(img)
img2 = transforms.Resize((image_size_before_geoaug, image_size_before_geoaug))(img)
img3 = transforms.Resize((image_size_before_geoaug, image_size_before_geoaug))(img)


if random.random() > 0.0:
    img1 = TF.crop(img1, i, j, h, w)
    img2 = TF.crop(img2, i, j, h, w)
    img3 = TF.crop(img3, i, j, h, w)
    hey = TF.crop(hey, i, j, h, w)
    print(hey.shape)

img1 = transforms.Resize((image_size, image_size))(img1)
img2 = transforms.Resize((image_size, image_size))(img2)
img3 = transforms.Resize((image_size, image_size))(img3)


img1.save("img1.png", "PNG")
img2.save("img2.png", "PNG")
img3.save("img3.png", "PNG")
