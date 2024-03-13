import json
import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import random

train_steps = 70000
lid_history_step = 50
lid_history_count = 60

saved_iterations = []
cur_iteration = train_steps - 1
for v in range(lid_history_count):
    saved_iterations.append(cur_iteration)
    cur_iteration = cur_iteration - lid_history_step

print(saved_iterations)
print(len(saved_iterations))
