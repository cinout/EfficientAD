import torch
import cv2
import numpy as np

resize_shape = [256, 256]

mask_path = "datasets/loco/breakfast_box/ground_truth/logical_anomalies/000/000.png"

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

mask = cv2.resize(mask, dsize=(resize_shape[1], resize_shape[0]))

cv2.imwrite("zebra.png", mask)

mask = mask / 255.0

mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)
print(np.unique(mask))
