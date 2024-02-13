import json
import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import random

output_dir = "outputs/folder_baseline/output_20240131_213923_16_56_sd10_[bb]"
jsonfile_path = os.path.join(output_dir, "metrics.json")

f = open(jsonfile_path)
data = json.load(f)


classification_results = data["classification"]["auc_roc"]
cls_logic = classification_results["logical_anomalies"]
cls_structure = classification_results["structural_anomalies"]
cls_mean = classification_results["mean"]


localization_results = data["localization"]["auc_spro"]
loc_logic = localization_results["logical_anomalies"]["0.05"]
loc_structure = localization_results["structural_anomalies"]["0.05"]
loc_mean = localization_results["mean"]["0.05"]

output_content = [
    cls_logic,
    cls_structure,
    cls_mean,
    loc_logic,
    loc_structure,
    loc_mean,
]

output_content = [f"{item * 100:.1f}\n" for item in output_content]

output_txt_file = os.path.join(output_dir, "format_table.txt")
with open(output_txt_file, "a") as the_file:
    the_file.writelines(output_content)
