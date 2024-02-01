import os
import torch
import glob
import numpy as np
import torch.multiprocessing
from torch.utils.data import Dataset
import glob
from PIL import Image, ImageOps
import random
import math
from torchvision import transforms


def transform_data(size):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train, std=std_train),
        ]
    )
    return data_transforms


class MyDummyDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class LogicalAnomalyDataset(Dataset):
    def __init__(
        self, logicano_select, num_logicano, percent_logicano, subdataset, image_size
    ) -> None:
        super().__init__()
        self.image_size = image_size
        logical_anomaly_path = "datasets/loco/" + subdataset + "/test/logical_anomalies"
        logical_anomaly_gt_path = (
            "datasets/loco/" + subdataset + "/ground_truth/logical_anomalies"
        )
        all_logical_anomalies = sorted(os.listdir(logical_anomaly_path))
        if logicano_select == "percent":
            selected_indices = [
                x.split(".png")[0]
                for x in random.sample(
                    all_logical_anomalies,
                    k=math.floor(percent_logicano * len(all_logical_anomalies)),
                )
            ]
        elif logicano_select == "absolute":
            selected_indices = [
                x.split(".png")[0]
                for x in random.sample(
                    all_logical_anomalies,
                    k=num_logicano,
                )
            ]
        self.images = [logical_anomaly_path + f"/{idx}.png" for idx in selected_indices]
        self.gt = [
            glob.glob(logical_anomaly_gt_path + f"/{idx}/*.png")
            for idx in selected_indices
        ]

    def __len__(self):
        return len(self.images)

    def transform_image(self, path):
        img = Image.open(path)
        img = img.convert("RGB")
        return transform_data(self.image_size)(img)

    def transform_gt(self, paths):
        overall_gt = None  # purpose is to determine all negative (normal) pixels
        individual_gts = []
        for each_path in paths:
            gt = Image.open(each_path)
            gt = np.array(gt)
            gt = torch.tensor(gt)
            gt = gt.unsqueeze(0)
            if overall_gt is not None:
                overall_gt = torch.logical_or(overall_gt, gt)
            else:
                overall_gt = gt

            individual_gts.append(gt)

        overall_gt = overall_gt.bool().to(torch.float32)
        return overall_gt, individual_gts

    def __getitem__(self, index):
        img_path = self.images[index]
        image = self.transform_image(img_path)

        gt_paths = self.gt[index]
        overall_gt, individual_gts = self.transform_gt(gt_paths)

        # overall_gt.shape: [1, orig.height, orig.width]

        sample = {
            "image": image,
            "overall_gt": overall_gt,
            "individual_gts": individual_gts,
            "img_path": img_path,
        }
        return sample
