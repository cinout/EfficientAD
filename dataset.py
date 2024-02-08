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


class NormalDatasetForGeoAug(Dataset):
    def __init__(self, path, image_size_before_geoaug, image_size) -> None:
        super().__init__()

        all_logical_anomalies = sorted(os.listdir(path))
        self.images = [path + f"/{item}" for item in all_logical_anomalies]
        self.image_size_before_geoaug = image_size_before_geoaug
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def transform_image(self, path):
        image = Image.open(path)
        image = image.convert("RGB")

        geoaug_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size_before_geoaug, self.image_size_before_geoaug)
                ),
                transforms.RandomApply(
                    [
                        transforms.RandomResizedCrop(
                            size=(self.image_size, self.image_size), scale=(0.85, 1)
                        )
                    ],
                    p=0.5,
                ),
            ]
        )
        default_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        transform_ae = transforms.RandomChoice(
            [
                transforms.ColorJitter(brightness=0.2),
                transforms.ColorJitter(contrast=0.2),
                transforms.ColorJitter(saturation=0.2),
            ]
        )

        geo_trans_img = geoaug_transform(image)
        return (
            default_transform(geo_trans_img),
            default_transform(transform_ae(geo_trans_img)),
        )

    def __getitem__(self, index):
        img_path = self.images[index]
        return self.transform_image(img_path)


class LogicalAnomalyDataset(Dataset):
    def __init__(
        self,
        logicano_select,
        num_logicano,
        percent_logicano,
        subdataset,
        image_size,
        loss_on_resize,
        device,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.loss_on_resize = loss_on_resize
        self.resize_operation = transforms.Resize(
            size=(self.image_size, self.image_size), antialias=True
        )
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
        # TODO: add geo_augment
        overall_gt = None  # purpose is to determine all negative (normal) pixels
        individual_gts = []
        for each_path in paths:
            gt = Image.open(each_path)
            gt = np.array(gt)
            gt = torch.tensor(gt)
            gt = gt.unsqueeze(0)  # [1, orig_h, orig_w]

            if overall_gt is not None:
                overall_gt = torch.logical_or(overall_gt, gt)
            else:
                overall_gt = gt
            if self.loss_on_resize:
                pixel_type = sorted(torch.unique(gt).detach().cpu().numpy())[-1]
                _, orig_height, orig_width = gt.shape

                gt = gt.bool().to(torch.float32)  # either 0. or 1.
                gt = self.resize_operation(gt)
                gt = gt.long()  # any value<1.0 is converted to 0
                gt = gt

                individual_gts.append(
                    {
                        "gt": gt,
                        "pixel_type": pixel_type,
                        "orig_height": orig_height,
                        "orig_width": orig_width,
                    }
                )
            else:
                individual_gts.append(gt)

        overall_gt = overall_gt.bool().to(
            torch.float32
        )  # overall_gt is either 0. or 1.

        if self.loss_on_resize:
            overall_gt = self.resize_operation(overall_gt)
            overall_gt = overall_gt.long()  # any value<1.0 is converted to 0

        return overall_gt, individual_gts

    def __getitem__(self, index):
        img_path = self.images[index]
        gt_paths = self.gt[index]
        image = self.transform_image(img_path)

        overall_gt, individual_gts = self.transform_gt(gt_paths)

        # overall_gt.shape: [1, orig.height, orig.width]

        sample = {
            "image": image,
            "overall_gt": overall_gt,
            "individual_gts": individual_gts,
            "img_path": img_path,
        }
        return sample
