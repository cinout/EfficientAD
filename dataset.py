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
import torchvision.transforms.functional as TF
from datetime import datetime

mean_tensor = [0.485, 0.456, 0.406]
std_tensor = [0.229, 0.224, 0.225]
scale = ((0.8, 1),)
ratio = ((3 / 4, 4 / 3),)


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
                            size=(self.image_size, self.image_size),
                            scale=scale,
                            ratio=ratio,
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
                transforms.Normalize(mean=mean_tensor, std=std_tensor),
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
        geo_augment,
        image_size_before_geoaug,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.image_size_before_geoaug = image_size_before_geoaug
        self.geo_augment = geo_augment

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
        self.default_img_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean_tensor, std=std_tensor),
            ]
        )

        if self.geo_augment:
            self.resize_before_geoaug = transforms.Resize(
                size=(self.image_size_before_geoaug, self.image_size_before_geoaug),
                antialias=True,
            )
            self.resize_to_image_size = transforms.Resize(
                size=(self.image_size, self.image_size),
                antialias=True,
            )

    def __len__(self):
        return len(self.images)

    def transform_image(self, img_path, gt_paths, file_name=None):
        """
        prepare transforms
        """
        if self.geo_augment:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                torch.rand(
                    size=(
                        1,
                        self.image_size_before_geoaug,
                        self.image_size_before_geoaug,
                    )
                ),
                scale=scale,
                ratio=ratio,
            )
            randomness = (
                random.random()
            )  # randomness threshold for applying geo_augment

        """
        transform input image itself
        """
        img = Image.open(img_path)
        img = img.convert("RGB")

        if self.geo_augment:
            img = self.resize_before_geoaug(img)
            if randomness > 0.2:
                img = TF.crop(img, i, j, h, w)

        # TODO: remove this debug
        # debug_img = transforms.Resize(
        #     (self.image_size, self.image_size), antialias=True
        # )(img)
        # debug_img.save(f"{file_name}_img.png", "PNG")

        img = self.default_img_transform(img)

        """
        transform GTs
        """
        overall_gt = None  # purpose is to determine all negative (normal) pixels
        individual_gts = []
        for idx, each_path in enumerate(gt_paths):
            gt = Image.open(each_path)
            orig_width, orig_height = gt.size
            pixel_type = sorted(np.unique(np.array(gt)))[-1]

            gt = np.array(gt)
            gt = torch.tensor(gt)
            gt = gt.unsqueeze(0)  # [1, h, w]
            gt = gt.bool().to(torch.float32)  # either 0. or 1.

            if self.geo_augment:
                gt = self.resize_before_geoaug(gt)
                if randomness > 0.2:
                    gt = TF.crop(gt, i, j, h, w)

            gt = self.resize_to_image_size(gt)  # (1, image_size, image_size)

            gt = gt.bool().to(torch.long)
            # gt =  gt.long()  # any value<1.0 is converted to 0; otherwise 1

            # TODO: remove this debug
            # data = np.array(torch.permute(gt * 255, (1, 2, 0)), dtype=np.uint8)
            # data = np.repeat(data, repeats=3, axis=2)
            # gt_img = Image.fromarray(data)
            # gt_img.save(f"{file_name}_gt_{idx}.png", "PNG")

            individual_gts.append(
                {
                    "gt": gt,
                    "pixel_type": pixel_type,
                    "orig_height": orig_height,
                    "orig_width": orig_width,
                }
            )

            if overall_gt is not None:
                overall_gt = torch.logical_or(overall_gt, gt)
            else:
                overall_gt = gt

        # overall_gt = overall_gt.bool().to(
        #     torch.float32
        # )  # overall_gt is either 0. or 1.

        # overall_gt = self.resize_operation(overall_gt)
        # overall_gt = overall_gt.long()  # any value<1.0 is converted to 0

        # TODO: remove this debug
        # data = np.array(torch.permute(overall_gt * 255, (1, 2, 0)), dtype=np.uint8)
        # data = np.repeat(data, repeats=3, axis=2)
        # overall_gt_img = Image.fromarray(data)
        # overall_gt_img.save(f"{file_name}_gt_overall.png", "PNG")

        return img, overall_gt, individual_gts

    def __getitem__(self, index):
        img_path = self.images[index]
        gt_paths = self.gt[index]

        # TODO: remove this debug
        # file_name = img_path.split("/")[-1][:-4]
        # timestamp = (
        #     datetime.now().strftime("%Y%m%d_%H%M%S")
        #     + "_"
        #     + str(random.randint(0, 100))
        #     + "_"
        #     + str(random.randint(0, 100))
        # )
        # file_name = file_name + "_" + timestamp

        image, overall_gt, individual_gts = self.transform_image(img_path, gt_paths)

        # overall_gt.shape: [1, orig.height, orig.width]

        sample = {
            "image": image,
            "overall_gt": overall_gt,
            "individual_gts": individual_gts,
            "img_path": img_path,
        }
        return sample
