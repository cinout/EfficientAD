#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
from tqdm import tqdm
from common import (
    Autoencoder,
    PDN_Small,
    get_pdn_medium,
    ImageFolderWithoutTarget,
    ImageFolderWithPath,
    InfiniteDataloader,
)
from sklearn.metrics import roc_auc_score
from datetime import datetime
from functools import partial
from pvt_v2 import pvt_v2_b2_li
import torch.nn.functional as F
import cv2

category_acronym = {
    "breakfast_box": "bb",
    "juice_bottle": "jb",
    "pushpins": "pp",
    "screw_bag": "sb",
    "splicing_connectors": "sc",
}


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", default="mvtec_ad", choices=["mvtec_ad", "mvtec_loco"]
    )
    parser.add_argument(
        "-s",
        "--subdataset",
        default="bottle",
        help="One of 15 sub-datasets of Mvtec AD or 5" + "sub-datasets of Mvtec LOCO",
    )
    parser.add_argument("-o", "--output_dir")
    parser.add_argument(
        "-m", "--model_size", default="small", choices=["small", "medium"]
    )
    parser.add_argument(
        "-i",
        "--imagenet_train_path",
        default="none",
        help='Set to "none" to disable ImageNet'
        + "pretraining penalty. Or see README.md to"
        + "download ImageNet and set to ImageNet path",
    )
    parser.add_argument(
        "-a",
        "--mvtec_ad_path",
        default="./datasets/mvtec",
        help="Downloaded Mvtec AD dataset",
    )
    parser.add_argument(
        "-b",
        "--mvtec_loco_path",
        default="./datasets/loco",
        help="Downloaded Mvtec LOCO dataset",
    )

    parser.add_argument(
        "-w",
        "--weights",
        default="pretrained_pdn/pretrained_pdn_wide_resnet101_2/teacher_small.pth",
        help="pretrained weights for structural branch teacher",
    )

    # TODO: for analysis
    parser.add_argument("--ana_id", type=str, help="identifier for analysis")

    # TODO: use the following option
    parser.add_argument("--logical_teacher", choices=["vit", "pvt2"], required=True)
    parser.add_argument("--logical_teacher_image_size", type=int, default=512)

    parser.add_argument(
        "--pvt2_stage3",
        action="store_true",
        help="if set to True, then use 3rd stage output",
    )
    parser.add_argument(
        "--pvt2_stage4",
        action="store_true",
        help="if set to True, then use final stage output",
    )

    parser.add_argument("-t", "--train_steps", type=int, default=70000)
    parser.add_argument("--note", type=str, default="")
    return parser.parse_args()


# constants
seed = 42
on_gpu = torch.cuda.is_available()
device = "cuda" if on_gpu else "cpu"
image_size = 256

# data loading


def process_vit_features(features):
    target_size = 64

    features = features[:, 1:, :]
    B, N, C = features.shape
    H = int(math.sqrt(N))
    W = int(math.sqrt(N))
    features = features.transpose(1, 2).view(
        B, C, H, W
    )  # shape: (bs, 768, 32, 32) if input_size is 512

    if H != target_size:
        features = torch.nn.functional.interpolate(
            features, (target_size, target_size), mode="bilinear"
        )
    return features


def process_pvt_features(features, config):
    target_size = 64

    if config.pvt2_stage4:
        # in this case, features have 4 elements
        features = features[3:]
    elif config.pvt2_stage3:
        # in this case, features have 3 elements
        features = features[2:]
    else:
        # in this case, features have 3 elements
        features = features[1:]  # remove 1st element

    for i in range(len(features)):
        # upsampling feature map
        _features = features[i]
        _features = F.interpolate(
            _features,
            size=(target_size, target_size),
            mode="bilinear",
        )
        features[i] = _features

    features = torch.cat(features, dim=1)  # shape: [bs, c_sum=128+320=448, 64, 64]

    return features


def train_transform(image, config):
    default_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    teacher_logical_transform = transforms.Compose(
        [
            transforms.Resize(
                (config.logical_teacher_image_size, config.logical_teacher_image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_logical = transforms.RandomChoice(
        [
            transforms.ColorJitter(brightness=0.2),
            transforms.ColorJitter(contrast=0.2),
            transforms.ColorJitter(saturation=0.2),
        ]
    )
    logical_image = transform_logical(image)
    return (
        default_transform(image),  # structural student and teacher
        default_transform(logical_image),  # logical student
        teacher_logical_transform(logical_image),  # logical teacher
    )


def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()

    # if config.subdataset == "breakfast_box":
    #     config.output_dir = config.output_dir + "_[bb]"
    # elif config.subdataset == "juice_bottle":
    #     config.output_dir = config.output_dir + "_[jb]"
    # elif config.subdataset == "pushpins":
    #     config.output_dir = config.output_dir + "_[pp]"
    # elif config.subdataset == "screw_bag":
    #     config.output_dir = config.output_dir + "_[sb]"
    # elif config.subdataset == "splicing_connectors":
    #     config.output_dir = config.output_dir + "_[sc]"
    # else:
    #     raise ValueError(f"unknown subdataset name {config.subdataset}")

    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(config)).items()))
    )

    if config.dataset == "mvtec_ad":
        dataset_path = config.mvtec_ad_path
    elif config.dataset == "mvtec_loco":
        dataset_path = config.mvtec_loco_path
    else:
        raise Exception("Unknown config.dataset")

    pretrain_penalty = True
    if config.imagenet_train_path == "none":
        pretrain_penalty = False

    # # create output dir

    # test_output_dir = os.path.join(
    #     config.output_dir, "anomaly_maps", config.dataset, config.subdataset, "test"
    # )
    # os.makedirs(train_output_dir)
    # os.makedirs(test_output_dir)

    # load data
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.subdataset, "train"),
        transform=transforms.Lambda(partial(train_transform, config=config)),
    )
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, config.subdataset, "test")
    )
    if config.dataset == "mvtec_ad":
        # mvtec dataset paper recommend 10% validation set
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(
            full_train_set, [train_size, validation_size], rng
        )
    elif config.dataset == "mvtec_loco":
        train_set = full_train_set
        validation_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, config.subdataset, "validation"),
            transform=transforms.Lambda(partial(train_transform, config=config)),
        )
    else:
        raise Exception("Unknown config.dataset")

    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
    )
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    # if pretrain_penalty:
    #     # load pretraining data for penalty
    #     penalty_transform = transforms.Compose(
    #         [
    #             transforms.Resize((2 * image_size, 2 * image_size)),
    #             transforms.RandomGrayscale(0.3),
    #             transforms.CenterCrop(image_size),
    #             transforms.ToTensor(),
    #             transforms.Normalize(
    #                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #             ),
    #         ]
    #     )
    #     penalty_set = ImageFolderWithoutTarget(
    #         config.imagenet_train_path, transform=penalty_transform
    #     )
    #     penalty_loader = DataLoader(
    #         penalty_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
    #     )
    #     penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    # else:
    #     penalty_loader_infinite = itertools.repeat(None)

    structural_channels = 384
    if config.logical_teacher == "vit":
        logical_channels = 768
    elif config.logical_teacher == "pvt2":
        if config.pvt2_stage3:
            logical_channels = 320
        elif config.pvt2_stage4:
            logical_channels = 512
        else:
            # default: stage2+3
            logical_channels = 448
    else:
        raise Exception("wrong logical_teacher")

    # create models
    if config.model_size == "small":
        # structural branch teacher
        teacher_structural = PDN_Small(out_channels=structural_channels, padding=True)
        state_dict = torch.load(config.weights, map_location=device)
        pretrained_teacher_model = {}
        for k, v in state_dict.items():
            if k == "0.weight":
                pretrained_teacher_model["conv1.weight"] = v
            elif k == "0.bias":
                pretrained_teacher_model["conv1.bias"] = v
            elif k == "3.weight":
                pretrained_teacher_model["conv2.weight"] = v
            elif k == "3.bias":
                pretrained_teacher_model["conv2.bias"] = v
            elif k == "6.weight":
                pretrained_teacher_model["conv3.weight"] = v
            elif k == "6.bias":
                pretrained_teacher_model["conv3.bias"] = v
            elif k == "8.weight":
                pretrained_teacher_model["conv4.weight"] = v
            elif k == "8.bias":
                pretrained_teacher_model["conv4.bias"] = v
            else:
                raise ValueError(f"unknown state_dict key {k}")
        teacher_structural.load_state_dict(pretrained_teacher_model, strict=False)

        # structural branch student
        student_structural = PDN_Small(out_channels=structural_channels, padding=True)

        # logical branch teacher
        if config.logical_teacher == "vit":
            from urllib.request import urlretrieve
            from vit_models.modeling import VisionTransformer, CONFIGS

            os.makedirs("vit_model_checkpoints", exist_ok=True)

            if not os.path.isfile("vit_model_checkpoints/ViT-B_16-224.npz"):
                urlretrieve(
                    "https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz",
                    "vit_model_checkpoints/ViT-B_16-224.npz",
                )
            model = VisionTransformer(
                config=CONFIGS["ViT-B_16"],
                num_classes=1000,
                zero_head=False,
                img_size=config.logical_teacher_image_size,
                vis=True,
                vit_mid=False,
            )
            model.load_from(np.load("vit_model_checkpoints/ViT-B_16-224.npz"))
            teacher_logical = torch.nn.Sequential(
                *[model.transformer.embeddings, model.transformer.encoder]
            )
        elif config.logical_teacher == "pvt2":
            pretrained_model = torch.load(
                "pvt_model_checkpoints/mask_rcnn_pvt_v2_b2_li_fpn_1x_coco.pth",
                map_location=device,
            )
            pretrained_weights = {}
            for k, v in pretrained_model["state_dict"].items():
                if k.startswith("backbone"):
                    pretrained_weights[k.replace("backbone.", "")] = v
                else:
                    continue

            teacher_logical = pvt_v2_b2_li(pretrained=False, stage4=config.pvt2_stage4)
            teacher_logical.load_state_dict(pretrained_weights, strict=False)
        else:
            raise Exception("wrong logical_teacher")

        # logical branch student
        student_logical = Autoencoder(out_channels=logical_channels)

    elif config.model_size == "medium":
        pass

    else:
        raise Exception()

    # teacher_structural.eval()
    # teacher_logical.eval()
    # student_structural.train()
    # student_logical.train()

    teacher_structural.to(device)
    student_structural.to(device)
    teacher_logical.to(device)
    student_logical.to(device)

    (
        teacher_structural_mean,
        teacher_structural_std,
        teacher_logical_mean,
        teacher_logical_std,
    ) = teacher_normalization(teacher_structural, teacher_logical, train_loader, config)

    # optimizer = torch.optim.Adam(
    #     itertools.chain(student_structural.parameters(), student_logical.parameters()),
    #     lr=1e-4,
    #     weight_decay=1e-5,
    # )

    # # step_size is 66500
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1
    # )

    # tqdm_obj = tqdm(range(config.train_steps))
    # for (
    #     iteration,
    #     train_images,
    #     image_penalty,
    # ) in zip(tqdm_obj, train_loader_infinite, penalty_loader_infinite):
    #     (
    #         img_structural,
    #         img_logical_student,
    #         img_logical_teacher,
    #     ) = train_images

    #     img_structural = img_structural.to(device)
    #     img_logical_student = img_logical_student.to(device)
    #     img_logical_teacher = img_logical_teacher.to(device)

    #     if image_penalty is not None:
    #         image_penalty = image_penalty.cuda()

    #     # structural branch
    #     with torch.no_grad():
    #         teacher_structural_output = teacher_structural(img_structural)
    #         teacher_structural_output = (
    #             teacher_structural_output - teacher_structural_mean
    #         ) / teacher_structural_std
    #     student_structural_output = student_structural(img_structural)
    #     distance_structural = (
    #         teacher_structural_output - student_structural_output
    #     ) ** 2
    #     d_hard = torch.quantile(distance_structural, q=0.999)
    #     loss_hard = torch.mean(distance_structural[distance_structural >= d_hard])
    #     if image_penalty is not None:
    #         student_output_penalty = student_structural(image_penalty)
    #         loss_penalty = torch.mean(student_output_penalty**2)
    #         loss_structural = loss_hard + loss_penalty
    #     else:
    #         loss_structural = loss_hard

    #     # logical branch
    #     with torch.no_grad():
    #         if config.logical_teacher == "vit":
    #             teacher_logical_output = teacher_logical(img_logical_teacher)[0]
    #             teacher_logical_output = process_vit_features(teacher_logical_output)
    #         elif config.logical_teacher == "pvt2":
    #             teacher_logical_output = teacher_logical(img_logical_teacher)
    #             teacher_logical_output = process_pvt_features(
    #                 teacher_logical_output, config
    #             )
    #         else:
    #             raise Exception("wrong logical_teacher")
    #         teacher_logical_output = (
    #             teacher_logical_output - teacher_logical_mean
    #         ) / teacher_logical_std
    #     student_logical_output = student_logical(img_logical_student)
    #     distance_logical = (teacher_logical_output - student_logical_output) ** 2
    #     loss_logical = torch.mean(distance_logical)

    #     loss_total = loss_structural + loss_logical
    #     optimizer.zero_grad()
    #     loss_total.backward()
    #     optimizer.step()
    #     scheduler.step()

    #     if iteration % 200 == 0:
    #         print(
    #             "Iteration: ",
    #             iteration,
    #             "Current loss: {:.4f}".format(loss_total.item()),
    #         )

    train_output_dir = os.path.join(
        config.output_dir, "trainings", config.dataset, config.subdataset
    )
    pretrained_student_structural = torch.load(
        os.path.join(train_output_dir, "student_structural.pth"), map_location=device
    )
    pretrained_student_logical = torch.load(
        os.path.join(train_output_dir, "student_logical.pth"), map_location=device
    )
    student_structural.load_state_dict(pretrained_student_structural)
    student_logical.load_state_dict(pretrained_student_logical)

    teacher_logical.eval()
    student_logical.eval()
    teacher_structural.eval()
    student_structural.eval()

    (
        q_structural_start,
        q_structural_end,
        q_logical_start,
        q_logical_end,
    ) = map_normalization(
        validation_loader=validation_loader,
        teacher_structural=teacher_structural,
        teacher_logical=teacher_logical,
        student_structural=student_structural,
        student_logical=student_logical,
        teacher_structural_mean=teacher_structural_mean,
        teacher_structural_std=teacher_structural_std,
        teacher_logical_mean=teacher_logical_mean,
        teacher_logical_std=teacher_logical_std,
        config=config,
        desc="Final map normalization",
    )

    auc = test(
        test_set=test_set,
        teacher_structural=teacher_structural,
        teacher_logical=teacher_logical,
        student_structural=student_structural,
        student_logical=student_logical,
        teacher_structural_mean=teacher_structural_mean,
        teacher_structural_std=teacher_structural_std,
        teacher_logical_mean=teacher_logical_mean,
        teacher_logical_std=teacher_logical_std,
        q_structural_start=q_structural_start,
        q_structural_end=q_structural_end,
        q_logical_start=q_logical_start,
        q_logical_end=q_logical_end,
        config=config,
        test_output_dir=None,
        desc="Final inference",
    )
    print("Final image auc: {:.4f}".format(auc))


def normalizeData(data, minval, maxval):
    return (data - minval) / (maxval - minval)


heatmap_alpha = 0.5


@torch.no_grad()
def test(
    test_set,
    teacher_structural,
    teacher_logical,
    student_structural,
    student_logical,
    teacher_structural_mean,
    teacher_structural_std,
    teacher_logical_mean,
    teacher_logical_std,
    q_structural_start,
    q_structural_end,
    q_logical_start,
    q_logical_end,
    config,
    test_output_dir=None,
    desc="Running inference",
):
    # y_true = []
    # y_score = []
    map_comb_min = None
    map_comb_max = None

    # obtain min and max values for map_combined
    for image, target, path in tqdm(test_set, desc=desc):
        images = train_transform(image, config=config)

        (
            img_structural,
            img_logical_student,
            img_logical_teacher,
        ) = images

        img_structural = img_structural.to(device)
        img_logical_student = img_logical_student.to(device)
        img_logical_teacher = img_logical_teacher.to(device)
        img_structural = img_structural.unsqueeze(0)
        img_logical_student = img_logical_student.unsqueeze(0)
        img_logical_teacher = img_logical_teacher.unsqueeze(0)

        map_combined, _, _ = predict(
            config=config,
            img_structural=img_structural,
            img_logical_student=img_logical_student,
            img_logical_teacher=img_logical_teacher,
            teacher_structural=teacher_structural,
            teacher_logical=teacher_logical,
            student_structural=student_structural,
            student_logical=student_logical,
            teacher_structural_mean=teacher_structural_mean,
            teacher_structural_std=teacher_structural_std,
            teacher_logical_mean=teacher_logical_mean,
            teacher_logical_std=teacher_logical_std,
            q_structural_start=q_structural_start,
            q_structural_end=q_structural_end,
            q_logical_start=q_logical_start,
            q_logical_end=q_logical_end,
        )

        if map_comb_min is None:
            map_comb_min = torch.min(map_combined)
            map_comb_max = torch.max(map_combined)
        else:
            map_comb_min = min(map_comb_min, torch.min(map_combined))
            map_comb_max = max(map_comb_max, torch.max(map_combined))

    heatmap_folder = f"analysis_heatmap_{config.ana_id}/{config.subdataset}/"
    os.makedirs(heatmap_folder, exist_ok=True)

    # output heatmaps for separate branches
    for image, target, path in tqdm(test_set, desc=desc):
        images = train_transform(image, config=config)

        (
            img_structural,
            img_logical_student,
            img_logical_teacher,
        ) = images

        img_structural = img_structural.to(device)
        img_logical_student = img_logical_student.to(device)
        img_logical_teacher = img_logical_teacher.to(device)
        img_structural = img_structural.unsqueeze(0)
        img_logical_student = img_logical_student.unsqueeze(0)
        img_logical_teacher = img_logical_teacher.unsqueeze(0)

        _, map_structural, map_logical = predict(
            config=config,
            img_structural=img_structural,
            img_logical_student=img_logical_student,
            img_logical_teacher=img_logical_teacher,
            teacher_structural=teacher_structural,
            teacher_logical=teacher_logical,
            student_structural=student_structural,
            student_logical=student_logical,
            teacher_structural_mean=teacher_structural_mean,
            teacher_structural_std=teacher_structural_std,
            teacher_logical_mean=teacher_logical_mean,
            teacher_logical_std=teacher_logical_std,
            q_structural_start=q_structural_start,
            q_structural_end=q_structural_end,
            q_logical_start=q_logical_start,
            q_logical_end=q_logical_end,
        )

        map_structural = map_structural.squeeze().cpu().numpy()  # shape: (256, 256)
        map_logical = map_logical.squeeze().cpu().numpy()
        map_structural = np.expand_dims(map_structural, axis=2)
        map_logical = np.expand_dims(map_logical, axis=2)

        raw_img_path = os.path.join(path)
        raw_img = np.array(cv2.imread(raw_img_path, cv2.IMREAD_COLOR))
        raw_img = cv2.resize(raw_img, dsize=(256, 256))

        # get heatmap
        pred_mask_structural = np.uint8(
            normalizeData(map_structural, map_comb_min, map_comb_max) * 255
        )
        heatmap_structural = cv2.applyColorMap(pred_mask_structural, cv2.COLORMAP_JET)
        hmap_overlay_gt_img_structural = (
            heatmap_structural * heatmap_alpha + raw_img * (1.0 - heatmap_alpha)
        )

        pred_mask_logical = np.uint8(
            normalizeData(map_logical, map_comb_min, map_comb_max) * 255
        )
        heatmap_logical = cv2.applyColorMap(pred_mask_logical, cv2.COLORMAP_JET)
        hmap_overlay_gt_img_logical = heatmap_logical * heatmap_alpha + raw_img * (
            1.0 - heatmap_alpha
        )

        cv2.imwrite(
            os.path.join(
                heatmap_folder,
                f"{'_'.join(path.split('/')[-2:])[:-4]}_structural.jpg",
            ),
            hmap_overlay_gt_img_structural,
        )
        cv2.imwrite(
            os.path.join(
                heatmap_folder,
                f"{'_'.join(path.split('/')[-2:])[:-4]}_logical.jpg",
            ),
            hmap_overlay_gt_img_logical,
        )

    auc = 0
    return auc * 100


@torch.no_grad()
def predict(
    config,
    img_structural,
    img_logical_student,
    img_logical_teacher,
    teacher_structural,
    teacher_logical,
    student_structural,
    student_logical,
    teacher_structural_mean,
    teacher_structural_std,
    teacher_logical_mean,
    teacher_logical_std,
    q_structural_start=None,
    q_structural_end=None,
    q_logical_start=None,
    q_logical_end=None,
):
    teacher_structural_output = teacher_structural(img_structural)
    if config.logical_teacher == "vit":
        teacher_logical_output = teacher_logical(img_logical_teacher)[0]
        teacher_logical_output = process_vit_features(teacher_logical_output)
    elif config.logical_teacher == "pvt2":
        teacher_logical_output = teacher_logical(img_logical_teacher)
        teacher_logical_output = process_pvt_features(teacher_logical_output, config)
    else:
        raise Exception("wrong logical_teacher")

    teacher_structural_output = (
        teacher_structural_output - teacher_structural_mean
    ) / teacher_structural_std
    teacher_logical_output = (
        teacher_logical_output - teacher_logical_mean
    ) / teacher_logical_std

    student_structural_output = student_structural(img_structural)
    student_logical_output = student_logical(img_logical_student)

    map_structural = torch.mean(
        (teacher_structural_output - student_structural_output) ** 2,
        dim=1,
        keepdim=True,
    )  # shape: (bs, 1, h, w)
    map_logical = torch.mean(
        (teacher_logical_output - student_logical_output) ** 2,
        dim=1,
        keepdim=True,
    )  # shape: (bs, 1, h, w)

    # upsample map_structural and map_logical to 256*256
    map_structural = torch.nn.functional.interpolate(
        map_structural, (image_size, image_size), mode="bilinear"
    )
    map_logical = torch.nn.functional.interpolate(
        map_logical, (image_size, image_size), mode="bilinear"
    )

    if q_structural_start is not None:
        map_structural = (
            0.1
            * (map_structural - q_structural_start)
            / (q_structural_end - q_structural_start)
        )
    if q_logical_start is not None:
        map_logical = (
            0.1 * (map_logical - q_logical_start) / (q_logical_end - q_logical_start)
        )
    map_combined = 0.5 * map_structural + 0.5 * map_logical

    return map_combined, map_structural, map_logical


@torch.no_grad()
def map_normalization(
    validation_loader,
    teacher_structural,
    teacher_logical,
    student_structural,
    student_logical,
    teacher_structural_mean,
    teacher_structural_std,
    teacher_logical_mean,
    teacher_logical_std,
    config,
    desc="Map normalization",
):
    maps_structural = []
    maps_logical = []
    # ignore augmented ae image
    for images in tqdm(validation_loader, desc=desc):
        (
            img_structural,
            img_logical_student,
            img_logical_teacher,
        ) = images

        img_structural = img_structural.to(device)
        img_logical_student = img_logical_student.to(device)
        img_logical_teacher = img_logical_teacher.to(device)

        map_combined, map_st, map_log = predict(
            config=config,
            img_structural=img_structural,
            img_logical_student=img_logical_student,
            img_logical_teacher=img_logical_teacher,
            teacher_structural=teacher_structural,
            teacher_logical=teacher_logical,
            student_structural=student_structural,
            student_logical=student_logical,
            teacher_structural_mean=teacher_structural_mean,
            teacher_structural_std=teacher_structural_std,
            teacher_logical_mean=teacher_logical_mean,
            teacher_logical_std=teacher_logical_std,
        )
        maps_structural.append(map_st)
        maps_logical.append(map_log)
    maps_structural = torch.cat(maps_structural)
    maps_logical = torch.cat(maps_logical)

    q_structural_start = torch.quantile(maps_structural, q=0.9)
    q_structural_end = torch.quantile(maps_structural, q=0.995)
    q_logical_start = torch.quantile(maps_logical, q=0.9)
    q_logical_end = torch.quantile(maps_logical, q=0.995)
    return q_structural_start, q_structural_end, q_logical_start, q_logical_end


@torch.no_grad()
def teacher_normalization(teacher_structural, teacher_logical, train_loader, config):
    mean_outputs_structural = []
    mean_outputs_logical = []
    for train_images in tqdm(train_loader, desc="Computing mean of features"):
        (
            img_structural,
            _,
            img_logical_teacher,
        ) = train_images

        img_structural = img_structural.to(device)
        img_logical_teacher = img_logical_teacher.to(device)

        teacher_structural_output = teacher_structural(img_structural)
        if config.logical_teacher == "vit":
            teacher_logical_output = teacher_logical(img_logical_teacher)[0]
            teacher_logical_output = process_vit_features(teacher_logical_output)
        elif config.logical_teacher == "pvt2":
            teacher_logical_output = teacher_logical(img_logical_teacher)
            teacher_logical_output = process_pvt_features(
                teacher_logical_output, config
            )
        else:
            raise Exception("wrong logical_teacher")

        mean_outputs_structural.append(
            torch.mean(teacher_structural_output, dim=[0, 2, 3])
        )
        mean_outputs_logical.append(torch.mean(teacher_logical_output, dim=[0, 2, 3]))

    channel_mean_structural = torch.mean(torch.stack(mean_outputs_structural), dim=0)
    channel_mean_structural = channel_mean_structural[None, :, None, None]
    channel_mean_logical = torch.mean(torch.stack(mean_outputs_logical), dim=0)
    channel_mean_logical = channel_mean_logical[None, :, None, None]

    mean_distances_structural = []
    mean_distances_logical = []
    for train_images in tqdm(train_loader, desc="Computing std of features"):
        (
            img_structural,
            _,
            img_logical_teacher,
        ) = train_images

        img_structural = img_structural.to(device)
        img_logical_teacher = img_logical_teacher.to(device)

        teacher_structural_output = teacher_structural(img_structural)
        if config.logical_teacher == "vit":
            teacher_logical_output = teacher_logical(img_logical_teacher)[0]
            teacher_logical_output = process_vit_features(teacher_logical_output)
        elif config.logical_teacher == "pvt2":
            teacher_logical_output = teacher_logical(img_logical_teacher)
            teacher_logical_output = process_pvt_features(
                teacher_logical_output, config
            )
        else:
            raise Exception("wrong logical_teacher")

        mean_distances_structural.append(
            torch.mean(
                (teacher_structural_output - channel_mean_structural) ** 2,
                dim=[0, 2, 3],
            )
        )
        mean_distances_logical.append(
            torch.mean(
                (teacher_logical_output - channel_mean_logical) ** 2, dim=[0, 2, 3]
            )
        )

    channel_var_structural = torch.mean(torch.stack(mean_distances_structural), dim=0)
    channel_var_structural = channel_var_structural[None, :, None, None]
    channel_std_structural = torch.sqrt(channel_var_structural)
    channel_var_logical = torch.mean(torch.stack(mean_distances_logical), dim=0)
    channel_var_logical = channel_var_logical[None, :, None, None]
    channel_std_logical = torch.sqrt(channel_var_logical)

    return (
        channel_mean_structural,
        channel_std_structural,
        channel_mean_logical,
        channel_std_logical,
    )


if __name__ == "__main__":
    main()
