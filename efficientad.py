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
    FocalLoss,
    IndividualGTLoss,
    PDN_Small,
    get_pdn_medium,
    ImageFolderWithoutTarget,
    ImageFolderWithTargetAndPath,
    InfiniteDataloader,
    l2_normalize,
)
from sklearn.metrics import roc_auc_score
from datetime import datetime
from functools import partial
import cv2
import torch.nn.functional as F
from dataset import LogicalAnomalyDataset, MyDummyDataset, NormalDatasetForGeoAug
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

timestamp = (
    datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_"
    + str(random.randint(0, 100))
    + "_"
    + str(random.randint(0, 100))
)


def normalizeData(data, minval, maxval):
    return (data - minval) / (maxval - minval)


heatmap_alpha = 0.5


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
    parser.add_argument("-o", "--output_dir", default=f"outputs/output_{timestamp}")
    parser.add_argument(
        "-m", "--model_size", default="small", choices=["small", "medium"]
    )
    parser.add_argument(
        "-w",
        "--weights",
        default="pretrained_pdn/pretrained_pdn_wide_resnet101_2/teacher_small.pth",
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
        "--pretrained_network",
        choices=["wide_resnet101_2"],
        type=str,
        default="wide_resnet101_2",
    )
    parser.add_argument(
        "--analysis_heatmap",
        action="store_true",
        help="if set to True, then generate branch-wise analysis heatmap",
    )
    parser.add_argument("-t", "--train_steps", type=int, default=70000)
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--seeds", type=int, default=[42], nargs="+")

    # NEW OPTIONS
    parser.add_argument(
        "--include_logicano",
        action="store_true",
        help="if set to True, then include a few logical anomalies into training",
    )
    parser.add_argument(
        "--logicano_select",
        type=str,
        choices=["percent", "absolute"],
        default="absolute",
    )
    parser.add_argument(
        "--percent_logicano",
        type=float,
        default=0.1,
        help="[logicano_select=percent] proportion of real logical anomalies used in training",
    )
    parser.add_argument(
        "--num_logicano",
        type=int,
        default=10,
        help="[logicano_select=absolute] number of real logical anomalies used in training",
    )

    parser.add_argument(
        "--geo_augment",
        action="store_true",
        help="if set to True, then apply RandomResizedCrop augmentations to the training images",
    )
    parser.add_argument(
        "--use_rotate_flip",
        action="store_true",
        help="if set to True, then apply HFlip/VFlip/Rotate augmentations to the training images",
    )
    parser.add_argument(
        "--geo_augment_only_on_logicano",
        action="store_true",
        help="if set to True, then apply RandomResizedCrop augmentations to only logicano images",
    )

    parser.add_argument(
        "--stg1_ckpt",
        type=str,
        help="should be the path of the parent folder of xxx.pth",
    )
    parser.add_argument(
        "--trained_folder",
        type=str,
        default="",
        help="path of trained model weights",
    )

    parser.add_argument(
        "--use_l1_loss",
        action="store_true",
        help="if set to True, then add l1 loss to focal loss",
    )
    parser.add_argument(
        "--w_f1loss",
        type=float,
        default=1.0,
        help="used if use_l1_loss is on",
    )
    parser.add_argument(
        "--limit_on_loss",
        action="store_true",
        help="if set to True, then limit the # of loss by the saturation_area",
    )

    parser.add_argument(
        "--lid_score_eval",
        action="store_true",
        help="if set to True, then add add lid score map",
    )
    parser.add_argument(
        "--lid_eval_onwhat",
        type=str,
        choices=["separate_mean", "diff_mean"],
        default="diff_mean",
    )
    parser.add_argument(
        "--lid_score_train",
        action="store_true",
        help="if set to True, then add lid score to loss",
    )
    parser.add_argument(
        "--lid_train_onwhat",
        type=str,
        choices=["separate_mean", "diff_mean"],
        default="separate_mean",
    )
    parser.add_argument(
        "--lid_regularizer",
        type=float,
        default=1.0,
        help="the denominator for loss of lid_score during training",
    )

    parser.add_argument(
        "--use_masked_conv",
        action="store_true",
        help="if set to True, then apply masked convolution on AE",
    )
    parser.add_argument(
        "--pos_masked_conv",
        type=str,
        choices=["d4", "d5", "d6", "d7"],
        default="d6",
    )
    parser.add_argument("--w_loss_masked_conv", type=float, default=1.0)
    parser.add_argument(
        "--masked_dilation",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--masked_kernel",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="if set to True, then enter debug mode",
    )

    parser.add_argument(
        "--lid_on_history",
        action="store_true",
        help="if set to True, then apply LID on loss history",
    )

    return parser.parse_args()


# constants
on_gpu = torch.cuda.is_available()
device = "cuda" if on_gpu else "cpu"
image_size = 256
image_size_before_geoaug = 512
out_channels = 384
acronym = {
    "breakfast_box": "bb",
    "juice_bottle": "jb",
    "pushpins": "pp",
    "screw_bag": "sb",
    "splicing_connectors": "sc",
}
lid_history_step = 50
lid_history_count = 60


def lid_mle_modified(
    data, reference, k=20, compute_mode="use_mm_for_euclid_dist_if_necessary"
):
    # for single input data
    k = min(k, reference.shape[0])
    data = torch.flatten(data, start_dim=1)  # [1, C]
    reference = torch.flatten(reference, start_dim=1)  # [N, C]
    r = torch.cdist(data, reference, p=2, compute_mode=compute_mode)  # [1, N]
    a, idx = torch.sort(r, dim=1)
    a = a.squeeze(0)
    lids = -k / torch.sum(torch.log(a[0:k] / a[k - 1] + 1.0e-4))
    return lids


def lid_mle(data, reference, k=20, compute_mode="use_mm_for_euclid_dist_if_necessary"):
    b = data.shape[0]
    k = min(k, b - 2)
    data = torch.flatten(data, start_dim=1)
    reference = torch.flatten(reference, start_dim=1)
    r = torch.cdist(data, reference, p=2, compute_mode=compute_mode)
    a, idx = torch.sort(r, dim=1)
    lids = -k / torch.sum(torch.log(a[:, 1:k] / a[:, k].view(-1, 1) + 1.0e-4), dim=1)
    return lids


def train_transform(image, config):
    default_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_ae = transforms.RandomChoice(
        [
            transforms.ColorJitter(brightness=0.2),
            transforms.ColorJitter(contrast=0.2),
            transforms.ColorJitter(saturation=0.2),
        ]
    )
    return (
        default_transform(image),
        default_transform(transform_ae(image)),
    )


def main(config, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    if config.subdataset == "breakfast_box":
        output_dir = config.output_dir + f"_sd{seed}" + "_[bb]"
    elif config.subdataset == "juice_bottle":
        output_dir = config.output_dir + f"_sd{seed}" + "_[jb]"
    elif config.subdataset == "pushpins":
        output_dir = config.output_dir + f"_sd{seed}" + "_[pp]"
    elif config.subdataset == "screw_bag":
        output_dir = config.output_dir + f"_sd{seed}" + "_[sb]"
    elif config.subdataset == "splicing_connectors":
        output_dir = config.output_dir + f"_sd{seed}" + "_[sc]"
    else:
        raise ValueError(f"unknown subdataset name {config.subdataset}")

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

    # create output dir
    train_output_dir = os.path.join(
        output_dir, "trainings", config.dataset, config.subdataset
    )
    test_output_dir = os.path.join(
        output_dir, "anomaly_maps", config.dataset, config.subdataset, "test"
    )
    os.makedirs(train_output_dir)
    os.makedirs(test_output_dir)

    # load data
    if config.geo_augment:
        # create normal images train set that uses geometric augmentation
        train_set_for_geoaug = NormalDatasetForGeoAug(
            path=os.path.join(dataset_path, config.subdataset, "train/good"),
            image_size_before_geoaug=image_size_before_geoaug,
            image_size=image_size,
            use_rotate_flip=config.use_rotate_flip,
        )
        train_loader_for_geoaug = DataLoader(
            train_set_for_geoaug,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        train_loader_for_geoaug_infinite = InfiniteDataloader(train_loader_for_geoaug)

    if config.include_logicano:
        logicano_data = LogicalAnomalyDataset(
            logicano_select=config.logicano_select,
            num_logicano=config.num_logicano,
            percent_logicano=config.percent_logicano,
            subdataset=config.subdataset,
            image_size=image_size,
            use_rotate_flip=config.use_rotate_flip,
            geo_augment=config.geo_augment or config.geo_augment_only_on_logicano,
            image_size_before_geoaug=image_size_before_geoaug,
        )
        # _, orig_height, orig_width = logicano_data[0]["overall_gt"].shape

        logicano_dataloader = DataLoader(
            logicano_data,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        logicano_dataloader_infite = InfiniteDataloader(logicano_dataloader)

    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.subdataset, "train"),
        transform=transforms.Lambda(
            partial(
                train_transform,
                config=config,
            )
        ),
    )
    test_set = ImageFolderWithTargetAndPath(
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
        train_set,
        batch_size=8 if config.lid_score_train else 1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=config.lid_score_train,
    )

    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    if pretrain_penalty:
        # load pretraining data for penalty
        penalty_transform = transforms.Compose(
            [
                transforms.Resize((2 * image_size, 2 * image_size)),
                transforms.RandomGrayscale(0.3),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        penalty_set = ImageFolderWithoutTarget(
            config.imagenet_train_path, transform=penalty_transform
        )
        penalty_loader = DataLoader(
            penalty_set,
            batch_size=8 if config.lid_score_train else 1,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    # create models
    if config.model_size == "small":
        teacher = PDN_Small(out_channels=out_channels, padding=True, config=config)
        student = PDN_Small(out_channels=2 * out_channels, padding=True, config=config)
        # teacher = get_pdn_small(out_channels, padding=True)
        # student = get_pdn_small(2 * out_channels, padding=True)
    elif config.model_size == "medium":
        teacher = get_pdn_medium(out_channels, padding=True)
        student = get_pdn_medium(2 * out_channels, padding=True)
    else:
        raise Exception()

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
    teacher.load_state_dict(pretrained_teacher_model, strict=False)

    # autoencoder = get_autoencoder(out_channels)
    autoencoder = Autoencoder(out_channels=out_channels, config=config)

    # teacher frozen
    teacher.eval()
    student.train()
    autoencoder.train()

    teacher = teacher.to(device)
    student = student.to(device)
    autoencoder = autoencoder.to(device)

    # TODO: uncomment below
    teacher_mean, teacher_std = teacher_normalization(
        teacher,
        train_loader,
        config,
    )
    # with open("teacher_mean.t", "rb") as f:
    #     teacher_mean = torch.load(f)
    # with open("teacher_std.t", "rb") as f:
    #     teacher_std = torch.load(f)

    if config.trained_folder == "":
        optimizer = torch.optim.Adam(
            itertools.chain(student.parameters(), autoencoder.parameters()),
            lr=1e-4,
            weight_decay=1e-5,
        )

        # step_size is 66500
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1
        )

        if config.include_logicano:
            loss_focal = FocalLoss()
            loss_individual_gt = IndividualGTLoss(config)

        writer = SummaryWriter(
            log_dir=f"./runs/{timestamp}_{config.subdataset}_sd{seed}"
        )  # Writer will output to ./runs/ directory by default. You can change log_dir in here

        tqdm_obj = tqdm(range(config.train_steps))

        if config.lid_on_history:
            # note down all the ckpts to store
            saved_iterations = []
            cur_iteration = config.train_steps - 1
            for v in range(lid_history_count):
                saved_iterations.append(cur_iteration)
                cur_iteration -= lid_history_step

        if config.include_logicano:
            for (
                iteration,
                normal,
                logicano,
                image_penalty,
            ) in zip(
                tqdm_obj,
                (
                    train_loader_for_geoaug_infinite
                    if config.geo_augment
                    else train_loader_infinite
                ),
                logicano_dataloader_infite,
                penalty_loader_infinite,
            ):
                # take turns to train normal and logicano
                criterion = iteration % 2 == 0

                if criterion:
                    # train normal

                    (image_st, image_ae) = normal

                    image_st = image_st.to(device)
                    image_ae = image_ae.to(device)

                    if image_penalty is not None:
                        image_penalty = image_penalty.to(device)

                    with torch.no_grad():
                        teacher_output_st = teacher(image_st)
                        teacher_output_st = (
                            teacher_output_st - teacher_mean
                        ) / teacher_std
                    student_output_st = student(image_st)[
                        :, :out_channels
                    ]  # the first half of student outputs
                    distance_st = (teacher_output_st - student_output_st) ** 2
                    d_hard = torch.quantile(distance_st, q=0.999)
                    loss_hard = torch.mean(distance_st[distance_st >= d_hard])

                    if image_penalty is not None:
                        student_output_penalty = student(image_penalty)[
                            :, :out_channels
                        ]
                        loss_penalty = torch.mean(student_output_penalty**2)
                        loss_st = loss_hard + loss_penalty
                    else:
                        loss_st = loss_hard

                    ae_output = autoencoder(image_ae)

                    with torch.no_grad():
                        teacher_output_ae = teacher(image_ae)
                        teacher_output_ae = (
                            teacher_output_ae - teacher_mean
                        ) / teacher_std
                    student_output_ae = student(image_ae)[
                        :, out_channels:
                    ]  # the second half of student outputs
                    distance_ae = (teacher_output_ae - ae_output) ** 2
                    distance_stae = (ae_output - student_output_ae) ** 2
                    loss_ae = torch.mean(distance_ae)
                    loss_stae = torch.mean(distance_stae)

                    loss_total = loss_st + loss_ae + loss_stae
                else:
                    # train logicano
                    logicano_image = logicano["image"]  # [1, 3, 256, 256]
                    overall_gt = logicano["overall_gt"]  # [1, 1, orig.h, orig.w]
                    individual_gts = logicano[
                        "individual_gts"
                    ]  # each item: [1, 1, orig.h, orig.w]

                    # _, _, orig_height, orig_width = overall_gt.shape

                    logicano_image = logicano_image.to(device)
                    overall_gt = overall_gt.to(device)
                    individual_gts = [
                        {
                            "gt": item["gt"].to(device),
                            "pixel_type": item["pixel_type"],
                            "orig_height": item["orig_height"],
                            "orig_width": item["orig_width"],
                        }
                        for item in individual_gts
                    ]

                    teacher_output = teacher(logicano_image)
                    teacher_output = (teacher_output - teacher_mean) / teacher_std
                    student_output = student(logicano_image)
                    autoencoder_output = autoencoder(logicano_image)

                    map_st = torch.mean(
                        (teacher_output - student_output[:, :out_channels]) ** 2,
                        dim=1,
                        keepdim=True,
                    )  # shape: (bs, 1, h, w)
                    map_ae = torch.mean(
                        (autoencoder_output - student_output[:, out_channels:]) ** 2,
                        dim=1,
                        keepdim=True,
                    )

                    map_st = torch.nn.functional.interpolate(
                        map_st, (image_size, image_size), mode="bilinear"
                    )
                    map_ae = torch.nn.functional.interpolate(
                        map_ae, (image_size, image_size), mode="bilinear"
                    )

                    map_combined = 0.5 * map_st + 0.5 * map_ae  # [1, 1, h, w]
                    _, _, h, w = map_combined.shape
                    map_combined = map_combined.reshape(1, 1, -1)
                    map_combined = F.normalize(map_combined, dim=2)
                    map_combined = map_combined.reshape(
                        1, 1, h, w
                    )  # prob of abnormalcy

                    map_combined_inverse = 1 - map_combined  # prob of normal
                    map_combined = torch.cat(
                        [map_combined_inverse, map_combined], dim=1
                    )

                    # loss_focal for overall negative target pixels only
                    loss_overall_negative = loss_focal(map_combined, overall_gt)
                    # loss for positive pixels in individual gts

                    loss_individual_positive = loss_individual_gt(
                        map_combined[0], individual_gts
                    )
                    loss_total = loss_overall_negative + loss_individual_positive

                    if config.use_l1_loss:
                        loss_total += config.w_f1loss * F.l1_loss(
                            map_combined[:, 1, :, :].unsqueeze(1),
                            overall_gt,
                            reduction="mean",
                        )
                writer.add_scalar("Loss/train", loss_total, iteration)
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
                scheduler.step()

                if iteration % 200 == 0:
                    print(
                        "Iteration: ",
                        iteration,
                        "Current loss: {:.4f}".format(loss_total.item()),
                    )
        else:
            # only normal images (may be augmented)
            if config.lid_score_train:
                # set up queues
                queue_size = 64
                queue_teacher = torch.randn(queue_size, 384, 64, 64)
                queue_student_st = torch.randn(queue_size, 384, 64, 64)
                queue_student_ae = torch.randn(queue_size, 384, 64, 64)
                queue_autoencoder = torch.randn(queue_size, 384, 64, 64)
                queue_teacher = queue_teacher.to(device)
                queue_student_st = queue_student_st.to(device)
                queue_student_ae = queue_student_ae.to(device)
                queue_autoencoder = queue_autoencoder.to(device)
                queue_ptr = 0

            for (
                iteration,
                normal,
                image_penalty,
            ) in zip(
                tqdm_obj,
                (
                    train_loader_for_geoaug_infinite
                    if config.geo_augment
                    else train_loader_infinite
                ),
                penalty_loader_infinite,
            ):
                (image_st, image_ae) = normal

                image_st = image_st.to(device)
                image_ae = image_ae.to(device)

                if image_penalty is not None:
                    image_penalty = image_penalty.to(device)

                with torch.no_grad():
                    teacher_output_st = teacher(image_st)
                    teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
                student_output_st = student(image_st)[
                    :, :out_channels
                ]  # the first half of student outputs
                distance_st = (teacher_output_st - student_output_st) ** 2
                d_hard = torch.quantile(distance_st, q=0.999)
                loss_hard = torch.mean(distance_st[distance_st >= d_hard])

                if image_penalty is not None:
                    student_output_penalty = student(image_penalty)[:, :out_channels]
                    loss_penalty = torch.mean(student_output_penalty**2)
                    loss_st = loss_hard + loss_penalty
                else:
                    loss_st = loss_hard

                if config.use_masked_conv:
                    ae_output, loss_ssmctb = autoencoder(image_ae)
                else:
                    ae_output = autoencoder(image_ae)

                with torch.no_grad():
                    teacher_output_ae = teacher(image_ae)
                    teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
                student_output_ae = student(image_ae)[
                    :, out_channels:
                ]  # the second half of student outputs
                distance_ae = (teacher_output_ae - ae_output) ** 2
                distance_stae = (ae_output - student_output_ae) ** 2
                loss_ae = torch.mean(distance_ae)
                loss_stae = torch.mean(distance_stae)

                if config.lid_score_train:
                    batch_size = teacher_output_st.shape[0]
                    assert queue_size % batch_size == 0  # for simplicity
                    queue_teacher[queue_ptr : queue_ptr + batch_size, :] = (
                        teacher_output_st.clone().detach()
                    )
                    queue_student_st[queue_ptr : queue_ptr + batch_size, :] = (
                        student_output_st.clone().detach()
                    )
                    queue_student_ae[queue_ptr : queue_ptr + batch_size, :] = (
                        student_output_ae.clone().detach()
                    )
                    queue_autoencoder[queue_ptr : queue_ptr + batch_size, :] = (
                        ae_output.clone().detach()
                    )
                    queue_ptr = (queue_ptr + batch_size) % queue_size

                    if config.lid_train_onwhat == "separate_mean":
                        # student_output_st
                        # student_output_ae
                        # ae_output
                        all_lid_scores = []

                        all_lid_scores.append(
                            lid_mle(
                                data=student_output_st,
                                reference=queue_student_st,
                            )
                        )

                        all_lid_scores.append(
                            lid_mle(
                                data=student_output_ae,
                                reference=queue_student_ae,
                            )
                        )
                        all_lid_scores.append(
                            lid_mle(
                                data=ae_output,
                                reference=queue_autoencoder,
                            )
                        )

                        # _, _, H, W = student_output_st.shape

                        # for i in range(H):
                        #     for j in range(W):
                        #         all_lid_scores.append(
                        #             lid_mle(
                        #                 data=student_output_st[:, :, i, j],
                        #                 reference=queue_student_st[:, :, i, j],
                        #             )
                        #         )

                        #         all_lid_scores.append(
                        #             lid_mle(
                        #                 data=student_output_ae[:, :, i, j],
                        #                 reference=queue_student_ae[:, :, i, j],
                        #             )
                        #         )
                        #         all_lid_scores.append(
                        #             lid_mle(
                        #                 data=ae_output[:, :, i, j],
                        #                 reference=queue_autoencoder[:, :, i, j],
                        #             )
                        #         )

                    elif config.lid_train_onwhat == "diff_mean":
                        all_lid_scores = []

                        all_lid_scores.append(
                            lid_mle(
                                data=distance_st,
                                reference=(queue_teacher - queue_student_st) ** 2,
                            )
                        )

                        all_lid_scores.append(
                            lid_mle(
                                data=distance_stae,
                                reference=(queue_autoencoder - queue_student_ae) ** 2,
                            )
                        )

                        # _, _, H, W = distance_st.shape
                        # for i in range(H):
                        #     for j in range(W):
                        #         all_lid_scores.append(
                        #             lid_mle(
                        #                 data=distance_st[:, :, i, j],
                        #                 reference=(
                        #                     (queue_teacher - queue_student_st) ** 2
                        #                 )[:, :, i, j],
                        #             )
                        #         )

                        #         all_lid_scores.append(
                        #             lid_mle(
                        #                 data=distance_stae[:, :, i, j],
                        #                 reference=(
                        #                     (queue_autoencoder - queue_student_ae) ** 2
                        #                 )[:, :, i, j],
                        #             )
                        #         )

                    all_lid_scores = torch.cat(all_lid_scores, dim=0)

                    all_lid_scores = 1.0 * torch.log(
                        all_lid_scores / config.lid_regularizer + 1.0e-4
                    )
                    loss_lid = torch.mean(all_lid_scores)

                    loss_total = loss_st + loss_ae + loss_stae + loss_lid
                else:
                    if config.use_masked_conv:
                        loss_total = (
                            loss_st
                            + loss_ae
                            + loss_stae
                            + config.w_loss_masked_conv * loss_ssmctb
                        )
                    else:
                        loss_total = loss_st + loss_ae + loss_stae

                writer.add_scalar("Loss/train", loss_total, iteration)
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()
                scheduler.step()

                if iteration % 200 == 0:
                    print(
                        "Iteration: ",
                        iteration,
                        "Current loss: {:.4f}".format(loss_total.item()),
                    )

                if iteration in saved_iterations:
                    torch.save(
                        student.state_dict(),
                        os.path.join(train_output_dir, f"student_ckpt_{iteration}.pth"),
                    )
                    torch.save(
                        autoencoder.state_dict(),
                        os.path.join(
                            train_output_dir, f"autoencoder_ckpt_{iteration}.pth"
                        ),
                    )

        writer.flush()  # Call flush() method to make sure that all pending events have been written to disk
        writer.close()  # if you do not need the summary writer anymore, call close() method.

        # torch.save(teacher, os.path.join(train_output_dir, "teacher_final.pth"))
        torch.save(student, os.path.join(train_output_dir, "student_final.pth"))
        torch.save(autoencoder, os.path.join(train_output_dir, "autoencoder_final.pth"))
    else:
        # load pretrained weights for AE and student
        folder_name = f"{config.trained_folder}_sd{seed}_[{acronym[config.subdataset]}]"

        if config.lid_on_history:
            # get all checkpoints
            autoencoder_student_pairs = []
            parent_dir = os.path.join(
                folder_name,
                "trainings",
                "mvtec_loco",
                config.subdataset,
            )
            ckpts = os.listdir(parent_dir)
            ckpts = list(filter(lambda x: x.startswith("autoencoder_ckpt"), ckpts))
            ckpts = [
                item.split("autoencoder_ckpt_")[1].split(".pth")[0] for item in ckpts
            ]
            for iter in ckpts:
                autoencoder = Autoencoder(out_channels=out_channels, config=config)
                student = PDN_Small(
                    out_channels=2 * out_channels, padding=True, config=config
                )
                autoencoder_dict = torch.load(
                    os.path.join(parent_dir, f"autoencoder_ckpt_{iter}.pth"),
                    map_location=device,
                )
                student_dict = torch.load(
                    os.path.join(parent_dir, f"student_ckpt_{iter}.pth"),
                    map_location=device,
                )
                # TODO: change back to autoencoder.load_state_dict(autoencoder_dict)
                autoencoder.load_state_dict(autoencoder_dict)
                autoencoder = autoencoder.to(device)
                autoencoder.eval()
                # TODO: change back to student.load_state_dict(student_dict)
                student.load_state_dict(student_dict)
                student = student.to(device)
                student.eval()
                autoencoder_student_pairs.append((autoencoder, student))

            # the final ckpt (to single out for convenience)
            autoencoder = Autoencoder(out_channels=out_channels, config=config)
            student = PDN_Small(
                out_channels=2 * out_channels, padding=True, config=config
            )

        autoencoder_dict = torch.load(
            os.path.join(
                folder_name,
                "trainings",
                "mvtec_loco",
                config.subdataset,
                "autoencoder_final.pth",
            ),
            map_location=device,
        )
        student_dict = torch.load(
            os.path.join(
                folder_name,
                "trainings",
                "mvtec_loco",
                config.subdataset,
                "student_final.pth",
            ),
            map_location=device,
        )
        autoencoder.load_state_dict(autoencoder_dict.state_dict())
        student.load_state_dict(student_dict.state_dict())
        autoencoder = autoencoder.to(device)
        student = student.to(device)

    teacher.eval()
    student.eval()
    autoencoder.eval()

    if config.lid_on_history:
        with torch.no_grad():
            # train set
            final_maps_for_train_imgs = []
            for normal in tqdm(train_loader):
                (image_st, image_ae) = normal
                image_st = image_st.to(device)
                image_ae = image_ae.to(device)

                final_map_for_single_train_img = generate_iso_forest_features(
                    autoencoder_student_pairs,
                    teacher,
                    student,
                    autoencoder,
                    image_st,
                    image_ae,
                    teacher_mean,
                    teacher_std,
                )

                final_maps_for_train_imgs.append(final_map_for_single_train_img)

            final_maps_for_train_imgs = torch.stack(
                final_maps_for_train_imgs, dim=0
            )  # [#train_img, 2, 64, 64]

            # create a dictionary of IsolationForests by key of spatial location
            iso_forest_dict = dict()
            COUNT_TRAIN, C, H, W = final_maps_for_train_imgs.shape
            for i in range(H):
                for j in range(W):
                    input = final_maps_for_train_imgs[:, :, i, j].cpu().numpy()
                    iforest = IsolationForest().fit(input)
                    iso_forest_dict[(i, j)] = iforest

            # test set
            final_maps_for_test_imgs = []
            paths = []
            orig_heights = []
            orig_widths = []
            for image, target, path in tqdm(test_set):
                orig_width = image.width
                orig_height = image.height
                (image_st, image_ae) = train_transform(image, config=config)
                image_st = image_st.to(device)
                image_ae = image_ae.to(device)
                paths.append(path)
                orig_heights.append(orig_height)
                orig_widths.append(orig_width)

                final_map_for_single_test_img = generate_iso_forest_features(
                    autoencoder_student_pairs,
                    teacher,
                    student,
                    autoencoder,
                    image_st,
                    image_ae,
                    teacher_mean,
                    teacher_std,
                )

                final_maps_for_test_imgs.append(final_map_for_single_test_img)

            final_maps_for_test_imgs = torch.stack(
                final_maps_for_test_imgs, dim=0
            )  # [#test_img, 2, 64, 64]

            COUNT_TEST, C, H, W = final_maps_for_test_imgs.shape
            map_isoforest_pred = torch.zeros(size=(COUNT_TEST, 1, H, W))
            map_isoforest_pred = map_isoforest_pred.to(device)

            for i in range(H):
                for j in range(W):
                    test_data = final_maps_for_test_imgs[:, :, i, j].cpu().numpy()
                    iforest = iso_forest_dict[(i, j)]
                    anomaly_score = -iforest.decision_function(
                        test_data
                    )  # decision_function(X): the lower, the more abnormal
                    # anomaly_score.shape: [#test_img]
                    map_isoforest_pred[:, :, i, j] = anomaly_score

            for idx, (path, orig_height, orig_width) in enumerate(
                zip(paths, orig_heights, orig_widths)
            ):
                defect_class = os.path.basename(os.path.dirname(path))

                pred = map_isoforest_pred[idx].unsqueeze(0)
                pred = torch.nn.functional.interpolate(
                    pred, (orig_height, orig_width), mode="bilinear"
                )
                pred = pred[0, 0].cpu().numpy()

                if test_output_dir is not None:
                    img_nm = os.path.split(path)[1].split(".")[0]
                    if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                        os.makedirs(os.path.join(test_output_dir, defect_class))
                    file = os.path.join(test_output_dir, defect_class, img_nm + ".tiff")
                    tifffile.imwrite(file, map_combined)

    # if config.lid_on_history:
    #     # TODO: to implement in the next step
    #     exit()

    if config.debug_mode:

        feature_maps_folder = os.path.join(
            output_dir, "feature_maps", config.subdataset
        )
        os.makedirs(feature_maps_folder)
        for image, target, path in tqdm(test_set):
            path_portions = path.split("/")
            anomaly_type = path_portions[-2]
            idx = path_portions[-1].split(".png")[0]

            images = train_transform(image, config=config)
            (image, _) = images
            image = image[None]
            image = image.to(device)

            autoencoder_output = autoencoder(image)
            autoencoder_output = autoencoder_output.squeeze(0)

            teacher_output = teacher(image)
            teacher_output = teacher_output.squeeze(0)

            student_output = student(image)
            student_output = student_output.squeeze(0)
            student_t_output = student_output[:out_channels, :, :]
            student_ae_output = student_output[out_channels:, :, :]

            gray_scale_autoencoder = torch.sum(autoencoder_output, 0)
            gray_scale_autoencoder = (
                gray_scale_autoencoder / autoencoder_output.shape[0]
            )
            gray_scale_autoencoder = gray_scale_autoencoder.data.cpu().numpy()

            gray_scale_teacher = torch.sum(teacher_output, 0)
            gray_scale_teacher = gray_scale_teacher / teacher_output.shape[0]
            gray_scale_teacher = gray_scale_teacher.data.cpu().numpy()

            gray_scale_student_t = torch.sum(student_t_output, 0)
            gray_scale_student_t = gray_scale_student_t / student_t_output.shape[0]
            gray_scale_student_t = gray_scale_student_t.data.cpu().numpy()

            gray_scale_student_ae = torch.sum(student_ae_output, 0)
            gray_scale_student_ae = gray_scale_student_ae / student_ae_output.shape[0]
            gray_scale_student_ae = gray_scale_student_ae.data.cpu().numpy()

            all_gray_scales = [
                gray_scale_autoencoder,
                gray_scale_student_ae,
                gray_scale_student_t,
                gray_scale_teacher,
            ]

            fig = plt.figure(figsize=(50, 200))

            for i in range(len(all_gray_scales)):
                title = ""
                if i == 0:
                    title = "AE"
                elif i == 1:
                    title = "S_AE"
                elif i == 2:
                    title = "S_T"
                elif i == 3:
                    title = "T"

                a = fig.add_subplot(1, 4, i + 1)
                imgplot = plt.imshow(all_gray_scales[i])
                a.set_title(title, fontsize=30)
                a.axis("off")

            plt.savefig(
                os.path.join(feature_maps_folder, f"{anomaly_type}_{idx}.jpg"),
                bbox_inches="tight",
            )
            plt.close()

        exit()

    if config.lid_score_eval:
        trained_features_st = []
        trained_features_sae = []
        for item in train_loader:
            (train_image, _) = item
            train_image = train_image.to(device)
            autoencoder_output = autoencoder(train_image)
            student_output = student(train_image)
            teacher_output = teacher(train_image)
            teacher_output = (teacher_output - teacher_mean) / teacher_std

            if config.lid_eval_onwhat == "diff_mean":
                diff_st = (teacher_output - student_output[:, :out_channels]) ** 2
                diff_sae = (autoencoder_output - student_output[:, out_channels:]) ** 2
                trained_features_st.append(diff_st)
                trained_features_sae.append(diff_sae)
            elif config.lid_eval_onwhat == "separate_mean":
                trained_features_st.append(student_output)
                trained_features_sae.append(autoencoder_output)
            else:
                raise Exception("Unimplemented lid_eval_onwhat")

        trained_features_st = torch.cat(
            trained_features_st, dim=0
        )  # [#train, 384, 64, 64]
        trained_features_sae = torch.cat(
            trained_features_sae, dim=0
        )  # [#train, 384, 64, 64]

        (
            q_st_start,
            q_st_end,
            q_ae_start,
            q_ae_end,
            q_lid_start,
            q_lid_end,
        ) = map_normalization(
            out_channels=out_channels,
            validation_loader=validation_loader,
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std,
            config=config,
            trained_features_st=trained_features_st,
            trained_features_sae=trained_features_sae,
            desc="Final map normalization",
        )

        auc = test(
            out_channels=out_channels,
            test_set=test_set,
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std,
            q_st_start=q_st_start,
            q_st_end=q_st_end,
            q_ae_start=q_ae_start,
            q_ae_end=q_ae_end,
            q_lid_start=q_lid_start,
            q_lid_end=q_lid_end,
            config=config,
            output_dir=output_dir,
            test_output_dir=test_output_dir,
            trained_features_st=trained_features_st,
            trained_features_sae=trained_features_sae,
            desc="Final inference",
        )
    else:
        q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
            out_channels=out_channels,
            validation_loader=validation_loader,
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std,
            config=config,
            desc="Final map normalization",
        )
        auc = test(
            out_channels=out_channels,
            test_set=test_set,
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std,
            q_st_start=q_st_start,
            q_st_end=q_st_end,
            q_ae_start=q_ae_start,
            q_ae_end=q_ae_end,
            config=config,
            output_dir=output_dir,
            test_output_dir=test_output_dir,
            desc="Final inference",
        )
    print("Final image auc: {:.4f}".format(auc))


@torch.no_grad()
def generate_iso_forest_features(
    autoencoder_student_pairs,
    teacher,
    student,
    autoencoder,
    image_st,
    image_ae,
    teacher_mean,
    teacher_std,
):
    # obtain all loss maps by different ckpts
    loss_maps_for_single_train_img = []
    for ckpt_autoencoder, ckpt_student in autoencoder_student_pairs:
        loss_map = generate_loss_map(
            teacher,
            ckpt_student,
            ckpt_autoencoder,
            image_st,
            image_ae,
            teacher_mean,
            teacher_std,
        )
        loss_maps_for_single_train_img.append(loss_map)

    loss_maps_for_single_train_img = torch.cat(
        loss_maps_for_single_train_img, dim=0
    )  # shape: [#history, 384, 64, 64]

    # calculate LID scores at each point on the maps
    HIS_COUNT, C, H, W = loss_maps_for_single_train_img.shape
    lid_center = torch.zeros(
        size=(1, C),
        device=loss_maps_for_single_train_img.device,
        dtype=loss_maps_for_single_train_img.dtype,
    )
    lid_center = lid_center.to(device)

    lid_map_for_single_train_img = torch.zeros(size=(H, W))
    lid_map_for_single_train_img = lid_map_for_single_train_img.to(device)

    for i in range(H):
        for j in range(W):
            lid_map_for_single_train_img[i, j] = lid_mle_modified(
                data=lid_center,
                reference=loss_maps_for_single_train_img[:, :, i, j],
            )

    # calculate final loss map
    final_loss_map_for_single_train_img = generate_loss_map(
        teacher,
        student,
        autoencoder,
        image_st,
        image_ae,
        teacher_mean,
        teacher_std,
    )  # [1, 384, 64, 64]
    final_loss_map_for_single_train_img = torch.mean(
        final_loss_map_for_single_train_img, dim=1
    ).squeeze(
        0
    )  # [64, 64]

    final_map_for_single_train_img = torch.stack(
        [lid_map_for_single_train_img, final_loss_map_for_single_train_img],
        dim=0,
    )  # [2, 64, 64], the first value is lid_score, second is loss

    return final_map_for_single_train_img


@torch.no_grad()
def generate_loss_map(
    teacher, student, autoencoder, image_st, image_ae, teacher_mean, teacher_std
):
    teacher_output_st = teacher(image_st)
    teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
    student_output_st = student(image_st)[
        :, :out_channels
    ]  # the first half of student outputs
    distance_st = (teacher_output_st - student_output_st) ** 2

    ae_output = autoencoder(image_ae)

    teacher_output_ae = teacher(image_ae)
    teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
    student_output_ae = student(image_ae)[
        :, out_channels:
    ]  # the second half of student outputs
    distance_ae = (teacher_output_ae - ae_output) ** 2
    distance_stae = (ae_output - student_output_ae) ** 2

    return distance_st + distance_ae + distance_stae  # shape: [1, 384, 64, 64]


@torch.no_grad()
def generate_map_lid(diffs_st, diffs_sae, trained_features_st, trained_features_sae):
    B, _, H, W = diffs_st.shape
    map_lid = torch.zeros(size=(B, 1, H, W))

    for i in range(H):
        for j in range(W):
            map_lid[:, :, i, j] = lid_mle(
                data=diffs_st[:, :, i, j],
                reference=trained_features_st[:, :, i, j],
            ).unsqueeze(1) + lid_mle(
                data=diffs_sae[:, :, i, j],
                reference=trained_features_sae[:, :, i, j],
            ).unsqueeze(
                1
            )
    map_lid = torch.nn.functional.interpolate(
        map_lid, (image_size, image_size), mode="bilinear"
    )
    map_lid = map_lid.to(device)
    return map_lid


@torch.no_grad()
def test(
    out_channels,
    test_set,
    teacher,
    student,
    autoencoder,
    teacher_mean,
    teacher_std,
    q_st_start,
    q_st_end,
    q_ae_start,
    q_ae_end,
    config,
    output_dir,
    test_output_dir=None,
    q_lid_start=None,
    q_lid_end=None,
    trained_features_st=None,
    trained_features_sae=None,
    desc="Running inference",
):
    y_true = []
    y_score = []

    if config.lid_score_eval:
        maps_st = []
        maps_ae = []
        diffs_st = []
        diffs_sae = []
        paths = []
        orig_heights = []
        orig_widths = []

    for image, target, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height

        images = train_transform(image, config=config)
        (image, _) = images
        image = image[None]
        image = image.to(device)

        if config.lid_score_eval:
            map_st, map_ae, diff_st, diff_sae = predict(
                config=config,
                out_channels=out_channels,
                image=image,
                teacher=teacher,
                student=student,
                autoencoder=autoencoder,
                teacher_mean=teacher_mean,
                teacher_std=teacher_std,
                q_st_start=q_st_start,
                q_st_end=q_st_end,
                q_ae_start=q_ae_start,
                q_ae_end=q_ae_end,
            )
            maps_st.append(map_st)
            maps_ae.append(map_ae)
            diffs_st.append(diff_st)
            diffs_sae.append(diff_sae)
            paths.append(path)
            orig_heights.append(orig_height)
            orig_widths.append(orig_width)
        else:
            map_combined, map_st, map_ae = predict(
                config=config,
                out_channels=out_channels,
                image=image,
                teacher=teacher,
                student=student,
                autoencoder=autoencoder,
                teacher_mean=teacher_mean,
                teacher_std=teacher_std,
                q_st_start=q_st_start,
                q_st_end=q_st_end,
                q_ae_start=q_ae_start,
                q_ae_end=q_ae_end,
            )

            defect_class = os.path.basename(os.path.dirname(path))
            y_true_image = 0 if defect_class == "good" else 1
            y_score_image = np.max(map_combined[0, 0].cpu().numpy())
            y_true.append(y_true_image)
            y_score.append(y_score_image)

            # save into riff format
            map_combined = torch.nn.functional.interpolate(
                map_combined, (orig_height, orig_width), mode="bilinear"
            )
            map_combined = map_combined[0, 0].cpu().numpy()

            if test_output_dir is not None:
                img_nm = os.path.split(path)[1].split(".")[0]
                if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                    os.makedirs(os.path.join(test_output_dir, defect_class))
                file = os.path.join(test_output_dir, defect_class, img_nm + ".tiff")
                tifffile.imwrite(file, map_combined)

    if config.lid_score_eval:
        diffs_st = torch.cat(diffs_st, dim=0)  # [#test, 384, 64, 64]
        diffs_sae = torch.cat(diffs_sae, dim=0)

        maps_lid = generate_map_lid(
            diffs_st, diffs_sae, trained_features_st, trained_features_sae
        )

        assert (
            q_lid_start is not None
        ), "in test(), the quantile of lid_score should be available"

        maps_lid = 0.1 * (maps_lid - q_lid_start) / (q_lid_end - q_lid_start)

        for map_st, map_ae, map_lid, path, orig_height, orig_width in zip(
            maps_st, maps_ae, maps_lid, paths, orig_heights, orig_widths
        ):
            map_combined = 1 / 3 * map_st + 1 / 3 * map_ae + 1 / 3 * map_lid

            defect_class = os.path.basename(os.path.dirname(path))
            y_true_image = 0 if defect_class == "good" else 1
            y_score_image = np.max(map_combined[0, 0].cpu().numpy())
            y_true.append(y_true_image)
            y_score.append(y_score_image)

            # save into riff format
            map_combined = torch.nn.functional.interpolate(
                map_combined, (orig_height, orig_width), mode="bilinear"
            )
            map_combined = map_combined[0, 0].cpu().numpy()

            if test_output_dir is not None:
                img_nm = os.path.split(path)[1].split(".")[0]
                if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                    os.makedirs(os.path.join(test_output_dir, defect_class))
                file = os.path.join(test_output_dir, defect_class, img_nm + ".tiff")
                tifffile.imwrite(file, map_combined)

    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100


# called in both test and map_normalization
@torch.no_grad()
def predict(
    config,
    out_channels,
    image,
    teacher,
    student,
    autoencoder,
    teacher_mean,
    teacher_std,
    q_st_start=None,
    q_st_end=None,
    q_ae_start=None,
    q_ae_end=None,
):
    teacher_output = teacher(image)

    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)

    if config.use_masked_conv:
        autoencoder_output, _ = autoencoder(image)
    else:
        autoencoder_output = autoencoder(image)

    map_st = torch.mean(
        (teacher_output - student_output[:, :out_channels]) ** 2, dim=1, keepdim=True
    )  # shape: (bs, 1, h, w)
    map_ae = torch.mean(
        (autoencoder_output - student_output[:, out_channels:]) ** 2,
        dim=1,
        keepdim=True,
    )  # shape: (bs, 1, h, w)

    # upsample map_st and map_ae to 256*256
    map_st = torch.nn.functional.interpolate(
        map_st, (image_size, image_size), mode="bilinear"
    )
    map_ae = torch.nn.functional.interpolate(
        map_ae, (image_size, image_size), mode="bilinear"
    )

    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)

    if config.lid_score_eval:
        if config.lid_eval_onwhat == "diff_mean":
            diff_st = (teacher_output - student_output[:, :out_channels]) ** 2
            diff_sae = (autoencoder_output - student_output[:, out_channels:]) ** 2
            return map_st, map_ae, diff_st, diff_sae
        elif config.lid_eval_onwhat == "separate_mean":
            return map_st, map_ae, student_output, autoencoder_output
        else:
            raise Exception("Unimplemented lid_eval_onwhat")
    else:
        map_combined = 0.5 * map_st + 0.5 * map_ae
        return map_combined, map_st, map_ae


@torch.no_grad()
def map_normalization(
    out_channels,
    validation_loader,
    teacher,
    student,
    autoencoder,
    teacher_mean,
    teacher_std,
    config,
    trained_features_st=None,
    trained_features_sae=None,
    desc="Map normalization",
):
    maps_st = []
    maps_ae = []
    if config.lid_score_eval:
        diffs_st = []
        diffs_sae = []

    # ignore augmented ae image
    for images in tqdm(validation_loader, desc=desc):
        (image, _) = images

        image = image.to(device)

        if config.lid_score_eval:
            map_st, map_ae, diff_st, diff_sae = predict(
                config=config,
                out_channels=out_channels,
                image=image,
                teacher=teacher,
                student=student,
                autoencoder=autoencoder,
                teacher_mean=teacher_mean,
                teacher_std=teacher_std,
            )
        else:
            map_combined, map_st, map_ae = predict(
                config=config,
                out_channels=out_channels,
                image=image,
                teacher=teacher,
                student=student,
                autoencoder=autoencoder,
                teacher_mean=teacher_mean,
                teacher_std=teacher_std,
            )
        maps_st.append(map_st)
        maps_ae.append(map_ae)

        if config.lid_score_eval:
            diffs_st.append(diff_st)  # each [1, 384, 64, 64]
            diffs_sae.append(diff_sae)

    maps_st = torch.cat(maps_st)
    q_st_start = torch.quantile(
        maps_st, q=0.9
    )  # means 90% of values lie below q_st_start
    q_st_end = torch.quantile(maps_st, q=0.995)

    maps_ae = torch.cat(maps_ae)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)

    if config.lid_score_eval:
        diffs_st = torch.cat(diffs_st, dim=0)  # [#validation, 384, 64, 64]
        diffs_sae = torch.cat(diffs_sae, dim=0)

        maps_lid = generate_map_lid(
            diffs_st, diffs_sae, trained_features_st, trained_features_sae
        )

        q_lid_start = torch.quantile(maps_lid, q=0.9)
        q_lid_end = torch.quantile(maps_lid, q=0.995)
        return q_st_start, q_st_end, q_ae_start, q_ae_end, q_lid_start, q_lid_end
    else:
        return q_st_start, q_st_end, q_ae_start, q_ae_end


@torch.no_grad()
def teacher_normalization(teacher, train_loader, config):
    mean_outputs = []
    for train_images in tqdm(train_loader, desc="Computing mean of features"):
        (train_image, _) = train_images

        train_image = train_image.to(device)

        teacher_output = teacher(train_image)

        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_images in tqdm(train_loader, desc="Computing std of features"):
        (train_image, _) = train_images

        train_image = train_image.to(device)

        teacher_output = teacher(train_image)

        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std


if __name__ == "__main__":
    config = get_argparse()
    for seed in config.seeds:
        main(config, seed)
