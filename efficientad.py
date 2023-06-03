#!/usr/bin/python
# -*- coding: utf-8 -*-
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

timestamp = (
    datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_"
    + str(random.randint(0, 100))
    + "_"
    + str(random.randint(0, 100))
)


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
    parser.add_argument("-o", "--output_dir", default=f"output_{timestamp}")
    parser.add_argument(
        "-m", "--model_size", default="small", choices=["small", "medium"]
    )
    parser.add_argument("-w", "--weights", default="models/teacher_small.pth")
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
    parser.add_argument("-t", "--train_steps", type=int, default=100)  # TODO: 70000
    parser.add_argument("--note", type=str, default="")
    return parser.parse_args()


# constants
seed = 42
on_gpu = torch.cuda.is_available()
device = "cuda" if on_gpu else "cpu"
out_channels = 384
image_size = 256

# data loading
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


def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))


def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()

    if config.subdataset == "breakfast_box":
        config.output_dir = config.output_dir + "_[bb]"
    elif config.subdataset == "juice_bottle":
        config.output_dir = config.output_dir + "_[jb]"
    elif config.subdataset == "pushpins":
        config.output_dir = config.output_dir + "_[pp]"
    elif config.subdataset == "screw_bag":
        config.output_dir = config.output_dir + "_[sb]"
    elif config.subdataset == "splicing_connectors":
        config.output_dir = config.output_dir + "_[sc]"
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
        config.output_dir, "trainings", config.dataset, config.subdataset
    )
    test_output_dir = os.path.join(
        config.output_dir, "anomaly_maps", config.dataset, config.subdataset, "test"
    )
    os.makedirs(train_output_dir)
    os.makedirs(test_output_dir)

    # load data
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.subdataset, "train"),
        transform=transforms.Lambda(train_transform),
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
            transform=transforms.Lambda(train_transform),
        )
    else:
        raise Exception("Unknown config.dataset")

    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
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
            penalty_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
        )
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    # create models
    if config.model_size == "small":
        teacher = PDN_Small(out_channels=out_channels, padding=True)
        student = PDN_Small(out_channels=2 * out_channels, padding=True)
        # teacher = get_pdn_small(out_channels, padding=True)
        # student = get_pdn_small(2 * out_channels, padding=True)
    elif config.model_size == "medium":
        teacher = get_pdn_medium(out_channels, padding=True)
        student = get_pdn_medium(2 * out_channels, padding=True)
    else:
        raise Exception()
    state_dict = torch.load(config.weights, map_location=device)

    pretrained_model = {}
    for k, v in state_dict.items():
        if k == "0.weight":
            pretrained_model["conv1.weight"] = v
        elif k == "0.bias":
            pretrained_model["conv1.bias"] = v
        elif k == "3.weight":
            pretrained_model["conv2.weight"] = v
        elif k == "3.bias":
            pretrained_model["conv2.bias"] = v
        elif k == "6.weight":
            pretrained_model["conv3.weight"] = v
        elif k == "6.bias":
            pretrained_model["conv3.bias"] = v
        elif k == "8.weight":
            pretrained_model["conv4.weight"] = v
        elif k == "8.bias":
            pretrained_model["conv4.bias"] = v
        else:
            raise ValueError("unknown state_dict key")

    teacher.load_state_dict(pretrained_model, strict=False)

    # autoencoder = get_autoencoder(out_channels)
    autoencoder = Autoencoder(out_channels=out_channels)

    # teacher frozen
    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    # TODO: uncomment
    (
        teacher_mean_2,
        teacher_std_2,
        teacher_mean_3,
        teacher_std_3,
        teacher_mean_4,
        teacher_std_4,
    ) = teacher_normalization(teacher, train_loader)

    #### TODO: hack code here, remove later
    # with open("teacher_mean.t", "rb") as f:
    #     teacher_mean = torch.load(f)
    # with open("teacher_std.t", "rb") as f:
    #     teacher_std = torch.load(f)

    optimizer = torch.optim.Adam(
        itertools.chain(student.parameters(), autoencoder.parameters()),
        lr=1e-4,
        weight_decay=1e-5,
    )

    # step_size is 66500
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1
    )

    tqdm_obj = tqdm(range(config.train_steps))
    for iteration, (image_st, image_ae), image_penalty in zip(
        tqdm_obj, train_loader_infinite, penalty_loader_infinite
    ):
        if on_gpu:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()
            if image_penalty is not None:
                image_penalty = image_penalty.cuda()
        with torch.no_grad():
            teacher_output_st_2, teacher_output_st_3, teacher_output_st_4 = teacher(
                image_st, trackmode=True
            )
            # print("==============teacher_output_st_3=============")
            # print(teacher_output_st_3)
            # print(torch.min(teacher_output_st_3), torch.max(teacher_output_st_3))
            teacher_output_st_2 = (teacher_output_st_2 - teacher_mean_2) / teacher_std_2
            teacher_output_st_3 = (teacher_output_st_3 - teacher_mean_3) / teacher_std_3
            teacher_output_st_4 = (teacher_output_st_4 - teacher_mean_4) / teacher_std_4
            # print("==============teacher_std_2=============")
            # print(teacher_std_2)
            # print(torch.min(teacher_std_2), torch.max(teacher_std_2))
            # print("==============teacher_std_3=============")
            # print(teacher_std_3)
            # print(torch.min(teacher_std_3), torch.max(teacher_std_3))
            # print("==============teacher_std_4=============")
            # print(teacher_std_4)
            # print(torch.min(teacher_std_4), torch.max(teacher_std_4))
            # print("==============teacher_output_st_3 (AGAIN)=============")
            # print(teacher_output_st_3)
            # print(torch.min(teacher_output_st_3), torch.max(teacher_output_st_3))

            # exit()
        student_output_st_2, student_output_st_3, student_output_st_4 = student(
            image_st
        )

        distance_st_2 = (teacher_output_st_2 - student_output_st_2) ** 2
        distance_st_3 = (teacher_output_st_3 - student_output_st_3) ** 2

        distance_st_4 = (
            teacher_output_st_4 - student_output_st_4[:, :out_channels]
        ) ** 2

        d_hard_2 = torch.quantile(distance_st_2, q=0.999)
        d_hard_3 = torch.quantile(distance_st_3, q=0.999)
        d_hard_4 = torch.quantile(distance_st_4, q=0.999)

        sorted_value, _ = torch.sort(torch.flatten(teacher_output_st_3))

        loss_hard_2 = torch.mean(distance_st_2[distance_st_2 >= d_hard_2])
        loss_hard_3 = torch.mean(distance_st_3[distance_st_3 >= d_hard_3])
        loss_hard_4 = torch.mean(distance_st_4[distance_st_4 >= d_hard_4])

        if image_penalty is not None:
            student_output_penalty = student(image_penalty)[2][:, :out_channels]
            loss_penalty = torch.mean(student_output_penalty**2)
            loss_st = (
                loss_hard_2 / 3.0 + loss_hard_3 / 3.0 + loss_hard_4 / 3.0 + loss_penalty
            )
        else:
            loss_st = loss_hard_2 / 3.0 + loss_hard_3 / 3.0 + loss_hard_4 / 3.0

        ae_output = autoencoder(image_ae)

        with torch.no_grad():
            teacher_output_ae_2, teacher_output_ae_3, teacher_output_ae_4 = teacher(
                image_ae
            )
            teacher_output_ae_2 = (teacher_output_ae_2 - teacher_mean_2) / teacher_std_2
            teacher_output_ae_3 = (teacher_output_ae_3 - teacher_mean_3) / teacher_std_3
            teacher_output_ae_4 = (teacher_output_ae_4 - teacher_mean_4) / teacher_std_4
        student_output_ae = student(image_ae)[2][
            :, out_channels:
        ]  # the second half of student outputs
        distance_ae = (teacher_output_ae_4 - ae_output) ** 2
        distance_stae = (ae_output - student_output_ae) ** 2

        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)

        loss_total = loss_st + loss_ae + loss_stae

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
            # tqdm_obj.set_description("Current loss: {:.4f}  ".format(loss_total.item()))

        # if iteration % 1000 == 0:
        #     torch.save(teacher, os.path.join(train_output_dir, "teacher_tmp.pth"))
        #     torch.save(student, os.path.join(train_output_dir, "student_tmp.pth"))
        #     torch.save(
        #         autoencoder, os.path.join(train_output_dir, "autoencoder_tmp.pth")
        #     )

        # if iteration % 10000 == 0 and iteration > 0:
        #     # run intermediate evaluation
        #     teacher.eval()
        #     student.eval()
        #     autoencoder.eval()

        #     q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        #         validation_loader=validation_loader,
        #         teacher=teacher,
        #         student=student,
        #         autoencoder=autoencoder,
        #         teacher_mean=teacher_mean,
        #         teacher_std=teacher_std,
        #         desc="Intermediate map normalization",
        #     )
        #     auc = test(
        #         test_set=test_set,
        #         teacher=teacher,
        #         student=student,
        #         autoencoder=autoencoder,
        #         teacher_mean=teacher_mean,
        #         teacher_std=teacher_std,
        #         q_st_start=q_st_start,
        #         q_st_end=q_st_end,
        #         q_ae_start=q_ae_start,
        #         q_ae_end=q_ae_end,
        #         test_output_dir=None,
        #         desc="Intermediate inference",
        #     )
        #     print("Intermediate image auc: {:.4f}".format(auc))

        #     # teacher frozen
        #     teacher.eval()
        #     student.train()
        #     autoencoder.train()

    teacher.eval()
    student.eval()
    autoencoder.eval()

    torch.save(teacher, os.path.join(train_output_dir, "teacher_final.pth"))
    torch.save(student, os.path.join(train_output_dir, "student_final.pth"))
    torch.save(autoencoder, os.path.join(train_output_dir, "autoencoder_final.pth"))

    (
        q_st_start_2,
        q_st_end_2,
        q_st_start_3,
        q_st_end_3,
        q_st_start_4,
        q_st_end_4,
        q_ae_start,
        q_ae_end,
    ) = map_normalization(
        validation_loader=validation_loader,
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,
        teacher_mean_2=teacher_mean_2,
        teacher_mean_3=teacher_mean_3,
        teacher_mean_4=teacher_mean_4,
        teacher_std_2=teacher_std_2,
        teacher_std_3=teacher_std_3,
        teacher_std_4=teacher_std_4,
        desc="Final map normalization",
    )
    auc = test(
        test_set=test_set,
        teacher=teacher,
        student=student,
        autoencoder=autoencoder,
        teacher_mean_2=teacher_mean_2,
        teacher_mean_3=teacher_mean_3,
        teacher_mean_4=teacher_mean_4,
        teacher_std_2=teacher_std_2,
        teacher_std_3=teacher_std_3,
        teacher_std_4=teacher_std_4,
        q_st_start_2=q_st_start_2,
        q_st_start_3=q_st_start_3,
        q_st_start_4=q_st_start_4,
        q_st_end_2=q_st_end_2,
        q_st_end_3=q_st_end_3,
        q_st_end_4=q_st_end_4,
        q_ae_start=q_ae_start,
        q_ae_end=q_ae_end,
        test_output_dir=test_output_dir,
        desc="Final inference",
    )
    print("Final image auc: {:.4f}".format(auc))


@torch.no_grad()
def test(
    test_set,
    teacher,
    student,
    autoencoder,
    teacher_mean_2,
    teacher_mean_3,
    teacher_mean_4,
    teacher_std_2,
    teacher_std_3,
    teacher_std_4,
    q_st_start_2,
    q_st_start_3,
    q_st_start_4,
    q_st_end_2,
    q_st_end_3,
    q_st_end_4,
    q_ae_start,
    q_ae_end,
    test_output_dir=None,
    desc="Running inference",
):
    y_true = []
    y_score = []
    for image, target, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        if on_gpu:
            image = image.cuda()

        map_combined, map_st_2, map_st_3, map_st_4, map_ae = predict(
            image=image,
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            teacher_mean_2=teacher_mean_2,
            teacher_mean_3=teacher_mean_3,
            teacher_mean_4=teacher_mean_4,
            teacher_std_2=teacher_std_2,
            teacher_std_3=teacher_std_3,
            teacher_std_4=teacher_std_4,
            q_st_start_2=q_st_start_2,
            q_st_start_3=q_st_start_3,
            q_st_start_4=q_st_start_4,
            q_st_end_2=q_st_end_2,
            q_st_end_3=q_st_end_3,
            q_st_end_4=q_st_end_4,
            q_ae_start=q_ae_start,
            q_ae_end=q_ae_end,
        )

        defect_class = os.path.basename(os.path.dirname(path))
        y_true_image = 0 if defect_class == "good" else 1
        y_score_image = np.max(map_combined[0, 0].cpu().numpy())
        y_true.append(y_true_image)
        y_score.append(y_score_image)

        # map_combined = torch.nn.functional.pad(
        #     map_combined, (4, 4, 4, 4)
        # )  # pad last dim by (4, 4) and 2nd to last by (4, 4), the value in padding area is 0

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


@torch.no_grad()
def predict(
    image,
    teacher,
    student,
    autoencoder,
    teacher_mean_2,
    teacher_mean_3,
    teacher_mean_4,
    teacher_std_2,
    teacher_std_3,
    teacher_std_4,
    q_st_start_2=None,
    q_st_start_3=None,
    q_st_start_4=None,
    q_st_end_2=None,
    q_st_end_3=None,
    q_st_end_4=None,
    q_ae_start=None,
    q_ae_end=None,
):
    teacher_output_2, teacher_output_3, teacher_output_4 = teacher(image)
    teacher_output_2 = (teacher_output_2 - teacher_mean_2) / teacher_std_2
    teacher_output_3 = (teacher_output_3 - teacher_mean_3) / teacher_std_3
    teacher_output_4 = (teacher_output_4 - teacher_mean_4) / teacher_std_4

    student_output_2, student_output_3, student_output_4 = student(image)

    autoencoder_output = autoencoder(image)

    map_st_2 = torch.mean(
        (teacher_output_2 - student_output_2) ** 2, dim=1, keepdim=True
    )
    map_st_3 = torch.mean(
        (teacher_output_3 - student_output_3) ** 2, dim=1, keepdim=True
    )
    map_st_4 = torch.mean(
        (teacher_output_4 - student_output_4[:, :out_channels]) ** 2,
        dim=1,
        keepdim=True,
    )  # shape: (bs, 1, h, w)

    map_ae = torch.mean(
        (autoencoder_output - student_output_4[:, out_channels:]) ** 2,
        dim=1,
        keepdim=True,
    )  # shape: (bs, 1, h, w)

    # upsample map_st and map_ae to 256*256
    map_st_2 = torch.nn.functional.interpolate(
        map_st_2, (image_size, image_size), mode="bilinear"
    )
    map_st_3 = torch.nn.functional.interpolate(
        map_st_3, (image_size, image_size), mode="bilinear"
    )
    map_st_4 = torch.nn.functional.interpolate(
        map_st_4, (image_size, image_size), mode="bilinear"
    )
    map_ae = torch.nn.functional.interpolate(
        map_ae, (image_size, image_size), mode="bilinear"
    )

    if q_st_start_2 is not None:
        map_st_2 = 0.1 * (map_st_2 - q_st_start_2) / (q_st_end_2 - q_st_start_2)
    if q_st_start_3 is not None:
        map_st_3 = 0.1 * (map_st_3 - q_st_start_3) / (q_st_end_3 - q_st_start_3)
    if q_st_start_4 is not None:
        map_st_4 = 0.1 * (map_st_4 - q_st_start_4) / (q_st_end_4 - q_st_start_4)

    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = (
        0.5 * (map_st_2 / 3.0 + map_st_3 / 3.0 + map_st_4 / 3.0) + 0.5 * map_ae
    )

    return map_combined, map_st_2, map_st_3, map_st_4, map_ae


@torch.no_grad()
def map_normalization(
    validation_loader,
    teacher,
    student,
    autoencoder,
    teacher_mean_2,
    teacher_mean_3,
    teacher_mean_4,
    teacher_std_2,
    teacher_std_3,
    teacher_std_4,
    desc="Map normalization",
):
    maps_st_2 = []
    maps_st_3 = []
    maps_st_4 = []
    maps_ae = []
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_combined, map_st_2, map_st_3, map_st_4, map_ae = predict(
            image=image,
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            teacher_mean_2=teacher_mean_2,
            teacher_mean_3=teacher_mean_3,
            teacher_mean_4=teacher_mean_4,
            teacher_std_2=teacher_std_2,
            teacher_std_3=teacher_std_3,
            teacher_std_4=teacher_std_4,
        )
        maps_st_2.append(map_st_2)
        maps_st_3.append(map_st_3)
        maps_st_4.append(map_st_4)
        maps_ae.append(map_ae)

    maps_st_2 = torch.cat(maps_st_2)
    maps_st_3 = torch.cat(maps_st_3)
    maps_st_4 = torch.cat(maps_st_4)
    maps_ae = torch.cat(maps_ae)

    q_st_start_2 = torch.quantile(maps_st_2, q=0.9)
    q_st_end_2 = torch.quantile(maps_st_2, q=0.995)
    q_st_start_3 = torch.quantile(maps_st_3, q=0.9)
    q_st_end_3 = torch.quantile(maps_st_3, q=0.995)
    q_st_start_4 = torch.quantile(maps_st_4, q=0.9)
    q_st_end_4 = torch.quantile(maps_st_4, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)

    return (
        q_st_start_2,
        q_st_end_2,
        q_st_start_3,
        q_st_end_3,
        q_st_start_4,
        q_st_end_4,
        q_ae_start,
        q_ae_end,
    )


@torch.no_grad()
def teacher_normalization(teacher, train_loader):
    mean_outputs_2 = []
    mean_outputs_3 = []
    mean_outputs_4 = []
    for train_image, _ in tqdm(train_loader, desc="Computing mean of features"):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output_2, teacher_output_3, teacher_output_4 = teacher(train_image)

        mean_output_2 = torch.mean(teacher_output_2, dim=[0, 2, 3])
        mean_output_3 = torch.mean(teacher_output_3, dim=[0, 2, 3])
        mean_output_4 = torch.mean(teacher_output_4, dim=[0, 2, 3])
        mean_outputs_2.append(mean_output_2)
        mean_outputs_3.append(mean_output_3)
        mean_outputs_4.append(mean_output_4)
    channel_mean_2 = torch.mean(torch.stack(mean_outputs_2), dim=0)
    channel_mean_3 = torch.mean(torch.stack(mean_outputs_3), dim=0)
    channel_mean_4 = torch.mean(torch.stack(mean_outputs_4), dim=0)
    channel_mean_2 = channel_mean_2[None, :, None, None]
    channel_mean_3 = channel_mean_3[None, :, None, None]
    channel_mean_4 = channel_mean_4[None, :, None, None]

    mean_distances_2 = []
    mean_distances_3 = []
    mean_distances_4 = []
    for train_image, _ in tqdm(train_loader, desc="Computing std of features"):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output_2, teacher_output_3, teacher_output_4 = teacher(train_image)
        distance_2 = (teacher_output_2 - channel_mean_2) ** 2
        distance_3 = (teacher_output_3 - channel_mean_3) ** 2
        distance_4 = (teacher_output_4 - channel_mean_4) ** 2
        mean_distance_2 = torch.mean(distance_2, dim=[0, 2, 3])
        mean_distance_3 = torch.mean(distance_3, dim=[0, 2, 3])
        mean_distance_4 = torch.mean(distance_4, dim=[0, 2, 3])
        mean_distances_2.append(mean_distance_2)
        mean_distances_3.append(mean_distance_3)
        mean_distances_4.append(mean_distance_4)
    channel_var_2 = torch.mean(torch.stack(mean_distances_2), dim=0)
    channel_var_3 = torch.mean(torch.stack(mean_distances_3), dim=0)
    channel_var_4 = torch.mean(torch.stack(mean_distances_4), dim=0)
    channel_var_2 = channel_var_2[None, :, None, None]
    channel_var_3 = channel_var_3[None, :, None, None]
    channel_var_4 = channel_var_4[None, :, None, None]
    channel_std_2 = torch.sqrt(channel_var_2)
    channel_std_3 = torch.sqrt(channel_var_3)
    channel_std_4 = torch.sqrt(channel_var_4)

    return (
        channel_mean_2,
        channel_std_2,
        channel_mean_3,
        channel_std_3,
        channel_mean_4,
        channel_std_4,
    )


if __name__ == "__main__":
    main()
