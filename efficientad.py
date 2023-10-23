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
    get_pdn_small,
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
    # TODO: import choice
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
    # TODO: important choice
    parser.add_argument(
        "--pretrained_network",
        choices=["wide_resnet101_2", "vit", "pvt2_b2li"],
        type=str,
        default="wide_resnet101_2",
    )
    parser.add_argument(
        "--avg_cdim",
        action="store_true",
        help="if set to True, then perform avg pooling on channel dim to 384",
    )
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
    parser.add_argument(
        "--patchify",
        action="store_true",
        help="if set to True, then augment features using PatchCore",
    )
    # TODO: add it to code and slurm file
    parser.add_argument(
        "--analysis_heatmap",
        action="store_true",
        help="if set to True, then generate branch-wise analysis heatmap",
    )

    # the following are necessary if teacher is directly a vit/pvt
    parser.add_argument(
        "--vit_teacher",
        action="store_true",
        help="if set to True, then use vit for teacher",
    )
    parser.add_argument("--image_size_vit_teacher", type=int, default=512)
    parser.add_argument(
        "--pvt2_teacher",
        action="store_true",
        help="if set to True, then use pvt for teacher",
    )
    parser.add_argument("--image_size_pvt2_teacher", type=int, default=512)

    parser.add_argument("-t", "--train_steps", type=int, default=70000)
    parser.add_argument("--note", type=str, default="")
    # TODO: add to slurm
    parser.add_argument("--seeds", type=int, default=[42], nargs="+")
    return parser.parse_args()


# constants
on_gpu = torch.cuda.is_available()
device = "cuda" if on_gpu else "cpu"
image_size = 256


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
    vit_teacher_transform = transforms.Compose(
        [
            transforms.Resize(
                (config.image_size_vit_teacher, config.image_size_vit_teacher)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    pvt2_teacher_transform = transforms.Compose(
        [
            transforms.Resize(
                (config.image_size_pvt2_teacher, config.image_size_pvt2_teacher)
            ),
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
    if config.vit_teacher:
        ae_image = transform_ae(image)
        return (
            default_transform(image),
            vit_teacher_transform(image),
            default_transform(ae_image),
            vit_teacher_transform(ae_image),
        )
    elif config.pvt2_teacher:
        ae_image = transform_ae(image)
        return (
            default_transform(image),
            pvt2_teacher_transform(image),
            default_transform(ae_image),
            pvt2_teacher_transform(ae_image),
        )
    else:
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

    if config.pretrained_network == "vit":
        out_channels = 768
    elif config.pretrained_network == "pvt2_b2li":
        # TODO: always pay attention to out_channels
        if config.avg_cdim:
            out_channels = 384
        else:
            if config.patchify:
                out_channels = 448
            else:
                if config.pvt2_stage3:
                    out_channels = 320
                elif config.pvt2_stage4:
                    out_channels = 512
                else:
                    out_channels = 448
    else:
        #  wide_resnet101_2
        out_channels = 384

    # create models
    if config.model_size == "small":
        if config.vit_teacher:
            from urllib.request import urlretrieve
            from vit_models.modeling import VisionTransformer, CONFIGS

            # if not on_gpu or dist.get_rank() == 0:
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
                img_size=config.image_size_vit_teacher,
                vis=True,
                vit_mid=False,
            )
            model.load_from(np.load("vit_model_checkpoints/ViT-B_16-224.npz"))
            teacher = torch.nn.Sequential(
                *[model.transformer.embeddings, model.transformer.encoder]
            )
        elif config.pvt2_teacher:
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

            teacher = pvt_v2_b2_li(pretrained=False, stage4=config.pvt2_stage4)
            teacher.load_state_dict(pretrained_weights, strict=False)
        else:
            teacher = PDN_Small(out_channels=out_channels, padding=True)

        student = PDN_Small(out_channels=2 * out_channels, padding=True)
        # teacher = get_pdn_small(out_channels, padding=True)
        # student = get_pdn_small(2 * out_channels, padding=True)
    elif config.model_size == "medium":
        teacher = get_pdn_medium(out_channels, padding=True)
        student = get_pdn_medium(2 * out_channels, padding=True)
    else:
        raise Exception()

    if config.pretrained_network == "wide_resnet101_2":
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
    elif config.pretrained_network in ["vit", "pvt2_b2li"]:
        if config.vit_teacher or config.pvt2_teacher:
            # ckpt already loaded
            pass
        else:
            state_dict = torch.load(config.weights, map_location=device)
            pretrained_teacher_model = {}
            for k, v in state_dict.items():
                if k == "module.conv1.weight":
                    pretrained_teacher_model["conv1.weight"] = v
                elif k == "module.conv1.bias":
                    pretrained_teacher_model["conv1.bias"] = v
                elif k == "module.conv2.weight":
                    pretrained_teacher_model["conv2.weight"] = v
                elif k == "module.conv2.bias":
                    pretrained_teacher_model["conv2.bias"] = v
                elif k == "module.conv3.weight":
                    pretrained_teacher_model["conv3.weight"] = v
                elif k == "module.conv3.bias":
                    pretrained_teacher_model["conv3.bias"] = v
                elif k == "module.conv4.weight":
                    pretrained_teacher_model["conv4.weight"] = v
                elif k == "module.conv4.bias":
                    pretrained_teacher_model["conv4.bias"] = v
                else:
                    raise ValueError(f"unknown state_dict key {k}")
            teacher.load_state_dict(pretrained_teacher_model, strict=False)

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

    # FIXME: should we update the teacher_mean, teacher_std on the fly? Different for each batch
    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader, config)

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
    for (
        iteration,
        train_images,
        image_penalty,
    ) in zip(tqdm_obj, train_loader_infinite, penalty_loader_infinite):
        if config.vit_teacher or config.pvt2_teacher:
            (
                image_st,
                image_st_teacher,
                image_ae,
                image_ae_teacher,
            ) = train_images
            if on_gpu:
                image_st = image_st.cuda()
                image_st_teacher = image_st_teacher.cuda()
                image_ae_teacher = image_ae_teacher.cuda()
                image_ae = image_ae.cuda()

        else:
            (image_st, image_ae) = train_images
            if on_gpu:
                image_st = image_st.cuda()
                image_ae = image_ae.cuda()

        if image_penalty is not None:
            image_penalty = image_penalty.cuda()

        with torch.no_grad():
            if config.vit_teacher:
                teacher_output_st = teacher(image_st_teacher)[0]
                teacher_output_st = process_vit_features(teacher_output_st)
            elif config.pvt2_teacher:
                teacher_output_st = teacher(image_st_teacher)
                teacher_output_st = process_pvt_features(teacher_output_st, config)
            else:
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

        ae_output = autoencoder(image_ae)

        with torch.no_grad():
            if config.vit_teacher:
                teacher_output_ae = teacher(image_ae_teacher)[0]
                teacher_output_ae = process_vit_features(teacher_output_ae)
            elif config.pvt2_teacher:
                teacher_output_ae = teacher(image_ae_teacher)
                teacher_output_ae = process_pvt_features(teacher_output_ae, config)
            else:
                teacher_output_ae = teacher(image_ae)

            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
        student_output_ae = student(image_ae)[
            :, out_channels:
        ]  # the second half of student outputs
        distance_ae = (teacher_output_ae - ae_output) ** 2
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

    teacher.eval()
    student.eval()
    autoencoder.eval()

    torch.save(teacher, os.path.join(train_output_dir, "teacher_final.pth"))
    torch.save(student, os.path.join(train_output_dir, "student_final.pth"))
    torch.save(autoencoder, os.path.join(train_output_dir, "autoencoder_final.pth"))

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
    desc="Running inference",
):
    y_true = []
    y_score = []

    if config.analysis_heatmap:
        map_comb_min = None
        map_comb_max = None

    for image, target, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height

        images = train_transform(image, config=config)

        if config.vit_teacher or config.pvt2_teacher:
            (
                image,
                image_teacher,
                _,
                _,
            ) = images
        else:
            (image, _) = images
            image_teacher = None

        image = image[None]
        if image_teacher is not None:
            image_teacher = image_teacher[None]

        if on_gpu:
            image = image.cuda()
            if image_teacher is not None:
                image_teacher = image_teacher.cuda()

        map_combined, map_st, map_ae = predict(
            config=config,
            out_channels=out_channels,
            image=image,
            image_teacher=image_teacher,
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

        if config.analysis_heatmap:
            if map_comb_min is None:
                map_comb_min = torch.min(map_combined)
                map_comb_max = torch.max(map_combined)
            else:
                map_comb_min = min(map_comb_min, torch.min(map_combined))
                map_comb_max = max(map_comb_max, torch.max(map_combined))

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

    if config.analysis_heatmap:
        map_comb_min = map_comb_min.cpu().numpy()
        map_comb_max = map_comb_max.cpu().numpy()

        heatmap_folder = os.path.join(output_dir, "analysis_heatmap/")
        os.makedirs(heatmap_folder, exist_ok=True)

        # output heatmaps for separate branches
        for image, target, path in tqdm(test_set, desc=desc):
            images = train_transform(image, config=config)

            if config.vit_teacher or config.pvt2_teacher:
                (
                    image,
                    image_teacher,
                    _,
                    _,
                ) = images
            else:
                (image, _) = images
                image_teacher = None

            image = image[None]
            if image_teacher is not None:
                image_teacher = image_teacher[None]

            if on_gpu:
                image = image.cuda()
                if image_teacher is not None:
                    image_teacher = image_teacher.cuda()

            _, map_structural, map_logical = predict(
                config=config,
                out_channels=out_channels,
                image=image,
                image_teacher=image_teacher,
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
            heatmap_structural = cv2.applyColorMap(
                pred_mask_structural, cv2.COLORMAP_JET
            )
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

    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100


@torch.no_grad()
def predict(
    config,
    out_channels,
    image,
    image_teacher,
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
    # no need as I commented out the code related to predict()
    if config.vit_teacher:
        teacher_output = teacher(image_teacher)[0]
        teacher_output = process_vit_features(teacher_output)
    elif config.pvt2_teacher:
        teacher_output = teacher(image_teacher)
        teacher_output = process_pvt_features(teacher_output, config)
    else:
        teacher_output = teacher(image)

    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
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
    desc="Map normalization",
):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for images in tqdm(validation_loader, desc=desc):
        if config.vit_teacher or config.pvt2_teacher:
            (
                image,
                image_teacher,
                _,
                _,
            ) = images
        else:
            (image, _) = images
            image_teacher = None

        if on_gpu:
            image = image.cuda()
            if image_teacher is not None:
                image_teacher = image_teacher.cuda()

        map_combined, map_st, map_ae = predict(
            config=config,
            out_channels=out_channels,
            image=image,
            image_teacher=image_teacher,
            teacher=teacher,
            student=student,
            autoencoder=autoencoder,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std,
        )
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(
        maps_st, q=0.9
    )  # means 90% of values lie below q_st_start
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end


@torch.no_grad()
def teacher_normalization(teacher, train_loader, config):
    mean_outputs = []
    for train_images in tqdm(train_loader, desc="Computing mean of features"):
        if config.vit_teacher or config.pvt2_teacher:
            (
                _,
                train_image,
                _,
                _,
            ) = train_images
        else:
            (train_image, _) = train_images

        if on_gpu:
            train_image = train_image.cuda()

        if config.vit_teacher:
            teacher_output = teacher(train_image)[0]
            teacher_output = process_vit_features(teacher_output)
        elif config.pvt2_teacher:
            teacher_output = teacher(train_image)
            teacher_output = process_pvt_features(teacher_output, config)
        else:
            teacher_output = teacher(train_image)

        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_images in tqdm(train_loader, desc="Computing std of features"):
        if config.vit_teacher or config.pvt2_teacher:
            (
                _,
                train_image,
                _,
                _,
            ) = train_images
        else:
            (train_image, _) = train_images

        if on_gpu:
            train_image = train_image.cuda()

        if config.vit_teacher:
            teacher_output = teacher(train_image)[0]
            teacher_output = process_vit_features(teacher_output)
        elif config.pvt2_teacher:
            teacher_output = teacher(train_image)
            teacher_output = process_pvt_features(teacher_output, config)
        else:
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
