#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import math
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import argparse
import itertools
import os
import random
from tqdm import tqdm
from common import (
    AESwap,
    PDN_Small,
    get_pdn_medium,
    ImageFolderWithoutTarget,
    ImageFolderWithPath,
    InfiniteDataloader,
)
from sklearn.metrics import roc_auc_score
from datetime import datetime
from functools import partial
from torchvision.models import Wide_ResNet101_2_Weights
import cv2


"""
Constants
"""
on_gpu = torch.cuda.is_available()
device = "cuda" if on_gpu else "cpu"
image_size = 256
image_size_ae = 512
heatmap_alpha = 0.5
timestamp = (
    datetime.now().strftime("%Y%m%d_%H%M%S")
    + "_"
    + str(random.randint(0, 100))
    + "_"
    + str(random.randint(0, 100))
)
ref_images = {
    "breakfast_box": ["160", "064", "101", "202", "058", "095", "329"],
    "juice_bottle": ["081", "136", "247", "160", "169", "217"],
    "pushpins": ["279", "035", "244", "002", "318", "141", "093"],
    "screw_bag": ["217", "147", "359", "311", "279", "246", "093"],
    "splicing_connectors": ["302", "023", "242", "214", "081", "331", "131"],
}


def normalizeData(data, minval, maxval):
    return (data - minval) / (maxval - minval)


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
    return parser.parse_args()


def train_transform(image, config):
    default_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_ae = transforms.Compose(
        [
            transforms.Resize((image_size_ae, image_size_ae)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return (
        default_transform(image),
        transform_ae(image),
    )


cos_sim = torch.nn.CosineSimilarity(dim=1)


def generate_ae_output(
    image_ae, ref_features, feature_extractor, autoencoder, path=None
):
    """
    image_ae: tensor
    ref_features: dict
    feature_extractor: model (in eval mode)
    autoencoder: model
    """
    image_ae_features = feature_extractor(image_ae)

    closest_ref_features = None
    max_similarity = -1000
    # closest_ref_id = None

    for ref_id, ref_feature in ref_features.items():
        similarity = (
            cos_sim(
                torch.mean(ref_feature, dim=(2, 3)),
                torch.mean(image_ae_features, dim=(2, 3)),
            )
            .mean()
            .item()
        )
        if similarity > max_similarity:
            max_similarity = similarity
            closest_ref_features = ref_feature
            # closest_ref_id = ref_id

    assert closest_ref_features is not None, "closest ref feature cannot be None"
    ## for debugging
    # if path is not None:
    #     print(f"input image path is: {path}")
    # similarity_dict = dict()
    # for ref_id, ref_feature in ref_features.items():
    #     # similarity_location_wise = cos_sim(ref_feature, image_ae_features).mean().item()
    #     similarity_average = (
    #         cos_sim(
    #             torch.mean(ref_feature, dim=(2, 3)),
    #             torch.mean(image_ae_features, dim=(2, 3)),
    #         )
    #         .mean()
    #         .item()
    #     )
    #     # print(ref_id, similarity_location_wise, similarity_average)
    #     similarity_dict[ref_id] = similarity_average
    #     # similarity_dict[ref_id] = similarity_location_wise + 2 * similarity_average
    # sorted_similarity_dict = sorted(
    #     similarity_dict.items(), key=lambda x: x[1], reverse=True
    # )

    B, C, H, W = image_ae_features.shape
    image_ae_features = image_ae_features.squeeze(0).reshape(
        C, -1
    )  # shape: [C, H*W] = [2048, 256]
    closest_ref_features = closest_ref_features.squeeze(0).reshape(
        C, -1
    )  # shape: [C, H*W]
    similarity_matrix = torch.mm(
        image_ae_features.T, closest_ref_features
    )  # shape: [H*W, H*W]
    similarity_matrix = similarity_matrix.flatten()
    descending_similarity = torch.argsort(similarity_matrix, descending=True).numpy()

    # turn absolute index into (row, col) index
    similarity_desc_index = [
        (math.floor(index / (H * W)), index % (H * W))
        for index in descending_similarity
    ]

    swap_guide = []  # each element will be (index_ae, index_ref)
    for t in similarity_desc_index:
        if len(swap_guide) == 0 or all(
            [t[0] != k[0] and t[1] != k[1] for k in swap_guide]
        ):
            swap_guide.append(t)
    
    # perform replacing
    for index_ae, index_ref in swap_guide:
        image_ae_features[:, index_ae] = closest_ref_features[:, index_ref]

    # ae features after replacing
    image_ae_features = image_ae_features.reshape(C, H, W).unsqueeze(0)

    return autoencoder(image_ae_features)


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

    """
    # load data
    """
    # FIXME: changed from ImageFolderWithoutTarget to ImageFolderWithPath
    full_train_set = ImageFolderWithPath(
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

    out_channels = 384

    """
    extract reference images' features
    """

    ref_ids = ref_images[config.subdataset]
    for ref_id in ref_ids:
        assert len(ref_id) == 3, "ref_id should contain 3 digits"
    backbone = torchvision.models.wide_resnet101_2(
        weights=Wide_ResNet101_2_Weights.IMAGENET1K_V1
    )
    feature_extractor = torch.nn.Sequential(
        backbone.conv1,
        backbone.bn1,
        backbone.relu,
        backbone.maxpool,
        backbone.layer1,
        backbone.layer2,
        backbone.layer3,
        backbone.layer4,
    )
    feature_extractor.eval()
    ref_loader = DataLoader(
        ImageFolderWithPath(
            os.path.join(dataset_path, config.subdataset, "train"),
            transform=transforms.Lambda(partial(train_transform, config=config)),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    ref_features = dict()
    for images, _, path in ref_loader:
        image_id = (path[0].split("/")[-1]).split(".")[0]
        if image_id in ref_ids:
            _, image = images
            feature = feature_extractor(image)  # shape: [1, 2048, 16, 16]
            ref_features[image_id] = feature
    """
    # create models
    """
    if config.model_size == "small":
        teacher = PDN_Small(out_channels=out_channels, padding=True)
        student = PDN_Small(out_channels=2 * out_channels, padding=True)

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
    else:
        raise Exception("pretrained_network is unrecognizable")

    autoencoder = AESwap(out_channels)

    # teacher frozen
    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader, config)
    # with open("teacher_mean_jb_respretrainedPDN.t", "rb") as f:
    #     teacher_mean = torch.load(f)
    # with open("teacher_std_jb_respretrainedPDN.t", "rb") as f:
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
    for (
        iteration,
        (train_images, _, path),
        image_penalty,
    ) in zip(tqdm_obj, train_loader_infinite, penalty_loader_infinite):
        (image_st, image_ae) = train_images
        image_st = image_st.to(device)
        image_ae = image_ae.to(device)

        if image_penalty is not None:
            image_penalty = image_penalty.to(device)

        with torch.no_grad():
            teacher_output_st = teacher(image_st)
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
        student_output = student(image_st)
        student_output_st = student_output[
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

        ae_output = generate_ae_output(
            image_ae, ref_features, feature_extractor, autoencoder, path
        )

        # with torch.no_grad():
        #     teacher_output_ae = teacher(image_ae)
        #     teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
        student_output_ae = student_output[:, out_channels:]
        distance_ae = (teacher_output_st - ae_output) ** 2
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
        ref_features=ref_features,
        feature_extractor=feature_extractor,
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
        ref_features=ref_features,
        feature_extractor=feature_extractor,
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
    ref_features,
    feature_extractor,
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

    for raw_image, _, path in tqdm(test_set, desc=desc):
        orig_width = raw_image.width
        orig_height = raw_image.height

        images = train_transform(raw_image, config=config)

        (image, image_ae) = images
        image = image.unsqueeze(0)
        image = image.to(device)
        image_ae = image_ae.unsqueeze(0)
        image_ae = image_ae.to(device)

        map_combined, map_st, map_ae = predict(
            config=config,
            ref_features=ref_features,
            feature_extractor=feature_extractor,
            out_channels=out_channels,
            image=image,
            image_ae=image_ae,
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
        for raw_image, _, path in tqdm(test_set, desc=desc):
            images = train_transform(raw_image, config=config)
            (image, image_ae) = images
            image = image.unsqueeze(0)
            image = image.to(device)
            image_ae = image_ae.unsqueeze(0)
            image_ae = image_ae.to(device)

            _, map_structural, map_logical = predict(
                config=config,
                ref_features=ref_features,
                feature_extractor=feature_extractor,
                out_channels=out_channels,
                image=image,
                image_ae=image_ae,
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
    ref_features,
    feature_extractor,
    out_channels,
    image,
    image_ae,
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
    autoencoder_output = generate_ae_output(
        image_ae, ref_features, feature_extractor, autoencoder
    )
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
    ref_features,
    feature_extractor,
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
        (image, image_ae) = images
        image = image.to(device)
        image_ae = image_ae.to(device)

        map_combined, map_st, map_ae = predict(
            config=config,
            ref_features=ref_features,
            feature_extractor=feature_extractor,
            out_channels=out_channels,
            image=image,
            image_ae=image_ae,
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
    for train_images, _, _ in tqdm(train_loader, desc="Computing mean of features"):
        (train_image, _) = train_images
        train_image = train_image.to(device)

        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_images, _, _ in tqdm(train_loader, desc="Computing std of features"):
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
