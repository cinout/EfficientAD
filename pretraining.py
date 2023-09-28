#!/usr/bin/python
# -*- coding: utf-8 -*-
import builtins
import math
import torchvision
import argparse
import os
import random
import copy
import torch.nn.functional as F
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from common import (
    PDN_Small,
    get_pdn_small,
    get_pdn_medium,
    ImageFolderWithoutTarget,
    InfiniteDataloader,
)
from functools import partial
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from pvt_v2 import pvt_v2_b2_li


def get_argparse():
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )
    parser.add_argument(
        "-o", "--output_folder", default="pretrained_pdn/pretrained_pdn"
    )
    parser.add_argument(
        "--network",
        choices=["wide_resnet101_2", "vit", "pvt2_b2li"],
        type=str,
        default="wide_resnet101_2",
    )
    parser.add_argument(
        "--pdn_size",
        choices=["small", "medium"],
        type=str,
        default="small",
    )
    parser.add_argument(
        "--imagenet_train_path",
        type=str,
        default="./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--extractor_input_size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--avg_cdim",
        action="store_true",
        help="if set to True, then perform avg pooling on channel dim to 384 (for PVT)",
    )
    parser.add_argument(
        "--pvt2_stage3",
        action="store_true",
        help="if set to True, then use 3rd stage output",
    )
    # TODO: use or not?
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
    parser.add_argument(
        "--vit_mid",
        action="store_true",
        help="if set to True, then use intermediate layer of ViT for knowledge distillation",
    )
    parser.add_argument("--note", default="", type=str)

    # ---- SLURM and DDP args ---- #
    parser.add_argument(
        "--world_size",
        default=-1,
        type=int,
        help="number of processes for distributed training",
    )
    parser.add_argument(
        "--global_rank",
        default=-1,
        type=int,
        help="global rank for distributed training",
    )
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
        help="local rank for torch.distributed.launch training",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        type=int,
        help="local rank for slurm training (c.f. local_rank)",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist_backend", default="nccl", type=str, help="distributed backend"
    )
    return parser.parse_args()


# variables
seed = 42
on_gpu = torch.cuda.is_available()
device = "cuda" if on_gpu else "cpu"


def process_pvt_features(features, args):
    target_size = 64
    out_channels = 448

    # FIXME: this assumes only first three stages' results are returned
    if args.pvt2_stage3:
        features = features[2:]
    else:
        features = features[1:]  # remove 1st element

    if args.patchify:
        # patchify the feature maps
        patch_maker = PatchMaker(3, stride=1)
        features = [patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]  # [64, 64] and [32, 32]
        features = [
            x[0] for x in features
        ]  # [bs, 4096, 128, 3, 3] and [bs, 1024, 320, 3, 3]

        # upsample the feature maps
        for i in range(len(features)):
            patch_dims = patch_shapes[i]
            if patch_dims[0] == target_size:
                # no need for upsampling
                continue

            _features = features[i]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )  # [4, 32, 32, 320, 3, 3]

            _features = _features.permute(0, -3, -2, -1, 1, 2)  # [4, 320, 3, 3, 32, 32]

            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])  # [36864, 32, 32]

            # upsample feature map to the largest size
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )  # [36864, 1, 64, 64]

            _features = _features.squeeze(1)  # [36864, 64, 64]

            _features = _features.reshape(
                *perm_base_shape[:-2], target_size, target_size
            )  # [4, 320, 3, 3, 64, 64]

            _features = _features.permute(0, -2, -1, 1, 2, 3)  # [4, 64, 64, 320, 3, 3]

            _features = _features.reshape(
                len(_features), -1, *_features.shape[-3:]
            )  # [bs, 4096, 320, 3, 3]

            features[i] = _features

        features = [
            x.reshape(-1, *x.shape[-3:]) for x in features
        ]  # each with shape: (bs*h*w, c, 3, 3), where h and w are the largest size

        # adapt channel dimension to 1024
        preprocessing = Preprocessing(len(features), 1024)
        features = preprocessing(features)  # shape: (bs*h*w, 2, output_dim==1024)

        # aggregate features of two maps
        preadapt_aggregator = Aggregator(target_dim=out_channels)
        preadapt_aggregator.to(device)
        features = preadapt_aggregator(features)  # (bs*h*w, out_channels)
        features = torch.reshape(
            features, (-1, 64, 64, out_channels)
        )  # (bs, h, w, out_channels)
        # features = torch.permute(features, (0, 3, 1, 2))  # (bs, out_channels, h, w)
        features = features.permute((0, 3, 1, 2))
        return features
    else:
        for i in range(0, len(features)):
            # upsampling feature map
            _features = features[i]
            _features = F.interpolate(
                _features,
                size=(target_size, target_size),
                mode="bilinear",
            )
            features[i] = _features

        features = torch.cat(features, dim=1)  # shape: [bs, c_sum=128+320=448, 64, 64]

        if args.avg_cdim:
            bs, c_size, h, w = features.shape
            features = features.reshape(bs, c_size, -1)
            features = features.transpose(1, 2)  # shape: [bs, h*w, c]
            features = F.adaptive_avg_pool1d(features, 384)
            features = features.transpose(1, 2)
            features = features.reshape(bs, 384, h, w)
        return features


def process_vit_features(features, args):
    target_size = 64
    out_channels = 768

    features = features[:, 1:, :]
    B, N, C = features.shape
    H = int(math.sqrt(N))
    W = int(math.sqrt(N))
    features = features.transpose(1, 2).view(
        B, C, H, W
    )  # shape: (bs, 768, 32, 32) if input_size is 512

    if args.patchify:
        # patchify the feature maps
        patch_maker = PatchMaker(3, stride=1)
        features = [patch_maker.patchify(features, return_spatial_info=True)]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]

        # upsample the feature maps
        for i in range(0, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )  # [bs, 32, 32, 768, 3, 3]

            _features = _features.permute(
                0, -3, -2, -1, 1, 2
            )  # [bs, 768, 3, 3, 32, 32]

            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])  # [36864, 32, 32]

            # upsample feature map to the largest size
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )  # [36864, 1, 64, 64]

            _features = _features.squeeze(1)  # [36864, 64, 64]

            _features = _features.reshape(
                *perm_base_shape[:-2], target_size, target_size
            )  # [bs, 768, 3, 3, 64, 64]

            _features = _features.permute(0, -2, -1, 1, 2, 3)  # [bs, 64, 64, 768, 3, 3]

            _features = _features.reshape(
                len(_features), -1, *_features.shape[-3:]
            )  # [bs, 4096, 768, 3, 3]

            features[i] = _features

        features = [
            x.reshape(-1, *x.shape[-3:]) for x in features
        ]  # each with shape: (bs*h*w, c, 3, 3), where h and w are the largest size

        # adapt channel dimension to 1024
        preprocessing = Preprocessing(len(features), 1024)
        features = preprocessing(features)  # shape: (bs*h*w, 2, output_dim==1024)

        # aggregate features of two maps
        preadapt_aggregator = Aggregator(target_dim=out_channels)
        preadapt_aggregator.to(device)
        features = preadapt_aggregator(features)  # (bs*h*w, out_channels)
        features = torch.reshape(
            features, (-1, 64, 64, out_channels)
        )  # (bs, h, w, out_channels)
        # features = torch.permute(features, (0, 3, 1, 2))  # (bs, out_channels, h, w)
        features = features.permute((0, 3, 1, 2))
        return features
    else:
        if H != target_size:
            features = torch.nn.functional.interpolate(
                features, (target_size, target_size), mode="bilinear"
            )
        return features


def train_transform(image, size):
    image = transforms.RandomGrayscale(0.1)(image)
    extractor_transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )  # for pretrained model
    pdn_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )  # for pdn teacher
    return extractor_transform(image), pdn_transform(image)


def main(args):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not on_gpu or dist.get_rank() == 0:
        os.makedirs(args.output_folder)

    if args.network == "wide_resnet101_2":
        from torchvision.models import Wide_ResNet101_2_Weights

        backbone = torchvision.models.wide_resnet101_2(
            weights=Wide_ResNet101_2_Weights.IMAGENET1K_V1
        )
        out_channels = 384

        extractor = FeatureExtractor(
            out_channels=out_channels,
            backbone=backbone,
            layers_to_extract_from=["layer2", "layer3"],
            device=device,
            input_shape=(
                3,
                512,
                512,
            ),
        )
        input_transform_func = partial(train_transform, size=512)
        suffix = "wres101_2"
    elif args.network == "pvt2_b2li":
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

        extractor = pvt_v2_b2_li(pretrained=False)
        extractor.load_state_dict(pretrained_weights, strict=False)
        extractor.eval()

        # TODO: alway pay attention to out_channels
        if args.avg_cdim:
            out_channels = 384
        else:
            if args.patchify:
                out_channels = 448
            else:
                if args.pvt2_stage3:
                    out_channels = 320
                else:
                    out_channels = 448

        input_transform_func = partial(train_transform, size=args.extractor_input_size)
        suffix = "pvt2_b2li"

    elif args.network == "vit":
        from urllib.request import urlretrieve
        from vit_models.modeling import VisionTransformer, CONFIGS

        if not on_gpu or dist.get_rank() == 0:
            os.makedirs("vit_model_checkpoints", exist_ok=True)

        if not os.path.isfile("vit_model_checkpoints/ViT-B_16-224.npz"):
            urlretrieve(
                "https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz",
                "vit_model_checkpoints/ViT-B_16-224.npz",
            )

        config = CONFIGS["ViT-B_16"]
        model = VisionTransformer(
            config,
            num_classes=1000,
            zero_head=False,
            img_size=args.extractor_input_size,
            vis=True,
            vit_mid=args.vit_mid,
        )
        model.load_from(np.load("vit_model_checkpoints/ViT-B_16-224.npz"))
        extractor = torch.nn.Sequential(
            *[model.transformer.embeddings, model.transformer.encoder]
        )
        extractor.eval()
        input_transform_func = partial(train_transform, size=args.extractor_input_size)
        out_channels = 768
        suffix = "vit_b16"

    if args.pdn_size == "small":
        # pdn = get_pdn_small(out_channels, padding=True)
        pdn = PDN_Small(out_channels=out_channels, padding=True)
    elif args.pdn_size == "medium":
        pdn = get_pdn_medium(out_channels, padding=True)
    else:
        raise Exception()
    pdn.train()

    if args.distributed:
        if args.gpu is None:
            extractor.cuda()
            extractor = DDP(extractor)
            pdn.cuda()
            pdn = DDP(pdn)
        else:
            extractor.cuda(args.gpu)
            extractor = DDP(extractor, device_ids=[args.gpu])
            pdn.cuda(args.gpu)
            pdn = DDP(pdn, device_ids=[args.gpu])
    else:
        extractor.to(device)
        pdn.to(device)

    train_set = ImageFolderWithoutTarget(
        args.imagenet_train_path, transform=input_transform_func
    )
    if args.distributed:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            # num_workers=7,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=7,
            pin_memory=True,
        )
    train_loader = InfiniteDataloader(train_loader)

    channel_mean, channel_std = feature_normalization(
        args, extractor=extractor, train_loader=train_loader
    )

    optimizer = torch.optim.Adam(pdn.parameters(), lr=1e-4, weight_decay=1e-5)

    tqdm_obj = tqdm(range(60000))
    for iteration, (image_fe, image_pdn) in zip(tqdm_obj, train_loader):
        # if isinstance(train_loader.sampler, DistributedSampler):
        # 		# calling the set_epoch() method at the beginning of each epoch before creating the DataLoader iterator is necessary to make shuffling work properly across multiple epochs. Otherwise, the same ordering will be always used.
        #     train_loader.sampler.set_epoch(iteration)

        if on_gpu:
            image_fe = image_fe.cuda()  # for extractor
            image_pdn = image_pdn.cuda()  # for pdn teacher

        if args.network == "vit":
            target = extractor(image_fe)[0]
            target = process_vit_features(target, args)
        elif args.network == "wide_resnet101_2":
            target = extractor.embed(image_fe)
        elif args.network == "pvt2_b2li":
            target = extractor(image_fe)
            target = process_pvt_features(target, args)

        target = (target - channel_mean) / channel_std
        prediction = pdn(image_pdn)
        loss = torch.mean((target - prediction) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tqdm_obj.set_description(f"{(loss.item())}")

        if iteration % 10000 == 0:
            if not on_gpu or dist.get_rank() == 0:
                # torch.save(
                #     pdn,
                #     os.path.join(
                #         args.output_folder, f"tmp_teacher_{args.pdn_size}_{suffix}.pth"
                #     ),
                # )
                torch.save(
                    pdn.state_dict(),
                    os.path.join(
                        args.output_folder,
                        f"tmp_teacher_{args.pdn_size}_state_{suffix}.pth",
                    ),
                )
    if not on_gpu or dist.get_rank() == 0:
        # torch.save(
        #     pdn,
        #     os.path.join(args.output_folder, f"teacher_{args.pdn_size}_{suffix}.pth"),
        # )
        torch.save(
            pdn.state_dict(),
            os.path.join(
                args.output_folder, f"teacher_{args.pdn_size}_state_{suffix}.pth"
            ),
        )


@torch.no_grad()
def feature_normalization(args, extractor, train_loader, steps=10000):
    mean_outputs = []
    normalization_count = 0
    with tqdm(desc="Computing mean of features", total=steps) as pbar:
        for image_fe, _ in train_loader:
            if on_gpu:
                image_fe = image_fe.cuda()  # shape: (16, 3, 512, 512)

            if args.network == "vit":
                output = extractor(image_fe)[0]
                output = process_vit_features(output, args)

            elif args.network == "wide_resnet101_2":
                output = extractor.embed(image_fe)
            elif args.network == "pvt2_b2li":
                output = extractor(image_fe)
                output = process_pvt_features(output, args)

            mean_output = torch.mean(output, dim=[0, 2, 3])
            mean_outputs.append(mean_output)
            normalization_count += len(image_fe)
            if normalization_count >= steps:
                pbar.update(steps - pbar.n)
                break
            else:
                pbar.update(len(image_fe))
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    normalization_count = 0
    with tqdm(desc="Computing variance of features", total=steps) as pbar:
        for image_fe, _ in train_loader:
            if on_gpu:
                image_fe = image_fe.cuda()

            if args.network == "vit":
                output = extractor(image_fe)[0]
                output = process_vit_features(output, args)
            elif args.network == "wide_resnet101_2":
                output = extractor.embed(image_fe)
            elif args.network == "pvt2_b2li":
                output = extractor(image_fe)
                output = process_pvt_features(output, args)

            distance = (output - channel_mean) ** 2
            mean_distance = torch.mean(distance, dim=[0, 2, 3])
            mean_distances.append(mean_distance)
            normalization_count += len(image_fe)
            if normalization_count >= steps:
                pbar.update(steps - pbar.n)
                break
            else:
                pbar.update(len(image_fe))
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std


class FeatureExtractor(torch.nn.Module):
    def __init__(
        self, out_channels, backbone, layers_to_extract_from, device, input_shape
    ):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.device = device
        self.input_shape = input_shape  # (3, 512, 512)
        self.out_channels = out_channels  # 384

        self.forward_modules = torch.nn.ModuleDict({})

        # aggregate features at different layers of pretrained model
        feature_aggregator = NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(
            input_shape
        )  # channel dims: [512, 1024]
        self.forward_modules["feature_aggregator"] = feature_aggregator

        # use padding to include neighborhood, and upsample smaller features to the largest
        self.patch_maker = PatchMaker(3, stride=1)  # modify features

        preprocessing = Preprocessing(len(feature_dimensions), 1024)
        self.forward_modules["preprocessing"] = preprocessing

        preadapt_aggregator = Aggregator(target_dim=out_channels)
        _ = preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.forward_modules.eval()

    @torch.no_grad()
    def embed(self, images):
        """Returns feature embeddings for images."""

        _ = self.forward_modules["feature_aggregator"].eval()
        features = self.forward_modules["feature_aggregator"](
            images
        )  # {"layer2": tensor, "layer3": tensor}

        features = [
            features[layer] for layer in self.layers_to_extract_from
        ]  # features[0].shape: [bs, 512, 64, 64], features[1].shape: [bs, 1024, 32, 32]
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]  # [(tensor.shape:[4, 4096, 512, 3, 3], number_of_total_patches:[64, 64]), ([4, 1024, 1024, 3, 3], [32, 32])]

        patch_shapes = [x[1] for x in features]  # [[64, 64], [32, 32]]
        features = [
            x[0] for x in features
        ]  # unfolded_features[], two elements, each.shape: [bs, #patches=h*w, c, 3, 3]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            # i is only 1, for upsampling feature map
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )  # [4, 32, 32, 1024, 3, 3]

            _features = _features.permute(
                0, -3, -2, -1, 1, 2
            )  # [4, 1024, 3, 3, 32, 32]

            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])  # [36864, 32, 32]

            # upsample feature map to the largest size
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )  # [36864, 1, 64, 64]

            _features = _features.squeeze(1)  # [36864, 64, 64]

            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )  # [4, 1024, 3, 3, 64, 64]

            _features = _features.permute(0, -2, -1, 1, 2, 3)  # [4, 64, 64, 1024, 3, 3]

            _features = _features.reshape(
                len(_features), -1, *_features.shape[-3:]
            )  # [4, 4096, 1024, 3, 3]

            features[i] = _features

        features = [
            x.reshape(-1, *x.shape[-3:]) for x in features
        ]  # each with shape: (bs*h*w, c, 3, 3), where h and w are the largest size

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](
            features
        )  # shape: (bs*h*w, 2, output_dim==1024)
        features = self.forward_modules["preadapt_aggregator"](
            features
        )  # (bs*h*w, 384)
        features = torch.reshape(
            features, (-1, 64, 64, self.out_channels)
        )  # (bs, h, w, 384)
        # features = torch.permute(features, (0, 3, 1, 2))  # (bs, 384, h, w)
        features = features.permute((0, 3, 1, 2))
        return features


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        """
        features.shape: torch.Size([4, 512, 64, 64])
        unfolded_features.shape: torch.Size([4, 4608, 4096])
        ----
        features.shape: torch.Size([4, 1024, 32, 32]) (1024*3*3)
        unfolded_features.shape: torch.Size([4, 9216, 1024]) 
        """
        number_of_total_patches = []  # [64, 64] or [32, 32]
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))

        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        """
        unfolded_features.shape: torch.Size([4, 512, 3, 3, 4096])
        ----
        unfolded_features.shape: torch.Size([4, 1024, 3, 3, 1024])
        """
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        """
        unfolded_features.shape: torch.Size([4, 4096, 512, 3, 3])
        ----
        unfolded_features.shape: torch.Size([4, 1024, 1024, 3, 3])
        """

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features


class Preprocessing(torch.nn.Module):
    def __init__(self, maps_count, output_dim):
        super(Preprocessing, self).__init__()
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for _ in range(maps_count):
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        # _features[x].shape: (bs*h*w, output_dim)
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        # features.shape: (bs*h*w, c, 3, 3)
        features = features.reshape(len(features), 1, -1)
        # features.shape: (bs*h*w, 1, c*3*3)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            # The backbone will throw an Exception once it reached the last
            # layer to compute features from. Computation will stop there.
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    pass


if __name__ == "__main__":
    args = get_argparse()

    devices_names = (
        f"{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}"
    )

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()
    torch.backends.cudnn.benchmark = True

    if args.distributed:
        if args.local_rank != -1:  # for torch.distributed.launch
            args.global_rank = args.local_rank
            args.gpu = args.local_rank
        elif "SLURM_PROCID" in os.environ:  # for slurm scheduler
            args.global_rank = int(os.environ["SLURM_PROCID"])
            args.gpu = args.global_rank % ngpus_per_node
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.global_rank,
        )
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)

        # suppress printing if not on master gpu
        if args.global_rank != 0:

            def print_pass(*args):
                pass

            builtins.print = print_pass

    print(devices_names)
    if not on_gpu or dist.get_rank() == 0:
        timestamp = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + str(random.randint(0, 100))
            + "_"
            + str(random.randint(0, 100))
        )
        args.output_folder = args.output_folder + "_" + timestamp

    options_str = "\n".join(
        "%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())
    )
    print(options_str)
    print("----------------------")
    main(args)
