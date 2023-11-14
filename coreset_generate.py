import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import torchvision
import random
from common import (
    ImageFolderWithPath,
    ImageFolderWithoutTarget,
)
from datetime import datetime
from torchvision.models import Wide_ResNet101_2_Weights
import copy
import torch.nn.functional as F

from sampler import GreedyCoresetSampler

# CONSTANTS
on_gpu = torch.cuda.is_available()
device = "cuda" if on_gpu else "cpu"
seed = 42
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
        "--mvtec_loco_path",
        default="./datasets/loco",
        help="Downloaded Mvtec LOCO dataset",
    )
    parser.add_argument("--output_dir", default=f"coreset")
    parser.add_argument(
        "--subdataset",
        default="breakfast_box",
        help="One of 15 sub-datasets of Mvtec AD or 5" + "sub-datasets of Mvtec LOCO",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    return parser.parse_args()


def train_transform(image):
    extractor_transform = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return extractor_transform(image)


def main(config):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    if config.subdataset == "breakfast_box":
        output_dir = config.output_dir + "_[bb]"
    elif config.subdataset == "juice_bottle":
        output_dir = config.output_dir + "_[jb]"
    elif config.subdataset == "pushpins":
        output_dir = config.output_dir + "_[pp]"
    elif config.subdataset == "screw_bag":
        output_dir = config.output_dir + "_[sb]"
    elif config.subdataset == "splicing_connectors":
        output_dir = config.output_dir + "_[sc]"
    else:
        raise ValueError(f"unknown subdataset name {config.subdataset}")

    """
    feature extractor
    """
    backbone = torchvision.models.wide_resnet101_2(
        weights=Wide_ResNet101_2_Weights.IMAGENET1K_V1
    )
    extractor = FeatureExtractor(
        # TODO: out_channels
        out_channels=16,
        backbone=backbone,
        layers_to_extract_from=["layer3", "layer4"],
        device=device,
        input_shape=(
            3,
            512,
            512,
        ),
    )

    extractor.to(device)
    extractor.eval()

    """
    dataset
    """
    train_set = ImageFolderWithoutTarget(
        os.path.join(config.mvtec_loco_path, config.subdataset, "train"),
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    # TODO: do I need to feature_normalization in channel direction?

    train_image_features = []
    for image in train_loader:
        image = image.to(device)
        output = extractor.embed(image)  # shape: (bs, out_channel*16*16)
        train_image_features.append(output)

    train_image_features = torch.cat(train_image_features, dim=0)
    print(train_image_features.shape)

    coresetsampler = GreedyCoresetSampler(
        percentage=0.02, device=device, dimension_to_project_features_to=1024
    )

    sample_indices = coresetsampler.run(train_image_features)
    print(sample_indices)


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
        )  # {"layerx": tensor, "layery": tensor}, layer2: [bs, 512, 64, 64], layer3: [bs, 1024, 32, 32], layer4: [bs, 2048, 16, 16]

        features = [
            features[layer] for layer in self.layers_to_extract_from
        ]  # in array form
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]  # [(tensor.shape:[4, 4096, 512, 3, 3], number_of_total_patches:[64, 64]), ([4, 1024, 1024, 3, 3], [32, 32])]

        patch_shapes = [x[1] for x in features]  # [[64, 64], [32, 32]]
        features = [
            x[0] for x in features
        ]  # unfolded_features[], two elements, each.shape: [bs, #patches=h*w, c, 3, 3]
        # TODO: notice that we use smaller size here
        ref_num_patches = patch_shapes[1]

        for i in range(0, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]
            if (
                patch_dims[0] == ref_num_patches[0]
                and patch_dims[1] == ref_num_patches[1]
            ):
                continue

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
        ]  # each with shape: (bs*h*w, c, 3, 3), where h and w are the ref size

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](
            features
        )  # shape: (bs*h*w, 2, output_dim==1024)
        features = self.forward_modules["preadapt_aggregator"](
            features
        )  # (bs*h*w, 384)
        features = torch.reshape(
            features, (-1, ref_num_patches[0], ref_num_patches[1], self.out_channels)
        )  # (bs, h, w, 384)
        # features = features.permute((0, 3, 1, 2))  # (bs, 384, h, w)
        features = features.flatten(start_dim=1)
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
    config = get_argparse()
    main(config)
