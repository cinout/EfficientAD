#!/usr/bin/python
# -*- coding: utf-8 -*-
from torch import nn
from torchvision.datasets import ImageFolder
import torch
import torch.nn.functional as F
from functools import partial
import json, os
import numpy as np


class IndividualGTLossForSphere(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        defects_config_path = os.path.join(
            "datasets/loco/", config.subdataset, "defects_config.json"
        )
        defects = json.load(open(defects_config_path))
        self.config = {e["pixel_value"]: e for e in defects}
        self.eps = 1e-6

    def forward(self, dist, gts):
        # predicted.shape: (2, orig.h, orig.w)
        loss_per_gt = []
        for gt in gts:
            # gt.shape: [1, 1, orig.h, orig.w]
            # find unique config for the gt
            unique_values = sorted(torch.unique(gt).detach().cpu().numpy())
            pixel_type = unique_values[-1]
            pixel_detail = self.config[pixel_type]
            saturation_threshold = pixel_detail["saturation_threshold"]
            relative_saturation = pixel_detail["relative_saturation"]

            # calculate saturation_area (max pixels needed)
            bool_array = gt.cpu().numpy().astype(np.bool_)
            defect_area = np.sum(bool_array)
            saturation_area = (
                int(saturation_threshold * defect_area)
                if relative_saturation
                else np.minimum(saturation_threshold, defect_area)
            )

            # calculate loss
            gt = gt.bool().to(torch.float32)
            mask = gt == 1

            loss = torch.masked_select((dist + self.eps) ** -0.2, mask)

            saturated_loss_values, _ = torch.topk(
                loss, k=saturation_area, largest=False
            )

            loss_per_gt.append(saturated_loss_values)

        loss_per_gt = torch.cat(loss_per_gt, dim=0)
        return loss_per_gt


class IndividualGTLoss(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        defects_config_path = os.path.join(
            "datasets/loco/", config.subdataset, "defects_config.json"
        )

        defects = json.load(open(defects_config_path))
        self.loss_on_resize = config.loss_on_resize
        self.config = {e["pixel_value"]: e for e in defects}

        self.gamma = 2
        self.smooth = 1e-5
        self.size_average = True

    def forward(self, predicted, gts):
        # predicted.shape: (2, orig.h, orig.w)
        loss_per_gt = []
        for gt in gts:
            # gt.shape: [1, 1, orig.h, orig.w]
            # find unique config for the gt
            if self.loss_on_resize:
                pixel_type = gt["pixel_type"].item()
                orig_width = gt["orig_width"].item()
                orig_height = gt["orig_height"].item()
                gt = gt["gt"]
            else:
                unique_values = sorted(torch.unique(gt).detach().cpu().numpy())
                pixel_type = unique_values[-1]

            pixel_detail = self.config[pixel_type]
            saturation_threshold = pixel_detail["saturation_threshold"]
            relative_saturation = pixel_detail["relative_saturation"]
            bool_array = gt.cpu().numpy().astype(np.bool_)
            defect_area = np.sum(bool_array)

            if self.loss_on_resize:
                _, _, h, w = gt.shape
                saturation_area = (
                    int(saturation_threshold * defect_area)
                    if relative_saturation
                    else np.minimum(
                        int(saturation_threshold * h * w / orig_width / orig_height),
                        defect_area,
                    )
                )

            else:
                # calculate saturation_area (max pixels needed)
                saturation_area = (
                    int(saturation_threshold * defect_area)
                    if relative_saturation
                    else np.minimum(saturation_threshold, defect_area)
                )
                gt = gt.bool().to(torch.float32)

            gt = gt.squeeze(0)
            gt = gt.view(gt.shape[0], -1)
            gt = gt.transpose(0, 1)

            mask = (gt == 1).squeeze(1)

            # num_class = predicted.shape[0]
            predicted = predicted.view(predicted.shape[0], -1)  # shape: (2, H*W)
            predicted = predicted[1]  # shape: (H*W, )
            predicted = torch.masked_select(predicted, mask)
            predicted = (
                (1.0 - self.smooth) * predicted
                + self.smooth * (1 - predicted)
                + self.smooth
            )
            logpt = predicted.log()
            loss = -1 * torch.pow((1 - predicted), self.gamma) * logpt

            # idx = gt.cpu().long()
            # one_hot_key = torch.FloatTensor(gt.size(0), num_class).zero_()
            # print(one_hot_key.shape)
            # one_hot_key = one_hot_key.scatter_(1, idx, 1)
            # if one_hot_key.device != predicted.device:
            #     one_hot_key = one_hot_key.to(predicted.device)
            # if self.smooth:
            #     one_hot_key = torch.clamp(
            #         one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
            #     )
            # pt = (one_hot_key * predicted).sum(1) + self.smooth

            # # use mask to only calculate loss of positive pixels
            # mask = (gt == 1).squeeze(1)
            # pt = torch.masked_select(pt, mask)
            # logpt = pt.log()
            # loss = -1 * torch.pow((1 - pt), self.gamma) * logpt

            # RANDOMLY choose k
            length = loss.size()[0]
            loss = loss[torch.randperm(length)]
            saturated_loss_values = loss[:saturation_area]

            # # choose k smallest losses
            # saturated_loss_values, _ = torch.topk(
            #     loss, k=saturation_area, largest=False
            # )

            loss_per_gt.append(saturated_loss_values)

        loss_per_gt = torch.cat(loss_per_gt, dim=0)
        if self.size_average:
            loss_per_gt = loss_per_gt.mean()
        return loss_per_gt


class FocalLoss(torch.nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py

    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    """

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.gamma = 2
        self.smooth = 1e-5
        self.size_average = True

    def forward(self, logit, target):
        # logit.shape: [bs, 2, h, w], sum of dim 1 is 1, because softmaxed
        # target.shape: [bs, 1, h, w], values are either 0 or 1, in float data type
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))  # flatten to [N*h*w, C]
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)  # [N*h*w, 1]

        idx = target.cpu().long()  # [N*h*w, 1]

        one_hot_key = torch.FloatTensor(
            target.size(0), num_class
        ).zero_()  # [N*h*w, C], all 0s
        one_hot_key = one_hot_key.scatter_(
            1, idx, 1
        )  # [N*h*w, C], with the right C marked with 1, and the other marked with 0
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
            )  # with values changed from {0, 1} to {smooth, 1-smooth}

        pt = (one_hot_key * logit).sum(1) + self.smooth

        # USE mask to only calculate loss of negative pixels
        mask = (target == 0).squeeze(1)
        pt = torch.masked_select(pt, mask)

        logpt = pt.log()

        loss = -1 * torch.pow((1 - pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class VectorQuantizerEMA(nn.Module):
    # Source for the VectorQuantizerEMA module: https://github.com/zalandoresearch/pytorch-vq-vae
    def __init__(self, num_embeddings, embedding_dim, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim  # C
        self._num_embeddings = num_embeddings  # N

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim
        )  # https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self._embedding.weight.data.normal_()  # the learnable weights of the module of shape (num_embeddings, embedding_dim) = (N, C)

        self.register_buffer(
            "_ema_cluster_size", torch.zeros(num_embeddings)
        )  # shape: (N, )
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()  # shape: (N, C)

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(
            -1, self._embedding_dim
        )  # shape: (BHW, C), C=self._embedding_dim

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )  # shape: (BHW, N), N=self._num_embeddings

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(
            1
        )  # shape: (BHW, 1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )  # shape: (BHW, N)
        encodings.scatter_(
            1, encoding_indices, 1
        )  # shape: (BHW, N), with value @encoding_indices changed to 1

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(
            input_shape
        )  # shape: (BHW, C) ->  (B, H, W, C)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(
                encodings, 0
            )  # shape: (N, )

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(
                encodings.t(), flat_input
            )  # shape: (BHW, N)^T * (BHW, C) -> (N, C)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )  # shape: (N, C)

            self._embedding.weight = nn.Parameter(
                self._ema_w
                / self._ema_cluster_size.unsqueeze(
                    1
                )  # normalized by the count of each N
            )  # shape: (N, C)

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        # convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous()


# def get_autoencoder(out_channels=384):
#     return nn.Sequential(
#         # encoder
#         nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
#         # decoder
#         nn.Upsample(size=3, mode="bilinear"),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.2),
#         nn.Upsample(size=8, mode="bilinear"),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.2),
#         nn.Upsample(size=15, mode="bilinear"),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.2),
#         nn.Upsample(size=32, mode="bilinear"),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.2),
#         nn.Upsample(size=63, mode="bilinear"),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.2),
#         nn.Upsample(size=127, mode="bilinear"),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2),
#         nn.ReLU(inplace=True),
#         nn.Dropout(0.2),
#         nn.Upsample(size=64, mode="bilinear"),
#         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(
#             in_channels=64,
#             out_channels=out_channels,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#         ),
#     )


class Autoencoder(nn.Module):
    def __init__(self, out_channels=384) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.enc_conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        self.enc_conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        self.enc_conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.enc_conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.enc_conv5 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.enc_conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8)

        self.dropout = nn.Dropout(0.2)

        self.dec_up1 = nn.Upsample(size=3, mode="bilinear")
        self.dec_up2 = nn.Upsample(size=8, mode="bilinear")
        self.dec_up3 = nn.Upsample(size=15, mode="bilinear")
        self.dec_up4 = nn.Upsample(size=32, mode="bilinear")
        self.dec_up5 = nn.Upsample(size=63, mode="bilinear")
        self.dec_up6 = nn.Upsample(size=127, mode="bilinear")
        self.dec_up7 = nn.Upsample(
            size=64, mode="bilinear"
        )  # size=64 if padding==True else size=64-8

        self.dec_conv1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2
        )
        self.dec_conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2
        )
        self.dec_conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2
        )
        self.dec_conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2
        )
        self.dec_conv5 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2
        )
        self.dec_conv6 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2
        )
        self.dec_conv7 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.dec_conv8 = nn.Conv2d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # self.enc_vq1 = VectorQuantizerEMA(num_embeddings=32, embedding_dim=64)
        # self.enc_vq2 = VectorQuantizerEMA(num_embeddings=16, embedding_dim=64)
        # self.enc_vq3 = VectorQuantizerEMA(num_embeddings=8, embedding_dim=64)

    def forward(self, x, return_bn=False, return_both=False):
        # encoder
        x = self.enc_conv1(x)
        x = self.relu(x)  # shape: [1, 32, 128, 128]

        x = self.enc_conv2(x)
        x = self.relu(x)  # shape: [1, 32, 64, 64]

        x = self.enc_conv3(x)
        x = self.relu(x)  # shape: [1, 64, 32, 32]
        # x = self.enc_vq1(x)

        x = self.enc_conv4(x)
        x = self.relu(x)  # shape: [1, 64, 16, 16]
        # x = self.enc_vq2(x)

        x = self.enc_conv5(x)
        x = self.relu(x)  # shape: [1, 64, 8, 8]
        # x = self.enc_vq3(x)

        x = self.enc_conv6(x)  # shape: [1, 64, 1, 1]

        if return_bn:
            return x
        elif return_both:
            bn = x.detach().clone()

        # decoder
        x = self.dec_up1(x)
        x = self.dec_conv1(x)
        x = self.relu(x)
        x = self.dropout(x)  # shape: [1, 64, 4, 4]

        x = self.dec_up2(x)
        x = self.dec_conv2(x)
        x = self.relu(x)
        x = self.dropout(x)  # shape: [1, 64, 9, 9]

        x = self.dec_up3(x)
        x = self.dec_conv3(x)
        x = self.relu(x)
        x = self.dropout(x)  # shape: [1, 64, 16, 16]

        x = self.dec_up4(x)
        x = self.dec_conv4(x)
        x = self.relu(x)
        x = self.dropout(x)  # shape: [1, 64, 33, 33]

        x = self.dec_up5(x)
        x = self.dec_conv5(x)
        x = self.relu(x)
        x = self.dropout(x)  # shape: [1, 64, 64, 64]

        x = self.dec_up6(x)
        x = self.dec_conv6(x)
        x = self.relu(x)
        x = self.dropout(x)  # shape: [1, 64, 128, 128]

        x = self.dec_up7(x)
        x = self.dec_conv7(x)
        x = self.relu(x)  # shape: [1, 64, 64, 64]

        x = self.dec_conv8(x)  # shape: [1, 384, 64, 64]

        if return_both:
            return (bn, x)
        else:
            return x


def get_pdn_small(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=4, padding=3 * pad_mult
        ),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1 * pad_mult
        ),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4),
    )


class PDN_Small(nn.Module):
    def __init__(self, out_channels=384, padding=False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=128, kernel_size=4, padding=3 * pad_mult
        )
        self.relu1 = nn.ReLU()
        self.avg1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=4, padding=3 * pad_mult
        )
        self.relu2 = nn.ReLU()
        self.avg2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1 * pad_mult
        )
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=out_channels, kernel_size=4
        )

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.relu1(x1)
        x2 = self.avg1(x2)
        x2 = self.conv2(x2)

        x3 = self.relu2(x2)
        x3 = self.avg2(x3)
        x3 = self.conv3(x3)

        x4 = self.relu3(x3)
        x4 = self.conv4(x4)

        return x4


def get_pdn_medium(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=4, padding=3 * pad_mult
        ),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1 * pad_mult
        ),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
    )


class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample


class ImageFolderWithTargetAndPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path


class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, path


def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        sr_ratio=1,  # to reduce spatial dim before K,V multiplication
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))

        return x


class DeConv(nn.Module):
    def __init__(
        self,
        in_dim=128,
        out_dim=2,
    ):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

        self.dec_1 = nn.ConvTranspose2d(
            in_channels=in_dim, out_channels=64, kernel_size=4, stride=4
        )
        self.dec_2 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=4, stride=4
        )
        self.dec_3 = nn.ConvTranspose2d(
            in_channels=32, out_channels=8, kernel_size=4, stride=4
        )
        self.dec_4 = nn.ConvTranspose2d(
            in_channels=8, out_channels=out_dim, kernel_size=4, stride=4
        )

        self.attn_module_1 = nn.ModuleList(
            [SelfAttentionBlock(dim=64) for j in range(4)]
        )
        self.attn_module_2 = nn.ModuleList(
            [SelfAttentionBlock(dim=32) for j in range(4)]
        )

    def forward(self, x):
        x = self.dec_1(x)
        x = self.relu(x)

        # pass through attention module
        B, C, H, W = x.shape  # ( 2 256 16 16)
        x = x.reshape(B, C, -1)
        x = x.permute(0, 2, 1)
        for blk in self.attn_module_1:
            x = blk(x, H, W)
        x = x.permute(0, 1, 2)
        x = x.reshape(B, C, H, W)

        x = self.dec_2(x)
        x = self.relu(x)

        # pass through attention module
        B, C, H, W = x.shape  # ( 2 32 64 64 )
        x = x.reshape(B, C, -1)
        x = x.permute(0, 2, 1)
        for blk in self.attn_module_2:
            x = blk(x, H, W)
        x = x.permute(0, 1, 2)
        x = x.reshape(B, C, H, W)

        x = self.dec_3(x)
        x = self.relu(x)

        x = self.dec_4(x)
        return x


class LogicalMaskProducer(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.deconv = DeConv(in_dim=128)

    def forward(self, x, ref_features=None):
        if self.training:
            """
            train mode
            """
            # x: [2, 64*2, 1, 1]
            x = self.deconv(x)
            x = torch.softmax(x, dim=1)
            return x
        else:
            """
            eval mode
            """
            with torch.no_grad():
                assert ref_features is not None, "ref_features should not be None"

                # find closest ref
                num_ref = ref_features.shape[0]
                max_sim = -1000
                max_index = None

                for i in range(num_ref):
                    ref = ref_features[i]
                    sim = F.cosine_similarity(ref, x[0], dim=0).mean()
                    if sim > max_sim:
                        max_sim = sim
                        max_index = i

                x = torch.cat([ref_features[max_index], x[0]]).unsqueeze(0)
                x = self.deconv(x)
                x = torch.softmax(x, dim=1)

                return x
