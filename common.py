#!/usr/bin/python
# -*- coding: utf-8 -*-
from torch import nn
from torchvision.datasets import ImageFolder
import torch


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
        self.dec_up7 = nn.Upsample(size=64, mode="bilinear")

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

        # TODO: check if it works
        # self.enc_vq1 = VectorQuantizerEMA(num_embeddings=32, embedding_dim=64)
        # self.enc_vq2 = VectorQuantizerEMA(num_embeddings=16, embedding_dim=64)
        # self.enc_vq3 = VectorQuantizerEMA(num_embeddings=8, embedding_dim=64)

    def forward(self, x):
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
        self.relu = nn.ReLU(inplace=True)
        self.avg1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=4, padding=3 * pad_mult
        )
        self.avg2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1 * pad_mult
        )
        self.avg3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=out_channels, kernel_size=4
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        x1 = self.avg1(x1)

        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        x2 = self.avg2(x2)

        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        x3 = self.avg3(x3)

        x4 = self.conv4(x3)
        return x2, x3, x4


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


class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path


def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
