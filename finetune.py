import glob
import math
import torch
import random, os, argparse, builtins
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
import torch.distributed as dist
import numpy as np
from torchvision import transforms
from torch import nn
from datetime import datetime
from torch.utils.data import Dataset
from PIL import Image
from torch.nn import functional as F

from pvt_v2 import pvt_v2_b2_li

on_gpu = torch.cuda.is_available()
device = "cuda" if on_gpu else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()

    # ---- SLURM and DDP ---- #
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

    # ---- Training Options ---- #

    parser.add_argument(
        "--mvtec_loco_path",
        default="./datasets/loco",
        help="Downloaded Mvtec LOCO dataset",
    )
    parser.add_argument(
        "--output_folder",
        default="./finetuned_vit",
        help="Downloaded Mvtec LOCO dataset",
    )
    parser.add_argument(
        "--model", type=str, default="vit", choices=["vit", "pvt2_b2li"]
    )
    parser.add_argument(
        "--image_size", type=int, default=512, help="input image size for training"
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        help="total training epochs",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="batch size for training on single process",
    )
    parser.add_argument(
        "--learning_rate",
        default=0.0001,
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "--seed",
        default=111,
        type=int,
        help="seed for randomization",
    )
    parser.add_argument("--note", type=str, default="")

    args = parser.parse_args()
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        self.dec_1 = nn.ConvTranspose2d(
            in_channels=768, out_channels=192, kernel_size=2, stride=2
        )
        self.dec_2 = nn.ConvTranspose2d(
            in_channels=192, out_channels=96, kernel_size=2, stride=2
        )
        self.dec_3 = nn.ConvTranspose2d(
            in_channels=96, out_channels=24, kernel_size=2, stride=2
        )
        self.dec_4 = nn.ConvTranspose2d(
            in_channels=24, out_channels=3, kernel_size=2, stride=2
        )

    def forward(self, x):
        # x.shape: [bs, 768, 32, 32]

        x = self.dec_1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.dec_2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.dec_3(x)
        x = self.relu(x)

        x = self.dec_4(x)

        return x  # target_x.shape: [bs, 3, 512, 512]


class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample


class MyDataset(Dataset):
    def __init__(self, mvtec_loco_path, image_size):
        super().__init__()
        self.images = glob.glob(mvtec_loco_path + "/*/train/good/*.png")
        self.default_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.transform_ae = transforms.RandomChoice(
            [
                transforms.ColorJitter(brightness=0.2),
                transforms.ColorJitter(contrast=0.2),
                transforms.ColorJitter(saturation=0.2),
            ]
        )

    def __len__(self):
        return len(self.images)

    def transform_image(self, img_path):
        image = Image.open(img_path)
        image = image.convert("RGB")
        return self.default_transform(self.transform_ae(image))

    def __getitem__(self, index):
        img_path = self.images[index]
        return self.transform_image(img_path)


def train(args):
    train_data = MyDataset(args.mvtec_loco_path, args.image_size)

    if args.distributed:
        train_sampler = DistributedSampler(train_data, shuffle=True)
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True
        )

    # initialize encoder
    if args.model == "vit":
        config = CONFIGS["ViT-B_16"]
        model = VisionTransformer(
            config,
            num_classes=1000,
            zero_head=False,
            img_size=args.image_size,
            vis=True,
        )
        model.load_from(np.load("vit_model_checkpoints/ViT-B_16-224.npz"))
        encoder = torch.nn.Sequential(
            *[model.transformer.embeddings, model.transformer.encoder]
        )
    elif args.model == "pvt2_b2li":
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

        encoder = pvt_v2_b2_li(pretrained=False)
        encoder.load_state_dict(pretrained_weights, strict=False)

    decoder = Decoder()

    if args.distributed:
        if args.gpu is None:
            encoder.cuda()
            encoder = torch.nn.parallel.DistributedDataParallel(encoder)
            decoder.cuda()
            decoder = torch.nn.parallel.DistributedDataParallel(decoder)

        else:
            encoder.cuda(args.gpu)
            encoder = torch.nn.parallel.DistributedDataParallel(
                encoder, device_ids=[args.gpu], find_unused_parameters=False
            )
            decoder.cuda(args.gpu)
            decoder = torch.nn.parallel.DistributedDataParallel(
                decoder, device_ids=[args.gpu], find_unused_parameters=False
            )
    else:
        encoder.to(device)
        decoder.to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )

    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()

        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)

        for img in train_dataloader:
            img = img.to(device)  # shape: [bs, 3, 512, 512]
            features = encoder(img)[0]
            features = features[:, 1:, :]
            B, N, C = features.shape
            H = int(math.sqrt(N))
            W = int(math.sqrt(N))
            features = features.transpose(1, 2).view(
                B, C, H, W
            )  # shape: [bs, 768, 32, 32]
            output = decoder(features)  # shape: [bs, 3, 512, 512]

            loss = F.mse_loss(output, img, reduction="mean")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(
                "epoch [{}/{}], loss:{:.4f}".format(epoch + 1, args.epochs, loss.item())
            )

        if epoch == args.epochs - 1:
            if not on_gpu or dist.get_rank() == 0:
                if args.model == "vit":
                    filename = "ft_vit.pth"
                elif args.model == "pvt2_b2li":
                    filename = "ft_pvt.pth"
                
                torch.save(
                    encoder.state_dict(),
                    os.path.join(
                        args.output_folder,
                        filename,
                    ),
                )


if __name__ == "__main__":
    args = parse_args()

    setup_seed(args.seed)

    # --- DDP setting --- #
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
    print(
        ">>> world_size:",
        args.world_size,
        "ngpus_per_node:",
        ngpus_per_node,
        f"device names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}",
    )

    print(f"===============")

    # if args.global_rank == 0:
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     args.anomaly_map_root_dir = args.anomaly_map_root_dir + "_" + timestamp
    #     os.makedirs(args.anomaly_map_root_dir, exist_ok=True)

    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )

    # if args.global_rank == 0:
    #     with open(
    #         os.path.join(args.anomaly_map_root_dir, "config.txt"), "a"
    #     ) as config_file:
    #         config_file.writelines(
    #             [f"\n{k}: {str(v)}" for k, v in sorted(dict(vars(args)).items())]
    #         )

    # item_list = [
    #     "breakfast_box",
    #     "juice_bottle",
    #     "pushpins",
    #     "screw_bag",
    #     "splicing_connectors",
    # ]

    # download pretrained vit
    from urllib.request import urlretrieve
    from vit_models.modeling import VisionTransformer, CONFIGS

    if not on_gpu or dist.get_rank() == 0:
        os.makedirs("vit_model_checkpoints", exist_ok=True)
    if not os.path.isfile("vit_model_checkpoints/ViT-B_16-224.npz"):
        urlretrieve(
            "https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz",
            "vit_model_checkpoints/ViT-B_16-224.npz",
        )

    if not on_gpu or dist.get_rank() == 0:
        timestamp = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + str(random.randint(0, 100))
            + "_"
            + str(random.randint(0, 100))
        )
        args.output_folder = args.output_folder + "_" + timestamp
        os.makedirs(args.output_folder)

    # for i in item_list:
    #     train(i, args)

    train(args)
