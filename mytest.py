import torch
from functools import reduce
import math
from PIL import Image
import os
from torchvision import transforms
import numpy as np

image_size_ae = 512
path = "./datasets/loco/splicing_connectors/test/good/000.png"
input_image = Image.open(path)
height = input_image.height
width = input_image.width
divider = 16
patch_resolution = int(image_size_ae / divider)
swap_guide = [
    (100, 99),
    (83, 98),
    (99, 115),
    (101, 117),
    (84, 141),
    (82, 142),
    (109, 126),
    (117, 116),
    (98, 114),
    (122, 140),
    (133, 133),
    (108, 125),
    (107, 100),
    (66, 82),
    (116, 131),
    (114, 130),
    (67, 83),
    (118, 139),
    (132, 132),
    (137, 155),
    (81, 97),
    (110, 127),
    (106, 124),
    (138, 156),
    (171, 189),
    (124, 113),
    (140, 157),
    (115, 147),
    (126, 143),
    (155, 172),
    (154, 171),
    (172, 190),
    (30, 30),
    (121, 121),
    (65, 81),
    (123, 148),
    (148, 163),
    (141, 173),
    (102, 123),
    (85, 110),
    (125, 146),
    (130, 129),
    (14, 14),
    (105, 122),
    (146, 162),
    (134, 154),
    (173, 206),
    (31, 31),
    (97, 145),
    (92, 111),
    (170, 188),
    (157, 174),
    (24, 27),
    (93, 109),
    (156, 158),
    (149, 205),
    (8, 11),
    (23, 25),
    (238, 17),
    (25, 28),
    (136, 137),
    (29, 29),
    (15, 1),
    (28, 26),
    (68, 164),
    (13, 9),
    (120, 138),
    (9, 7),
    (174, 191),
    (94, 15),
    (27, 24),
    (26, 23),
    (7, 8),
    (12, 12),
    (11, 13),
    (10, 10),
    (91, 108),
    (142, 175),
    (153, 149),
    (1, 2),
    (145, 161),
    (228, 230),
    (188, 207),
    (227, 222),
    (3, 6),
    (229, 231),
    (254, 46),
    (2, 3),
    (111, 223),
    (22, 21),
    (119, 120),
    (239, 16),
    (6, 5),
    (225, 232),
    (17, 19),
    (139, 159),
    (230, 229),
    (4, 4),
    (150, 179),
    (0, 0),
    (50, 66),
    (135, 136),
    (5, 20),
    (231, 233),
    (244, 241),
    (21, 22),
    (187, 204),
    (233, 47),
    (226, 225),
    (16, 18),
    (80, 96),
    (243, 247),
    (232, 228),
    (90, 84),
    (249, 246),
    (147, 101),
    (245, 248),
    (189, 178),
    (51, 65),
    (104, 105),
    (129, 112),
    (235, 234),
    (247, 243),
    (241, 242),
    (246, 249),
    (242, 254),
    (127, 187),
    (248, 245),
    (236, 238),
    (234, 62),
    (103, 119),
    (251, 244),
    (255, 224),
    (237, 227),
    (95, 80),
    (190, 177),
    (252, 250),
    (250, 63),
    (86, 107),
    (162, 221),
    (224, 239),
    (96, 144),
    (253, 255),
    (49, 67),
    (175, 170),
    (19, 78),
    (128, 160),
    (113, 128),
    (20, 79),
    (240, 240),
    (18, 226),
    (46, 45),
    (163, 165),
    (206, 235),
    (40, 43),
    (164, 180),
    (64, 32),
    (213, 216),
    (47, 44),
    (204, 236),
    (158, 153),
    (62, 60),
    (41, 41),
    (191, 251),
    (151, 152),
    (63, 95),
    (214, 215),
    (215, 39),
    (169, 203),
    (205, 252),
    (152, 150),
    (42, 61),
    (131, 118),
    (112, 33),
    (144, 176),
    (78, 94),
    (39, 59),
    (212, 214),
    (209, 237),
    (89, 89),
    (161, 220),
    (216, 42),
    (211, 40),
    (222, 253),
    (56, 57),
    (159, 106),
    (223, 200),
    (79, 37),
    (217, 217),
    (38, 56),
    (179, 195),
    (43, 55),
    (69, 88),
    (199, 38),
    (143, 64),
    (207, 58),
    (52, 151),
    (87, 134),
    (44, 36),
    (210, 213),
    (57, 76),
    (45, 35),
    (200, 199),
    (203, 196),
    (55, 77),
    (208, 208),
    (88, 85),
    (198, 201),
    (186, 169),
    (197, 212),
    (76, 68),
    (165, 166),
    (58, 54),
    (77, 53),
    (168, 219),
    (48, 48),
    (180, 181),
    (201, 198),
    (160, 104),
    (178, 194),
    (61, 34),
    (54, 75),
    (177, 209),
    (221, 218),
    (193, 197),
    (75, 135),
    (166, 168),
    (37, 52),
    (167, 202),
    (196, 211),
    (32, 49),
    (219, 184),
    (202, 193),
    (185, 186),
    (60, 51),
    (53, 90),
    (220, 183),
    (218, 182),
    (192, 192),
    (195, 210),
    (59, 74),
    (33, 102),
    (36, 50),
    (176, 167),
    (74, 93),
    (70, 87),
    (181, 185),
    (72, 73),
    (183, 103),
    (73, 91),
    (35, 72),
    (34, 71),
    (71, 92),
    (194, 86),
    (184, 69),
    (182, 70),
]

ref_image_path = os.path.join(
    "./datasets/loco/",
    "splicing_connectors",
    "train",
    "good",
    "023.png",
)
ref_image = Image.open(ref_image_path)


transform_ae = transforms.Compose(
    [
        transforms.Resize((image_size_ae, image_size_ae)),
        transforms.ToTensor(),
    ]
)

input_image = transform_ae(input_image)
ref_image = transform_ae(ref_image)

unfolding_op = torch.nn.Unfold(kernel_size=patch_resolution, stride=patch_resolution)
input_image = unfolding_op(
    input_image
)  # [C*patch_resolution*patch_resolution, divider*divider]
ref_image = unfolding_op(ref_image)

for index_ae, index_ref in swap_guide:
    input_image[:, index_ae] = ref_image[:, index_ref]

fold_op = torch.nn.Fold(
    output_size=image_size_ae, kernel_size=patch_resolution, stride=patch_resolution
)
input_image = fold_op(input_image)


input_image = np.array(input_image * 255, dtype=np.uint8)
input_image = np.transpose(input_image, (1, 2, 0))
input_image = Image.fromarray(input_image)
input_image = input_image.resize(size=(width, height))
input_image.save(f"yuck.png", "PNG")

# input_image = input_image.unfold(1, patch_resolution, patch_resolution).unfold(
#     2, patch_resolution, patch_resolution
# )
# ref_image = ref_image.unfold(1, patch_resolution, patch_resolution).unfold(
#     2, patch_resolution, patch_resolution
# )

# C, H_N, W_N, H, W = input_image.shape  # [3, 16, 16, 32, 32]
# input_image = input_image.reshape(C, -1, H, W)
# ref_image = ref_image.reshape(C, -1, H, W)

# for index_ae, index_ref in swap_guide:
#     input_image[:, index_ae, :, :] = ref_image[:, index_ref, :, :]

# input_image = input_image.reshape(C, H_N, W_N, H, W)
