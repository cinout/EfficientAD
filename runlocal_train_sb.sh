python -u efficientad_aeswap.py \
  --dataset mvtec_loco \
  --subdataset screw_bag \
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  --debug_mode \
  --recontrast \
  --image_size_ae 256 \
  # --analysis_heatmap \
  # --norm_c \
  # --reduce_channel_dim \
  # --seeds 10 20 30 \

# breakfast_box
# juice_bottle
# pushpins
# screw_bag
# splicing_connectors