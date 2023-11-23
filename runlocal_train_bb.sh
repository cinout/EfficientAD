python -u efficientad_aeswap.py \
  --dataset mvtec_loco \
  --subdataset breakfast_box \
  --imagenet_train_path ./datasets/Imagenet/ILSVRC/Data/CLS-LOC/train \
  --recontrast \
  --image_size_ae 256 \
  --loose_ae \
  # --debug_mode \
  # --analysis_heatmap \
  # --norm_c \
  # --reduce_channel_dim \
  # --seeds 10 20 30 \

# breakfast_box
# juice_bottle
# pushpins
# screw_bag
# splicing_connectors