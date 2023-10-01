python -u pretraining_after_ft.py \
  --network vit \
  --extractor_input_size 512 \
  --batch_size 2 \
  --subdataset breakfast_box \
  --ft_folder finetuned_vit_20231001_112430_27_40 \
### vit, pvt2_b2li