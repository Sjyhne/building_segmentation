# Building Segmentation
Building segmentation in cooperation with the KartAi research project. This repository will explore BEiT and a multi-scale model to be used for building segmentation from orthophotos.

## Dataset

Download [this](https://www.kaggle.com/balraj98/massachusetts-buildings-dataset/download) dataset and place it in a data folder located in the main directory.

Unzip it and run the rename_files.py file. This is due to the labels are spelled tif with only one f, and pillow does not like that.

python run_beit_pretraining.py --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} --num_mask_patches 75 --model beit_base_patch16_224_8k_vocab --discrete_vae_weight_path ${TOKENIZER_PATH} --batch_size 4 --lr 1.5e-3 --warmup_steps 10000 --epochs 10 --clip_grad 3.0 --drop_path 0.1 --layer_scale_init_value 0.1