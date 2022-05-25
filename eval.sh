#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train.py \
    --data_path='/home/zjm/Data/HaN_OAR_raw/four_organ/' \
    --output_dir='/home/zjm/Project/segmentation/BLSC_2D/outputs/' \
    --val_subset='subset5' \
    --epochs=1000 \
    --learning_rate=1e-4 \
    --num_organ=4 \
    --slice_size=128 \
    --resolution 512 512 \
    --HU_upper_threshold=300 \
    --HU_lower_threshold=-100 \
    --model_name='SC_UNet'\
    --repeat_times='1' \
    --slice_expand=0 \
    --loss='ce' \
    --crop_size 128 128 \
    --test \
    --test_model_path='/home/zjm/Project/segmentation/BLSC_2D/outputs/' \
    --test_best_val \



