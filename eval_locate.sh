#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train_locate.py \
    --data_path='/home/zjm/Data/MICCAI_2015/3organs/preprocess/test/' \
    --output_dir='/home/zjm/Project/segmentation/MICCAI 2015 H&N/LocationNet' \
    --epochs=1000 \
    --learning_rate=1e-4 \
    --num_organ=2 \
    --HU_upper_threshold=100 \
    --HU_lower_threshold=-100 \
    --model_name='UNet'\
    --repeat_times='1' \
    --slice_expand=-1 \
    --loss='ce' \
    --test \
    --test_model_path='/home/zjm/Project/segmentation/MICCAI 2015 H&N/LocationNet/' \
#    --test_best_val \



