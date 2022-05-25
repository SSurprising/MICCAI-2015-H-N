#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train_locate.py \
    --model_name='UNet'\
    --data_path='/home/zjm/Data/MICCAI_2015/3organs/preprocess/train/' \
    --output_dir='/home/zjm/Project/segmentation/MICCAI 2015 H&N/LocationNet/' \
    --epochs=500\
    --learning_rate=1e-4 \
    --num_organ=2 \
    --HU_upper_threshold=100 \
    --HU_lower_threshold=-100 \
    --train \
    --loss='ce' \
    --batch_size=1 \
    --repeat_times='1' \
    --slice_expand=4 \
