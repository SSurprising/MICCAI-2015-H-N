#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name='SC_UNet_all'\
    --data_path='/home/zjm/Data/MICCAI_2015/3organs/' \
    --output_dir='/home/zjm/Project/segmentation/BLSC_2D/outputs/' \
    --val_subset='subset5' \
    --epochs=1000\
    --learning_rate=1e-4 \
    --num_organ=4 \
    --slice_size=128 \
    --resolution 512 512 \
    --HU_upper_threshold=300 \
    --HU_lower_threshold=-100 \
    --train \
    --loss='ce' \
    --batch_size=1 \
    --repeat_times='1' \
    --slice_expand=0 \
    --show_test_loss \
    --auxiliary_loss \
    --lw=0.1 \
