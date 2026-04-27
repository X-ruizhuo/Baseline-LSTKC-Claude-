#!/bin/bash

# Bi-C2R + LSTKC++ 融合框架训练脚本
# 使用混合精度训练和梯度累积优化显存使用

CUDA_VISIBLE_DEVICES=0 python continual_train.py \
    --logs-dir logs-lstkc-setting1/ \
    --setting 1 \
    --enable_lstkc \
    --use_amp \
    --gradient_accumulation_steps 2 \
    --batch-size 48 \
    --weight_long_term 0.5 \
    --weight_short_term 0.3 \
    --epochs0 80 \
    --epochs 60
