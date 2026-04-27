#!/bin/bash

# 单卡低显存配置 - 适用于 V100-32GB
# 主要优化:
# 1. batch_size 从 64 降到 32
# 2. workers 从 8 降到 4
# 3. 使用低显存配置文件
# 4. 启用梯度累积效果 (通过减小batch size实现)

CUDA_VISIBLE_DEVICES=0 python continual_train.py \
    --config_file config/low_memory.yml \
    --logs-dir logs-lstkc-low-memory/ \
    --batch-size 32 \
    --workers 4 \
    --setting 1 \
    --AF_weight 1.0 \
    --weight_trans 100 \
    --weight_anti 1 \
    --weight_discri 0.007 \
    --weight_transx 0.0005 \
    --epochs0 80 \
    --epochs 60 \
    --milestones 30 \
    --middle_test
