#!/bin/bash

# LSTKC++融合版本训练脚本 - Setting 2
# Duke → MSMT17 → Market → CUHK-SYSU → CUHK03

CUDA_VISIBLE_DEVICES=0 python continual_train.py \
    --config_file config/lstkc_fusion.yml \
    --logs-dir logs-lstkc-fusion-setting2/ \
    --setting 2 \
    --batch-size 64 \
    --epochs0 80 \
    --epochs 60 \
    --weight_trans 100 \
    --weight_anti 1.0 \
    --weight_discri 0.007 \
    --weight_transx 0.0005 \
    --seed 24 \
    --trans_feat \
    --middle_test

echo "Training completed for Setting 2"
