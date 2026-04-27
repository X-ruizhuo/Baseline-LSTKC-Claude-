#!/bin/bash

# Baseline BI-C2R 训练脚本 (不使用 LSTKC++) - Setting 1
# 用于对比实验

CUDA_VISIBLE_DEVICES=0 python continual_train.py \
    --config_file config/base.yml \
    --data-dir /home/data/PRID \
    --logs-dir logs-baseline-setting1 \
    --setting 1 \
    --epochs0 80 \
    --epochs 60 \
    --eval_epoch 10 \
    -b 64 \
    --num-instances 4 \
    --lr 0.008 \
    --weight-decay 0.0001 \
    --momentum 0.9 \
    --optimizer SGD \
    --milestones 40 70 \
    --warmup-step 10 \
    --weight_trans 100 \
    --weight_anti 1 \
    --weight_discri 0.007 \
    --weight_transx 0.0005 \
    --AF_weight 1.0 \
    --trans_feat \
    --middle_test \
    --use_lstkc False

echo "Baseline 训练完成! 日志保存在: logs-baseline-setting1/"
