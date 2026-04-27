#!/bin/bash

# LSTKC++ 融合训练脚本 - Setting 1
# 数据集顺序: Market1501 → DukeMTMC → CUHK03 → MSMT17 (从小到大)

CUDA_VISIBLE_DEVICES=0 python continual_train.py \
    --config_file config/base.yml \
    --data-dir /home/data/PRID \
    --logs-dir logs-lstkc-fusion-setting1 \
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
    --decouple_epochs 5 \
    --decouple_lr 0.001 \
    --use_lstkc True \
    --long_align_weight 1.0 \
    --long_relation_weight 2.0 \
    --short_adapt_weight 0.5 \
    --short_relation_weight 0.5

echo "训练完成! 日志保存在: logs-lstkc-fusion-setting1/"
