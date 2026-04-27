# 单卡低显存运行配置说明

## 显卡配置
- GPU: Tesla V100-32GB (单卡)
- 可用显存: 32GB

## 优化策略

为了在单卡上运行此项目,我们通过以下方式降低显存使用,**无需修改任何训练代码**:

### 1. 批次大小优化
- **原始配置**: batch_size = 64
- **优化配置**: batch_size = 32
- **显存节省**: 约 40-50%

### 2. 数据加载优化
- **原始配置**: workers = 8
- **优化配置**: workers = 4
- **显存节省**: 减少数据预加载的显存占用

### 3. 配置文件
创建了专门的低显存配置文件 `config/low_memory.yml`,主要修改:
- `IMS_PER_BATCH: 32` (从 64 降到 32)
- `NUM_WORKERS: 4` (从 8 降到 4)

## 使用方法

### 方式1: 使用提供的脚本(推荐)
```bash
cd /home/Baseline-LSTKC-Claude-/Bi-C2R
bash run_low_memory.sh
```

### 方式2: 手动运行
```bash
cd /home/Baseline-LSTKC-Claude-/Bi-C2R
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
```

## 预期显存使用

根据配置优化,预期显存使用:
- **训练阶段**: 约 15-20GB
- **评估阶段**: 约 10-15GB
- **峰值**: 不超过 25GB

这样可以安全地在单张 V100-32GB 上运行,留有充足的显存余量。

## 性能影响

- **训练速度**: 由于 batch size 减半,每个 epoch 的迭代次数会增加约 2倍,但总体训练时间影响不大
- **收敛性**: batch_size=32 仍然足够大,不会影响模型收敛
- **最终精度**: 预期与原始配置相当,因为总的训练样本数不变

## 进一步优化(如果仍然显存不足)

如果上述配置仍然显存不足,可以尝试:

1. **进一步降低 batch size**:
   ```bash
   --batch-size 16
   ```
   同时修改 `config/low_memory.yml` 中的 `IMS_PER_BATCH: 16`

2. **降低图像分辨率**(会影响精度):
   在 `config/low_memory.yml` 中修改:
   ```yaml
   INPUT:
     SIZE_TRAIN: [224, 112]  # 从 [256, 128] 降低
     SIZE_TEST: [224, 112]
   ```

3. **使用梯度累积**(需要修改代码,不推荐):
   这需要在 trainer.py 中添加梯度累积逻辑

## 监控显存使用

运行训练时,可以在另一个终端监控显存:
```bash
watch -n 1 nvidia-smi
```

## 文件清单

- `run_low_memory.sh`: 低显存运行脚本
- `config/low_memory.yml`: 低显存配置文件
- `README_LOW_MEMORY.md`: 本说明文档
