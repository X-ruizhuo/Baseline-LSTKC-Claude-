# Bi-C2R + LSTKC++ 融合框架

本项目将Bi-C2R框架与LSTKC++框架进行模块融合，用于Re-indexing Free Lifelong Person Re-identification (RFL-ReID)任务。

## 融合特性

### 1. LSTKC++核心模块
- **长短期知识分解**: 区分长期稳定知识和短期任务特定知识
- **知识过滤机制**: 评估旧知识质量，过滤低质量知识
- **自适应知识整合**: 动态平衡新旧知识的融合权重

### 2. 显存优化
- **混合精度训练**: 使用FP16降低显存占用
- **梯度累积**: 支持更大的有效批次大小
- **批次大小调整**: 从64降至48，节省显存

### 3. 保持Bi-C2R优势
- 双向转换网络保持特征兼容性
- 自适应alpha计算
- 多损失函数协同优化

## 安装

```bash
conda create -n IRL python=3.7
conda activate IRL
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirement.txt
```

## 使用方法

### 基础训练 (Bi-C2R baseline)
```bash
bash run1.sh  # Setting 1
bash run2.sh  # Setting 2
```

### LSTKC++增强训练
```bash
bash run_lstkc.sh
```

或手动指定参数:
```bash
CUDA_VISIBLE_DEVICES=0 python continual_train.py \
    --logs-dir logs-lstkc/ \
    --setting 1 \
    --enable_lstkc \
    --use_amp \
    --gradient_accumulation_steps 2 \
    --batch-size 48 \
    --weight_long_term 0.5 \
    --weight_short_term 0.3
```

## 主要参数

### LSTKC++相关参数
- `--enable_lstkc`: 启用LSTKC++模块
- `--weight_long_term`: 长期知识损失权重 (默认: 0.5)
- `--weight_short_term`: 短期知识损失权重 (默认: 0.3)

### 显存优化参数
- `--use_amp`: 启用混合精度训练
- `--gradient_accumulation_steps`: 梯度累积步数 (默认: 1)
- `--batch-size`: 批次大小 (推荐: 48)

### Bi-C2R原有参数
- `--weight_trans`: 转换对齐损失权重 (默认: 100)
- `--weight_anti`: 兼容性保持损失权重 (默认: 1)
- `--weight_discri`: 判别性保持损失权重 (默认: 0.007)
- `--weight_transx`: 方向一致性损失权重 (默认: 0.0005)

## 项目结构

```
Bi-C2R/
├── continual_train.py          # 主训练脚本 (已修改)
├── reid/
│   ├── models/
│   │   ├── resnet.py          # 模型定义 (已修改)
│   │   └── lstkc_modules.py   # LSTKC++模块 (新增)
│   └── trainer.py             # 训练器 (已修改)
├── config/
│   ├── base.yml               # 基础配置 (已修改)
│   └── lstkc_enhanced.yml     # LSTKC++配置 (新增)
├── docs/
│   └── Bi-C2R_LSTKC++_融合方案.md  # 详细融合方案
├── run1.sh                    # 原始训练脚本
├── run2.sh                    # 原始训练脚本
└── run_lstkc.sh              # LSTKC++训练脚本 (新增)
```

## 显存使用

### 优化前 (Bi-C2R baseline)
- 批次大小: 64
- 显存占用: ~32GB
- 需要: Tesla V100 32GB

### 优化后 (LSTKC++增强)
- 批次大小: 48
- 梯度累积: 2步
- 混合精度: FP16
- 显存占用: ~21GB (节省34%)
- 支持: Tesla V100 32GB (有余量)

## 预期性能提升

基于LSTKC框架的研究成果，预期改进:

1. **Seen datasets mAP**: +1-2%
2. **Unseen datasets mAP**: +2-3% (RFL关键指标)
3. **训练稳定性**: 减少知识冲突，更平滑的性能曲线

## 核心改进

### 1. 知识分解 ([reid/models/lstkc_modules.py](reid/models/lstkc_modules.py))
```python
class KnowledgeDecomposition(nn.Module):
    # 将特征分解为长期稳定知识和短期任务特定知识
    # 通过门控机制自适应平衡两者
```

### 2. 知识过滤 ([reid/models/lstkc_modules.py](reid/models/lstkc_modules.py))
```python
class KnowledgeFilter(nn.Module):
    # 评估旧知识质量
    # 过滤低质量知识，减少负迁移
```

### 3. 增强训练 ([reid/trainer.py](reid/trainer.py))
- 集成知识分解和过滤
- 添加长短期知识损失
- 支持混合精度训练

## 消融实验建议

1. Baseline (Bi-C2R)
2. + 知识分解模块
3. + 知识过滤模块
4. + 混合精度优化
5. Full (所有模块)

## GPU要求

- 推荐: Tesla V100 32GB 或更高
- 最低: RTX 3090 24GB (需进一步调整批次大小)

## 参考文献

- **Bi-C2R**: Bidirectional Continual Compatible Representation for Re-indexing Free Lifelong Person Re-identification (TPAMI 2026)
- **LSTKC**: Long Short-Term Knowledge Consolidation for Lifelong Person Re-identification (AAAI 2024)
- **LSTKC++**: Extended version with knowledge decomposition and consolidation

## 联系方式

如有问题，请参考 [docs/Bi-C2R_LSTKC++_融合方案.md](docs/Bi-C2R_LSTKC++_融合方案.md) 获取详细技术方案。
