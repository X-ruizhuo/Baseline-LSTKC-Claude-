# 改进框架（Bi-C2R + LSTKC++）与 Baseline 对比说明

## 一、概述

本文档对比 **原始 Baseline（Bi-C2R）** 与 **改进框架（Bi-C2R + LSTKC++）** 在框架结构、核心模块、损失函数、模型管理策略等维度的差异，并说明每处改进的动机与创新点。

改进框架在 Bi-C2R 的基础上，融合了 LSTKC++（IEEE TPAMI 2025）的三个核心机制：
1. **知识纠正蒸馏**（STKR / C-STKR）
2. **旧知识反向纠正**（M\*\_o）
3. **关系相似度引导的模型融合**（M\*\_e，公式15）

同时完整保留了 Bi-C2R 的双向特征变换网络（TransNet\_adaptive）和特征重索引机制。

---

## 二、框架结构对比

### 2.1 Baseline（Bi-C2R）框架

```
每个训练步 t：
┌─────────────────────────────────────────────────────┐
│  维护：model（单一当前模型）                          │
│        model_old（上一步推理模型的拷贝）              │
│        model_trans / model_trans2（双向变换网络）     │
└─────────────────────────────────────────────────────┘
         ↓ 训练
  L = L_ce + L_tp
    + L_kl（TP/FP/TN/FN加权软目标KL）
    + L_trans（双向MSE变换对齐）
    + L_cr（对比关系损失）
    + L_discri（知识蒸馏判别损失）
    + L_transx（变换方向余弦损失）
         ↓ 训练后
  alpha = 1 - |Affin_new - Affin_old|.sum(-1).mean()
  model = alpha * model_new + (1-alpha) * model_old
         ↓
  更新旧 gallery 特征（trans_feat）
```

**模型状态**：每步只维护 1 个旧模型（`model_old`），融合后直接作为下一步的旧模型。

---

### 2.2 改进框架（Bi-C2R + LSTKC++）

```
每个训练步 t：
┌─────────────────────────────────────────────────────────────────┐
│  维护：model（当前训练模型 Θ_t^s）                               │
│        model_long（长期模型 Θ_t^l，多域平衡知识）               │
│        model_short_old（上一步训练模型 Θ_{t-1}^s，最新域知识）  │
│        model_trans / model_trans2（双向变换网络，保留）          │
└─────────────────────────────────────────────────────────────────┘
         ↓ 训练（使用 C-STKR 纠正蒸馏）
  L = L_ce + L_tp
    + L_kl（C-STKR纠正后的关系矩阵KL）   ← 替换
    + L_trans（双向MSE变换对齐）           ← 保留
    + L_cr（对比关系损失）                 ← 保留
    + L_discri（知识蒸馏判别损失）         ← 保留
    + L_transx（变换方向余弦损失）         ← 保留
         ↓ 训练后
  【M*_o】在新数据上线性搜索最优 α，更新长期模型
  model_long = (1-α*) * model_short_old + α* * model_long_old
         ↓
  【M*_e】δ = 1 - mean_i<R^s_i, R^l_i>  （公式15，严格[0,1]）
  model = δ * model_long + (1-δ) * model_new
         ↓
  更新旧 gallery 特征（trans_feat，保留）
```

**模型状态**：每步维护 2 个旧模型（长期 + 短期），通过反向搜索动态更新长期模型。

---

## 三、逐模块创新点对比

### 3.1 知识蒸馏模块（核心替换）

#### Baseline：TP/FP/TN/FN 加权软目标 KL（`cal_KL`）

**位置**：`reid/trainer.py:207-237`

```python
# 构建 TP/FP/TN/FN 四类软目标
Old_Keep  = attri_old['TN'] + attri_old['TP']   # 旧模型预测正确的部分
New_keep  = (attri_new['TN'] + attri_new['TP']) * (attri_old['FN'] + attri_old['FP'])  # 旧错新对
Hard_pos  = attri_new['FN'] * attri_old['FN']   # 双方都漏检的正样本
Hard_neg  = attri_new['FP'] * attri_old['FP']   # 双方都误检的负样本
Target = (Target_1 + Target_2 + Target_3 + Target_4) / sum
KLDiv(log(Affinity_new), Target)
```

**问题**：
- 软目标混合了旧模型的正确预测和错误预测，错误知识仍然被蒸馏进新模型
- 对 Hard\_pos 和 Hard\_neg 的处理依赖阈值，存在边界模糊问题
- 新旧模型的 TP/FP 判断相互耦合，逻辑复杂

#### 改进框架：STKR / C-STKR 纠正蒸馏

**位置**：`reid/trainer.py:63-130`（新增方法）

**STKR（单模型纠正，t=2 时使用）**：
```python
# 阈值挖掘
sn = min{R_old[i,j]} for same-ID j   # 同ID最小亲和度 → 负阈值
sp = max{R_old[i,j]} for diff-ID j   # 异ID最大亲和度 → 正阈值

# 阈值纠正
R_tilde[i,j] = max(R_old[i,j], sp)  if same-ID   # 同ID分数不低于sp
R_tilde[i,j] = min(R_old[i,j], sn)  if diff-ID   # 异ID分数不高于sn
# L1归一化 → 蒸馏目标
```

**C-STKR（双模型互补纠正，t≥3 时使用）**：
```python
# 分别纠正长期和短期旧模型
R_tilde_s = stkr_rectify(R_short, targets)
R_tilde_l = stkr_rectify(R_long,  targets)

# 互补融合：谁对用谁
if both_correct or both_wrong:  R_tilde = (R_tilde_s + R_tilde_l) / 2
if only_short_correct:          R_tilde = R_tilde_s
if only_long_correct:           R_tilde = R_tilde_l
```

**创新点**：
- **主动识别并剔除错误知识**：通过阈值判断每个关系分数是否正确，只蒸馏正确的知识
- **互补利用双模型**：长期模型和短期模型各有擅长的域，C-STKR 取两者之长
- **实现更简洁**：完全向量化，无 Python 循环，比 `cal_KL` 的 TP/FP/TN/FN 逻辑更清晰

---

### 3.2 模型融合权重计算（关键修复）

#### Baseline：绝对差公式（`get_adaptive_alpha`）

**位置**：`continual_train.py:184-200`（原始版本）

```python
Difference = |Affin_new - Affin_old|.sum(-1).mean()
alpha = 1 - Difference
```

**问题**：
- 理论上 `Difference ∈ [0, 2]`（两个概率向量的 L1 距离上界为 2）
- 当数据集域间隔较大时，`Difference > 1`，导致 `alpha < 0`
- `alpha < 0` 意味着新模型权重为负，融合结果不可控，性能严重下降

#### 改进框架：点积相似度公式（公式15）

**位置**：`continual_train.py:219-244`

```python
# 点积相似度
dot_sim = (Affin_new * Affin_old).sum(dim=1).mean()
delta = 1.0 - dot_sim
delta = max(0.0, min(1.0, delta))  # 严格裁剪到 [0,1]
```

**数学保证**：
- 每行 `r ∈ R^n` 满足 `Σr_j = 1, r_j ≥ 0`（softmax 输出）
- 点积 `<r^s, r^l> = Σ r^s_j * r^l_j ∈ [0, 1]`（Cauchy-Schwarz 不等式）
- 因此 `delta = 1 - <r^s, r^l> ∈ [0, 1]`，严格有界

**创新点**：
- **理论保证融合权重有界**：彻底解决大域间隔时 alpha 越界导致的训练不稳定问题
- **语义更合理**：点积相似度直接度量两个关系分布的一致性，比绝对差更能反映知识差异程度
- **实验验证**：论文 Fig.6 显示公式(15)在 5 种训练顺序下均比公式(9)更稳定

---

### 3.3 长短期模型分解（结构性创新）

#### Baseline：单一旧模型

每步训练结束后，融合得到的推理模型直接作为下一步的旧模型：

```
t=1: model_1 → 训练 → Θ_1
t=2: Θ_1 → 训练 → Θ_2^s → 融合(Θ_2^s, Θ_1) → Θ_2
t=3: Θ_2 → 训练 → Θ_3^s → 融合(Θ_3^s, Θ_2) → Θ_3
```

**问题**：
- 单一旧模型随着训练步数增加，会逐渐偏向最近学习的域（短期偏置）
- 融合权重 alpha 是基于最新域数据估算的，对历史域的遗忘程度估计不准确

#### 改进框架：长短期双模型分解

```
t=1: 训练 → Θ_1^s;  model_long = Θ_1^s
t=2: 训练(STKR) → Θ_2^s;  M*_e融合 → Θ_2;  model_long = Θ_2
t=3: 训练(C-STKR, 用Θ_1^s和Θ_2) → Θ_3^s
     M*_o(Θ_1^s, Θ_2) → Θ_3^l（反向搜索最优α）
     M*_e(Θ_3^l, Θ_3^s) → Θ_3（推理模型）
     model_short_old = Θ_2^s;  model_long = Θ_3
```

**状态管理**（`continual_train.py:134-174`）：
```python
model_long      = None  # 长期模型：多域平衡知识
model_short_old = None  # 短期旧模型：上一步训练模型

# t=1
model_long = copy.deepcopy(model)
model_short_old = copy.deepcopy(model)

# t=2
best_alpha = get_adaptive_alpha(...)  # M*_e
model = linear_combination(model_long, model, best_alpha)
model_long = copy.deepcopy(model)

# t>=3
model_long = backward_rectification(model_short_old, model_long, init_loader)  # M*_o
best_alpha = get_adaptive_alpha(model, model_long, ...)  # M*_e
model = linear_combination(model_long, model, best_alpha)
```

**创新点**：
- **解耦短期与长期知识**：短期模型专注最新域，长期模型积累多域平衡知识，两者互补
- **消除最新域偏置**：长期模型通过 M\*\_o 反向纠正，不再单纯依赖最新域数据估算融合权重

---

### 3.4 旧知识反向纠正（M\*\_o，全新模块）

#### Baseline：无此机制

Baseline 没有对旧模型融合权重进行反向优化的机制，每步的 alpha 仅基于当前新数据估算，无法回溯修正历史融合决策。

#### 改进框架：M\*\_o 反向搜索

**位置**：`continual_train.py:421-461`（新增函数）

```python
def backward_rectification(args, model_short_old, model_long_old, init_loader):
    best_mAP, best_alpha = 0.0, 0.0
    for alpha in [0.0, 0.1, ..., 1.0]:  # 11次搜索
        model_fused = (1-alpha)*model_short_old + alpha*model_long_old
        # 用新数据（对两个旧模型都是未见数据）做自排序评估
        mAP = compute_self_ranking_mAP(model_fused, init_loader)
        if mAP > best_mAP:
            best_alpha = alpha
    model_long_new = (1-best_alpha)*model_short_old + best_alpha*model_long_old
    return model_long_new
```

**关键设计**：新数据 `D_t` 对 `model_short_old` 和 `model_long_old` 都是**未见数据**，因此可以作为无偏参考来评估融合模型的泛化能力，避免了用训练数据评估带来的过拟合偏差。

**创新点**：
- **无偏参考评估**：利用新数据的"未见性"作为客观评估标准，这是 Baseline 完全没有的机制
- **反向优化历史决策**：每步都重新优化上一步的长期模型融合权重，实现跨步的知识平衡
- **计算开销可控**：11 次前向推理（无梯度），论文实测额外开销约 30%，A40 单卡完全可承受

---

### 3.5 保留的 Bi-C2R 特有机制

以下模块在改进框架中**完整保留**，是 Bi-C2R 相对于 LSTKC++ 的独特贡献：

#### TransNet\_adaptive（双向特征变换网络）

**位置**：`reid/models/resnet.py:170-183`

```
model_trans:  old_features → new_space（旧→新空间映射）
model_trans2: new_features → old_space（新→旧空间映射）
```

每个 TransNet 由 2 层 `RBTBlock_dual` 组成，每层融合：
- 4 路并行 MLP 变换路径（2048→32→32→2048）
- 16 个可学习原型的注意力加权
- 自适应门控混合两者输出 + 残差连接

**作用**：在特征空间层面实现跨步对齐，LSTKC++ 没有此机制。

#### 特征重索引（trans\_feat）

**位置**：`continual_train.py:182-206`

```python
# 每步训练后，用 model_trans 更新旧数据集的 gallery 特征
updated = best_alpha * trans(old_feat) + (1-best_alpha) * old_feat
```

**作用**：无需重新推理旧数据集，通过特征变换实现 re-indexing-free 评估，LSTKC++ 没有此机制。

#### 对比关系损失（L\_cr）、判别损失（L\_discri）、方向损失（L\_transx）

这三个损失与 TransNet 配合，确保特征变换的质量，在改进框架中完整保留。

---

## 四、损失函数全景对比

| 损失项 | Baseline | 改进框架 | 变化 |
|--------|----------|----------|------|
| L\_ce（标签平滑交叉熵） | ✓ | ✓ | 不变 |
| L\_tp（Triplet 硬挖掘） | ✓ | ✓ | 不变 |
| L\_kl（关系矩阵KL蒸馏） | TP/FP/TN/FN 加权软目标 | STKR/C-STKR 纠正后目标 | **替换** |
| L\_trans（双向MSE变换） | ✓ | ✓ | 不变 |
| L\_cr（对比关系损失） | ✓ | ✓ | 不变 |
| L\_discri（KD判别损失） | ✓ | ✓ | 不变 |
| L\_transx（变换方向损失） | ✓ | ✓ | 不变 |

**总损失公式**：

Baseline：
```
L = L_ce + L_tp + L_kl(TP/FP加权) + L_trans + L_cr + L_discri + L_transx
```

改进框架：
```
L = L_ce + L_tp + L_kl(C-STKR纠正) + L_trans + L_cr + L_discri + L_transx
```

损失项数量不变，仅 L\_kl 的目标分布构建方式发生改变。

---

## 五、训练流程对比

### Baseline 训练流程

```
for t in [1, 2, 3, 4, 5]:
    model_old = deepcopy(model)
    train(model, old_model=model_old)          # 7个损失
    alpha = 1 - |Affin_new - Affin_old|        # 可能越界
    model = alpha*model + (1-alpha)*model_old  # 单模型融合
    save_features(model)
    update_old_gallery(model_trans, alpha)
    evaluate(model)
```

### 改进框架训练流程

```
model_long = None
model_short_old = None

for t in [1, 2, 3, 4, 5]:
    model_old = deepcopy(model)
    old_model_long = model_long if t >= 3 else None

    train(model, old_model=model_old,
          old_model_long=old_model_long)        # 7个损失，L_kl用C-STKR

    if t == 1:
        model_long = deepcopy(model)
        model_short_old = deepcopy(model)
    elif t == 2:
        delta = 1 - mean<R^s, R^l>             # 公式(15)，严格[0,1]
        model = delta*model_long + (1-delta)*model
        model_short_old = deepcopy(model_old)
        model_long = deepcopy(model)
    else:  # t >= 3
        model_long = M*_o(model_short_old,      # 反向搜索最优α
                          model_long, init_loader)
        delta = 1 - mean<R^s, R^l>             # 公式(15)
        model = delta*model_long + (1-delta)*model
        model_short_old = deepcopy(model_old)
        model_long = deepcopy(model)

    save_features(model)
    update_old_gallery(model_trans, delta)
    evaluate(model)
```

---

## 六、计算资源对比

| 资源维度 | Baseline | 改进框架 | 增量 |
|----------|----------|----------|------|
| 训练中模型数 | 2（model + model\_old） | 3（model + model\_old + model\_long） | +1 个冻结模型 |
| 显存占用（估算） | ~23 GB | ~26 GB | +~3 GB |
| 每 batch 前向次数 | 2（new + old） | 3（new + old\_short + old\_long，t≥3） | +1 次无梯度前向 |
| M\*\_o 搜索开销 | 无 | 11次无梯度前向（每步一次） | 约 +30% 步间开销 |
| 可学习参数量 | ResNet50 + 2×TransNet | 同左 | 不变 |
| 存储模型数 | 1 | 2（model\_long + model\_short\_old） | +1 个模型文件 |

**A40 单卡 48GB 显存评估**：改进框架峰值显存约 26-28 GB，远低于 48 GB 上限，完全可行。

---

## 七、创新点汇总

| 编号 | 创新点 | 解决的问题 | 对应代码位置 |
|------|--------|------------|-------------|
| C1 | **STKR 知识纠正蒸馏** | 旧模型错误知识被无选择蒸馏，污染新模型学习 | `trainer.py: stkr_rectify()` |
| C2 | **C-STKR 互补双模型纠正** | 单模型纠正能力有限，长短期模型各有盲区 | `trainer.py: complementary_stkr()` |
| C3 | **公式(15)点积相似度 δ** | 绝对差公式在大域间隔时 δ>1，导致融合权重越界 | `continual_train.py: get_adaptive_alpha()` |
| C4 | **M\*\_o 反向纠正长期模型** | 历史融合权重无法回溯优化，长期模型存在最新域偏置 | `continual_train.py: backward_rectification()` |
| C5 | **长短期双模型状态管理** | 单一旧模型无法同时表达多域通用知识和最新域知识 | `continual_train.py: main_worker()` |

**保留的 Bi-C2R 独特贡献**：

| 编号 | 机制 | 作用 |
|------|------|------|
| B1 | TransNet\_adaptive 双向变换 | 特征空间跨步对齐，LSTKC++ 无此机制 |
| B2 | 特征重索引（trans\_feat） | Re-indexing-free 评估，无需重推理旧数据 |
| B3 | L\_cr + L\_discri + L\_transx | 保证变换网络质量的辅助损失 |

---

## 八、改动文件清单

| 文件 | 改动类型 | 具体内容 |
|------|----------|----------|
| `reid/trainer.py` | 新增方法 + 修改训练逻辑 | 新增 `stkr_rectify`、`complementary_stkr`、`get_affinity`；`train()` 新增 `old_model_long` 参数，替换 `cal_KL` 调用 |
| `continual_train.py` | 替换函数 + 新增函数 + 修改主循环 | 替换 `get_adaptive_alpha`（公式15）；新增 `backward_rectification`（M\*\_o）；`main_worker` 主循环加入长短期模型状态管理 |
| `run1.sh` / `run2.sh` | 更新启动参数 | 指向 GPU2，输出到新目录 `logs-lstkc-setting*/` |

**未改动文件**：`reid/models/resnet.py`、`reid/evaluators.py`、`reid/loss/`、`lreid_dataset/`、`config/`、所有数据处理模块。
