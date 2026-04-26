# Bi-C2R 训练流程与数据流详细说明

## 目录
1. [整体架构](#整体架构)
2. [训练流程](#训练流程)
3. [数据流](#数据流)
4. [模型管理](#模型管理)
5. [损失函数](#损失函数)
6. [特征保存与加载](#特征保存与加载)
7. [评估流程](#评估流程)

---

## 整体架构

### 核心组件
- **主模型 (model)**: ReID 特征提取网络，基于 ResNet
- **转换网络 (model_trans, model_trans2)**: 特征域适配网络 (TransNet_adaptive)
- **长短期模型状态**:
  - `model_long` (Θ_t^l): 长期模型，多域平衡知识
  - `model_short_old` (Θ_{t-1}^s): 上一步的短期训练模型

### 连续学习设置
- **Setting 1**: market1501 → cuhk_sysu → dukemtmc → msmt17 → cuhk03
- **Setting 2**: dukemtmc → msmt17 → market1501 → cuhk_sysu → cuhk03

---

## 训练流程

### 1. 初始化阶段 ([continual_train.py:55-95](../continual_train.py#L55-L95))

```
加载数据集 → 构建数据加载器 → 初始化模型
```

**关键步骤**:
1. 根据 `args.setting` 确定训练数据集顺序
2. 构建训练集和测试集的数据加载器
3. 初始化主模型和两个转换网络
4. 将模型移至 GPU 并使用 DataParallel

### 2. 连续学习主循环 ([continual_train.py:137-211](../continual_train.py#L137-L211))

对每个数据集 `set_index` (t=1,2,3,...):

#### 2.1 训练前准备
```python
for set_index in range(0, len(training_set)):
    model_old = copy.deepcopy(model)  # 保存当前模型作为旧模型
    
    # 确定 old_model_long 用于 C-STKR
    # t=1: 无旧模型
    # t=2: 单一旧模型 (STKR)
    # t≥3: 短期和长期旧模型 (C-STKR)
    old_model_long_for_train = model_long if set_index >= 2 else None
```

#### 2.2 数据集训练 ([continual_train.py:245-333](../continual_train.py#L245-L333))

**分类器扩展** (set_index > 0):
```python
# 计算累积类别数
add_num = sum([all_train_sets[i][1] for i in range(set_index)])

# 扩展分类器
org_classifier_params = model.module.classifier.weight.data
model.module.classifier = nn.Linear(out_channel, add_num + num_classes, bias=False)
model.module.classifier.weight.data[:add_num].copy_(org_classifier_params)

# 初始化新类别的分类器权重
class_centers = initial_classifier(model, init_loader)
model.module.classifier.weight.data[add_num:].copy_(class_centers)
```

**优化器配置**:
- 第一个数据集 (set_index=0): 只优化主模型
- 后续数据集 (set_index>0): 优化主模型 + 两个转换网络

**训练循环** ([continual_train.py:305-332](../continual_train.py#L305-L332)):
```python
for epoch in range(0, Epochs):
    train_loader.new_epoch()
    trainer.train(epoch, train_loader, optimizer, 
                  training_phase=set_index + 1,
                  train_iters=len(train_loader), 
                  add_num=add_num, 
                  old_model=old_model,
                  old_model_long=old_model_long_for_train)
    
    lr_scheduler.step()
    
    # 保存检查点
    save_checkpoint({...}, fpath='{}_checkpoint.pth.tar'.format(name))
```

#### 2.3 训练后模型管理 ([continual_train.py:149-174](../continual_train.py#L149-L174))

**t=1 (第一个数据集)**:
```python
if set_index == 0:
    model_long = copy.deepcopy(model)
    model_short_old = copy.deepcopy(model)
```

**t=2 (第二个数据集)**:
```python
elif set_index == 1:
    # M*_e 融合: Θ_2^l = α·Θ_1^s + (1-α)·Θ_2^s
    best_alpha = get_adaptive_alpha(args, model, model_long, all_train_sets, set_index)
    model = linear_combination(args, model_long, model, best_alpha)
    
    model_short_old = copy.deepcopy(model_old)  # Θ_1^s
    model_long = copy.deepcopy(model)            # Θ_2^l
```

**t≥3 (后续数据集)**:
```python
else:
    # M*_o: 后向修正更新长期模型
    model_long = backward_rectification(args, model_short_old, model_long, init_loader)
    
    # M*_e: 自适应融合
    best_alpha = get_adaptive_alpha(args, model, model_long, all_train_sets, set_index)
    model = linear_combination(args, model_long, model, best_alpha)
    
    model_short_old = copy.deepcopy(model_old)  # Θ_{t-1}^s
    model_long = copy.deepcopy(model)            # Θ_t^l
```

#### 2.4 特征保存 ([continual_train.py:176-180](../continual_train.py#L176-L180))

```python
features, _ = extract_features(model, test_loader, training_phase=set_index+1)
torch.save({'features': features}, args.logs_dir + name + '_features.pth.tar')
```

#### 2.5 旧特征转换 ([continual_train.py:182-206](../continual_train.py#L182-L206))

如果 `args.trans_feat=True` 且 `set_index > 0`:
```python
for each_data in training_set[0:set_index]:
    # 加载旧特征
    each_old_gallery = torch.load(args.logs_dir + str(each_data) + '_features.pth.tar')
    
    # 使用 model_trans 转换旧特征
    trans_features = model_trans(old_features)
    trans_features_norm = F.normalize(trans_features, p=2, dim=1)
    
    # 融合: α·转换特征 + (1-α)·原始特征
    updated_features = best_alpha * trans_features_norm + (1 - best_alpha) * old_features
    
    # 保存更新后的特征
    torch.save(updated_features, args.logs_dir + str(each_data) + '_features.pth.tar')
```

#### 2.6 测试评估 ([continual_train.py:210-211](../continual_train.py#L210-L211))

```python
if set_index > 0:
    test_model(model, all_train_sets, all_test_only_sets, set_index, 
               logger_res=logger_res, feats_dir=args.logs_dir)
```

---

## 数据流

### 单次训练迭代数据流 ([reid/trainer.py:143-240](../reid/trainer.py#L143-L240))

```
输入图像 (s_inputs)
    ↓
主模型前向传播
    ↓
s_features, bn_feat, cls_outputs, feat_final_layer
    ↓
计算基础损失 (loss_ce + loss_tp)
    ↓
[如果有旧模型] 知识蒸馏分支
    ↓
总损失反向传播
    ↓
参数更新
```

### 知识蒸馏数据流 (set_index > 0)

#### 1. 旧模型特征提取
```python
with torch.no_grad():
    s_features_old, bn_feat_old, cls_outputs_old, feat_final_layer_old = old_model(s_inputs)
```

#### 2. 关系矩阵计算与修正

**STKR (t=2, 单一旧模型)**:
```python
R_old = self.get_affinity(old_model, s_inputs)  # 旧模型关系矩阵
R_tilde = self.stkr_rectify(R_old, targets)     # 修正
```

**C-STKR (t≥3, 互补修正)**:
```python
R_short = self.get_affinity(old_model, s_inputs)       # 短期模型关系矩阵
R_long = self.get_affinity(old_model_long, s_inputs)   # 长期模型关系矩阵
R_tilde = self.complementary_stkr(R_short, R_long, targets)  # 互补修正
```

**关系蒸馏损失**:
```python
Affinity_matrix_new = self.get_normal_affinity(s_features)  # 新模型关系矩阵
divergence = KLDivLoss(log(Affinity_matrix_new), R_tilde)
loss += divergence * AF_weight
```

#### 3. 特征转换与对齐

**双向转换**:
```python
# 旧特征 → 新域
trans_old_features = model_trans(s_features_old)
trans_old_features_norm = F.normalize(trans_old_features, p=2, dim=1)

# 新特征 → 旧域
trans_new_features = model_trans2(s_features)
trans_new_features_norm = F.normalize(trans_new_features, p=2, dim=1)
```

**转换损失 (CA Loss)**:
```python
trans_loss = weight_trans * MSE(trans_old_features_norm, s_features) +
             weight_trans * MSE(trans_new_features_norm, s_features_old)
```

**关系保持损失 (CR Loss)**:
```python
anti_loss = loss_cr(targets, s_features_old, trans_old_features_norm) +
            loss_cr(targets, s_features, trans_new_features_norm)
```

#### 4. 判别损失 (DC Loss)

```python
# 反归一化转换特征
mean_old = s_features_old_origin.mean(dim=-1, keepdim=True)
std_old = s_features_old_origin.std(dim=-1, keepdim=True)
trans_old_features_unnorm = trans_old_features_norm * std_old + mean_old

# 判别损失
discri_loss = weight_discri * MSE(trans_old_features_unnorm, s_features_old_origin)
```

---

## 模型管理

### 模型状态转换图

```
t=1: 训练 D1
    model → model_long, model_short_old

t=2: 训练 D2
    old_model = model (from t=1)
    使用 STKR
    融合: model = α·model_long + (1-α)·model
    model → model_long
    model_old → model_short_old

t≥3: 训练 Dt
    old_model = model (from t-1)
    old_model_long = model_long (from t-1)
    使用 C-STKR
    后向修正: model_long = backward_rectify(model_short_old, model_long)
    融合: model = α·model_long + (1-α)·model
    model → model_long
    model_old → model_short_old
```

### 自适应融合权重计算 ([continual_train.py:219-241](../continual_train.py#L219-L241))

```python
def get_adaptive_alpha(args, model, model_old, all_train_sets, set_index):
    """
    计算 M*_e 融合权重 delta (Eq.15)
    delta = 1 - mean_i(dot(R^s_i, R^l_i))
    
    model: 短期新模型 Θ_t^s
    model_old: 长期旧模型 Θ_t^l
    """
    # 提取特征
    features_new = extract_features_voro(model, init_loader)
    features_old = extract_features_voro(model_old, init_loader)
    
    # 计算关系矩阵
    Affin_new = get_normal_affinity(features_new)  # R^s
    Affin_old = get_normal_affinity(features_old)  # R^l
    
    # 计算相似度
    dot_sim = (Affin_new * Affin_old).sum(dim=1).mean()
    alpha = 1.0 - dot_sim
    alpha = max(0.0, min(1.0, alpha))  # 限制在 [0,1]
    
    return alpha
```

---

## 损失函数

### 总损失组成

**第一个数据集 (t=1)**:
```
Loss = Loss_CE + Loss_Triplet
```

**后续数据集 (t≥2)**:
```
Loss = Loss_CE + Loss_Triplet + Loss_KD + Loss_CA + Loss_CR + Loss_DC
```

### 各损失函数详解

#### 1. 分类损失 (Loss_CE)
- **类型**: CrossEntropyLabelSmooth
- **作用**: 监督学习，分类身份
- **位置**: [reid/trainer.py:177](../reid/trainer.py#L177)

#### 2. 三元组损失 (Loss_Triplet)
- **类型**: TripletLoss
- **作用**: 度量学习，拉近同类样本，推远异类样本
- **位置**: [reid/trainer.py:177](../reid/trainer.py#L177)

#### 3. 知识蒸馏损失 (Loss_KD)
- **类型**: KLDivLoss
- **作用**: 关系知识蒸馏，使用 STKR/C-STKR 修正
- **公式**: `KL(log(R_new), R_tilde)`
- **位置**: [reid/trainer.py:199](../reid/trainer.py#L199)

#### 4. 跨域对齐损失 (Loss_CA)
- **类型**: MSELoss
- **作用**: 特征域转换对齐
- **公式**: 
  ```
  MSE(trans(f_old), f_new) + MSE(trans(f_new), f_old)
  ```
- **位置**: [reid/trainer.py:209-210](../reid/trainer.py#L209-L210)

#### 5. 关系保持损失 (Loss_CR)
- **类型**: KLDivLoss
- **作用**: 保持转换后的关系结构
- **位置**: [reid/trainer.py:213](../reid/trainer.py#L213)

#### 6. 判别损失 (Loss_DC)
- **类型**: MSELoss
- **作用**: 确保转换特征在原始特征空间中可判别
- **位置**: [reid/trainer.py:224-227](../reid/trainer.py#L224-L227)

---

## 特征保存与加载

### 特征文件命名规则
```
{logs_dir}/{dataset_name}_features.pth.tar
```

例如:
- `logs-lstkc-setting1/market1501_features.pth.tar`
- `logs-lstkc-setting1/cuhk_sysu_features.pth.tar`

### 特征保存时机
每个数据集训练完成后立即保存 ([continual_train.py:176-180](../continual_train.py#L176-L180))

### 特征文件结构
```python
{
    'features': OrderedDict({
        (fname, pid, camid): tensor([2048]),  # 特征向量
        ...
    })
}
```

### 特征加载与使用

#### 1. 评估时加载 ([reid/evaluators.py:218-225](../reid/evaluators.py#L218-L225))

```python
def evaluate_rfl(self, ..., old_feat=None):
    # 提取当前特征
    features, _ = extract_features(self.model, data_loader)
    
    # 加载旧特征（带文件存在性检查）
    if old_feat is None or not os.path.exists(old_feat):
        features_old = features  # 第一个数据集，使用当前特征
    else:
        features_old = torch.load(old_feat)['features']
        for key, value in features_old.items():
            features_old[key] = value.data.cpu()
    
    # 计算距离矩阵
    distmat, query_features, gallery_features = pairwise_distance_rfl(
        features, features_old, query, gallery, metric=metric)
```

#### 2. 特征更新 ([continual_train.py:182-206](../continual_train.py#L182-L206))

训练新数据集后，使用 `model_trans` 更新所有旧数据集的特征:
```python
for each_data in training_set[0:set_index]:
    # 加载旧特征
    old_gallery = torch.load(args.logs_dir + each_data + '_features.pth.tar')
    
    # 批量转换
    for batch in batches(old_features):
        trans_features = model_trans(batch)
        trans_features = F.normalize(trans_features, p=2, dim=1)
        updated_features = alpha * trans_features + (1-alpha) * batch
    
    # 保存更新后的特征
    torch.save(updated_features, args.logs_dir + each_data + '_features.pth.tar')
```

---

## 评估流程

### 测试函数 ([continual_train.py:336-389](../continual_train.py#L336-L389))

```python
def test_model(model, all_train_sets, all_test_sets, set_index, 
               logger_res=None, feats_dir=None):
    evaluator = Evaluator(model)
    
    # 1. 评估已见数据集（使用当前特征）
    for i in range(0, set_index + 1):
        dataset, _, _, test_loader, _, name = all_train_sets[i]
        R1, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
    
    # 2. 评估已见数据集（使用旧特征，RFL）
    for i in range(0, set_index + 1):
        dataset, _, _, test_loader, _, name = all_train_sets[i]
        R1, mAP = evaluator.evaluate_rfl(
            test_loader, dataset.query, dataset.gallery,
            old_feat=feats_dir + name + '_features.pth.tar')
    
    # 3. 评估未见数据集
    for test_set in all_test_sets:
        dataset, _, _, test_loader, _, name = test_set
        R1, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
```

### 评估指标
- **Rank-1 (R1)**: 第一个检索结果正确的概率
- **mAP**: 平均精度均值

### 评估时机
- **中间测试**: 每个 epoch 后（如果 `args.middle_test=True`）
- **最终测试**: 每个数据集训练完成后（如果 `set_index > 0`）

---

## 关键流程图

### 完整训练流程

```
初始化
  ↓
for each dataset t:
  ├─ 保存旧模型 (model_old)
  ├─ 扩展分类器（如果 t>1）
  ├─ 训练循环
  │   ├─ 前向传播
  │   ├─ 计算基础损失
  │   ├─ [如果 t>1] 知识蒸馏
  │   │   ├─ STKR/C-STKR 修正
  │   │   ├─ 关系蒸馏损失
  │   │   ├─ 特征转换损失
  │   │   └─ 判别损失
  │   └─ 反向传播 + 更新
  ├─ 模型融合
  │   ├─ [t=1] 初始化 model_long
  │   ├─ [t=2] M*_e 融合
  │   └─ [t≥3] M*_o 后向修正 + M*_e 融合
  ├─ 保存特征
  ├─ 更新旧特征（使用 model_trans）
  └─ 测试评估
```

### 特征流转

```
训练 D1 → 保存 features_D1
训练 D2 → 保存 features_D2 → 更新 features_D1 (使用 trans)
训练 D3 → 保存 features_D3 → 更新 features_D1, features_D2 (使用 trans)
...
```

---

## 常见问题

### Q1: 为什么第一个数据集不调用 test_model？
**A**: 因为 `test_model` 中的 `evaluate_rfl` 需要加载旧特征文件，而第一个数据集训练时还没有旧特征。修复后的代码会检查文件是否存在，如果不存在则使用当前特征。

### Q2: model_trans 和 model_trans2 的区别？
**A**: 
- `model_trans`: 将旧域特征转换到新域
- `model_trans2`: 将新域特征转换到旧域
- 两者共同实现双向特征对齐

### Q3: STKR 和 C-STKR 的区别？
**A**:
- **STKR** (t=2): 使用单一旧模型的关系矩阵进行修正
- **C-STKR** (t≥3): 使用短期和长期两个旧模型的关系矩阵进行互补修正，更鲁棒

### Q4: 为什么需要更新旧特征？
**A**: 训练新数据集后，特征空间发生变化。通过 `model_trans` 将旧特征转换到新的特征空间，确保评估时的公平性和一致性。

---

## 参考文件

- 主训练脚本: [continual_train.py](../continual_train.py)
- 训练器: [reid/trainer.py](../reid/trainer.py)
- 评估器: [reid/evaluators.py](../reid/evaluators.py)
- 损失函数: [reid/utils/make_loss.py](../reid/utils/make_loss.py)
- 模型定义: [reid/models/resnet.py](../reid/models/resnet.py)
