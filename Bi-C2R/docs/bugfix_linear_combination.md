# Bug 修复：linear_combination 参数顺序错误

## 报错信息

```
RuntimeError: The size of tensor a (500) must match the size of tensor b (1000) at non-singleton dimension 0
```

发生位置：[continual_train.py:417](../continual_train.py#L417)

## 报错原因

### 问题分析

在 `linear_combination` 函数中，参数顺序的假设是：
- **第一个参数 `model`**：新模型（分类器类别数更多）
- **第二个参数 `model_old`**：旧模型（分类器类别数较少）

函数逻辑：
```python
def linear_combination(args, model, model_old, alpha, model_old_id=-1):
    model_old_state_dict = model_old.state_dict()
    model_state_dict = model.state_dict()
    
    model_new = copy.deepcopy(model)  # 基于第一个参数创建新模型
    model_new_state_dict = model_new.state_dict()
    
    for k, v in model_state_dict.items():
        if model_old_state_dict[k].shape == v.shape:
            # 形状相同：直接融合
            model_new_state_dict[k] = alpha * v + (1 - alpha) * model_old_state_dict[k]
        else:
            # 形状不同：假设第一个参数更大，用旧模型权重填充前 num_class_old 个
            num_class_old = model_old_state_dict[k].shape[0]
            model_new_state_dict[k][:num_class_old] = alpha * v[:num_class_old] + (1 - alpha) * model_old_state_dict[k]
```

### 错误的调用

在 [continual_train.py:159](../continual_train.py#L159) 和 [continual_train.py:171](../continual_train.py#L171)：

```python
# 错误：参数顺序颠倒
model = linear_combination(args, model_long, model, best_alpha)
```

**实际情况**：
- `model_long`：500 类（第一个数据集）
- `model`：1000 类（累积了两个数据集）

**导致的问题**：
1. 函数基于 `model_long`（500 类）创建新模型
2. 尝试执行 `model_new_state_dict[k][:500] = ...`
3. 但 `model_old_state_dict[k]` 有 1000 维
4. 导致维度不匹配错误

## 修复方案

### 修复 1：M*_e 融合（t=2）

**位置**：[continual_train.py:159](../continual_train.py#L159)

**修改前**：
```python
model = linear_combination(args, model_long, model, best_alpha)
```

**修改后**：
```python
model = linear_combination(args, model, model_long, best_alpha)
```

### 修复 2：M*_e 融合（t≥3）

**位置**：[continual_train.py:171](../continual_train.py#L171)

**修改前**：
```python
model = linear_combination(args, model_long, model, best_alpha)
```

**修改后**：
```python
model = linear_combination(args, model, model_long, best_alpha)
```

## 融合公式说明

修复后的融合公式为：

```
model_fused = α * model + (1 - α) * model_long
```

其中：
- `model`：当前训练的短期模型 Θ_t^s（新知识）
- `model_long`：长期模型 Θ_t^l（旧知识）
- `α`：融合权重（delta），由 `get_adaptive_alpha` 计算

**融合权重的含义**：
- `α` 接近 1：更多保留新模型（域差距小，新知识可靠）
- `α` 接近 0：更多保留旧模型（域差距大，需要更多旧知识）

## 验证

修复后，融合逻辑正确：
1. 基于 `model`（1000 类）创建新模型
2. 对于前 500 个类别：融合新旧模型的权重
3. 对于后 500 个类别：保留新模型的权重（旧模型没有这些类别）

## 相关代码

- 融合函数：[continual_train.py:401-419](../continual_train.py#L401-L419)
- 自适应权重计算：[continual_train.py:219-241](../continual_train.py#L219-L241)
- M*_e 融合（t=2）：[continual_train.py:154-161](../continual_train.py#L154-L161)
- M*_e 融合（t≥3）：[continual_train.py:168-171](../continual_train.py#L168-L171)

## 注意事项

`backward_rectification` 函数中的调用（[continual_train.py:436](../continual_train.py#L436) 和 [continual_train.py:460](../continual_train.py#L460)）在 t≥3 时，`model_long_old` 和 `model_short_old` 应该有相同的分类器大小，因此不会出现此问题。但为了保持一致性，建议也检查这些调用的参数顺序。
