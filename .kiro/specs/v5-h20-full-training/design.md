# 设计文档：H20 96GB 全量训练（v5）

## 概述

v5 在 v4 代码基础上，针对 H20 96GB 训练环境进行优化。核心原则：**只改训练过程，不改模型架构**。所有改动仅影响 train.py 的训练逻辑，predict.py 仅做向前兼容的小改动（动态读取 Transformer 架构参数）。

**改动范围**：
- `train.py`：移除采样限制、增大 batch、增加 epochs、AMP 训练、on-the-fly 窗口、合并全局训练调用
- `predict.py`：Transformer 加载时从 checkpoint 读取架构参数（向前兼容）

**不改动**：
- `factor.py`：特征生成完全不变（165维）
- 模型架构：GRU (h=64, l=2) 和 Transformer (d=64, l=4) 不变
- 推理逻辑：predict.py 的推理流程不变

---

## 架构

### 训练侧变更（train.py）

```
v4 训练流程                          v5 训练流程（H20 优化）
─────────────                        ─────────────────────
GRU/TF 采样 150K~300K 条             → 全量训练（无采样限制）
预构建全部滑动窗口 (~80GB)            → on-the-fly batch-wise 构建 (~0.6GB)
验证集窗口在 CPU                      → 预加载到 GPU (~17GB)
batch_size = 4096                    → batch_size = 16384
fp32 训练                            → Mixed Precision (fp16/fp32)
GRU 20 epochs / patience 5          → 30 epochs / patience 7
TF 30 epochs / patience 7           → 40 epochs / patience 10
全局 LGB: 每数据集限 200K 行          → 无限制（全量）
_build_global_dataset 调用 3 次       → 合并为 1 次
```

### 推理侧（不变）

```
predict.py 推理流程（v4 = v5，完全一致）
─────────────────────────────────────
factor.py → (T, 165) float32
predict.py:
  ├── LightGBM 推理（CPU，batch=全量）
  ├── GRU 推理（GPU，batch=65536）
  └── Transformer 推理（GPU，batch=65536）
→ (T, 2) float32
```

---

## 组件变更详情

### 1. 常量变更（train.py）

| 常量 | v4 值 | v5 值 | 说明 |
|------|-------|-------|------|
| `GRU_BATCH_SIZE` | 4096 | 16384 | 训练专用，推理 batch 不变 |
| `GRU_EPOCHS` | 20 | 30 | 全量数据需要更多迭代 |
| `GRU_PATIENCE` | 5 | 7 | 配合更多 epochs |
| `GRU_MAX_TRAIN_SAMPLES` | 150000 | None | 无限制 |
| `GRU_MAX_VAL_SAMPLES` | 40000 | None | 无限制 |
| `TRANSFORMER_BATCH_SIZE` | 4096 | 16384 | 训练专用 |
| `TRANSFORMER_EPOCHS` | 30 | 40 | 全量数据需要更多迭代 |
| `TRANSFORMER_PATIENCE` | 7 | 10 | 配合更多 epochs |
| `TRANSFORMER_MAX_TRAIN_SAMPLES` | 50000 | None | 无限制 |
| `TRANSFORMER_MAX_VAL_SAMPLES` | 15000 | None | 无限制 |

### 2. `_compute_adaptive_max_samples()` 变更

```python
# v4: 按显存公式计算
max_samples = floor(available_vram * 0.6 / (batch * window * features * 4)) * batch

# v5: 直接返回全量
return train_set_size  # OOM retry 是安全网
```

### 3. `_build_global_dataset()` 变更

```python
# v4: max_per_dataset=200000
# v5: max_per_dataset=0（无限制）
# 条件保留：max_per_dataset > 0 时仍执行采样（向后兼容）
```

### 4. `train_gru_model()` 变更

**v4**：预构建全部训练窗口 `build_sliding_windows_for_indices(features, train_indices)` → ~80GB

**v5**：
```python
# 训练循环中 on-the-fly 构建
for batch_idx in batches:
    batch_windows = build_sliding_windows_for_indices(
        features, train_indices[batch_idx], GRU_WINDOW_SIZE
    )  # 仅 ~0.6 GB

# 验证集预加载到 GPU
val_windows_gpu = torch.from_numpy(val_windows_np).to(device)

# Mixed Precision
scaler = torch.amp.GradScaler("cuda", enabled=True)
with torch.amp.autocast("cuda", enabled=True):
    pred = model(x_batch)
    loss = criterion(pred, y_batch)
scaler.scale(loss).backward()
```

### 5. `train_transformer_model()` 变更

与 GRU 相同的模式：on-the-fly 窗口 + GPU 预加载验证集 + AMP。

### 6. 全局 LightGBM 训练优化

**v4**：调用 `_build_global_dataset` 3 次（第1次取 X，第2次取 y_r5，第3次取 y_r60）

**v5**：调用 1 次取 X + y_r5，然后用轻量循环提取 y_r60（不重建 X 矩阵）

### 7. predict.py 变更

```python
# v5: 从 checkpoint 动态读取架构参数
ckpt_d_model = checkpoint.get("d_model", 64)
ckpt_nhead = checkpoint.get("nhead", 4)
ckpt_num_layers = checkpoint.get("num_layers", 4)
ckpt_dim_ff = checkpoint.get("dim_feedforward", 256)
ckpt_dropout = checkpoint.get("dropout", 0.1)

model = _TransformerPredictor(
    input_size=input_size, d_model=ckpt_d_model, nhead=ckpt_nhead,
    num_layers=ckpt_num_layers, dim_feedforward=ckpt_dim_ff,
    dropout=ckpt_dropout, max_seq_len=window_size,
)
```

默认值与 v4 架构一致，确保加载旧模型时行为不变。

---

## 资源估算

### H20 训练环境（96GB VRAM / 150GB RAM）

| 环节 | GPU 显存峰值 | 系统内存峰值 |
|------|-------------|-------------|
| GRU 训练（最大数据集） | ~5 GB + 17 GB val = 22 GB | ~20 GB |
| Transformer 训练（最大数据集） | ~23 GB + 17 GB val = 40 GB | ~20 GB |
| 全局 LightGBM 训练 | 0 GB | ~77 GB |
| **峰值** | **~40 GB / 96 GB (42%)** | **~77 GB / 150 GB (51%)** |

### RTX 4090 推理环境（不变）

| 指标 | 值 | 限制 |
|------|-----|------|
| 提交包大小 | ~99 MB | ≤ 150 MB |
| 推理时间 | ~26 min | ≤ 60 min |
| GPU 显存 | < 4 GB | ≤ 24 GB |

---

## 正确性属性

### 属性1：模型架构不变量

*对任意* v5 训练产出的 GRU 模型文件，其 hidden_size=64, num_layers=2；*对任意* Transformer 模型文件，其 d_model=64, nhead=4, num_layers=4, dim_feedforward=256。

**验证：需求 5.1、5.2**

### 属性2：推理输出等价性

*对任意* 数据集和相同的模型文件，v5 的 predict.py 产出的信号与 v4 的 predict.py 产出的信号完全一致（bit-exact）。

**验证：需求 5.4、5.6**

### 属性3：训练内存安全性

*对任意* 数据集（包括最大的 dataset11，224万行），v5 的训练过程中系统内存峰值不超过 150 GB，GPU 显存峰值不超过 96 GB。

**验证：需求 6.1、6.2**

### 属性4：全量训练不变量

*对任意* 数据集，v5 训练时实际使用的训练样本数等于该数据集前 80% 有效样本数（不做任何子采样），除非 OOM 重试触发了自动减半。

**验证：需求 1.1、1.2、1.3、1.4**

---

## 错误处理

| 场景 | 处理方式 |
|------|---------|
| GRU 训练 GPU OOM | 自动将训练样本减半，最多重试 2 次 |
| Transformer 训练 GPU OOM | 自动将训练样本减半，最多重试 2 次 |
| 显存检测失败 | 记录日志，继续使用全量数据 |
| AMP 产生 NaN loss | GradScaler 自动跳过该 step，不影响训练 |
| 验证集窗口无法加载到 GPU | fallback 到 CPU 验证（逐 batch 传输） |
