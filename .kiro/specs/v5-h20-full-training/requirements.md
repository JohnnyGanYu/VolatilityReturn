# 需求文档：H20 96GB 全量训练（v5）

## 简介

本文档定义 v5 迭代的需求。v5 基于 v4 的代码基础（165维特征、全局LightGBM、GRU+Transformer三模型集成），针对 H20 96GB GPU 训练环境进行优化：**移除所有训练采样限制，使用全量数据训练，同时保持模型架构不变以确保推理侧完全兼容 RTX 4090 测试环境**。

v4 的评测结果（nR5=0.0860, nR60=0.2211）全面低于 v3（nR5=0.1665, nR60=0.4705），其中一个关键原因是序列模型（GRU/Transformer）在大数据集上仅使用了 15万~30万条采样数据训练，丢失了大量时序信息，导致 30 个数据集中仅 4 个获得了非零的序列模型权重。v5 通过全量训练解决这一问题。

**训练环境**：H20 GPU，96GB 显存，150GB 系统内存，无时间限制。
**推理环境（不可改变）**：RTX 4090，24GB 显存，80GB 内存，16核 CPU，60分钟时间限制，提交包 ≤ 150MB。

---

## 词汇表

- **全量训练**：使用数据集的全部训练样本（前80%时序划分），不做任何子采样
- **On-the-fly window construction**：每个训练 batch 现场构建滑动窗口，而非预先构建全部窗口数组
- **Mixed precision (AMP)**：使用 float16 进行前向/反向传播，float32 维护参数，在 H20 上约 2x 加速
- **Val windows GPU preload**：将验证集滑动窗口预加载到 GPU 显存，避免每 epoch 的 CPU→GPU 传输
- 其余术语沿用 v4 需求文档定义

---

## 需求

### 需求1：移除序列模型训练采样限制

**用户故事：** 作为模型训练工程师，我希望在 H20 96GB 上使用全量训练数据训练 GRU 和 Transformer，让序列模型在大数据集上学到更完整的时序模式，提升序列模型的验证 IC，使更多数据集获得非零的集成权重。

#### 验收标准

1. THE Training_Pipeline SHALL 将 `GRU_MAX_TRAIN_SAMPLES` 和 `TRANSFORMER_MAX_TRAIN_SAMPLES` 设为 `None`，不再限制序列模型的训练样本数。

2. THE Training_Pipeline SHALL 将 `GRU_MAX_VAL_SAMPLES` 和 `TRANSFORMER_MAX_VAL_SAMPLES` 设为 `None`，不再限制序列模型的验证样本数。

3. THE `_compute_adaptive_max_samples()` 函数 SHALL 直接返回 `train_set_size`（全量训练集大小），不再按显存公式计算采样上限。

4. THE Training_Pipeline SHALL 移除大数据集（dataset6-19）的 300,000 条采样上限，使用全量训练数据。

5. THE Training_Pipeline SHALL 保留 GPU OOM 自动减半重试逻辑（最多 2 次）作为安全网，在显存不足时自动降级。

---

### 需求2：移除全局 LightGBM 训练采样限制

**用户故事：** 作为模型训练工程师，我希望全局 LightGBM 训练时使用每个数据集的全量训练数据，而非每个数据集限制 200,000 行，让全局模型学到更完整的跨数据集规律。

#### 验收标准

1. THE `_build_global_dataset()` 函数 SHALL 将 `max_per_dataset` 默认值设为 0（无限制），不再对每个数据集的训练行数做子采样。

2. THE `_build_global_dataset()` 函数 SHALL 在 `max_per_dataset > 0` 时保留原有的均匀间隔采样逻辑，确保向后兼容。

3. THE Training_Pipeline SHALL 将 3 次 `_build_global_dataset()` 调用合并为 1 次，避免重复构建相同的特征矩阵，将全局训练的内存峰值从 ~116 GB 降至 ~77 GB。

---

### 需求3：On-the-fly 滑动窗口构建

**用户故事：** 作为模型训练工程师，我希望序列模型训练时不预先构建全部滑动窗口数组（最大数据集需要 ~80GB），而是每个 batch 现场构建，使系统内存占用保持在可控范围内。

#### 验收标准

1. THE `train_gru_model()` 函数 SHALL 在训练循环中对每个 batch 调用 `build_sliding_windows_for_indices()` 现场构建窗口，不预先构建全部训练窗口。

2. THE `train_transformer_model()` 函数 SHALL 采用与 GRU 相同的 on-the-fly 窗口构建方式。

3. THE 验证集滑动窗口 SHALL 仍然预先构建（验证集为后 20%，内存可控），以加速每 epoch 的验证。

4. THE 训练过程中的系统内存占用 SHALL 保持在单 batch 级别（约 0.6 GB），加上验证集窗口（最大 ~17 GB），总计不超过 25 GB。

---

### 需求4：训练加速优化

**用户故事：** 作为模型训练工程师，我希望充分利用 H20 96GB 的计算能力加速训练，在不改变模型架构的前提下缩短训练时间。

#### 验收标准

1. THE Training_Pipeline SHALL 将 GRU 和 Transformer 的训练 batch_size 从 4096 增大到 16384，利用 H20 的大显存加速训练。

2. THE Training_Pipeline SHALL 将 GRU 的最大 epoch 从 20 增加到 30，patience 从 5 增加到 7，让全量数据有足够的迭代次数收敛。

3. THE Training_Pipeline SHALL 将 Transformer 的最大 epoch 从 30 增加到 40，patience 从 7 增加到 10。

4. THE `train_gru_model()` 和 `train_transformer_model()` SHALL 在 CUDA 设备上启用 Mixed Precision (AMP) 训练，使用 `torch.amp.autocast` 和 `torch.amp.GradScaler`。

5. THE `train_gru_model()` 和 `train_transformer_model()` SHALL 在 CUDA 设备上将验证集滑动窗口预加载到 GPU 显存，避免每 epoch 的 CPU→GPU 数据传输。

6. THE 推理侧的 batch_size（predict.py 中的 `GRU_BATCH_SIZE = 65536`）SHALL 保持不变，不受训练 batch_size 变更影响。

---

### 需求5：模型架构不变约束

**用户故事：** 作为系统工程师，我希望 v5 的所有改动仅影响训练过程，不改变模型架构和推理逻辑，确保在 RTX 4090 测试环境下的推理时间、模型大小和显存需求与 v4 完全一致。

#### 验收标准

1. THE GRU 模型架构 SHALL 保持 hidden_size=64, num_layers=2, dropout=0.1 不变。

2. THE Transformer 模型架构 SHALL 保持 d_model=64, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1 不变。

3. THE 提交包大小 SHALL 不超过 150 MB（平台限制）。模型文件大小与 v4 相同（架构不变）。

4. THE 推理时间 SHALL 不超过 60 分钟（平台限制）。推理逻辑与 v4 完全一致。

5. THE RTX 4090 推理显存 SHALL 不超过 24 GB。推理 batch_size 和模型大小与 v4 相同。

6. THE predict.py SHALL 从 checkpoint 动态读取 Transformer 架构参数（d_model, nhead, num_layers, dim_feedforward, dropout），在参数缺失时使用默认值（64, 4, 4, 256, 0.1），确保向前兼容。

---

### 需求6：资源安全约束

**用户故事：** 作为系统工程师，我希望 v5 的训练过程在 H20 96GB / 150GB RAM 环境下不会 OOM，并有合理的安全余量。

#### 验收标准

1. THE H20 训练 GPU 显存峰值 SHALL 不超过 96 GB。估算值：Transformer 训练 ~23 GB + 验证集窗口 ~17 GB ≈ 40 GB（42%）。

2. THE H20 训练系统内存峰值 SHALL 不超过 150 GB。估算值：全局 LightGBM 训练 ~77 GB（51%）。

3. THE Training_Pipeline SHALL 在每个数据集的 GRU 训练完成后、Transformer 训练开始前，执行 `gc.collect()` 和 `torch.cuda.empty_cache()` 清理 GPU 资源。

4. THE Training_Pipeline SHALL 在训练日志中记录每个数据集的可用显存和实际使用样本数。
