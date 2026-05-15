# 需求文档：模型优化 v6

## 简介

本文档定义 v6 迭代的需求。v6 基于 v5 的代码基础（165维特征、LightGBM + GRU + Transformer 四模型集成），在五个方向进行优化提升：

1. **IC-aware 损失函数**：序列模型（GRU/Transformer）将 MSE 损失替换为排序/相关性损失，直接优化评测指标 IC
2. **更大容量 GRU**：hidden_size 从 64 扩大到 128（保持 num_layers=2），提升序列模型表达能力
3. **双窗口策略**：Ret5 使用 20 步窗口（短期模式），Ret60 使用 240 步窗口（长期趋势）
4. **极端行情专用 LightGBM**：为极端行情区间训练独立模型，利用极端行情更可预测的特性
5. **迭代权重优化**：支持训练一次、多次优化权重的工作流（本地验证 + 平台反馈迭代）

**v5 基线**：nR5=0.1341, nR60=0.4101, eR5=0.2929, eR60=0.6089，推理 8 分钟，提交包 99 MB。

**训练环境**：H20 GPU，96GB 显存，150GB 系统内存，无时间限制。
**推理环境（不可改变）**：RTX 4090，24GB 显存，80GB 内存，16核 CPU，60 分钟时间限制，提交包 ≤ 150 MB，特征 ≤ 512 维。

---

## 词汇表

- **IC (Information Coefficient)**：Pearson 相关系数，评测指标，衡量预测信号与实际收益的线性相关性
- **IC-aware Loss**：直接优化 IC 的损失函数，包括 Pearson 相关损失（1 - corr）和 ListMLE 排序损失
- **Pearson_Correlation_Loss**：1 - pearson_correlation(pred, target)，可微分的相关性损失
- **ListMLE_Loss**：基于 Plackett-Luce 模型的列表级排序损失，优化预测值的排序一致性
- **Extreme_Regime**：极端行情区间，由收益率绝对值超过阈值（如 2σ）定义的时间段
- **Normal_Regime**：普通行情区间，非极端行情的时间段
- **Dual_Window**：双窗口策略，Ret5 和 Ret60 使用不同长度的滑动窗口输入序列模型
- **Reoptimize_Script**：权重重优化脚本（reoptimize_v6.py），支持在不重新训练模型的情况下多次调整集成权重
- **Submission_Package**：提交包，包含所有模型文件和代码的压缩包，平台限制 ≤ 150 MB
- **Training_Pipeline**：训练流水线（train.py），负责离线训练所有模型
- **Inference_Pipeline**：推理流水线（predict.py），负责在评测平台上生成预测信号
- **Ensemble_Weights**：集成权重，控制各模型预测信号的加权比例
- **GRU_Model**：门控循环单元序列模型
- **Transformer_Model**：基于自注意力的序列模型
- **LGB_Local**：每个数据集独立训练的 LightGBM 模型
- **LGB_Extreme**：仅在极端行情区间训练的 LightGBM 模型

---

## 需求

### 需求1：IC-aware 损失函数

**用户故事：** 作为模型训练工程师，我希望序列模型（GRU/Transformer）使用直接优化 IC 的损失函数替代 MSE，使模型训练目标与评测指标对齐，提升验证集 IC。

#### 验收标准

1. THE Training_Pipeline SHALL 实现 Pearson_Correlation_Loss 函数，计算公式为 `loss = 1 - pearson_correlation(pred, target)`，其中 pearson_correlation 为批次内预测值与目标值的 Pearson 相关系数。

2. THE Training_Pipeline SHALL 实现 ListMLE_Loss 函数，基于 Plackett-Luce 概率模型计算列表级排序损失，优化预测值排序与目标值排序的一致性。

3. THE Training_Pipeline SHALL 使用组合损失 `loss = α * Pearson_Correlation_Loss + (1-α) * ListMLE_Loss` 训练 GRU_Model 和 Transformer_Model，其中 α 默认值为 0.5。

4. WHEN 批次内目标值方差为零（所有目标值相同），THE Pearson_Correlation_Loss SHALL 返回 0.0（跳过该批次的相关性计算），避免除零错误。

5. THE Training_Pipeline SHALL 在训练日志中记录每个 epoch 的 IC-aware loss 值和验证集 IC 值，用于监控训练收敛情况。

6. THE Training_Pipeline SHALL 保留 MSE 损失作为可选配置（通过 `LOSS_TYPE` 参数切换），确保可回退到 v5 行为。

7. THE Training_Pipeline SHALL 在使用 IC-aware Loss 时将学习率从 v5 的默认值降低 50%（IC-aware loss 梯度波动较大），并启用梯度裁剪（max_norm=1.0）。

8. THE Pearson_Correlation_Loss 和 ListMLE_Loss SHALL 对 NaN 目标值进行掩码处理：仅在非 NaN 样本上计算损失，WHEN 批次内非 NaN 样本数少于 32，SHALL 跳过该批次。

---

### 需求2：更大容量 GRU 模型

**用户故事：** 作为模型训练工程师，我希望将 GRU 的 hidden_size 从 64 扩大到 128，在保持文件大小可控（~1MB/模型）的前提下提升序列模型的表达能力。

#### 验收标准

1. THE Training_Pipeline SHALL 将 GRU_Model 的 hidden_size 从 64 增大到 128，num_layers 保持为 2。

2. THE Inference_Pipeline SHALL 从 checkpoint 文件动态读取 `hidden_size` 参数，在参数缺失时使用默认值 64，确保向前兼容 v5 模型。

3. THE 单个 GRU_Model checkpoint 文件大小 SHALL 不超过 1.5 MB（估算：128*165*2层 + 128*128*2层 + 128*2 ≈ 1.0 MB）。

4. THE Inference_Pipeline SHALL 在 RTX 4090 24GB 上以 batch_size=32768 运行 GRU_Model（hidden_size=128）时，GPU 显存占用不超过 6 GB。

5. THE Training_Pipeline SHALL 在 checkpoint 中保存 `hidden_size=128` 元数据，供推理侧动态构建模型。

---

### 需求3：双窗口策略

**用户故事：** 作为模型训练工程师，我希望 Ret5 预测使用 20 步短窗口（捕捉短期微观结构），Ret60 预测使用 240 步长窗口（捕捉长期趋势），使窗口长度与预测目标的时间尺度匹配。

#### 验收标准

1. THE Training_Pipeline SHALL 为每个数据集训练两组 GRU_Model：一组使用 window_size=20 用于 Ret5 预测，一组使用 window_size=240 用于 Ret60 预测。

2. THE Training_Pipeline SHALL 为每个数据集训练两组 Transformer_Model：一组使用 window_size=20 用于 Ret5 预测，一组使用 window_size=240 用于 Ret60 预测。

3. THE checkpoint 文件命名 SHALL 遵循格式 `gru_ret5_{dataset_name}.pt`（20步窗口）和 `gru_ret60_{dataset_name}.pt`（240步窗口），Transformer 同理。

4. THE Inference_Pipeline SHALL 根据预测目标分别加载对应窗口的模型：Ret5 预测使用 `*_ret5_*.pt` 模型，Ret60 预测使用 `*_ret60_*.pt` 模型。

5. THE Inference_Pipeline SHALL 为 Ret5 模型构建 20 步滑动窗口，为 Ret60 模型构建 240 步滑动窗口，分别进行推理。

6. THE checkpoint 文件 SHALL 保存 `window_size` 元数据（20 或 240），供推理侧验证窗口配置一致性。

7. WHEN 数据集长度小于 240 行，THE Inference_Pipeline SHALL 对 Ret60 序列模型输出零值（窗口不足），仅使用 LightGBM 预测。

---

### 需求4：极端行情专用 LightGBM 模型

**用户故事：** 作为模型训练工程师，我希望为极端行情区间训练独立的 LightGBM 模型，利用极端行情更可预测的特性（eR60 远高于 nR60），在极端行情时段提供更精准的预测。

#### 验收标准

1. THE Training_Pipeline SHALL 为每个数据集的 Ret5 和 Ret60 分别训练极端行情专用模型（LGB_Extreme），仅使用极端行情区间的样本。

2. THE Training_Pipeline SHALL 使用滚动窗口标准差的 2 倍作为极端行情阈值：当 |Ret| > 2 * rolling_std(Ret, window=60) 时，标记该时刻为极端行情。

3. THE LGB_Extreme 模型文件命名 SHALL 遵循格式 `lgb_extreme_ret5_{dataset_name}.txt` 和 `lgb_extreme_ret60_{dataset_name}.txt`。

4. THE Inference_Pipeline SHALL 在推理时使用严格因果方式检测极端行情：基于已观测的历史 OHLCV 数据计算滚动波动率（rolling_std(close_return, window=60)），仅使用当前时刻之前的数据，不引入未来信息。

5. THE Inference_Pipeline SHALL 对每个时刻计算极端行情指示信号（0 或 1），当检测到极端行情时，将 LGB_Extreme 预测以 `w_extreme` 权重融合到最终预测中。

6. THE Ensemble_Weights SHALL 扩展为包含极端行情模型权重：`ret5_w_extreme`、`ret60_w_extreme`，默认值为 0.0（不使用）。

7. IF 某数据集的极端行情样本数少于 1000 条，THEN THE Training_Pipeline SHALL 跳过该数据集的 LGB_Extreme 训练，不生成极端行情模型文件。

8. THE 全部 LGB_Extreme 模型文件总大小 SHALL 不超过 30 MB（30 数据集 × 2 目标 × ~500 KB/模型）。

9. THE LGB_Extreme 训练 SHALL 使用与 LGB_Local 相同的 IC 早停策略，num_boost_round 上限为 500（极端样本较少，防止过拟合）。

---

### 需求5：迭代权重优化工作流

**用户故事：** 作为模型训练工程师，我希望在模型训练完成后，能够多次重新优化集成权重（无需重新训练模型），支持本地验证集优化和平台反馈迭代两种模式，快速找到最优权重组合。

#### 验收标准

1. THE Reoptimize_Script SHALL 支持 `--mode local` 模式：基于本地验证集（后 20% 时序数据）的 IC 进行网格搜索或贝叶斯优化，输出最优权重到 `ensemble_weights.json`。

2. THE Reoptimize_Script SHALL 支持 `--mode feedback` 模式：读取平台反馈结果（`feedback_state/` 目录下的 JSON 文件），基于实际平台 IC 调整权重。

3. THE Reoptimize_Script SHALL 在优化时考虑所有模型类型的权重：LGB_Local、LGB_Extreme、GRU_Ret5、GRU_Ret60、Transformer_Ret5、Transformer_Ret60。

4. THE Reoptimize_Script SHALL 在输出权重前验证约束：所有权重 ≥ 0，且同一目标的权重之和为 1.0。

5. THE Reoptimize_Script SHALL 支持 `--prune` 选项：将权重为 0 的模型从提交包中删除，减小提交包体积。

6. WHEN `--prune` 选项启用，THE Reoptimize_Script SHALL 生成 `submission_manifest.json`，列出需要包含在提交包中的所有模型文件。

7. THE Reoptimize_Script SHALL 在每次优化后输出预估提交包大小，WHEN 预估大小超过 144 MB，THE Reoptimize_Script SHALL 发出警告。

---

### 需求6：提交包大小约束

**用户故事：** 作为系统工程师，我希望 v6 的所有模型文件总大小控制在 150 MB 以内（平台硬限制），并留有安全余量。

#### 验收标准

1. THE Submission_Package 总大小 SHALL 不超过 150 MB（平台硬限制）。

2. THE Submission_Package 预估组成 SHALL 为：LGB_Local ~99 MB + LGB_Extreme ~30 MB + 选中的 GRU/Transformer ~15 MB ≈ 144 MB。

3. THE Training_Pipeline SHALL 在训练完成后输出所有模型文件的大小汇总报告，标注总大小和剩余余量。

4. IF 总模型大小超过 144 MB，THEN THE Reoptimize_Script SHALL 自动裁剪权重最低的序列模型文件，直到总大小降至 144 MB 以下。

5. THE GRU_Model（hidden_size=128, num_layers=2）单个 checkpoint 文件大小 SHALL 不超过 1.5 MB。

6. THE Transformer_Model（d_model=64, num_layers=4）单个 checkpoint 文件大小 SHALL 不超过 1.0 MB。

---

### 需求7：推理时间约束

**用户故事：** 作为系统工程师，我希望 v6 的推理时间在 RTX 4090 上不超过 60 分钟（平台硬限制），并保持合理余量。

#### 验收标准

1. THE Inference_Pipeline 在 30 个数据集上的总推理时间 SHALL 不超过 60 分钟（平台硬限制）。

2. THE Inference_Pipeline 对单个数据集的推理时间 SHALL 不超过 3 分钟（v5 实测最大单数据集 ~30 秒）。

3. WHEN 数据集行数超过 3,000,000 且序列模型权重 ≤ 0.2，THE Inference_Pipeline SHALL 跳过序列模型推理（沿用 v5 安全逻辑）。

4. THE Inference_Pipeline SHALL 对 Ret5（20步窗口）和 Ret60（240步窗口）的序列模型分别推理，240 步窗口的推理时间约为 20 步窗口的 2-3 倍（窗口更大但 batch 数相同）。

5. THE Inference_Pipeline 预估总推理时间 SHALL 不超过 15 分钟（v5 实测 8 分钟，双窗口增加约 50-80% 序列模型推理时间）。

---

### 需求8：推理显存约束

**用户故事：** 作为系统工程师，我希望 v6 的推理过程在 RTX 4090 24GB 上不会 OOM，即使使用更大的 GRU 和更长的 240 步窗口。

#### 验收标准

1. THE Inference_Pipeline 在 RTX 4090 上的 GPU 显存峰值 SHALL 不超过 24 GB。

2. THE Inference_Pipeline SHALL 对 240 步窗口的序列模型使用较小的 batch_size（16384），对 20 步窗口保持 batch_size=32768。

3. THE Inference_Pipeline SHALL 在每个序列模型推理完成后执行 `del model` 和 `torch.cuda.empty_cache()`，释放 GPU 资源。

4. THE 240 步窗口 + GRU(h=128) 的单 batch 显存占用 SHALL 不超过 4 GB（估算：16384 × 240 × 165 × 4B ≈ 2.5 GB 输入 + 模型参数 ~4 MB + 中间状态 ~1 GB）。

5. THE Inference_Pipeline SHALL 在推理开始前检查可用 GPU 显存，WHEN 可用显存低于 8 GB，SHALL 将 batch_size 减半。

---

### 需求9：训练环境资源约束

**用户故事：** 作为系统工程师，我希望 v6 的训练过程在 H20 96GB / 150GB RAM 环境下不会 OOM，考虑到双窗口和更大 GRU 带来的额外资源需求。

#### 验收标准

1. THE H20 训练 GPU 显存峰值 SHALL 不超过 96 GB。

2. THE H20 训练系统内存峰值 SHALL 不超过 150 GB。

3. THE Training_Pipeline SHALL 对 240 步窗口的训练使用 on-the-fly 窗口构建（不预先构建全部窗口），单 batch 内存占用不超过 2 GB。

4. THE Training_Pipeline SHALL 在 GRU_Ret5 和 GRU_Ret60 训练之间执行 GPU 资源清理（`gc.collect()` + `torch.cuda.empty_cache()`）。

5. THE Training_Pipeline SHALL 在训练日志中记录每个模型的 GPU 显存峰值和系统内存使用量。

6. IF 训练过程中发生 GPU OOM，THEN THE Training_Pipeline SHALL 自动将 batch_size 减半并重试（最多 2 次）。

---

### 需求10：向前兼容性

**用户故事：** 作为系统工程师，我希望 v6 的推理代码能够同时兼容 v5 和 v6 的模型文件格式，在 v6 模型不存在时自动回退到 v5 行为。

#### 验收标准

1. THE Inference_Pipeline SHALL 在 `gru_ret5_{dataset_name}.pt` 不存在时，尝试加载 `gru_{dataset_name}.pt`（v5 格式），使用其输出同时作为 Ret5 和 Ret60 的 GRU 预测。

2. THE Inference_Pipeline SHALL 在 `lgb_extreme_*` 文件不存在时，将极端行情模型权重视为 0，仅使用基础模型预测。

3. THE Ensemble_Weights JSON 格式 SHALL 向后兼容 v5 格式（`ret5_w_local`, `ret5_w_global`, `ret5_w_gru`, `ret5_w_tf`），新增字段（`ret5_w_extreme`, `ret5_w_gru_ret5`, `ret60_w_gru_ret60` 等）为可选。

4. THE Inference_Pipeline SHALL 从 checkpoint 动态读取所有模型架构参数（hidden_size, window_size, d_model 等），在参数缺失时使用 v5 默认值。

5. THE Inference_Pipeline SHALL 在加载任何模型文件失败时输出警告日志并回退到零值预测，不中断整体推理流程。

---

### 需求11：端到端本地验证

**用户故事：** 作为模型训练工程师，我希望有一个完整的本地验证流程，能够在提交前验证 v6 所有新功能的正确性和性能指标。

#### 验收标准

1. THE evaluate_local.py SHALL 支持 v6 的所有新模型类型（LGB_Extreme、双窗口 GRU/Transformer），正确加载并评测。

2. THE evaluate_local.py SHALL 分别报告普通行情 IC（nR5, nR60）和极端行情 IC（eR5, eR60），与平台评测指标一致。

3. THE evaluate_local.py SHALL 报告每个数据集的推理时间和 GPU 显存峰值，用于预估平台运行情况。

4. THE evaluate_local.py SHALL 在评测完成后输出提交包大小检查结果，WHEN 超过 144 MB 发出警告，超过 150 MB 报错。

5. THE Training_Pipeline SHALL 在全部训练完成后自动运行 evaluate_local.py，输出完整的 v6 评测报告。

---

### 需求12：训练时间预期

**用户故事：** 作为模型训练工程师，我希望了解 v6 双窗口 + 更大 GRU 带来的训练时间增长，确保在 H20 上的总训练时间可接受。

#### 验收标准

1. THE Training_Pipeline 在 H20 上的总训练时间 SHALL 不超过 48 小时（v5 约 8-12 小时，v6 增加 4 个序列模型方向 + 极端 LGB）。

2. THE Training_Pipeline SHALL 支持断点续训：在训练中断后，能够跳过已完成的数据集/模型，从中断点继续。

3. THE Training_Pipeline SHALL 按以下顺序训练以优化资源利用：LGB_Local → LGB_Extreme → GRU_Ret5(20步) → GRU_Ret60(240步) → Transformer_Ret5(20步) → Transformer_Ret60(240步)。

4. THE Training_Pipeline SHALL 在每个模型训练完成后立即保存 checkpoint，不等待全部训练结束。

5. THE Training_Pipeline SHALL 在训练日志中记录每个数据集每个模型的训练耗时，用于后续优化。
