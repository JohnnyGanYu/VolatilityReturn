# 实现计划：H20 96GB 全量训练（v5）

## 概述

在 v4 代码基础上修改 train.py 和 predict.py，移除所有训练采样限制，针对 H20 96GB 环境优化训练速度和内存使用。模型架构不变，推理侧完全兼容 RTX 4090。

核心原则：**只改训练过程，不改模型架构。**

---

## 任务

- [x] 1. 移除序列模型采样限制
  - [x] 1.1 将 `GRU_MAX_TRAIN_SAMPLES` 和 `GRU_MAX_VAL_SAMPLES` 设为 `None`
    - _需求：1.1、1.2_
  - [x] 1.2 将 `TRANSFORMER_MAX_TRAIN_SAMPLES` 和 `TRANSFORMER_MAX_VAL_SAMPLES` 设为 `None`
    - _需求：1.1、1.2_
  - [x] 1.3 修改 `_compute_adaptive_max_samples()` 直接返回 `train_set_size`
    - 保留 VRAM 检测日志输出（仅供参考）
    - _需求：1.3_
  - [x] 1.4 移除大数据集 300,000 条采样上限
    - 在 `train_all_models()` 中移除 `max_samples = min(max_samples, 300000)` 逻辑
    - 保留波动率权重计算用于日志记录
    - _需求：1.4_
  - [x] 1.5 移除小数据集的 `GRU_MAX_TRAIN_SAMPLES` 子采样逻辑
    - _需求：1.1_
  - [x] 1.6 移除验证集的 `GRU_MAX_VAL_SAMPLES` 子采样逻辑
    - _需求：1.2_
  - [x] 1.7 移除 Transformer 的独立子采样逻辑（`TRANSFORMER_MAX_TRAIN_SAMPLES`/`TRANSFORMER_MAX_VAL_SAMPLES`）
    - Transformer 使用与 GRU 相同的全量训练/验证索引
    - _需求：1.1、1.2_

- [x] 2. 移除全局 LightGBM 采样限制
  - [x] 2.1 将 `_build_global_dataset()` 的 `max_per_dataset` 默认值改为 0
    - 保留 `max_per_dataset > 0` 时的采样逻辑（向后兼容）
    - _需求：2.1、2.2_
  - [x] 2.2 合并 3 次 `_build_global_dataset()` 调用为 1 次
    - 第 1 次调用返回 X_train/X_val + y_train_r5/y_val_r5
    - 用轻量循环提取 y_train_r60/y_val_r60（不重建 X 矩阵）
    - 更新 IC 对比逻辑使用统一的 `per_dataset_val_info`
    - _需求：2.3_

- [x] 3. 实现 on-the-fly 滑动窗口构建
  - [x] 3.1 改造 `train_gru_model()` 为 on-the-fly 窗口构建
    - 训练循环中每个 batch 调用 `build_sliding_windows_for_indices()`
    - 验证集窗口仍预先构建
    - 删除预构建全部训练窗口的代码
    - _需求：3.1、3.3_
  - [x] 3.2 改造 `train_transformer_model()` 为 on-the-fly 窗口构建
    - 与 GRU 相同的模式
    - _需求：3.2、3.3_

- [x] 4. 实现训练加速优化
  - [x] 4.1 增大训练 batch_size 至 16384
    - `GRU_BATCH_SIZE = 16384`，`TRANSFORMER_BATCH_SIZE = 16384`
    - _需求：4.1_
  - [x] 4.2 增加 GRU epochs/patience
    - `GRU_EPOCHS = 30`，`GRU_PATIENCE = 7`
    - _需求：4.2_
  - [x] 4.3 增加 Transformer epochs/patience
    - `TRANSFORMER_EPOCHS = 40`，`TRANSFORMER_PATIENCE = 10`
    - _需求：4.3_
  - [x] 4.4 在 `train_gru_model()` 中实现 Mixed Precision (AMP) 训练
    - 使用 `torch.amp.GradScaler` 和 `torch.amp.autocast`
    - 仅在 CUDA 设备上启用
    - _需求：4.4_
  - [x] 4.5 在 `train_transformer_model()` 中实现 Mixed Precision (AMP) 训练
    - _需求：4.4_
  - [x] 4.6 在 `train_gru_model()` 中实现验证集 GPU 预加载
    - 构建验证集窗口后立即 `.to(device)`，释放 CPU 副本
    - 验证循环直接从 GPU 切片，无需 CPU→GPU 传输
    - _需求：4.5_
  - [x] 4.7 在 `train_transformer_model()` 中实现验证集 GPU 预加载
    - _需求：4.5_
  - [x] 4.8 新增 `_batch_predict_gpu()` 辅助函数
    - 接受已在 GPU 上的 tensor，支持 AMP autocast
    - 用于验证集推理
    - _需求：4.5_

- [x] 5. predict.py 向前兼容改动
  - [x] 5.1 修改 Transformer 加载逻辑，从 checkpoint 动态读取架构参数
    - 读取 `d_model`、`nhead`、`num_layers`、`dim_feedforward`、`dropout`
    - 缺失时使用默认值（64, 4, 4, 256, 0.1）
    - _需求：5.6_

- [ ] 6. 验证
  - [ ] 6.1 在 H20 上运行全量训练，验证无 OOM
    - 记录每个数据集的训练样本数、显存使用、训练时间
    - _需求：6.1、6.2、6.4_
  - [ ] 6.2 验证训练产出的模型文件大小与 v4 一致
    - GRU .pt 文件大小应与 v4 相同（架构不变）
    - Transformer .pt 文件大小应与 v4 相同
    - _需求：5.1、5.2、5.3_
  - [ ] 6.3 在 RTX 4090 上运行推理评测，验证推理时间 ≤ 60 分钟
    - _需求：5.4_
  - [ ] 6.4 对比 v5 vs v4 的 IC 指标
    - 期望：更多数据集获得非零的 GRU/Transformer 集成权重
    - 期望：大数据集的序列模型验证 IC 提升
    - _需求：1.1_

---

## 备注

- 任务 1-5 已在 train.py 和 predict.py 中实现完成
- 任务 6（验证）需要在 H20 训练环境和 RTX 4090 推理环境上分别执行
- 所有改动均不影响 factor.py
- 模型架构完全不变，推理侧与 v4 完全兼容
