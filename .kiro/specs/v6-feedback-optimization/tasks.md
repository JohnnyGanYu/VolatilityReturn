# 实施计划：模型优化 v6

## 概述

基于 v5 代码基础，按训练流水线阶段顺序实施五大优化方向：IC-aware 损失函数、GRU h=128、双窗口训练、极端行情 LGB、推理流水线更新、权重优化脚本、评估验证、提交打包。所有改动集中在 `train.py`、`predict.py`、`reoptimize_v6.py`（新增）和 `evaluate_local.py`。

## 任务

- [x] 1. 实现 IC-aware 损失函数（train.py）
  - [x] 1.1 实现 Pearson Correlation Loss 函数
    - 在 `train.py` 中新增 `pearson_correlation_loss(pred, target, mask)` 函数
    - 计算公式：`loss = 1 - pearson_corr(pred, target)`
    - 处理边界条件：有效样本 < 32 时返回 0.0，目标方差为零时返回 0.0
    - 使用 log-sum-exp trick 保证数值稳定性
    - _需求：1.1、1.4、1.8_

  - [x] 1.2 实现 ListMLE Loss 函数
    - 在 `train.py` 中新增 `listmle_loss(pred, target, mask)` 函数
    - 基于 Plackett-Luce 模型：按 target 降序排列，计算列表级排序损失
    - 使用 log-sum-exp trick 防止 exp 溢出
    - 有效样本 < 32 时返回 0.0
    - _需求：1.2、1.8_

  - [x] 1.3 实现组合损失 `ic_aware_loss` 及训练配置切换
    - 新增 `ic_aware_loss(pred, target, mask, alpha=0.5)` 组合函数
    - 新增 `LOSS_TYPE` 配置参数（"mse" / "ic_aware"），保留 MSE 回退能力
    - IC-aware 模式下学习率降为 5e-4，启用梯度裁剪 max_norm=1.0
    - 新增 `IC_LOSS_ALPHA`、`IC_LOSS_MIN_SAMPLES` 配置常量
    - 修改 `train_gru_model` 和 `train_transformer_model` 使用新损失函数
    - _需求：1.3、1.5、1.6、1.7_

  - [ ]* 1.4 编写 IC-aware 损失函数属性测试
    - **属性 1：损失函数数学正确性** — 生成随机 pred/target 张量，验证组合损失等于 α*(1-pearson) + (1-α)*listmle
    - **验证：需求 1.1、1.2、1.3**

  - [ ]* 1.5 编写 NaN 安全损失计算属性测试
    - **属性 2：NaN 安全损失计算** — 生成含 NaN 的批次，验证掩码处理正确、不产生 NaN/Inf
    - **验证：需求 1.4、1.8**

- [x] 2. GRU h=128 容量升级（train.py）
  - [x] 2.1 修改 GRU 模型配置和 checkpoint 保存
    - 将 `GRU_HIDDEN_SIZE` 常量从 64 改为 128
    - 修改 `GRUPredictor.__init__` 使用新 hidden_size
    - 在 checkpoint 保存中新增 `hidden_size` 元数据字段
    - 验证单模型 checkpoint 大小 ≤ 1.5 MB
    - _需求：2.1、2.3、2.5_

  - [ ]* 2.2 编写 GRU checkpoint 大小单元测试
    - 构建 h=128, l=2 模型，保存 checkpoint，验证文件大小 < 1.5 MB
    - 验证 checkpoint 包含 hidden_size=128 元数据
    - _需求：2.3、2.5_

- [x] 3. 双窗口训练策略（train.py）
  - [x] 3.1 实现 Ret5 专用序列模型训练（window=20）
    - 修改 `train_gru_model` 支持单目标输出（output_dim=1）
    - 新增 Ret5 训练流程：window_size=20，仅预测 Ret5
    - 模型文件命名：`gru_ret5_{dataset}.pt`、`transformer_ret5_{dataset}.pt`
    - checkpoint 保存 `window_size=20` 和 `target="ret5"` 元数据
    - _需求：3.1、3.2、3.3、3.6_

  - [x] 3.2 实现 Ret60 专用序列模型训练（window=240）
    - 新增 Ret60 训练流程：window_size=240，仅预测 Ret60
    - 模型文件命名：`gru_ret60_{dataset}.pt`、`transformer_ret60_{dataset}.pt`
    - checkpoint 保存 `window_size=240` 和 `target="ret60"` 元数据
    - 使用 on-the-fly 窗口构建避免内存溢出（单 batch ≤ 2 GB）
    - _需求：3.1、3.2、3.3、3.6、9.3_

  - [x] 3.3 修改 `train_all_models` 训练顺序集成双窗口
    - 训练顺序：LGB_Local → LGB_Extreme → GRU_Ret5 → GRU_Ret60 → TF_Ret5 → TF_Ret60
    - 每个序列模型训练完成后执行 `gc.collect()` + `torch.cuda.empty_cache()`
    - 每个模型训练完成后立即保存 checkpoint
    - 在训练日志中记录每个模型的训练耗时和 GPU 显存峰值
    - _需求：9.4、9.5、12.3、12.4、12.5_

  - [ ]* 3.4 编写双窗口路由正确性属性测试
    - **属性 3：双窗口模型路由正确性** — 验证 Ret5 仅使用 *_ret5_* 模型和 20 步窗口，Ret60 仅使用 *_ret60_* 模型和 240 步窗口
    - **验证：需求 3.3、3.4、3.5**

- [x] 4. 极端行情 LightGBM 训练（train.py）
  - [x] 4.1 实现因果极端行情检测函数
    - 新增 `detect_extreme_regime(close_prices, window=60, threshold_mult=2.0)` 函数
    - 算法：计算 log return → rolling_std(window=60) → |ret| > 2*rolling_std 标记为极端
    - 严格因果：仅使用 t 时刻之前的数据
    - _需求：4.2、4.4_

  - [x] 4.2 实现 LGB_Extreme 训练逻辑
    - 使用极端行情掩码筛选训练样本
    - 极端样本 < 1000 条时跳过训练，不生成模型文件
    - 使用 IC 早停策略，num_boost_round 上限 500
    - 文件命名：`lgb_extreme_ret5_{dataset}.txt`、`lgb_extreme_ret60_{dataset}.txt`
    - 集成到 `train_all_models` 的 LGB 训练阶段
    - _需求：4.1、4.3、4.7、4.8、4.9_

  - [ ]* 4.3 编写极端行情检测属性测试
    - **属性 5：极端行情样本过滤正确性** — 验证掩码精确标记 |ret| > 2*rolling_std 的时刻
    - **验证：需求 4.1、4.2**

  - [ ]* 4.4 编写因果性属性测试
    - **属性 6：因果极端行情推理** — 验证 t 时刻输出仅依赖 t 之前数据
    - **验证：需求 4.4、4.5**

  - [ ]* 4.5 编写极端训练跳过阈值属性测试
    - **属性 7：极端训练跳过阈值** — 验证极端样本 < 1000 时跳过训练
    - **验证：需求 4.7**

- [x] 5. 检查点 - 训练流水线验证
  - 确保所有训练侧代码通过测试，如有疑问请询问用户。

- [x] 6. 推理流水线更新（predict.py）
  - [x] 6.1 实现双窗口推理逻辑
    - 修改 `generate_signals` 支持分别加载 Ret5（w=20）和 Ret60（w=240）模型
    - Ret5 推理：batch_size=32768，构建 20 步窗口
    - Ret60 推理：batch_size=16384，构建 240 步窗口
    - 数据集长度 < 240 时，Ret60 序列模型输出零向量
    - 每个模型推理完成后执行 `del model` + `torch.cuda.empty_cache()`
    - _需求：3.4、3.5、3.7、7.4、8.2、8.3_

  - [x] 6.2 实现极端行情推理融合
    - 在推理时调用 `detect_extreme_regime` 进行因果检测
    - 加载 LGB_Extreme 模型，计算极端行情预测
    - 以 `w_extreme * extreme_indicator` 权重融合到最终预测
    - 极端模型文件不存在时权重视为 0
    - _需求：4.4、4.5、4.6、10.2_

  - [x] 6.3 实现 v5 兼容回退逻辑
    - `gru_ret5_*.pt` 不存在时回退到 `gru_*.pt`（v5 格式）
    - 从 checkpoint 动态读取 hidden_size（缺失时默认 64）
    - v5 格式权重 JSON 字段自动映射到 v6 字段
    - 任何模型加载失败时输出警告 + 零向量，不中断推理
    - _需求：10.1、10.2、10.3、10.4、10.5_

  - [x] 6.4 实现自适应 Batch 大小和大数据集跳过
    - 推理前检查可用 GPU 显存，< 8 GB 时 batch_size 减半
    - 数据集行数 > 3,000,000 且序列模型权重 ≤ 0.2 时跳过序列模型
    - _需求：7.3、8.5_

  - [x] 6.5 实现扩展集成权重加载
    - 支持 v6 权重格式（含 `ret5_w_extreme`、`ret5_w_gru_ret5` 等新字段）
    - 兼容 v5 权重格式（`ret5_w_gru`、`ret5_w_tf` 映射到对应 v6 字段）
    - 所有权重约束验证：≥ 0，同目标权重和 = 1.0
    - _需求：4.6、5.4、10.3_

  - [ ]* 6.6 编写短数据集安全降级属性测试
    - **属性 4：短数据集安全降级** — 验证 T < 240 时 Ret60 序列模型输出全零
    - **验证：需求 3.7**

  - [ ]* 6.7 编写自适应 Batch 大小属性测试
    - **属性 10：自适应 Batch 大小选择** — 验证 w=20 时 batch=32768，w=240 时 batch=16384，显存 < 8GB 时减半
    - **验证：需求 8.2、8.5**

  - [ ]* 6.8 编写向后兼容属性测试
    - **属性 11：向后兼容与优雅降级** — 验证 v6 模型缺失时回退 v5，加载失败时零向量不中断
    - **验证：需求 10.1、10.2、10.3、10.4、10.5**

- [x] 7. 检查点 - 推理流水线验证
  - 确保所有推理侧代码通过测试，如有疑问请询问用户。

- [x] 8. 权重优化脚本（reoptimize_v6.py）
  - [x] 8.1 创建 reoptimize_v6.py 基础框架
    - 实现 CLI 参数解析：`--mode local|feedback`、`--models-dir`、`--output`、`--prune`
    - 实现 v6 扩展权重格式的读写（含所有 6 种模型类型权重）
    - 实现权重约束验证：所有权重 ≥ 0，同目标权重和 = 1.0
    - _需求：5.1、5.2、5.3、5.4_

  - [x] 8.2 实现 local 模式权重优化
    - 基于本地验证集（后 20% 时序数据）的 IC 进行优化
    - 支持网格搜索或贝叶斯优化
    - 输出最优权重到 `ensemble_weights.json`
    - 优化后输出预估提交包大小，超过 144 MB 发出警告
    - _需求：5.1、5.7_

  - [x] 8.3 实现 feedback 模式权重优化
    - 读取 `feedback_state/` 目录下的平台反馈 JSON
    - 基于实际平台 IC 调整权重
    - _需求：5.2_

  - [x] 8.4 实现 --prune 裁剪逻辑
    - 将权重为 0 的模型从提交包中删除
    - 总大小超过 144 MB 时按权重从低到高移除序列模型
    - 不移除 LightGBM 模型
    - 生成 `submission_manifest.json` 列出需包含的文件
    - _需求：5.5、5.6、6.4_

  - [ ]* 8.5 编写权重约束属性测试
    - **属性 8：权重约束验证** — 验证输出权重 ≥ 0 且同目标权重和 = 1.0
    - **验证：需求 5.4**

  - [ ]* 8.6 编写提交包裁剪属性测试
    - **属性 9：提交包自动裁剪** — 验证超过 144 MB 时按权重从低到高移除序列模型
    - **验证：需求 6.4**

- [x] 9. 训练流水线鲁棒性增强（train.py）
  - [x] 9.1 实现 OOM 恢复机制
    - GPU OOM 时自动将 batch_size 减半并重试，最多 2 次
    - 3 次均失败时记录错误并跳过该模型
    - _需求：9.6_

  - [x] 9.2 实现断点续训逻辑
    - 训练前检查 checkpoint 文件是否已存在
    - 已完成的模型跳过，从中断点继续
    - _需求：12.2_

  - [x] 9.3 实现训练完成后大小汇总报告
    - 输出所有模型文件的大小汇总
    - 标注总大小和剩余余量（相对 150 MB 限制）
    - _需求：6.3_

  - [ ]* 9.4 编写 OOM 恢复属性测试
    - **属性 12：OOM 恢复机制** — 验证 OOM 时 batch_size 减半重试，最多 2 次
    - **验证：需求 9.6**

  - [ ]* 9.5 编写断点续训属性测试
    - **属性 13：断点续训正确性** — 验证已完成模型被跳过
    - **验证：需求 12.2**

- [x] 10. 检查点 - 鲁棒性验证
  - 确保所有鲁棒性相关代码通过测试，如有疑问请询问用户。

- [x] 11. 评估与验证（evaluate_local.py）
  - [x] 11.1 更新 evaluate_local.py 支持 v6 模型
    - 支持加载 LGB_Extreme、双窗口 GRU/Transformer 所有新模型类型
    - 分别报告普通行情 IC（nR5, nR60）和极端行情 IC（eR5, eR60）
    - 报告每个数据集的推理时间和 GPU 显存峰值
    - 评测完成后输出提交包大小检查（> 144 MB 警告，> 150 MB 报错）
    - _需求：11.1、11.2、11.3、11.4_

  - [x] 11.2 在 train_all_models 末尾集成自动评估
    - 全部训练完成后自动运行 evaluate_local.py
    - 输出完整 v6 评测报告
    - _需求：11.5_

  - [ ]* 11.3 编写大数据集跳过属性测试
    - **属性 14：大数据集序列模型跳过** — 验证行数 > 3,000,000 且序列模型权重 ≤ 0.2 时跳过序列模型
    - **验证：需求 7.3**

- [x] 12. 提交打包与最终验证
  - [x] 12.1 更新 check_submission_size.py 支持 v6 文件结构
    - 识别所有 v6 新增模型文件类型
    - 验证总大小 ≤ 150 MB
    - 输出各类模型文件的分类大小统计
    - _需求：6.1、6.2_

  - [x] 12.2 创建测试配置和属性测试入口文件
    - 创建 `tests/test_v6_properties.py` 属性测试文件框架
    - 配置 pytest markers：property、unit、integration、slow
    - 确保所有属性测试使用 Hypothesis 库，最少 100 次迭代
    - _需求：设计文档测试策略_

- [x] 13. 最终检查点 - 全流程验证
  - 确保所有测试通过，运行完整训练+推理流水线验证，如有疑问请询问用户。

## 备注

- 标记 `*` 的子任务为可选，可跳过以加速 MVP 交付
- 每个任务引用具体需求编号，确保可追溯性
- 检查点任务确保增量验证，及时发现问题
- 属性测试验证正确性属性的普遍性，单元测试验证具体示例和边界条件
- 实现语言：Python（与现有代码库一致）
- 所有序列模型输出维度从 2 变为 1（每个模型只预测一个目标）
