# 实现计划：模型优化 v4

## 概述

按方向B → 方向A → 方向C/E 的顺序实现，确保每一步都能独立验证。先扩展特征（factor.py），再改造推理（predict.py），最后改造训练（train.py）。所有 LightGBM 模型以 `.txt.gz` 格式保存，提交包总大小 ≤ 150 MB。

---

## 任务

- [x] 1. 方向B：在 factor.py 中新增长周期特征（18维）
  - [x] 1.1 实现 `_compute_long_period_features()` Numba JIT 函数
    - 在 factor.py 中新增 `@njit(cache=True)` 函数，接受 `(open_, high, low, close, volume)` 五个 1D 数组
    - 窗口列表 `[240, 480]`，输出 18 维 float64 数组，列顺序严格按设计文档表格：
      - 列 0–3：长周期动量（240/480 步对数收益率 + 变化率，共 4 维）
      - 列 4–9：长周期波动率（240/480 步滚动标准差 + Parkinson + ATR，共 6 维）
      - 列 10–11：长周期 EMA 偏离度（240/480 步，共 2 维）
      - 列 12–15：长周期已实现方差（240/480 步 RV + log-RV，共 4 维）
      - 列 16–17：长周期价格区间位置（240/480 步，共 2 维）
    - 所有计算仅使用 `i` 及之前的数据（因果性）
    - 前 `w-1` 行输出 NaN；分母为零时价格区间位置输出 0.5，对数计算参数 ≤ 0 时输出 NaN
    - _需求：2.1、2.2、2.3、2.4、2.5、2.7、2.8、2.9、2.11_

  - [x] 1.2 在 `generate_factors()` 中追加长周期特征列
    - 调用 `_compute_long_period_features()` 并将结果转换为 float32
    - 用 `np.hstack` 将 18 维特征追加到现有 147 维末尾，总输出 165 维
    - 在函数 docstring 和代码注释中标注 `# v4: 147 + 18 = 165维`
    - _需求：2.6、5.2、5.8_

  - [x]* 1.3 为属性1编写属性测试：特征矩阵维数不变量
    - **Property 1: 特征矩阵维数不变量**
    - **验证：需求 2.6、5.2**
    - 使用 `@given(st.arrays(..., shape=(T, 5), elements=st.floats(min_value=0.01, max_value=1000.0, allow_nan=False)))` 生成随机 OHLCV，断言 `result.shape[1] == 165`
    - `@settings(max_examples=100)`，测试标签 `Feature: model-optimization-v4, Property 1: 特征矩阵维数不变量`

  - [x]* 1.4 为属性2编写属性测试：特征矩阵精度不变量
    - **Property 2: 特征矩阵精度不变量**
    - **验证：需求 5.8**
    - 断言 `result.dtype == np.float32`
    - 测试标签 `Feature: model-optimization-v4, Property 2: 特征矩阵精度不变量`

  - [x]* 1.5 为属性3编写属性测试：长周期特征因果性
    - **Property 3: 长周期特征因果性**
    - **验证：需求 2.9**
    - 生成长度 ≥ 500 的 OHLCV，记录时刻 `i` 的特征值；随机修改 `i+1` 之后的数据；断言时刻 `i` 的所有特征值（含列 147–164）不变
    - 测试标签 `Feature: model-optimization-v4, Property 3: 长周期特征因果性`

  - [x]* 1.6 为属性4编写属性测试：长周期动量特征数学正确性
    - **Property 4: 长周期动量特征数学正确性**
    - **验证：需求 2.1**
    - 对任意 `i ≥ w`（`w ∈ {240, 480}`），断言列 147/148（w=240）和列 149/150（w=480）的值分别等于 `log(close[i]/close[i-w])` 和 `close[i]/close[i-w]-1`
    - 测试标签 `Feature: model-optimization-v4, Property 4: 长周期动量特征数学正确性`

  - [x]* 1.7 为属性5编写属性测试：价格区间位置有界性
    - **Property 5: 价格区间位置有界性**
    - **验证：需求 2.5、2.11**
    - 断言所有非 NaN 的列 163–164 值均在 `[0.0, 1.0]` 范围内
    - 构造高低价相等的特殊输入，断言对应位置输出恰好为 0.5
    - 测试标签 `Feature: model-optimization-v4, Property 5: 价格区间位置有界性`

- [x] 2. 检查点——验证 factor.py 变更
  - 运行 `python -c "import factor; import numpy as np; d=np.random.rand(600,5).astype(np.float32)+0.1; r=factor.generate_factors('dataset0',d); assert r.shape==(600,165) and r.dtype==np.float32"` 确认输出形状和精度正确
  - 确保所有属性测试通过，如有问题请向用户反馈。

- [x] 3. 方向A：改造 predict.py 支持全局模型推理
  - [x] 3.1 实现 `_load_lgb_model()` 辅助函数
    - 在 predict.py 中新增函数，优先加载 `.txt.gz`，fallback 到 `.txt`，两者均缺失时抛出 `FileNotFoundError`
    - 使用 `lgb.Booster(model_file=str(path))` 加载（LightGBM 原生支持 gzip）
    - _需求：1.6、5.7_

  - [x] 3.2 在 `generate_signals()` 中读取 `use_global_model_ret5/ret60` 字段
    - 从 `ensemble_weights.json` 的 `ds_w` 字典中读取 `use_global_model_ret5`（默认 `False`）和 `use_global_model_ret60`（默认 `False`）
    - 字段缺失时静默使用默认值，不抛异常
    - _需求：1.6、5.5_

  - [x] 3.3 实现全局模型推理分支
    - 解析 `dataset_id = int(dataset_name.replace("dataset", ""))`
    - 当 `use_global_model_ret5=True` 时：构造 `id_col = np.full((T, 1), dataset_id, dtype=np.int32)`，`factors_with_id = np.hstack([factors, id_col])`（165→166维），调用 `_load_lgb_model(MODEL_DIR/"lgb_ret5_global.txt.gz", MODEL_DIR/"lgb_ret5_global.txt")`；全局模型文件缺失时自动 fallback 到本地模型
    - Ret60 同理
    - 将原有本地模型加载也改用 `_load_lgb_model()` 函数（支持 `.txt.gz`）
    - _需求：1.2、1.6、1.10、5.7_

  - [x]* 3.4 为属性6编写属性测试：数据集ID解析正确性
    - **Property 6: 数据集ID解析正确性**
    - **验证：需求 1.2**
    - 对 `dataset_name ∈ {"dataset0", ..., "dataset29"}`，断言解析出的整数 N 等于 `int(dataset_name.replace("dataset", ""))`，且追加的 ID 列所有值均等于 N
    - 测试标签 `Feature: model-optimization-v4, Property 6: 数据集ID解析正确性`

  - [x]* 3.5 为属性7编写属性测试：模型选择逻辑正确性
    - **Property 7: 模型选择逻辑正确性**
    - **验证：需求 1.6**
    - Mock 全局模型和本地模型，随机生成 `use_global_model_ret5/ret60` 布尔值，断言实际调用的模型路径与标志一致
    - 测试标签 `Feature: model-optimization-v4, Property 7: 模型选择逻辑正确性`

  - [x]* 3.6 为属性8编写属性测试：权重文件向后兼容性
    - **Property 8: 权重文件向后兼容性**
    - **验证：需求 5.5**
    - 生成不含 `use_global_model_*` 字段的 v3 格式权重字典，断言读取后 `use_global_model_ret5 == False` 且 `use_global_model_ret60 == False`
    - 测试标签 `Feature: model-optimization-v4, Property 8: 权重文件向后兼容性`

  - [x]* 3.7 为属性9编写属性测试：降级健壮性
    - **Property 9: 降级健壮性**
    - **验证：需求 5.7**
    - 随机组合模型文件缺失场景（全局模型缺失、GRU缺失、Transformer缺失），断言 `generate_signals()` 返回形状 `(T, 2)`、dtype `float32`、不含 NaN/Inf
    - 测试标签 `Feature: model-optimization-v4, Property 9: 降级健壮性`

- [x] 4. 检查点——验证 predict.py 变更
  - 使用 mock 模型文件验证 `_load_lgb_model()` 的 gz/txt fallback 逻辑
  - 验证 `use_global_model=False` 时行为与 v3 完全一致
  - 确保所有属性测试通过，如有问题请向用户反馈。

- [x] 5. 方向A：改造 train.py 实现全局模型训练
  - [x] 5.1 实现全局训练集构建函数
    - 新增 `_build_global_dataset(datasets_features, datasets_labels, max_per_dataset=200000)` 函数
    - 对每个数据集按前 80% 划分训练集，若训练行数 > `max_per_dataset` 则按时序均匀间隔采样（步长 `k = ceil(train_rows / max_per_dataset)`，取索引 `[0, k, 2k, ...]`）
    - 追加整数类型数据集 ID 列（`np.full((N_i,), dataset_id, dtype=np.int32)`）
    - 纵向拼接所有数据集，返回全局特征矩阵（含 ID 列，166维）和标签
    - 当总行数 > 500 万时触发采样逻辑
    - _需求：1.1、1.2、1.3、1.9_

  - [x] 5.2 实现全局 LightGBM 模型训练（两阶段 + IC 早停）
    - 新增 `train_global_lgb(global_X_train, global_y_train, global_X_val, global_y_val, target)` 函数
    - 将 ID 列（最后一列，索引 165）声明为 `categorical_feature`
    - 使用与本地模型相同的 IC 自定义评估函数和两阶段训练机制（最少 30 棵树）
    - Ret5 和 Ret60 分别训练独立全局模型，超参数与对应本地模型保持一致
    - 训练完成后用 gzip 压缩保存：`model.save_model(str(path_txt)); gzip_compress(path_txt, path_gz)`
    - _需求：1.3、1.7、1.8_

  - [x] 5.3 实现全局 IC vs 本地 IC 对比，写入 ensemble_weights.json
    - 训练完成后，在验证集上分别计算每个数据集的全局模型 IC 和本地模型 IC
    - 若全局 IC > 本地 IC，则在 `ensemble_weights.json` 中将该数据集的 `use_global_model_ret5/ret60` 设为 `true`，否则设为 `false`
    - 将对比结果记录到训练日志
    - 若全局模型训练失败，记录错误日志并继续，不中断整体训练流程
    - _需求：1.5、1.6、5.4_

  - [x] 5.4 实现 LightGBM 模型 gzip 压缩保存工具函数
    - 新增 `save_lgb_model_gz(booster, path_gz)` 函数：先保存为临时 `.txt`，再用 `gzip` 模块压缩为 `.txt.gz`，删除临时文件
    - 在所有本地模型和全局模型保存处统一调用此函数
    - _需求：5.3_

- [x] 6. 方向C：在 train.py 中实现高波动加权采样
  - [x] 6.1 实现 `_compute_volatility_weights()` 函数
    - 计算 20-bar 滚动波动率 `σ_20(i)`（1-bar 对数收益率的滚动标准差）
    - 计算 120-bar 滚动中位数 `Median_120(σ_20)(i)`
    - 权重 `w_i = σ_20(i) / Median_120(σ_20)(i)`；NaN 或零值位置替换为 1.0
    - 返回与输入等长的正数权重数组
    - _需求：3.2、3.5_

  - [x] 6.2 在 Large_Datasets 序列模型训练中启用加权采样
    - 对 dataset6–dataset19，在计算 `max_samples` 后进一步限制 `max_samples = min(max_samples, 300000)`
    - 调用 `_compute_volatility_weights()` 获取权重，用 `np.random.choice(indices, size=max_samples, replace=False, p=weights/weights.sum())` 加权采样
    - 对采样结果排序（`np.sort`）以保持时序顺序
    - 记录采样前后样本数和高波动覆盖率（`w_i > 1.5` 的比例）到训练日志
    - Small_Datasets 保持原有全量训练方式不变
    - _需求：3.1、3.3、3.4、3.6_

  - [x]* 6.3 为属性10编写属性测试：采样上限不变量
    - **Property 10: 采样上限不变量**
    - **验证：需求 3.3、4.2、4.3**
    - 对任意 `available_vram`（正整数）和 `train_set_size`（正整数），断言计算所得 `max_samples ≤ min(max_samples_by_vram, train_set_size)`；对 Large_Datasets 且方向C启用时，还断言 `actual_samples ≤ 300000`
    - 测试标签 `Feature: model-optimization-v4, Property 10: 采样上限不变量`

  - [x]* 6.4 为属性11编写属性测试：采样时序保持性
    - **Property 11: 采样时序保持性**
    - **验证：需求 3.4**
    - 对任意波动率权重序列，断言加权采样后返回的索引数组是严格单调递增的
    - 测试标签 `Feature: model-optimization-v4, Property 11: 采样时序保持性`

  - [x]* 6.5 为属性12编写属性测试：采样权重计算正确性
    - **Property 12: 采样权重计算正确性**
    - **验证：需求 3.2、3.5**
    - 生成含 NaN 和零值的波动率序列，断言：非 NaN/零位置的权重等于 `σ_20(i) / Median_120(σ_20)(i)`；NaN 或零位置的权重等于 1.0；所有权重均为正数
    - 测试标签 `Feature: model-optimization-v4, Property 12: 采样权重计算正确性`

- [x] 7. 方向E：在 train.py 中实现自适应采样上限
  - [x] 7.1 实现 `_compute_adaptive_max_samples()` 函数
    - 调用 `torch.cuda.mem_get_info()[0]` 获取可用显存（字节）；异常时 fallback 到 150000
    - 按公式计算：`max_samples = floor(available_vram * 0.6 / (4096 * 60 * 165 * 4)) * 4096`
    - 限制 `max_samples = min(max_samples, train_set_size)`
    - 若 `max_samples < 50000`，返回 0（调用方跳过序列模型训练并记录日志）
    - CPU 模式（`not torch.cuda.is_available()`）时固定返回 150000
    - 每次调用重新检测，不缓存
    - _需求：4.1、4.2、4.3、4.5、4.7、4.9_

  - [x] 7.2 实现按批次构建滑动窗口的训练循环
    - 替换原有预先展开全部窗口数组的方式
    - 每个训练步骤仅对当前 batch 的索引现场切片：`factors[idx-59:idx+1]`（`idx` 为采样后的样本索引）
    - 系统内存占用保持在单 batch 级别（约 0.15 GB）
    - _需求：4.4_

  - [x] 7.3 实现 GPU OOM 自动减半重试逻辑
    - 捕获 `torch.cuda.OutOfMemoryError`，将 `max_samples` 减半后重试，最多重试 2 次
    - 每次重试记录日志
    - _需求：4.1（错误处理章节）_

  - [x] 7.4 在训练日志中记录显存检测和采样信息
    - 记录每个数据集的：可用显存（GB）、计算所得采样上限、实际使用样本数
    - _需求：4.6_

- [x] 8. 检查点——验证 train.py 变更
  - 用小规模合成数据（3个数据集，各1000行）运行全局模式训练，验证：全局训练集构建正确、IC对比逻辑正确、ensemble_weights.json 写入正确
  - 验证 gzip 压缩保存和加载的完整性（保存后重新加载，预测结果一致）
  - 确保所有属性测试通过，如有问题请向用户反馈。

- [x] 9. 提交包大小验证
  - [x] 9.1 编写 `check_submission_size.py` 脚本
    - 遍历 `models_v4/` 目录，统计所有文件总大小（MB）
    - 断言总大小 ≤ 140 MB，否则打印超出文件列表并退出非零状态码
    - _需求：5.3、5.4_

  - [x] 9.2 验证所有 LightGBM 模型均以 `.txt.gz` 格式保存
    - 断言 `models_v4/` 中不存在未压缩的 `.txt` LightGBM 模型文件
    - _需求：5.3_

- [x] 10. 最终检查点——端到端验证
  - 对所有 30 个数据集运行 `generate_factors()` + `generate_signals()`，验证输出形状 `(T, 2)`、dtype `float32`、不含 NaN/Inf
  - 验证 v3 格式的 `ensemble_weights.json` 在 v4 中能正确加载（向后兼容性）
  - 确保所有属性测试通过，如有问题请向用户反馈。

---

## 备注

- 标有 `*` 的子任务为可选属性测试，可跳过以加快 MVP 进度
- 每个任务均引用了具体需求条款，便于追溯
- 方向B（factor.py）是其他所有方向的基础，必须最先完成
- 方向C 和方向E 均在 train.py 中实现，可并行开发
- 属性测试文件建议命名为 `test_properties_v4.py`，放在项目根目录
