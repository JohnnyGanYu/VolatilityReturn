# 需求文档

## 简介

本文档定义模型优化迭代 v4 的需求。v4 基于论文《高波动标的收益率预测：基于多模型集成的特征工程与机器学习方法》第六章所识别的局限性，筛选低风险高收益的优化方向，在不破坏现有系统稳定性、不大幅增加推理时间（当前约18.5分钟，限制120分钟）的前提下，提升4个IC指标（nR5、nR60、eR5、eR60）。

**核心优化方向（必须实现）：**
- **方向A**：跨数据集联合建模——将数据集ID作为类别特征加入LightGBM，训练全局共享模型
- **方向B**：Ret60长周期特征扩展——新增240步/480步窗口特征，修复4个数据集Ret60负IC问题

**可选优化方向（条件实现）：**
- **方向C**：大数据集高波动子采样——训练序列模型时优先保留高波动时段样本
- **方向E**：序列模型自适应采样上限——根据运行环境的可用显存自动计算训练采样上限，采用按批次构建窗口方式消除系统内存瓶颈，保持60步窗口不变

---

## 词汇表

- **IC（Information Coefficient）**：预测信号与真实收益率之间的Pearson相关系数，是本系统的核心评测指标
- **nR5**：普通行情下Ret5（未来5分钟收益率）的IC均值
- **nR60**：普通行情下Ret60（未来60分钟收益率）的IC均值
- **eR5**：极端行情下Ret5的IC均值
- **eR60**：极端行情下Ret60的IC均值
- **Ret5**：未来5分钟对数收益率，$\log(\text{close}[i+5]/\text{close}[i])$
- **Ret60**：未来60分钟对数收益率，$\log(\text{close}[i+60]/\text{close}[i])$
- **LightGBM_Global**：以数据集ID为类别特征、在所有数据集上联合训练的全局LightGBM模型
- **LightGBM_Local**：仅在单个数据集上训练的原有LightGBM模型（v3方案）
- **Factor_Generator**：factor.py中的特征生成模块
- **Signal_Generator**：predict.py中的信号生成模块
- **Training_Pipeline**：train.py中的离线训练流程
- **Ensemble_Selector**：基于验证集IC网格搜索确定集成权重的模块
- **Negative_IC_Datasets**：Ret60 LightGBM验证IC为负的数据集，即dataset6、dataset13、dataset15、dataset28
- **Large_Datasets**：行数超过150万的数据集，即dataset6至dataset19
- **Small_Datasets**：行数不超过150万的数据集，即dataset0至dataset5和dataset20至dataset29
- **High_Volatility_Sampler**：按波动率优先级对训练样本进行子采样的模块
- **Sequence_Model**：GRU或Transformer序列模型的统称
- **Sliding_Window**：为序列模型构建的因果滑动窗口，长度为 window_size 步
- **Validation_IC_Threshold**：判断序列模型是否有效的验证IC阈值，当前为0.01
- **OHLCV**：开盘价（Open）、最高价（High）、最低价（Low）、收盘价（Close）、成交量（Volume）

---

## 需求

### 需求1：跨数据集联合建模（方向A）

**用户故事：** 作为模型训练工程师，我希望将30个数据集的训练数据合并，以数据集ID作为类别特征训练全局LightGBM模型，从而让弱数据集借助强数据集的共性规律，提升整体IC表现。

#### 验收标准

1. THE Training_Pipeline SHALL 支持"全局模式"训练选项，在该模式下将所有30个数据集的特征矩阵和标签纵向拼接为一个全局训练集。

2. WHEN 全局模式启用时，THE Factor_Generator SHALL 在每个数据集的特征矩阵末尾追加一列整数类型的数据集ID特征（取值范围0至29），该列在推理时同样追加。

3. THE LightGBM_Global SHALL 将数据集ID列声明为LightGBM的`categorical_feature`，使模型能够学习跨数据集的差异化规律。

4. THE Training_Pipeline SHALL 对全局训练集执行时序划分：每个数据集内部按前80%训练、后20%验证的比例划分，拼接后的验证集用于全局模型的IC早停。

5. WHEN 全局模式训练完成后，THE Training_Pipeline SHALL 在验证集上分别计算每个数据集的IC，并与对应的LightGBM_Local验证IC进行对比，记录到训练日志。

6. THE Ensemble_Selector SHALL 在推理时支持从全局模型或本地模型中选择：WHEN 某数据集的LightGBM_Global验证IC高于LightGBM_Local验证IC时，THE Signal_Generator SHALL 使用LightGBM_Global的预测结果；OTHERWISE THE Signal_Generator SHALL 回退至LightGBM_Local的预测结果。

7. THE LightGBM_Global SHALL 使用与LightGBM_Local相同的IC自定义评估函数和两阶段训练机制（最少30棵树）。

8. THE LightGBM_Global SHALL 对Ret5和Ret60分别训练独立的全局模型，超参数与对应目标的LightGBM_Local保持一致。

9. IF 全局训练集拼接后总行数超过500万行，THEN THE Training_Pipeline SHALL 对每个数据集按时序均匀采样，使每个数据集贡献不超过200000行，以控制训练时间。

10. THE Signal_Generator SHALL 在推理时保持因果性：WHEN 为数据集X生成信号时，THE Signal_Generator SHALL 仅使用数据集X的特征矩阵（含数据集ID列），不访问其他数据集的数据。

---

### 需求2：Ret60长周期特征扩展（方向B）

**用户故事：** 作为特征工程师，我希望为Ret60预测新增240步和480步的长周期特征，以修复dataset6、dataset13、dataset15、dataset28四个数据集Ret60验证IC为负的问题，提升长周期收益率预测能力。

#### 验收标准

1. THE Factor_Generator SHALL 新增长周期动量特征：对回望窗口 $w \in \{240, 480\}$，计算对数收益率 $\log(\text{close}[i]/\text{close}[i-w])$ 和变化率 $\text{close}[i]/\text{close}[i-w]-1$，共4维。

2. THE Factor_Generator SHALL 新增长周期波动率特征：对窗口 $w \in \{240, 480\}$，计算滚动标准差（1-bar对数收益率）、Parkinson波动率和ATR，共6维。

3. THE Factor_Generator SHALL 新增长周期趋势强度特征：对窗口 $w \in \{240, 480\}$，计算EMA偏离度 $\text{close}[i]/\text{EMA}(\text{close}, w)_i - 1$，共2维。

4. THE Factor_Generator SHALL 新增长周期已实现方差特征：对窗口 $w \in \{240, 480\}$，计算已实现方差RV和log-RV，共4维。

5. THE Factor_Generator SHALL 新增长周期价格区间位置特征：对窗口 $w \in \{240, 480\}$，计算 $(\text{close}[i] - \min(\text{close}, w)_i) / (\max(\text{close}, w)_i - \min(\text{close}, w)_i)$，共2维。

6. THE Factor_Generator SHALL 将上述新增特征（共18维）追加至现有147维特征矩阵末尾，使总特征维数达到165维，且不超过平台限制的512维。

7. WHEN 数据集行数不足480行时，THE Factor_Generator SHALL 对超出历史长度的位置输出NaN，与现有短窗口特征的NaN处理方式保持一致。

8. THE Factor_Generator SHALL 对所有新增长周期特征使用Numba `@njit(cache=True)` JIT编译，保证计算效率与现有特征一致。

9. THE Factor_Generator SHALL 保证所有新增特征的因果性：特征计算仅使用时刻i及之前的数据，不使用未来信息。

10. WHEN 新增特征计算完成后，THE Training_Pipeline SHALL 对Negative_IC_Datasets（dataset6、dataset13、dataset15、dataset28）的Ret60模型单独记录验证IC，以验证长周期特征是否改善了负IC问题。

11. THE Factor_Generator SHALL 对新增特征中的零分母情况做安全处理：WHEN 窗口内最高价等于最低价时，THE Factor_Generator SHALL 将价格区间位置特征输出为0.5而非NaN或Inf。

---

### 需求3：大数据集高波动子采样（方向C，可选）

**用户故事：** 作为模型训练工程师，我希望在训练序列模型时优先保留高波动时段的样本，而非均匀随机采样，从而让序列模型在大数据集上学到更有价值的时序模式，改善当前18/30个数据集退化为纯LightGBM的问题。

#### 验收标准

1. THE High_Volatility_Sampler SHALL 对Large_Datasets（dataset6至dataset19）的序列模型训练启用高波动优先采样，对Small_Datasets保持原有全量训练方式不变。

2. THE High_Volatility_Sampler SHALL 按以下规则计算每个时刻的采样权重：$w_i = \sigma_{20}(i) / \text{Median}_{120}(\sigma_{20})(i)$，其中 $\sigma_{20}(i)$ 为20-bar滚动波动率，$\text{Median}_{120}$ 为120-bar滚动中位数。

3. WHEN 对Large_Datasets进行序列模型训练采样时，THE High_Volatility_Sampler SHALL 按采样权重进行加权随机采样，采样数量上限为300000条（相比原有150000条翻倍），以保留更多高波动时段信息。

4. THE High_Volatility_Sampler SHALL 保证采样后的样本集仍按时间顺序排列，不打乱时序关系，以保证滑动窗口构建的因果性。

5. WHEN 采样权重中存在NaN或零值时，THE High_Volatility_Sampler SHALL 将该位置的权重替换为1.0（均匀权重），避免采样失败。

6. THE Training_Pipeline SHALL 在训练日志中记录每个Large_Dataset的序列模型采样前后的样本数量和高波动时段覆盖率（定义为采样集中 $w_i > 1.5$ 的样本比例）。

7. WHERE 方向C被启用，THE Ensemble_Selector SHALL 重新对所有Large_Datasets执行集成权重网格搜索，以反映新采样策略下序列模型性能的变化。

---

### 需求4：序列模型自适应采样上限（方向E，可选）

**用户故事：** 作为模型训练工程师，我希望训练序列模型时根据当前运行环境的可用显存自动计算最大训练样本数，采用按批次构建窗口的方式消除系统内存瓶颈，在保持60步窗口不变的前提下，让大数据集尽可能使用更多训练样本，改善序列模型在大数据集上的预测效果。

#### 验收标准

1. THE Training_Pipeline SHALL 在序列模型训练开始前，自动检测当前可用GPU显存（通过 `torch.cuda.mem_get_info()`），以确定训练采样上限。

2. THE Training_Pipeline SHALL 按以下公式计算基于显存的采样上限：
   $$\text{max\_samples} = \left\lfloor \frac{\text{available\_vram} \times r_{\text{gpu}}}{\text{batch\_size} \times \text{window\_size} \times \text{feature\_dim} \times 4} \right\rfloor \times \text{batch\_size}$$
   其中 $r_{\text{gpu}} = 0.6$（为模型参数、梯度、优化器状态预留40%显存），`window_size=60`，`feature_dim=165`，`batch_size=4096`。

3. THE Training_Pipeline SHALL 将采样上限限制在不超过该数据集训练集实际行数（前80%部分）：
   $$\text{max\_samples} = \min(\text{max\_samples},\ \text{train\_set\_size})$$
   当采样上限大于等于训练集实际行数时，使用全量训练数据，不做采样。

4. THE Training_Pipeline SHALL 采用按批次构建滑动窗口的方式训练序列模型：每个训练步骤仅对当前batch的索引现场切片构建窗口 `factors[i-59:i+1]`，不预先展开全部采样数据的窗口数组，系统内存占用始终保持在单batch级别（约0.15 GB）。

5. THE Training_Pipeline SHALL 设置采样下限为50000条：WHEN 计算所得 `max_samples` 低于50000时，THE Training_Pipeline SHALL 跳过该数据集的序列模型训练，并在日志中记录"显存不足，跳过序列模型训练"。

6. THE Training_Pipeline SHALL 在训练日志中记录每个数据集检测到的可用显存、计算所得采样上限和实际使用的样本数量。

7. WHEN GPU不可用时（CPU训练模式），THE Training_Pipeline SHALL 将采样上限固定为150000条（与v3保持一致），不执行自适应计算。

8. THE Training_Pipeline SHALL 对Small_Datasets（行数不超过150万）保持全量训练方式不变，自适应采样上限仅对Large_Datasets生效。

9. THE Training_Pipeline SHALL 在每次训练开始时重新检测可用显存，不缓存上次的检测结果，以适应多任务并行时显存动态变化的情况。

---

### 需求5：系统约束与兼容性

**用户故事：** 作为系统工程师，我希望v4的所有改动在满足平台约束的前提下向后兼容v3，确保提交包大小、推理时间和特征维数均不超过平台限制。

#### 验收标准

1. THE Signal_Generator SHALL 在完成v4所有核心改动（方向A+B）后，总推理时间不超过40分钟（平台限制120分钟的1/3），为可选方向C/E留有余量。

2. THE Factor_Generator SHALL 在完成方向B后，输出特征维数不超过200维，满足平台512维限制，并在代码注释中标注当前维数。

3. THE Training_Pipeline SHALL 在完成v4训练后，生成的提交包总大小不超过140 MB（平台限制150 MB，预留10 MB余量）。

4. IF 全局模型文件（方向A）导致提交包超过140 MB，THEN THE Training_Pipeline SHALL 仅保存验证IC高于对应本地模型的全局模型，其余数据集继续使用本地模型。

5. THE Signal_Generator SHALL 在v3的 `ensemble_weights.json` 格式基础上扩展，新增 `use_global_model` 字段（布尔值，按数据集和目标独立配置），保持与v3权重文件的向后兼容性：WHEN `use_global_model` 字段缺失时，THE Signal_Generator SHALL 默认使用本地模型。

6. THE Factor_Generator 和 THE Signal_Generator SHALL 在入口处保留固定随机种子（seed=42）设置，保证v4的可复现性。

7. WHEN v4的任意模型文件缺失时，THE Signal_Generator SHALL 按v3的降级逻辑处理：优先使用可用的序列模型，最终回退至纯LightGBM预测，不抛出异常。

8. THE Factor_Generator SHALL 对新增的165维特征矩阵保持float32精度输出，不引入float64，以控制内存占用。
