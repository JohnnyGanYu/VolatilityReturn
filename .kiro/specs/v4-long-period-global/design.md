# 设计文档：模型优化 v4

## 概述

本文档描述模型优化迭代 v4 的技术设计。v4 基于 v3 方案（LightGBM + GRU + Transformer 三模型集成，147维特征，149.18 MB提交包）进行四个方向的优化：

- **方向A（核心）**：跨数据集联合建模——将数据集ID作为类别特征，训练全局LightGBM模型（LightGBM_Global），推理时按验证IC选择全局或本地模型
- **方向B（核心）**：Ret60长周期特征扩展——新增240步/480步窗口特征18维，特征矩阵从147维扩展到165维
- **方向C（可选）**：大数据集高波动子采样——训练序列模型时按波动率加权采样，采样上限300000条
- **方向E（可选）**：序列模型自适应采样上限——根据可用GPU显存自动计算采样上限，采用按批次构建窗口方式

**系统约束**：提交包 ≤ 140 MB（通过gzip压缩LightGBM模型实现）、推理时间 ≤ 40 分钟、特征维数 ≤ 200 维、测试服务器 RTX 4090（24GB显存）。

---

## 架构

### 整体架构

v4 保持 v3 的三层架构不变，在各层内部进行扩展：

```
┌─────────────────────────────────────────────────────────────────┐
│                        离线训练（train.py）                       │
│                                                                   │
│  ┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │  方向B        │   │  方向A            │   │  方向C/E        │  │
│  │  特征扩展     │   │  全局LGB训练      │   │  序列模型训练   │  │
│  │  147→165维   │   │  + 本地LGB训练    │   │  自适应采样     │  │
│  └──────────────┘   └──────────────────┘   └─────────────────┘  │
│                              ↓                                    │
│              ensemble_weights_v4.json（扩展格式）                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      在线推理（predict.py）                       │
│                                                                   │
│  factor.py                                                        │
│  generate_factors(dataset_name, data)                             │
│  → (T, 165) float32                                               │
│                                                                   │
│  predict.py                                                       │
│  generate_signals(dataset_name, factors)                          │
│  ├── 方向A：按IC选择全局/本地LGB模型                              │
│  ├── 方向B：165维特征直接输入（含长周期特征）                     │
│  └── 方向C/E：GRU/Transformer推理（不变）                        │
│  → (T, 2) float32                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 模型文件结构（v4）

```
models_v4/
├── lgb_ret5_global.txt.gz          # 全局LGB Ret5模型（方向A，1个）
├── lgb_ret60_global.txt.gz         # 全局LGB Ret60模型（方向A，1个）
├── lgb_ret5_dataset{0..29}.txt.gz  # 本地LGB Ret5模型（30个，gzip压缩）
├── lgb_ret60_dataset{0..29}.txt.gz # 本地LGB Ret60模型（30个，gzip压缩）
├── gru_dataset{0..29}.pt           # GRU模型（30个，方向C/E训练）
└── ensemble_weights.json           # 扩展格式权重配置
```

**提交包大小估算**：
- 60个LGB本地模型（gzip）：~45-60 MB
- 2个LGB全局模型（gzip）：~1-3 MB
- 30个GRU模型（.pt）：~45 MB
- 合计：~91-108 MB，满足 ≤ 140 MB 约束

---

## 组件与接口

### 组件1：Factor_Generator（factor.py）

**接口**（不变）：
```python
def generate_factors(dataset_name: str, data: np.ndarray) -> np.ndarray:
    """
    输入: data (T, 5) float32，列顺序 [open, high, low, close, volume]
    输出: (T, 165) float32  # v4: 147 + 18 = 165维
    """
```

**变更**：新增 `_compute_long_period_features()` 函数，输出18维长周期特征，追加在147维末尾（列索引147-164）。

**特征维数分配（v4）**：

| 列索引 | 特征组 | 维数 |
|--------|--------|------|
| 0–146 | 原有特征（v3不变） | 147 |
| 147–150 | 长周期动量（240/480步，对数收益率+变化率） | 4 |
| 151–156 | 长周期波动率（240/480步，滚动标准差+Parkinson+ATR） | 6 |
| 157–158 | 长周期EMA偏离度（240/480步） | 2 |
| 159–162 | 长周期已实现方差（240/480步，RV+log-RV） | 4 |
| 163–164 | 长周期价格区间位置（240/480步） | 2 |
| **合计** | | **165** |

### 组件2：Signal_Generator（predict.py）

**接口**（不变）：
```python
def generate_signals(dataset_name: str, factors: np.ndarray) -> np.ndarray:
    """
    输入: factors (T, 165) float32  # v4: 165维
    输出: (T, 2) float32，列0=Ret5预测，列1=Ret60预测
    """
```

**变更**：
1. 支持加载 `.txt.gz` 格式的LightGBM模型（带fallback到 `.txt`）
2. 读取 `ensemble_weights.json` 中新增的 `use_global_model_ret5` / `use_global_model_ret60` 字段
3. 当 `use_global_model_ret5=true` 时，在调用全局模型前追加数据集ID列（165→166维）

**推理流程（v4）**：

```
factors (T, 165)
    │
    ├── 读取 ensemble_weights.json
    │       ├── use_global_model_ret5: bool
    │       └── use_global_model_ret60: bool
    │
    ├── LGB Ret5推理
    │   ├── IF use_global_model_ret5:
    │   │   └── factors_with_id = hstack([factors, dataset_id_col])  # (T, 166)
    │   │       → lgb_ret5_global.txt.gz
    │   └── ELSE:
    │       → lgb_ret5_dataset{N}.txt.gz
    │
    ├── LGB Ret60推理（同上）
    │
    └── GRU/Transformer推理（不变，使用165维输入）
```

### 组件3：Training_Pipeline（train.py）

**新增功能**：

1. **全局模式训练（方向A）**：
   - 拼接所有数据集的特征矩阵，追加数据集ID列
   - 按数据集内部时序80/20划分，拼接后训练全局模型
   - 训练完成后对比每个数据集的全局IC vs 本地IC

2. **长周期特征训练（方向B）**：
   - 调用更新后的 `generate_factors()`，自动获得165维特征
   - 无需额外修改，特征扩展对训练流程透明

3. **高波动子采样（方向C）**：
   - 对Large_Datasets（dataset6-19）的序列模型训练启用加权采样
   - 采样上限300000条，保持时序顺序

4. **自适应采样上限（方向E）**：
   - 训练前检测可用GPU显存
   - 按公式计算采样上限，采用按批次构建窗口方式

### 组件4：Ensemble_Selector

**ensemble_weights.json 扩展格式（v4）**：

```json
{
  "dataset0": {
    "ret5_alpha": 1.0,
    "ret5_beta": 0.0,
    "ret5_gamma": 0.0,
    "ret60_alpha": 1.0,
    "ret60_beta": 0.0,
    "ret60_gamma": 0.0,
    "use_global_model_ret5": false,
    "use_global_model_ret60": true
  },
  ...
}
```

**向后兼容性**：
- `use_global_model_ret5` / `use_global_model_ret60` 字段缺失时，默认为 `false`（使用本地模型）
- `ret5_beta` / `ret5_gamma` 等字段缺失时，默认为 `0.0`（与v3行为一致）

---

## 数据模型

### 特征矩阵

| 版本 | 形状 | dtype | 说明 |
|------|------|-------|------|
| v3 | (T, 147) | float32 | 原有特征 |
| v4（本地模型推理） | (T, 165) | float32 | 追加18维长周期特征 |
| v4（全局模型推理） | (T, 166) | float32 | 额外追加1维数据集ID |

**数据集ID编码**：
- `dataset0` → 0, `dataset1` → 1, ..., `dataset29` → 29
- 类型：int32（LightGBM categorical_feature要求）
- 在predict.py中通过 `int(dataset_name.replace("dataset", ""))` 解析

### 全局训练集构建

```
全局训练集 = concat([
    dataset_i_features_with_id  # (N_i, 166)，N_i ≤ 200000
    for i in range(30)
])
```

**采样策略**（当总行数 > 500万时）：
- 对每个数据集，按时序均匀间隔采样至200000行
- 采样步长 `k = ceil(train_rows / 200000)`，取索引 `[0, k, 2k, ...]`
- 保持时序顺序，不打乱

### 序列模型训练数据（方向C/E）

```python
# 方向E：自适应采样上限计算
available_vram = torch.cuda.mem_get_info()[0]  # 字节
r_gpu = 0.6
batch_size = 4096
window_size = 60
feature_dim = 165  # v4使用165维

max_samples = floor(
    available_vram * r_gpu / (batch_size * window_size * feature_dim * 4)
) * batch_size
max_samples = min(max_samples, train_set_size)

# 方向C：高波动加权采样（在max_samples基础上）
# 对Large_Datasets：max_samples = min(max_samples, 300000)
# 按波动率权重采样，保持时序顺序
```

---

## 正确性属性

*属性（Property）是在系统所有有效执行中都应成立的特征或行为——本质上是关于系统应该做什么的形式化陈述。属性是人类可读规范与机器可验证正确性保证之间的桥梁。*

### 属性1：特征矩阵维数不变量

*对任意* 形状为 (T, 5) 的 OHLCV 输入（T ≥ 1），`generate_factors()` 的输出列数恰好为 165。

**验证：需求 2.6、5.2**

### 属性2：特征矩阵精度不变量

*对任意* 形状为 (T, 5) 的 OHLCV 输入，`generate_factors()` 的输出 dtype 恰好为 float32。

**验证：需求 5.8**

### 属性3：长周期特征因果性

*对任意* OHLCV 序列和任意时刻 i，修改时刻 i 之后的任意数据，时刻 i 的所有特征值（包括新增的长周期特征）保持不变。

**验证：需求 2.9**

### 属性4：长周期动量特征数学正确性

*对任意* close 序列和任意 i ≥ w（w ∈ {240, 480}），当 close[i-w] > 0 时，对数收益率特征值等于 log(close[i] / close[i-w])，变化率特征值等于 close[i] / close[i-w] - 1。

**验证：需求 2.1**

### 属性5：价格区间位置有界性

*对任意* OHLCV 序列，所有非 NaN 的价格区间位置特征值均在 [0.0, 1.0] 范围内；当窗口内最高价等于最低价时，输出恰好为 0.5。

**验证：需求 2.5、2.11**

### 属性6：数据集ID解析正确性

*对任意* dataset_name（格式为 "datasetN"，N ∈ {0, ..., 29}），predict.py 追加的 ID 列值等于从名称中解析出的整数 N。

**验证：需求 1.2**

### 属性7：模型选择逻辑正确性

*对任意* 数据集，当 `use_global_model_ret5=true` 时，Ret5 推理使用全局模型；当 `use_global_model_ret5=false` 时，使用本地模型。Ret60 同理。

**验证：需求 1.6**

### 属性8：权重文件向后兼容性

*对任意* v3 格式的 ensemble_weights.json（不含 `use_global_model_*` 字段），`generate_signals()` 能正确读取并将缺失字段默认为 `false`，行为与 v3 完全一致。

**验证：需求 5.5**

### 属性9：降级健壮性

*对任意* 模型文件缺失的组合（全局模型缺失、GRU缺失、Transformer缺失），`generate_signals()` 返回形状为 (T, 2)、dtype 为 float32、不含 NaN/Inf 的有效预测数组。

**验证：需求 5.7**

### 属性10：采样上限不变量

*对任意* 数据集和任意可用显存值，方向E计算所得实际使用样本数 ≤ min(max_samples_by_vram, train_set_size)；对 Large_Datasets 且方向C启用时，实际样本数还满足 ≤ 300000。

**验证：需求 3.3、4.2、4.3**

### 属性11：采样时序保持性

*对任意* 高波动加权采样结果，采样后的样本索引序列是严格单调递增的（时序顺序不被打乱）。

**验证：需求 3.4**

### 属性12：采样权重计算正确性

*对任意* 波动率序列（含 NaN 和零值），采样权重计算结果满足：非 NaN/零位置的权重等于 σ_20(i) / Median_120(σ_20)(i)；NaN 或零位置的权重替换为 1.0；所有权重均为正数。

**验证：需求 3.2、3.5**

---

## 错误处理

### 模型文件加载

```python
def _load_lgb_model(path_gz: Path, path_txt: Path) -> lgb.Booster:
    """加载LightGBM模型，优先.gz，fallback到.txt。"""
    if path_gz.exists():
        try:
            return lgb.Booster(model_file=str(path_gz))
        except Exception:
            pass  # fallback
    if path_txt.exists():
        return lgb.Booster(model_file=str(path_txt))
    raise FileNotFoundError(f"模型文件不存在: {path_gz} 或 {path_txt}")
```

**降级策略**（predict.py）：

| 缺失文件 | 降级行为 |
|---------|---------|
| 全局模型（.gz/.txt均缺失） | 自动使用本地模型，不抛异常 |
| 本地模型缺失 | 抛出 FileNotFoundError（本地模型是必需的） |
| GRU模型缺失 | beta权重置0，退化为LGB+Transformer |
| Transformer模型缺失 | gamma权重置0，退化为LGB+GRU |
| ensemble_weights.json缺失 | 使用默认权重（alpha=1.0，use_global=false） |

### 特征计算异常

- 除零：所有除法操作检查分母 > 1e-10，否则输出 NaN 或安全默认值（如0.5）
- 负对数：所有 log() 调用前检查参数 > 0，否则输出 NaN
- NaN传播：Numba JIT函数中 NaN 通过计算自然传播，不做强制填充
- 最终输出：`generate_signals()` 在返回前执行 `np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)`

### 训练异常

- GPU OOM：序列模型训练时捕获 `torch.cuda.OutOfMemoryError`，自动将 max_samples 减半重试，最多重试2次
- 显存检测失败：`torch.cuda.mem_get_info()` 异常时，fallback到150000条固定上限
- 全局模型训练失败：记录错误日志，继续使用本地模型，不中断整体训练流程

---

## 测试策略

### 单元测试

**factor.py 测试**：
- 验证 `generate_factors()` 输出形状为 (T, 165)，dtype 为 float32
- 验证长周期特征（列147-164）的数学正确性（与手动计算对比）
- 验证因果性：修改未来数据不影响当前时刻特征
- 验证边界条件：T < 480 时前479行长周期特征为 NaN
- 验证零分母处理：高低价相等时价格区间位置输出0.5

**predict.py 测试**：
- 验证数据集ID解析：`"dataset0"→0`，`"dataset29"→29`
- 验证模型选择逻辑：`use_global_model=true` 时使用全局模型
- 验证降级逻辑：各种模型文件缺失组合下不抛异常
- 验证权重文件向后兼容：v3格式文件能正确读取

**train.py 测试**：
- 验证采样上限计算公式
- 验证时序采样保持单调递增
- 验证高波动权重计算（含NaN/零值处理）

### 属性测试

使用 [Hypothesis](https://hypothesis.readthedocs.io/) 库实现属性测试，最少100次迭代。

**属性测试配置**：
```python
from hypothesis import given, settings
from hypothesis import strategies as st

@settings(max_examples=100)
@given(st.arrays(dtype=np.float32, shape=st.tuples(
    st.integers(min_value=500, max_value=2000),  # T
    st.just(5)  # OHLCV
), elements=st.floats(min_value=0.01, max_value=1000.0, allow_nan=False)))
def test_property_1_feature_dim(ohlcv):
    """Feature: model-optimization-v4, Property 1: 特征矩阵维数不变量"""
    result = generate_factors("dataset0", ohlcv)
    assert result.shape[1] == 165
```

**各属性对应的测试标签**：

| 属性 | 测试标签 |
|------|---------|
| 属性1 | `Feature: model-optimization-v4, Property 1: 特征矩阵维数不变量` |
| 属性2 | `Feature: model-optimization-v4, Property 2: 特征矩阵精度不变量` |
| 属性3 | `Feature: model-optimization-v4, Property 3: 长周期特征因果性` |
| 属性4 | `Feature: model-optimization-v4, Property 4: 长周期动量特征数学正确性` |
| 属性5 | `Feature: model-optimization-v4, Property 5: 价格区间位置有界性` |
| 属性6 | `Feature: model-optimization-v4, Property 6: 数据集ID解析正确性` |
| 属性7 | `Feature: model-optimization-v4, Property 7: 模型选择逻辑正确性` |
| 属性8 | `Feature: model-optimization-v4, Property 8: 权重文件向后兼容性` |
| 属性9 | `Feature: model-optimization-v4, Property 9: 降级健壮性` |
| 属性10 | `Feature: model-optimization-v4, Property 10: 采样上限不变量` |
| 属性11 | `Feature: model-optimization-v4, Property 11: 采样时序保持性` |
| 属性12 | `Feature: model-optimization-v4, Property 12: 采样权重计算正确性` |

### 集成测试

- **推理时间测试**：在RTX 4090上运行完整30数据集推理，验证总时间 ≤ 40分钟
- **提交包大小测试**：训练完成后检查 `models_v4/` 目录总大小 ≤ 140 MB
- **IC对比测试**：对Negative_IC_Datasets（dataset6/13/15/28）验证v4 Ret60 IC高于v3

### 回归测试

- 对所有30个数据集运行v4推理，与v3结果对比，确保无数据集出现IC显著下降（阈值：-0.01）
- 验证v3格式的 `ensemble_weights.json` 在v4中能正确加载（向后兼容性）
