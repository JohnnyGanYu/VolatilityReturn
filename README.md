# VolatilityReturn

![Python](https://img.shields.io/badge/python-3.12-blue) ![License](https://img.shields.io/badge/license-MIT-green)

华东杯数学建模竞赛 C 题解决方案。对 30 支高波动标的预测 Ret5（5分钟收益率）和 Ret60（60分钟收益率），评测指标为 Pearson IC，覆盖 normal 和 extreme 两种行情，共 4 个 IC 取均值。

最终成绩：avg IC ≈ 0.0663，历史最佳 0.0590。

## 方法概述

基于 1 分钟 K 线数据构造 165 维特征（Numba JIT 加速），训练多版本模型集成：

- **LightGBM**：per-dataset 局部模型 + 全局模型 + 极端行情专用模型
- **GRU**：双目标（Ret5+Ret60 同时预测），window=60
- **Transformer**：双目标，window=60

9 个信号源通过 per-dataset 独立权重集成，权重由平台反馈贪心爬坡自动优化。

## 项目结构

```
factor.py              特征工程（165维，Numba JIT）
train.py               模型训练入口
predict_v7.py          推理脚本（9信号源集成）
evaluate_local.py      本地评估

scripts/
  auto_iterate_v7.py   自动贪心爬坡优化（平台反馈驱动）
  package_v7_smart.py  智能打包（只打包权重非零的模型）
  submit_to_platform.py 平台提交 + 轮询结果
  feedback_optimize.py  权重优化
  generate_submissions.py 批量生成提交包
  train_global.py      全局模型训练

tests/                 单元测试 + 属性测试
.kiro/specs/           各版本设计文档（v1-v6 演进记录）
```

## 模型演进

| 版本 | 日期 | 核心改动 |
|------|------|---------|
| v1 | 2026-05-01 | 初始 LGB pipeline，109维特征 |
| v2 | 2026-05-05 | IC feval + 特征扩展至147维 + GRU |
| v3 | 2026-05-08 | Transformer ensemble + GPU训练 |
| v4 | 2026-05-10 | 全局LGB模型 + 165维长周期特征 |
| v5 | 2026-05-12 | H20全量训练，双目标GRU/TF |
| v6 | 2026-05-13 | IC-aware loss + 极端行情专用模型 |
| v7 | 2026-05-15 | 9信号源集成 + 自动贪心爬坡 |

## 关键结论

- v6 单目标序列模型（GRU/TF single-target）在平台上过拟合，加入后 eR5 暴跌，不可用
- v5 双目标 GRU/TF（window=60）对 eR60 有显著贡献
- 逐 dataset 独立优化 + 平台反馈贪心爬坡是最可靠的提升路径
- 提交包需 ≤ 150MB，智能打包只包含权重非零的模型文件

## 环境

```bash
pip install -e .
```

依赖：Python 3.12，LightGBM，PyTorch，Numba，NumPy，pandas。

训练需要竞赛数据集（不含在本仓库中）。模型文件因体积过大不纳入版本控制，见 `.gitignore`。
