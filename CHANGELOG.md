# Changelog

## v7 (2026-05-15)
- 9-source ensemble: v5 LGB local/global, v5 GRU/TF dual-target, v6 LGB local/global/extreme, v6 GRU/TF single-target
- Per-dataset independent weight optimization
- Automated greedy hill climbing driven by platform feedback (avg IC: 0.0663)
- Smart packaging: only bundle models with non-zero weights to stay under 150MB

## v6 (2026-05-11)
- Replace MSE loss with α·Pearson_Loss + (1-α)·ListMLE (α=0.5)
- Scale GRU hidden size 64 → 128
- Dual-window strategy: Ret5 w=20, Ret60 w=240
- Add LGB_Extreme: dedicated model trained on extreme-market samples only
- Iterative weight reoptimization scripts

## v5 (2026-05-08)
- H20 96GB training environment: remove all sampling caps
- On-the-fly batch-wise window construction (~0.6GB peak vs ~80GB)
- Preload validation windows to GPU (~17GB)
- Increase batch_size 4096 → 16384, add AMP (fp16) training
- Dual-target GRU/Transformer: predict Ret5 + Ret60 simultaneously

## v4 (2026-05-05)
- Cross-dataset global LightGBM (dataset_id as categorical feature)
- Extend features to 165 dims: 18 new long-period features (window=240/480) for Ret60
- Inference: select global vs local model by validation IC per dataset
- Gzip-compress LGB models to fit 140MB submission limit

## v3 (2026-04-30)
- Add Transformer model (d=64, nhead=4, layers=4) alongside GRU
- GPU-accelerated sequence model training on RTX 4090
- Fix Ret5 overfitting: tune num_leaves per target type
- Three-model ensemble: LGB + GRU + Transformer

## v2 (2026-04-25)
- Replace MAE early stopping with custom Pearson IC feval
- Expand features 109 → 147 dims (EMA ratios, skewness/kurtosis, volume-weighted returns, realized variance)
- Add per-dataset 2-layer GRU on 60-bar sliding windows
- Multi-threaded LightGBM training

## v1 (2026-04-20)
- Initial pipeline: feature engineering + LightGBM + inference
- 109-dim OHLCV features with Numba JIT acceleration
- Per-dataset LightGBM models for Ret5 and Ret60
- Temporal 80/20 train/val split
