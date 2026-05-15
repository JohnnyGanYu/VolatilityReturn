#!/usr/bin/env python3
"""
提交迭代管理脚本。

功能：
1. 打包提交（根据 ensemble_weights.json 自动选择需要的模型文件）
2. 记录平台反馈
3. 根据反馈生成下一轮权重
4. 对比多轮结果

用法：
    # 打包当前方案
    python scripts/submit_iterate.py pack --round 1

    # 记录平台反馈（手动填入或从文件读取）
    python scripts/submit_iterate.py record --round 1 --feedback feedback_state/iter_1.json

    # 根据反馈生成下一轮权重
    python scripts/submit_iterate.py next --round 1

    # 查看历史对比
    python scripts/submit_iterate.py compare
"""

import os
import sys
import json
import tarfile
import shutil
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

MODEL_DIR = Path("models_v6")
FEEDBACK_DIR = Path("feedback_state")
SUBMISSION_DIR = Path("submissions")
MAX_SIZE_MB = 150.0

NUM_DATASETS = 30
RET5_KEYS = ["ret5_w_local", "ret5_w_global", "ret5_w_gru_ret5", "ret5_w_tf_ret5", "ret5_w_extreme"]
RET60_KEYS = ["ret60_w_local", "ret60_w_global", "ret60_w_gru_ret60", "ret60_w_tf_ret60", "ret60_w_extreme"]


# =============================================================================
# 工具函数
# =============================================================================

def get_needed_files(weights: dict) -> list:
    """根据权重文件确定需要打包的模型文件列表。"""
    files = set()
    
    # 始终包含 LGB local
    for i in range(NUM_DATASETS):
        for target in ["ret5", "ret60"]:
            for ext in [".txt.gz", ".txt"]:
                p = MODEL_DIR / f"lgb_{target}_dataset{i}{ext}"
                if p.exists():
                    files.add(p.name)
                    break
    
    # 始终包含 LGB global（如果存在）
    for f in ["lgb_ret5_global.txt.gz", "lgb_ret60_global.txt.gz"]:
        if (MODEL_DIR / f).exists():
            files.add(f)
    
    # 始终包含 LGB extreme（predict.py 自动融合）
    for i in range(NUM_DATASETS):
        for target in ["ret5", "ret60"]:
            f = f"lgb_extreme_{target}_dataset{i}.txt"
            if (MODEL_DIR / f).exists():
                files.add(f)
    
    # 只包含权重 > 0 的序列模型
    for i in range(NUM_DATASETS):
        ds = f"dataset{i}"
        w = weights.get(ds, {})
        
        if w.get("ret5_w_gru_ret5", 0) > 0:
            f = f"gru_ret5_dataset{i}.pt"
            if (MODEL_DIR / f).exists(): files.add(f)
        
        if w.get("ret60_w_gru_ret60", 0) > 0:
            f = f"gru_ret60_dataset{i}.pt"
            if (MODEL_DIR / f).exists(): files.add(f)
        
        if w.get("ret5_w_tf_ret5", 0) > 0:
            f = f"transformer_ret5_dataset{i}.pt"
            if (MODEL_DIR / f).exists(): files.add(f)
        
        if w.get("ret60_w_tf_ret60", 0) > 0:
            f = f"transformer_ret60_dataset{i}.pt"
            if (MODEL_DIR / f).exists(): files.add(f)
    
    return sorted(files)


def estimate_size(files: list) -> float:
    """估算打包后大小（MB）。tar.gz 对 .txt 压缩率约 60%，对 .gz/.pt 几乎不压缩。"""
    total = 0
    for f in files:
        p = MODEL_DIR / f
        if p.exists():
            size = p.stat().st_size
            # .txt 文件会被 tar.gz 压缩约 60%
            if f.endswith(".txt"):
                size = int(size * 0.4)
            total += size
    # 加上代码文件 (~100KB)
    total += 100 * 1024
    return total / (1024 * 1024)


def normalize_weights(w: dict, keys: list):
    """归一化权重到 sum=1.0。"""
    total = sum(max(0, w.get(k, 0)) for k in keys)
    if total > 0:
        for k in keys:
            w[k] = round(max(0, w.get(k, 0)) / total, 3)
    else:
        w[keys[0]] = 1.0
        for k in keys[1:]:
            w[k] = 0.0


# =============================================================================
# pack: 打包提交
# =============================================================================

def cmd_pack(args):
    """打包当前方案为提交包。"""
    round_num = args.round
    weights_path = MODEL_DIR / "ensemble_weights.json"
    
    if not weights_path.exists():
        print(f"ERROR: {weights_path} not found")
        sys.exit(1)
    
    with open(weights_path) as f:
        weights = json.load(f)
    
    # 确定需要的文件
    model_files = get_needed_files(weights)
    est_size = estimate_size(model_files)
    
    print(f"Round {round_num} 打包")
    print(f"  模型文件: {len(model_files)}")
    print(f"  预估大小: {est_size:.1f} MB (限制 {MAX_SIZE_MB} MB)")
    
    if est_size > MAX_SIZE_MB:
        print(f"  ERROR: 超过 {MAX_SIZE_MB} MB 限制！")
        print(f"  需要减少序列模型或去掉 extreme。")
        sys.exit(1)
    
    # 创建提交包
    SUBMISSION_DIR.mkdir(exist_ok=True)
    output = SUBMISSION_DIR / f"submission_round{round_num}.tar.gz"
    
    with tarfile.open(str(output), "w:gz") as tar:
        # 代码文件
        tar.add("factor.py", arcname="factor.py")
        tar.add("predict.py", arcname="predict.py")
        
        # 权重文件
        tar.add(str(weights_path), arcname="ensemble_weights.json")
        
        # 模型文件
        for f in model_files:
            tar.add(str(MODEL_DIR / f), arcname=f)
    
    actual_size = output.stat().st_size / (1024 * 1024)
    print(f"\n  ✅ 打包完成: {output}")
    print(f"  实际大小: {actual_size:.1f} MB")
    print(f"  文件数: {len(model_files) + 3} (模型 + 代码 + 权重)")
    
    # 保存本轮权重快照
    snapshot = FEEDBACK_DIR / f"weights_round{round_num}.json"
    FEEDBACK_DIR.mkdir(exist_ok=True)
    shutil.copy(str(weights_path), str(snapshot))
    print(f"  权重快照: {snapshot}")


# =============================================================================
# record: 记录平台反馈
# =============================================================================

def cmd_record(args):
    """记录平台反馈结果。"""
    round_num = args.round
    FEEDBACK_DIR.mkdir(exist_ok=True)
    
    if args.feedback and Path(args.feedback).exists():
        # 从文件读取
        with open(args.feedback) as f:
            feedback = json.load(f)
    else:
        # 交互式输入
        print(f"请输入 Round {round_num} 的平台反馈 IC 值。")
        print("格式: nR5 nR60 eR5 eR60 (空格分隔，每行一个 dataset)")
        print("输入 'done' 结束\n")
        
        feedback = {}
        for i in range(NUM_DATASETS):
            ds = f"dataset{i}"
            line = input(f"  {ds}: ").strip()
            if line.lower() == "done":
                break
            parts = line.split()
            if len(parts) >= 4:
                feedback[ds] = {
                    "nR5": float(parts[0]),
                    "nR60": float(parts[1]),
                    "eR5": float(parts[2]),
                    "eR60": float(parts[3]),
                }
    
    # 保存
    output = FEEDBACK_DIR / f"iter_{round_num}.json"
    result = {
        "round": round_num,
        "timestamp": datetime.now().isoformat(),
        "results": feedback,
    }
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n  ✅ 反馈已保存: {output}")
    print(f"  记录了 {len(feedback)} 个 dataset 的结果")


# =============================================================================
# next: 生成下一轮权重
# =============================================================================

def cmd_next(args):
    """根据反馈生成下一轮权重。"""
    round_num = args.round
    next_round = round_num + 1
    
    # 加载当前权重
    current_path = FEEDBACK_DIR / f"weights_round{round_num}.json"
    if not current_path.exists():
        current_path = MODEL_DIR / "ensemble_weights.json"
    with open(current_path) as f:
        current = json.load(f)
    
    # 加载反馈
    feedback_path = FEEDBACK_DIR / f"iter_{round_num}.json"
    if not feedback_path.exists():
        print(f"ERROR: 反馈文件不存在: {feedback_path}")
        print(f"请先运行: python scripts/submit_iterate.py record --round {round_num}")
        sys.exit(1)
    with open(feedback_path) as f:
        feedback_data = json.load(f)
    feedback = feedback_data.get("results", {})
    
    # 加载历史最佳（如果有）
    best_path = FEEDBACK_DIR / "best_weights.json"
    best_ic = FEEDBACK_DIR / "best_ic.json"
    if best_path.exists():
        with open(best_path) as f:
            best_weights = json.load(f)
        with open(best_ic) as f:
            best_results = json.load(f)
    else:
        best_weights = current.copy()
        best_results = feedback.copy()
    
    # 生成下一轮权重
    rng = np.random.default_rng(42 + next_round)
    aggressive = next_round <= 5  # 前5轮大幅探索
    
    next_weights = {}
    actions = []
    
    for i in range(NUM_DATASETS):
        ds = f"dataset{i}"
        w = current.get(ds, {}).copy()
        ds_fb = feedback.get(ds, {})
        ds_best = best_results.get(ds, {})
        
        # 计算平均 IC
        avg_r5 = (ds_fb.get("nR5", 0) + ds_fb.get("eR5", 0)) / 2
        avg_r60 = (ds_fb.get("nR60", 0) + ds_fb.get("eR60", 0)) / 2
        
        # 如果本轮比历史最佳好，更新最佳
        best_avg_r5 = (ds_best.get("nR5", 0) + ds_best.get("eR5", 0)) / 2
        best_avg_r60 = (ds_best.get("nR60", 0) + ds_best.get("eR60", 0)) / 2
        
        if avg_r5 > best_avg_r5:
            best_results[ds] = ds_fb
            best_weights[ds] = current.get(ds, {}).copy()
        
        # --- Ret5 策略 ---
        if aggressive:
            # 前几轮：大幅随机跳
            for k in RET5_KEYS:
                w[k] = max(0, w.get(k, 0) + rng.uniform(-0.3, 0.3))
            action_r5 = f"explore(avg={avg_r5:.3f})"
        elif avg_r5 >= 0.10:
            # IC 高：大幅跳，探索更优
            for k in RET5_KEYS:
                w[k] = max(0, w.get(k, 0) + rng.uniform(-0.3, 0.3))
            action_r5 = f"big_jump(avg={avg_r5:.3f})"
        elif avg_r5 < 0.03:
            # IC 低：回退到历史最佳
            if ds in best_weights:
                for k in RET5_KEYS:
                    w[k] = best_weights[ds].get(k, 0)
            action_r5 = f"revert_best(avg={avg_r5:.3f})"
        else:
            # IC 中等：小幅扰动
            for k in RET5_KEYS:
                w[k] = max(0, w.get(k, 0) + rng.uniform(-0.1, 0.1))
            action_r5 = f"perturb(avg={avg_r5:.3f})"
        
        # --- Ret60 策略 ---
        if aggressive:
            for k in RET60_KEYS:
                w[k] = max(0, w.get(k, 0) + rng.uniform(-0.3, 0.3))
            action_r60 = f"explore(avg={avg_r60:.3f})"
        elif avg_r60 >= 0.10:
            for k in RET60_KEYS:
                w[k] = max(0, w.get(k, 0) + rng.uniform(-0.3, 0.3))
            action_r60 = f"big_jump(avg={avg_r60:.3f})"
        elif avg_r60 < 0.03:
            if ds in best_weights:
                for k in RET60_KEYS:
                    w[k] = best_weights[ds].get(k, 0)
            action_r60 = f"revert_best(avg={avg_r60:.3f})"
        else:
            for k in RET60_KEYS:
                w[k] = max(0, w.get(k, 0) + rng.uniform(-0.1, 0.1))
            action_r60 = f"perturb(avg={avg_r60:.3f})"
        
        # 归一化
        normalize_weights(w, RET5_KEYS)
        normalize_weights(w, RET60_KEYS)
        
        next_weights[ds] = w
        actions.append(f"  {ds}: R5={action_r5} | R60={action_r60}")
    
    # 保存下一轮权重
    output = MODEL_DIR / "ensemble_weights.json"
    with open(output, "w") as f:
        json.dump(next_weights, f, indent=2)
    
    # 保存历史最佳
    with open(best_path, "w") as f:
        json.dump(best_weights, f, indent=2)
    with open(best_ic, "w") as f:
        json.dump(best_results, f, indent=2)
    
    # 检查包大小
    model_files = get_needed_files(next_weights)
    est_size = estimate_size(model_files)
    
    print(f"Round {next_round} 权重已生成")
    print(f"  输出: {output}")
    print(f"  预估包大小: {est_size:.1f} MB")
    if est_size > MAX_SIZE_MB:
        print(f"  ⚠️ 超过 {MAX_SIZE_MB} MB！需要裁剪序列模型。")
        # 自动裁剪：去掉权重最小的序列模型直到 < 150MB
        _auto_trim(next_weights, est_size)
        with open(output, "w") as f:
            json.dump(next_weights, f, indent=2)
        model_files = get_needed_files(next_weights)
        est_size = estimate_size(model_files)
        print(f"  裁剪后: {est_size:.1f} MB")
    
    print(f"\n策略:")
    for a in actions:
        print(a)
    
    print(f"\n下一步: python scripts/submit_iterate.py pack --round {next_round}")


def _auto_trim(weights: dict, current_size: float):
    """自动裁剪低权重序列模型直到包大小 < 150MB。"""
    # 收集所有序列模型权重
    seq_entries = []
    for i in range(NUM_DATASETS):
        ds = f"dataset{i}"
        w = weights.get(ds, {})
        for key, prefix in [
            ("ret5_w_gru_ret5", f"gru_ret5_dataset{i}.pt"),
            ("ret5_w_tf_ret5", f"transformer_ret5_dataset{i}.pt"),
            ("ret60_w_gru_ret60", f"gru_ret60_dataset{i}.pt"),
            ("ret60_w_tf_ret60", f"transformer_ret60_dataset{i}.pt"),
        ]:
            val = w.get(key, 0)
            if val > 0:
                fpath = MODEL_DIR / prefix
                fsize = fpath.stat().st_size / (1024*1024) if fpath.exists() else 0
                seq_entries.append((val, fsize, ds, key))
    
    # 按权重从小到大排序，逐个去掉
    seq_entries.sort(key=lambda x: x[0])
    for val, fsize, ds, key in seq_entries:
        if current_size <= MAX_SIZE_MB:
            break
        weights[ds][key] = 0.0
        normalize_weights(weights[ds], RET5_KEYS if "ret5" in key else RET60_KEYS)
        current_size -= fsize
        print(f"    裁剪: {ds}.{key} (weight={val:.2f}, {fsize:.1f}MB)")


# =============================================================================
# compare: 对比历史结果
# =============================================================================

def cmd_compare(args):
    """对比所有轮次的平台反馈。"""
    FEEDBACK_DIR.mkdir(exist_ok=True)
    
    # 收集所有反馈文件
    iter_files = sorted(FEEDBACK_DIR.glob("iter_*.json"))
    if not iter_files:
        print("没有反馈记录。请先提交并记录反馈。")
        return
    
    rounds = []
    for f in iter_files:
        with open(f) as fp:
            data = json.load(fp)
        rounds.append(data)
    
    # 打印对比表
    print(f"{'Round':>6} {'nR5':>8} {'nR60':>8} {'eR5':>8} {'eR60':>8} {'Avg':>8}")
    print("-" * 50)
    
    for data in rounds:
        r = data.get("round", "?")
        results = data.get("results", {})
        
        nR5s = [results[ds].get("nR5", 0) for ds in results]
        nR60s = [results[ds].get("nR60", 0) for ds in results]
        eR5s = [results[ds].get("eR5", 0) for ds in results]
        eR60s = [results[ds].get("eR60", 0) for ds in results]
        
        mean_nR5 = np.mean(nR5s) if nR5s else 0
        mean_nR60 = np.mean(nR60s) if nR60s else 0
        mean_eR5 = np.mean(eR5s) if eR5s else 0
        mean_eR60 = np.mean(eR60s) if eR60s else 0
        avg = (mean_nR5 + mean_nR60 + mean_eR5 + mean_eR60) / 4
        
        print(f"{r:>6} {mean_nR5:>8.4f} {mean_nR60:>8.4f} {mean_eR5:>8.4f} {mean_eR60:>8.4f} {avg:>8.4f}")
    
    # 显示每个 dataset 的最佳轮次
    if len(rounds) > 1:
        print(f"\n每个 dataset 的最佳轮次:")
        for i in range(NUM_DATASETS):
            ds = f"dataset{i}"
            best_round = -1
            best_avg = -1
            for data in rounds:
                r = data.get("round", 0)
                ds_result = data.get("results", {}).get(ds, {})
                avg = sum(ds_result.values()) / 4 if ds_result else 0
                if avg > best_avg:
                    best_avg = avg
                    best_round = r
            print(f"  {ds}: Round {best_round} (avg IC={best_avg:.4f})")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="提交迭代管理")
    sub = parser.add_subparsers(dest="cmd")
    
    # pack
    p_pack = sub.add_parser("pack", help="打包提交")
    p_pack.add_argument("--round", type=int, required=True)
    
    # record
    p_record = sub.add_parser("record", help="记录平台反馈")
    p_record.add_argument("--round", type=int, required=True)
    p_record.add_argument("--feedback", default=None, help="反馈 JSON 文件路径")
    
    # next
    p_next = sub.add_parser("next", help="生成下一轮权重")
    p_next.add_argument("--round", type=int, required=True, help="当前轮次（基于此轮反馈生成下一轮）")
    
    # compare
    p_compare = sub.add_parser("compare", help="对比历史结果")
    
    args = parser.parse_args()
    
    if args.cmd == "pack":
        cmd_pack(args)
    elif args.cmd == "record":
        cmd_record(args)
    elif args.cmd == "next":
        cmd_next(args)
    elif args.cmd == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
