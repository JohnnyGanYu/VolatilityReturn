#!/usr/bin/env python3
"""
智能迭代优化：每个 dataset 独立优化，记录最佳配置。

策略：
- 前 N 轮：大幅随机探索（v5/v6 模型随机组合）
- 后续轮：贪心爬坡（在当前最佳基础上小幅扰动）
- 每轮提交后：逐 dataset 对比，更新各自最佳
- 最终提交：每个 dataset 用各自历史最佳

用法：
    # 生成下一轮权重
    python scripts/smart_iterate.py next --round 4

    # 查看每个 dataset 的最佳配置
    python scripts/smart_iterate.py best

    # 生成最终提交（每个 dataset 用最佳配置）
    python scripts/smart_iterate.py final
"""

import json
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from copy import deepcopy

NUM_DATASETS = 30
FEEDBACK_DIR = Path("feedback_state")
STATE_FILE = FEEDBACK_DIR / "smart_state.json"
MODEL_DIR = Path("models_v6")

# 所有可用的权重 key（v6 only，不混合 v5）
ALL_RET5_KEYS = ["ret5_w_local", "ret5_w_global", "ret5_w_gru_ret5", "ret5_w_tf_ret5"]
ALL_RET60_KEYS = ["ret60_w_local", "ret60_w_global", "ret60_w_gru_ret60", "ret60_w_tf_ret60"]

EXPLORE_ROUNDS = 6  # 前6轮大幅探索（已完成），第7轮起贪心爬坡


def load_state():
    """加载迭代状态。"""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    # 初始化
    state = {
        "best_weights": {},  # 每个 dataset 的最佳权重
        "best_ic": {},       # 每个 dataset 的最佳 IC (avg of 4 metrics)
        "history": {},       # 每个 dataset 的所有尝试: [{weights, ic, round}]
        "round": 0,
    }
    for i in range(NUM_DATASETS):
        ds = f"dataset{i}"
        state["best_weights"][ds] = {
            "ret5_w_local": 1.0, "ret5_w_global": 0.0,
            "ret5_w_gru": 0.0, "ret5_w_tf": 0.0,
            "ret5_w_gru_ret5": 0.0, "ret5_w_tf_ret5": 0.0,
            "ret60_w_local": 1.0, "ret60_w_global": 0.0,
            "ret60_w_gru": 0.0, "ret60_w_tf": 0.0,
            "ret60_w_gru_ret60": 0.0, "ret60_w_tf_ret60": 0.0,
        }
        state["best_ic"][ds] = {"nR5": 0, "nR60": 0, "eR5": 0, "eR60": 0, "avg": 0}
        state["history"][ds] = []
    return state


def save_state(state):
    """保存迭代状态。"""
    FEEDBACK_DIR.mkdir(exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def normalize(w, keys):
    """归一化权重。"""
    total = sum(max(0, w.get(k, 0)) for k in keys)
    if total > 0:
        for k in keys:
            w[k] = round(max(0, w.get(k, 0)) / total, 3)
    else:
        w[keys[0]] = 1.0
        for k in keys[1:]:
            w[k] = 0.0


def random_weights(rng):
    """生成随机权重配置（大幅探索）。"""
    w = {}
    # Ret5: 随机选择 2-3 个模型
    r5_vals = rng.dirichlet(np.ones(len(ALL_RET5_KEYS)) * 0.5)
    for k, v in zip(ALL_RET5_KEYS, r5_vals):
        w[k] = round(float(v), 3)
    # 随机置零一些（稀疏化）
    for k in ALL_RET5_KEYS:
        if rng.random() < 0.4:
            w[k] = 0.0
    normalize(w, ALL_RET5_KEYS)

    # Ret60: 同理
    r60_vals = rng.dirichlet(np.ones(len(ALL_RET60_KEYS)) * 0.5)
    for k, v in zip(ALL_RET60_KEYS, r60_vals):
        w[k] = round(float(v), 3)
    for k in ALL_RET60_KEYS:
        if rng.random() < 0.4:
            w[k] = 0.0
    normalize(w, ALL_RET60_KEYS)

    return w


def perturb_weights(w, rng, magnitude=0.15):
    """在当前权重基础上小幅扰动（贪心爬坡）。"""
    w = deepcopy(w)
    for k in ALL_RET5_KEYS:
        w[k] = max(0, w.get(k, 0) + rng.uniform(-magnitude, magnitude))
    normalize(w, ALL_RET5_KEYS)

    for k in ALL_RET60_KEYS:
        w[k] = max(0, w.get(k, 0) + rng.uniform(-magnitude, magnitude))
    normalize(w, ALL_RET60_KEYS)

    return w


def is_duplicate(w, history, threshold=0.05):
    """检查是否和历史配置重复。"""
    for entry in history:
        hw = entry["weights"]
        diff = sum(abs(w.get(k, 0) - hw.get(k, 0)) for k in ALL_RET5_KEYS + ALL_RET60_KEYS)
        if diff < threshold:
            return True
    return False


def update_best(state, feedback, round_num):
    """根据反馈更新每个 dataset 的最佳配置。"""
    current_weights_path = FEEDBACK_DIR / f"weights_round{round_num}.json"
    if not current_weights_path.exists():
        print(f"WARNING: {current_weights_path} not found")
        return

    with open(current_weights_path) as f:
        current_weights = json.load(f)

    improved = []
    degraded = []

    for i in range(NUM_DATASETS):
        ds = f"dataset{i}"
        fb = feedback.get(ds, {})
        avg = (fb.get("nR5", 0) + fb.get("nR60", 0) + fb.get("eR5", 0) + fb.get("eR60", 0)) / 4

        # 记录到历史
        entry = {
            "weights": current_weights.get(ds, {}),
            "ic": fb,
            "avg": avg,
            "round": round_num,
        }
        state["history"][ds].append(entry)

        # 更新最佳
        if avg > state["best_ic"][ds].get("avg", 0):
            state["best_ic"][ds] = {**fb, "avg": avg}
            state["best_weights"][ds] = current_weights.get(ds, {})
            improved.append(f"  {ds}: avg {state['best_ic'][ds]['avg']:.4f} → {avg:.4f} ✅")
        else:
            degraded.append(ds)

    if improved:
        print(f"改善 ({len(improved)} datasets):")
        for s in improved:
            print(s)
    print(f"未改善: {len(degraded)} datasets")


def generate_next(state, round_num):
    """生成下一轮权重。"""
    rng = np.random.default_rng(42 + round_num)
    exploring = round_num <= EXPLORE_ROUNDS

    weights = {}
    for i in range(NUM_DATASETS):
        ds = f"dataset{i}"
        history = state["history"][ds]

        if exploring:
            # 大幅随机探索，避免重复
            for _ in range(100):
                w = random_weights(rng)
                if not is_duplicate(w, history):
                    break
        else:
            # 贪心爬坡：在最佳基础上小幅扰动
            best_w = state["best_weights"][ds]
            for _ in range(100):
                w = perturb_weights(best_w, rng, magnitude=0.1)
                if not is_duplicate(w, history):
                    break

        weights[ds] = w

    return weights


def cmd_next(args):
    """生成下一轮权重。"""
    round_num = args.round
    next_round = round_num + 1

    state = load_state()

    # 如果有反馈，先更新最佳
    feedback_path = FEEDBACK_DIR / f"iter_{round_num}.json"
    if feedback_path.exists():
        with open(feedback_path) as f:
            feedback = json.load(f).get("results", {})
        update_best(state, feedback, round_num)
    else:
        print(f"WARNING: No feedback for round {round_num}")

    state["round"] = next_round

    # 生成下一轮
    weights = generate_next(state, next_round)

    # 保存
    output = MODEL_DIR / "ensemble_weights.json"
    with open(output, "w") as f:
        json.dump(weights, f, indent=2)

    save_state(state)

    # 统计
    exploring = next_round <= EXPLORE_ROUNDS
    mode = "探索" if exploring else "爬坡"
    print(f"\nRound {next_round} 权重已生成 (模式: {mode})")
    print(f"  输出: {output}")

    # 显示几个 dataset 的权重
    for ds in ["dataset0", "dataset4", "dataset11", "dataset15"]:
        w = weights[ds]
        active = [(k.replace("ret5_w_", "").replace("ret60_w_", ""), v) for k, v in w.items() if v > 0]
        print(f"  {ds}: {active}")


def cmd_best(args):
    """显示每个 dataset 的最佳配置。"""
    state = load_state()

    print(f"每个 dataset 的历史最佳 (共 {state['round']} 轮):")
    print(f"{'Dataset':<12} {'avg':>6} {'nR5':>6} {'nR60':>6} {'eR5':>6} {'eR60':>6}  配置")
    print("-" * 90)

    for i in range(NUM_DATASETS):
        ds = f"dataset{i}"
        ic = state["best_ic"][ds]
        w = state["best_weights"][ds]
        active = [f"{k.split('_w_')[1]}={v:.1f}" for k, v in w.items() if v > 0]
        print(f"{ds:<12} {ic.get('avg',0):>6.3f} {ic.get('nR5',0):>6.3f} {ic.get('nR60',0):>6.3f} "
              f"{ic.get('eR5',0):>6.3f} {ic.get('eR60',0):>6.3f}  {' '.join(active)}")

    # 总平均
    all_avg = [state["best_ic"][f"dataset{i}"].get("avg", 0) for i in range(NUM_DATASETS)]
    print(f"\n总平均 (每个 dataset 取最佳): {np.mean(all_avg):.4f}")


def cmd_final(args):
    """生成最终提交：每个 dataset 用各自最佳配置。"""
    state = load_state()

    weights = {}
    for i in range(NUM_DATASETS):
        ds = f"dataset{i}"
        weights[ds] = state["best_weights"][ds]

    output = MODEL_DIR / "ensemble_weights.json"
    with open(output, "w") as f:
        json.dump(weights, f, indent=2)

    print(f"最终权重已生成: {output}")
    print(f"每个 dataset 使用各自历史最佳配置")

    all_avg = [state["best_ic"][f"dataset{i}"].get("avg", 0) for i in range(NUM_DATASETS)]
    print(f"预期平均 IC: {np.mean(all_avg):.4f}")


def cmd_init(args):
    """用已有的反馈初始化状态。"""
    state = load_state()

    # 导入所有已有的反馈
    for iter_file in sorted(FEEDBACK_DIR.glob("iter_*.json")):
        with open(iter_file) as f:
            data = json.load(f)
        round_num = data.get("round", 0)
        feedback = data.get("results", {})

        weights_file = FEEDBACK_DIR / f"weights_round{round_num}.json"
        if weights_file.exists():
            with open(weights_file) as f:
                current_weights = json.load(f)

            for i in range(NUM_DATASETS):
                ds = f"dataset{i}"
                fb = feedback.get(ds, {})
                avg = (fb.get("nR5", 0) + fb.get("nR60", 0) + fb.get("eR5", 0) + fb.get("eR60", 0)) / 4

                entry = {
                    "weights": current_weights.get(ds, {}),
                    "ic": fb,
                    "avg": avg,
                    "round": round_num,
                }
                state["history"][ds].append(entry)

                if avg > state["best_ic"][ds].get("avg", 0):
                    state["best_ic"][ds] = {**fb, "avg": avg}
                    state["best_weights"][ds] = current_weights.get(ds, {})

            state["round"] = max(state["round"], round_num)
            print(f"  导入 Round {round_num}")

    save_state(state)
    print(f"\n初始化完成，已导入 {state['round']} 轮数据")
    cmd_best(args)


def main():
    parser = argparse.ArgumentParser(description="智能迭代优化")
    sub = parser.add_subparsers(dest="cmd")

    p_next = sub.add_parser("next", help="生成下一轮权重")
    p_next.add_argument("--round", type=int, required=True)

    p_best = sub.add_parser("best", help="查看最佳配置")

    p_final = sub.add_parser("final", help="生成最终提交权重")

    p_init = sub.add_parser("init", help="从已有反馈初始化状态")

    args = parser.parse_args()

    if args.cmd == "next":
        cmd_next(args)
    elif args.cmd == "best":
        cmd_best(args)
    elif args.cmd == "final":
        cmd_final(args)
    elif args.cmd == "init":
        cmd_init(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
