#!/usr/bin/env python3
"""
v7 自动迭代优化：贪心爬坡 + 自动提交 + 自动等待结果。

策略：
- 以当前最佳权重为基线
- 每轮对所有 30 个 dataset 独立做小幅扰动
- 提交到平台，等待结果
- 逐 dataset 对比，只保留改善的变更
- 循环 N 轮

用法：
    python scripts/auto_iterate_v7.py --rounds 20 --token YOUR_TOKEN
"""

import json
import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))
from submit_to_platform import submit, poll_result

NUM_DATASETS = 30
FEEDBACK_DIR = Path("feedback_state")
SUBMISSIONS_DIR = Path("submissions")

# All weight keys in v7 format
R5_KEYS = ['ret5_w_local', 'ret5_w_global', 'ret5_w_gru', 'ret5_w_tf',
           'ret5_w_v6_local', 'ret5_w_v6_global', 'ret5_w_extreme',
           'ret5_w_gru_ret5', 'ret5_w_tf_ret5']
R60_KEYS = ['ret60_w_local', 'ret60_w_global', 'ret60_w_gru', 'ret60_w_tf',
            'ret60_w_v6_local', 'ret60_w_v6_global', 'ret60_w_extreme',
            'ret60_w_gru_ret60', 'ret60_w_tf_ret60']


def normalize(d, keys):
    """Normalize weights to sum to 1."""
    total = sum(max(0, d.get(k, 0)) for k in keys)
    if total > 0:
        for k in keys:
            d[k] = round(max(0, d.get(k, 0)) / total, 4)
    else:
        d[keys[0]] = 1.0
        for k in keys[1:]:
            d[k] = 0.0


def perturb_weights(w, rng, magnitude=0.12):
    """Small random perturbation on one dataset's weights."""
    w = deepcopy(w)

    # Perturb ret5
    for k in R5_KEYS:
        if rng.random() < 0.4:
            w[k] = max(0, w.get(k, 0) + rng.uniform(-magnitude, magnitude))
    normalize(w, R5_KEYS)

    # Perturb ret60
    for k in R60_KEYS:
        if rng.random() < 0.4:
            w[k] = max(0, w.get(k, 0) + rng.uniform(-magnitude, magnitude))
    normalize(w, R60_KEYS)

    return w


def trim_to_fit(trial_weights):
    """If package would be too large, drop the lowest-weight sequence models
    until it fits. Modifies trial_weights in place."""
    # Collect all (dataset_idx, key, weight) for sequence model keys
    seq_keys_r5 = ['ret5_w_gru', 'ret5_w_tf', 'ret5_w_gru_ret5', 'ret5_w_tf_ret5']
    seq_keys_r60 = ['ret60_w_gru', 'ret60_w_tf', 'ret60_w_gru_ret60', 'ret60_w_tf_ret60']

    seq_entries = []
    for i in range(NUM_DATASETS):
        ds = f"dataset{i}"
        d = trial_weights[ds]
        for k in seq_keys_r5 + seq_keys_r60:
            if d.get(k, 0) > 0:
                seq_entries.append((d[k], ds, k))

    # Sort by weight ascending (lowest first = least important)
    seq_entries.sort(key=lambda x: x[0])

    # Drop lowest-weight sequence models one by one until we fit
    dropped = 0
    for weight, ds, k in seq_entries:
        trial_weights[ds][k] = 0.0
        if k in seq_keys_r5:
            normalize(trial_weights[ds], R5_KEYS)
        else:
            normalize(trial_weights[ds], R60_KEYS)
        dropped += 1
        if dropped >= 4:  # Drop up to 4 at a time, then re-check
            break

    return dropped


def load_state(state_file):
    """Load iteration state."""
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return None


def save_state(state, state_file):
    """Save iteration state."""
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def package_submission(weights, output_path):
    """Package submission using the smart packager."""
    # Write temp weights file
    tmp_weights = Path("_tmp_weights.json")
    with open(tmp_weights, 'w') as f:
        json.dump(weights, f, indent=2)

    # Run packager
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/package_v7_smart.py",
         "--weights", str(tmp_weights),
         "--output", str(output_path)],
        capture_output=True, text=True
    )

    tmp_weights.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  PACKAGING FAILED: {result.stderr[-500:]}")
        return False
    return True


def parse_result(result):
    """Parse platform result into per-dataset IC dict."""
    details = result.get("details", {})
    normal_ic = details.get("normal_ic", {})
    extreme_ic = details.get("extreme_ic", {})

    parsed = {}
    for i in range(NUM_DATASETS):
        ds = f"dataset{i}"
        n = normal_ic.get(ds, {})
        e = extreme_ic.get(ds, {})

        nR5 = n.get("Ret5")
        nR60 = n.get("Ret60")
        eR5 = e.get("Ret5")
        eR60 = e.get("Ret60")

        if any(v is None for v in [nR5, nR60, eR5, eR60]):
            parsed[ds] = None
        else:
            parsed[ds] = {
                "nR5": nR5, "nR60": nR60,
                "eR5": eR5, "eR60": eR60,
                "avg": (nR5 + nR60 + eR5 + eR60) / 4
            }

    return parsed


def main():
    parser = argparse.ArgumentParser(description="v7 自动迭代优化")
    parser.add_argument("--rounds", type=int, default=20, help="迭代轮数")
    parser.add_argument("--token", required=True, help="平台 token")
    parser.add_argument("--magnitude", type=float, default=0.12, help="扰动幅度")
    parser.add_argument("--state-file", default="feedback_state/v7_auto_state.json")
    parser.add_argument("--start-weights", default="weights_v5_cherrypick.json",
                        help="初始权重文件")
    args = parser.parse_args()

    state_file = Path(args.state_file)
    FEEDBACK_DIR.mkdir(exist_ok=True)
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    # Load or initialize state
    state = load_state(state_file)
    if state is None:
        # Initialize from start weights
        with open(args.start_weights) as f:
            start_w = json.load(f)

        state = {
            "best_weights": start_w,
            "best_ic": {},  # per-dataset best IC
            "round": 0,
            "history": [],
        }

        # Initialize best_ic from the known baseline (iter_14 result)
        # Will be updated after first submission
        for i in range(NUM_DATASETS):
            ds = f"dataset{i}"
            state["best_ic"][ds] = {"nR5": 0, "nR60": 0, "eR5": 0, "eR60": 0, "avg": 0}

    start_round = state["round"] + 1
    end_round = start_round + args.rounds - 1

    print("=" * 70)
    print(f"  v7 Auto-Iterate: rounds {start_round} to {end_round}")
    print(f"  Magnitude: {args.magnitude}")
    print(f"  State: {state_file}")
    print("=" * 70)

    rng = np.random.default_rng(42 + start_round)

    for round_num in range(start_round, end_round + 1):
        print(f"\n{'='*70}")
        print(f"  Round {round_num}/{end_round}")
        print(f"{'='*70}")

        # Generate perturbed weights for all datasets
        trial_weights = {}
        for i in range(NUM_DATASETS):
            ds = f"dataset{i}"
            base = state["best_weights"][ds]
            trial_weights[ds] = perturb_weights(base, rng, args.magnitude)

        # Package
        sub_path = SUBMISSIONS_DIR / f"submission_auto_r{round_num}.tar.gz"
        print(f"  Packaging {sub_path.name}...")

        # If over 150MB, trim lowest-weight sequence models until it fits
        pack_attempts = 0
        while not package_submission(trial_weights, sub_path):
            pack_attempts += 1
            if pack_attempts > 10:
                print("  SKIP: cannot fit in 150MB after 10 trim attempts")
                break
            dropped = trim_to_fit(trial_weights)
            if pack_attempts % 5 == 0:
                print(f"  Over 150MB, trimmed {dropped} seq models (attempt {pack_attempts})...")
        else:
            pass

        if pack_attempts > 10:
            continue

        size_mb = sub_path.stat().st_size / 1024 / 1024
        print(f"  Size: {size_mb:.1f} MB")

        # Save trial weights before submitting (so we can recover if crash)
        trial_weights_path = FEEDBACK_DIR / f"trial_weights_r{round_num}.json"
        with open(trial_weights_path, 'w') as f:
            json.dump(trial_weights, f, indent=2)

        # Submit
        print(f"  Submitting...")
        try:
            submission_id = submit(args.token, str(sub_path))
        except Exception as e:
            err_msg = str(e)
            print(f"  SUBMIT FAILED: {err_msg}")
            # Parse wait time from error message
            import re
            wait_match = re.search(r'(\d+)\s*seconds', err_msg)
            if wait_match or "rate" in err_msg.lower() or "Rate" in err_msg:
                wait_secs = int(wait_match.group(1)) + 60 if wait_match else 2700
                print(f"  Rate limited, waiting {wait_secs}s ({wait_secs//60} min)...")
                time.sleep(wait_secs)
                try:
                    submission_id = submit(args.token, str(sub_path))
                except Exception as e2:
                    print(f"  RETRY FAILED: {e2}")
                    continue
            elif err_msg.strip().endswith("failed:") or not err_msg.strip():
                # Empty error = likely network timeout, retry after short wait
                print("  Empty error (network issue?), retrying in 60s...")
                time.sleep(60)
                try:
                    submission_id = submit(args.token, str(sub_path))
                except Exception as e2:
                    print(f"  RETRY FAILED: {e2}")
                    continue
            else:
                continue

        # Poll for result
        print(f"  Waiting for result (ID: {submission_id})...")
        try:
            result = poll_result(args.token, submission_id)
        except Exception as e:
            print(f"  POLL ERROR: {e}")
            print(f"  Will query result next round. Saving submission_id...")
            state["pending_id"] = submission_id
            save_state(state, state_file)
            time.sleep(120)
            # Try once more
            try:
                result = poll_result(args.token, submission_id)
            except Exception:
                print(f"  Still failing. Skipping round.")
                continue

        if result is None or result.get("status") != "completed":
            print(f"  FAILED: {result.get('status', 'timeout') if result else 'no response'}")
            continue

        # Save raw result immediately (before any processing)
        result_path = FEEDBACK_DIR / f"auto_iter_{round_num}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        # Parse result
        parsed = parse_result(result)

        # Compare with best, update per-dataset
        improved = []
        unchanged = []
        degraded = []

        for i in range(NUM_DATASETS):
            ds = f"dataset{i}"
            if parsed[ds] is None:
                unchanged.append(ds)
                continue

            new_avg = parsed[ds]["avg"]
            old_avg = state["best_ic"][ds].get("avg", 0)

            if new_avg > old_avg:
                state["best_ic"][ds] = parsed[ds]
                state["best_weights"][ds] = trial_weights[ds]
                improved.append((ds, old_avg, new_avg))
            else:
                # Keep old weights (revert this dataset's perturbation)
                degraded.append((ds, old_avg, new_avg))

        state["round"] = round_num

        # Report
        all_avg = [state["best_ic"][f"dataset{i}"].get("avg", 0) for i in range(NUM_DATASETS)]
        overall_avg = np.mean(all_avg)

        print(f"\n  Results:")
        print(f"    Improved: {len(improved)} datasets")
        for ds, old, new in improved[:10]:
            print(f"      {ds}: {old:.4f} → {new:.4f} (+{new-old:.4f})")
        print(f"    Degraded: {len(degraded)} datasets (reverted)")
        print(f"    Overall best avg: {overall_avg:.6f}")

        # Save state
        state["history"].append({
            "round": round_num,
            "improved": len(improved),
            "degraded": len(degraded),
            "overall_avg": overall_avg,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        save_state(state, state_file)

        print(f"    State saved. Next round in 5s...")
        time.sleep(5)

    # Final summary
    print(f"\n{'='*70}")
    print(f"  DONE: {end_round - start_round + 1} rounds completed")
    print(f"{'='*70}")
    all_avg = [state["best_ic"][f"dataset{i}"].get("avg", 0) for i in range(NUM_DATASETS)]
    print(f"  Final overall avg: {np.mean(all_avg):.6f}")
    print(f"  Best weights saved in: {state_file}")

    # Export final weights
    final_path = Path("weights_v7_final.json")
    with open(final_path, 'w') as f:
        json.dump(state["best_weights"], f, indent=2)
    print(f"  Final weights exported to: {final_path}")


if __name__ == "__main__":
    main()
