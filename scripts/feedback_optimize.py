#!/usr/bin/env python3
"""
Iterative feedback-based ensemble weight optimizer.

Strategy:
- Phase 1 (first N iterations): Random exploration with LGB >= 0.3 constraint
- Phase 2 (after N iterations): Greedy hill-climbing from historical best

All 60 slots are independent and tested in parallel each submission.
After each result, per-slot comparison: keep if better than historical best.

Usage:
    python feedback_optimize.py --result <result.json>
"""
import json
import os
import sys
import copy
import random
import argparse
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models_v5")
WEIGHTS_PATH = MODEL_DIR / "ensemble_weights.json"
STATE_DIR = Path("feedback_state")

RANDOM_SEED_BASE = 42
RANDOM_PHASE_ITERATIONS = 10  # First 10 iterations: random exploration
STEP = 0.1


def generate_all_candidates():
    """
    Generate all valid weight configs with LGB total >= 0.3.
    w_local + w_global + w_gru + w_tf = 1.0, step 0.1, local+global >= 0.3.
    """
    configs = []
    for wl_int in range(11):
        for wg_int in range(11 - wl_int):
            for wb_int in range(11 - wl_int - wg_int):
                wt_int = 10 - wl_int - wg_int - wb_int
                if wt_int < 0:
                    continue
                wl = wl_int / 10
                wg = wg_int / 10
                wb = wb_int / 10
                wt = wt_int / 10
                # Safety: LGB total >= 0.3
                if wl + wg >= 0.3 - 1e-9:
                    configs.append((wl, wg, wb, wt))
    return configs


ALL_CANDIDATES = generate_all_candidates()
print(f"[INFO] Total valid candidates (LGB>=0.3): {len(ALL_CANDIDATES)}")


def combined_ic(n_ic, e_ic):
    """Weighted combination: 0.6*normal + 0.4*extreme."""
    return 0.6 * n_ic + 0.4 * e_ic


def get_ic(result, ds_name, target, ic_type):
    """Extract IC from result JSON."""
    details = result.get("details", {})
    ic_map = details.get(f"{ic_type}_ic", {})
    ds_data = ic_map.get(ds_name, {})
    key = "Ret5" if target == "ret5" else "Ret60"
    val = ds_data.get(key, None)
    return float(val) if val is not None else 0.0


def get_slot_key(ds_name, target):
    return f"{ds_name}_{target}"


def config_to_weights(config, target):
    """Convert (wl, wg, wb, wt) tuple to weight dict entries."""
    prefix = target
    return {
        f"{prefix}_w_local": config[0],
        f"{prefix}_w_global": config[1],
        f"{prefix}_w_gru": config[2],
        f"{prefix}_w_tf": config[3],
    }


def weights_to_config(weights, ds_name, target):
    """Extract (wl, wg, wb, wt) tuple from weights dict."""
    ds_w = weights.get(ds_name, {})
    prefix = target
    return (
        ds_w.get(f"{prefix}_w_local", 1.0),
        ds_w.get(f"{prefix}_w_global", 0.0),
        ds_w.get(f"{prefix}_w_gru", 0.0),
        ds_w.get(f"{prefix}_w_tf", 0.0),
    )


def load_state():
    """Load optimization state."""
    state_path = STATE_DIR / "state.json"
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {
        "best_weights": {},
        "best_ic": {},
        "tried": {},
        "iteration": 0,
        "history": [],
    }


def save_state(state):
    """Save optimization state."""
    STATE_DIR.mkdir(exist_ok=True)
    with open(STATE_DIR / "state.json", "w") as f:
        json.dump(state, f, indent=2)


def pick_random_candidate(state, ds_name, target, rng):
    """
    Pick a random untried candidate (Phase 1: exploration).
    Constraint: LGB total >= 0.3 (already enforced in ALL_CANDIDATES).
    """
    slot_key = get_slot_key(ds_name, target)
    tried = set(tuple(x) for x in state["tried"].get(slot_key, []))

    untried = [c for c in ALL_CANDIDATES if c not in tried]
    if not untried:
        return None
    return untried[rng.randint(0, len(untried) - 1)]


def pick_hillclimb_candidate(state, ds_name, target):
    """
    Pick the next untried neighbor of current best (Phase 2: exploitation).
    Neighbors = configs that differ by exactly 0.1 in one or two dimensions.
    """
    slot_key = get_slot_key(ds_name, target)
    tried = set(tuple(x) for x in state["tried"].get(slot_key, []))

    # Current best
    current = weights_to_config(state["best_weights"], ds_name, target)

    # Sort all candidates by L1 distance from current best (nearest first)
    def distance(c):
        return sum(abs(a - b) for a, b in zip(c, current))

    sorted_candidates = sorted(ALL_CANDIDATES, key=distance)

    for c in sorted_candidates:
        if c not in tried and c != current:
            return c
    return None


def load_weights():
    with open(WEIGHTS_PATH) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, help="Path to result JSON")
    args = parser.parse_args()

    STATE_DIR.mkdir(exist_ok=True)
    state = load_state()

    # Load result
    with open(args.result) as f:
        result = json.load(f)

    if result.get("status") != "completed":
        print(f"ERROR: Result status is '{result.get('status')}', not 'completed'")
        sys.exit(1)

    state["iteration"] += 1
    iteration = state["iteration"]

    print("=" * 70)
    print(f"Feedback Optimization — Iteration {iteration}")
    print(f"  Phase: {'RANDOM exploration' if iteration <= RANDOM_PHASE_ITERATIONS else 'HILL-CLIMBING'}")
    print(f"  Result: {args.result}")
    print("=" * 70)

    # --- Step 1: Update state with results ---
    current_weights = load_weights()
    improvements = 0

    for ds_idx in range(30):
        ds_name = f"dataset{ds_idx}"
        for target in ["ret5", "ret60"]:
            slot_key = get_slot_key(ds_name, target)

            n_ic = get_ic(result, ds_name, target, "normal")
            e_ic = get_ic(result, ds_name, target, "extreme")
            c_ic = combined_ic(n_ic, e_ic)

            # Record what we tried
            config = weights_to_config(current_weights, ds_name, target)
            if slot_key not in state["tried"]:
                state["tried"][slot_key] = []
            config_list = list(config)
            if config_list not in state["tried"][slot_key]:
                state["tried"][slot_key].append(config_list)

            # Update best if improved
            prev_best = state["best_ic"].get(slot_key, float("-inf"))
            if c_ic > prev_best:
                state["best_ic"][slot_key] = c_ic
                if ds_name not in state["best_weights"]:
                    state["best_weights"][ds_name] = {}
                w_entries = config_to_weights(config, target)
                state["best_weights"][ds_name].update(w_entries)
                improvements += 1

    # --- Step 2: Print status ---
    print(f"\nImprovements: {improvements}/60 slots")

    all_combined = []
    all_n_r5, all_n_r60, all_e_r5, all_e_r60 = [], [], [], []

    print(f"\n{'Dataset':<12} {'cR5':>7} {'cR60':>7} | {'bestR5':>7} {'bestR60':>7}")
    print("-" * 60)

    for ds_idx in range(30):
        ds_name = f"dataset{ds_idx}"
        n_r5 = get_ic(result, ds_name, "ret5", "normal")
        e_r5 = get_ic(result, ds_name, "ret5", "extreme")
        n_r60 = get_ic(result, ds_name, "ret60", "normal")
        e_r60 = get_ic(result, ds_name, "ret60", "extreme")
        c_r5 = combined_ic(n_r5, e_r5)
        c_r60 = combined_ic(n_r60, e_r60)

        best_r5 = state["best_ic"].get(get_slot_key(ds_name, "ret5"), c_r5)
        best_r60 = state["best_ic"].get(get_slot_key(ds_name, "ret60"), c_r60)

        all_combined.extend([c_r5, c_r60])
        all_n_r5.append(n_r5)
        all_n_r60.append(n_r60)
        all_e_r5.append(e_r5)
        all_e_r60.append(e_r60)

        print(f"{ds_name:<12} {c_r5:>7.4f} {c_r60:>7.4f} | {best_r5:>7.4f} {best_r60:>7.4f}")

    print(f"\nThis submission mean ICs:")
    print(f"  Normal  Ret5:  {np.mean(all_n_r5):.6f}")
    print(f"  Normal  Ret60: {np.mean(all_n_r60):.6f}")
    print(f"  Extreme Ret5:  {np.mean(all_e_r5):.6f}")
    print(f"  Extreme Ret60: {np.mean(all_e_r60):.6f}")
    print(f"  Combined mean: {np.mean(all_combined):.6f}")

    # Historical best
    best_ics = [state["best_ic"].get(get_slot_key(f"dataset{i}", t), 0)
                for i in range(30) for t in ["ret5", "ret60"]]
    print(f"\n  Historical best combined mean: {np.mean(best_ics):.6f}")

    # Tried stats
    tried_counts = [len(state["tried"].get(get_slot_key(f"dataset{i}", t), []))
                    for i in range(30) for t in ["ret5", "ret60"]]
    print(f"  Configs tried per slot: avg={np.mean(tried_counts):.1f}, "
          f"min={min(tried_counts)}, max={max(tried_counts)}")
    print(f"  Total candidate pool: {len(ALL_CANDIDATES)}")

    # --- Step 3: Prepare next submission ---
    print(f"\n{'=' * 70}")
    print("Preparing next submission weights...")

    rng = random.Random(RANDOM_SEED_BASE + iteration)
    new_weights = {}
    changes = 0
    exhausted = 0

    for ds_idx in range(30):
        ds_name = f"dataset{ds_idx}"
        new_weights[ds_name] = {}

        for target in ["ret5", "ret60"]:
            if iteration <= RANDOM_PHASE_ITERATIONS:
                # Phase 1: Random exploration
                candidate = pick_random_candidate(state, ds_name, target, rng)
            else:
                # Phase 2: Hill-climbing from best
                candidate = pick_hillclimb_candidate(state, ds_name, target)

            if candidate is not None:
                w_entries = config_to_weights(candidate, target)
                new_weights[ds_name].update(w_entries)
                changes += 1
            else:
                # Exhausted: use historical best
                best_config = weights_to_config(state["best_weights"], ds_name, target)
                w_entries = config_to_weights(best_config, target)
                new_weights[ds_name].update(w_entries)
                exhausted += 1

    print(f"  New candidates: {changes}, Exhausted (using best): {exhausted}")

    # Save new weights
    with open(WEIGHTS_PATH, "w") as f:
        json.dump(new_weights, f, indent=2)
    print(f"  Saved to {WEIGHTS_PATH}")

    # Save state
    state["history"].append({
        "iteration": iteration,
        "result_file": args.result,
        "improvements": improvements,
        "mean_combined_ic": float(np.mean(all_combined)),
        "best_combined_mean": float(np.mean(best_ics)),
    })
    save_state(state)

    # Save iteration snapshot
    snapshot_path = STATE_DIR / f"iter_{iteration:03d}.json"
    with open(snapshot_path, "w") as f:
        json.dump({
            "iteration": iteration,
            "phase": "random" if iteration <= RANDOM_PHASE_ITERATIONS else "hillclimb",
            "mean_combined": float(np.mean(all_combined)),
            "best_combined_mean": float(np.mean(best_ics)),
            "improvements": improvements,
            "changes": changes,
        }, f, indent=2)

    print(f"  Saved snapshot to {snapshot_path}")
    print(f"\nDone. Repackage and submit next iteration.")


if __name__ == "__main__":
    main()
