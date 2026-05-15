#!/usr/bin/env python3
"""
Iterate ensemble weights based on platform feedback.

Reads platform IC results, compares with baseline, and generates
next-round weights with three strategies per dataset:
- Keep: IC is good, don't change
- Switch: IC is bad, try sequence models or different blend
- Perturb: IC is medium, random perturbation to explore

Usage:
    python scripts/iterate_from_feedback.py \
        --feedback feedback_state/iter_1.json \
        --current models_v6/ensemble_weights.json \
        --local-seq-weights models_v6/ensemble_weights_with_seq.json \
        --output models_v6/ensemble_weights_next.json

Feedback JSON format:
{
    "dataset0": {"nR5": 0.05, "nR60": 0.16, "eR5": 0.48, "eR60": 0.95},
    "dataset1": {"nR5": 0.13, "nR60": 0.22, "eR5": 0.15, "eR60": 0.57},
    ...
}
"""

import json
import argparse
import numpy as np
from pathlib import Path


SEED = 42
NUM_DATASETS = 30

# Thresholds for decision
IC_GOOD_THRESHOLD = 0.10    # IC > this → already good, worth big exploration
IC_BAD_THRESHOLD = 0.03     # IC < this → bad, switch strategy conservatively
# Between bad and good → small perturbation

RET5_KEYS = ["ret5_w_local", "ret5_w_global", "ret5_w_gru_ret5", "ret5_w_tf_ret5", "ret5_w_extreme"]
RET60_KEYS = ["ret60_w_local", "ret60_w_global", "ret60_w_gru_ret60", "ret60_w_tf_ret60", "ret60_w_extreme"]


def normalize_weights(w_dict, keys):
    """Normalize weights to sum to 1.0, clip negatives."""
    total = sum(max(0, w_dict.get(k, 0)) for k in keys)
    if total > 0:
        for k in keys:
            w_dict[k] = round(max(0, w_dict.get(k, 0)) / total, 3)
    else:
        w_dict[keys[0]] = 1.0
        for k in keys[1:]:
            w_dict[k] = 0.0


def perturb_weights(w_dict, keys, rng, magnitude=0.1):
    """Random perturbation of weights."""
    for k in keys:
        delta = rng.uniform(-magnitude, magnitude)
        w_dict[k] = max(0, w_dict.get(k, 0) + delta)
    normalize_weights(w_dict, keys)


def switch_to_seq(w_dict, seq_weights, target_prefix):
    """Switch to sequence model weights from local validation."""
    keys = RET5_KEYS if target_prefix == "ret5" else RET60_KEYS
    for k in keys:
        w_dict[k] = seq_weights.get(k, 0.0)
    normalize_weights(w_dict, keys)


def main():
    parser = argparse.ArgumentParser(description="Iterate weights from platform feedback")
    parser.add_argument("--feedback", required=True, help="Platform feedback JSON file")
    parser.add_argument("--current", required=True, help="Current ensemble_weights.json")
    parser.add_argument("--local-seq-weights", default=None,
                        help="Weights with seq models from local validation (optional)")
    parser.add_argument("--output", required=True, help="Output next-round weights")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--perturb-magnitude", type=float, default=0.1)
    parser.add_argument("--round", type=int, default=1,
                        help="Current iteration round. Round 1-5: aggressive exploration. Round 6+: conservative.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed + args.round)  # Different seed each round

    # Early rounds: always big jumps. Later rounds: use IC-based logic.
    aggressive = args.round <= 5

    # Load feedback
    with open(args.feedback) as f:
        feedback = json.load(f)

    # Load current weights
    with open(args.current) as f:
        current = json.load(f)

    # Load seq weights if available
    seq_weights = {}
    if args.local_seq_weights and Path(args.local_seq_weights).exists():
        with open(args.local_seq_weights) as f:
            seq_weights = json.load(f)

    next_weights = {}
    actions = []

    for ds_idx in range(NUM_DATASETS):
        ds = f"dataset{ds_idx}"
        ds_feedback = feedback.get(ds, {})
        ds_current = current.get(ds, {}).copy()
        ds_seq = seq_weights.get(ds, {})

        # Get platform ICs
        nR5 = ds_feedback.get("nR5", 0)
        nR60 = ds_feedback.get("nR60", 0)
        eR5 = ds_feedback.get("eR5", 0)
        eR60 = ds_feedback.get("eR60", 0)

        # Average IC for this dataset
        avg_r5 = (nR5 + eR5) / 2
        avg_r60 = (nR60 + eR60) / 2

        action_r5 = "keep"
        action_r60 = "keep"

        # --- Ret5 decision ---
        if aggressive:
            # Early rounds: always big random jump for exploration
            perturb_weights(ds_current, RET5_KEYS, rng, magnitude=0.3)
            action_r5 = "explore"
        elif avg_r5 >= IC_GOOD_THRESHOLD:
            # Good IC: worth big exploration — large random jump
            perturb_weights(ds_current, RET5_KEYS, rng, magnitude=0.3)
            action_r5 = "big_jump"
        elif avg_r5 < IC_BAD_THRESHOLD:
            # Bad: try switching to seq model weights or pure local
            if ds_seq:
                switch_to_seq(ds_current, ds_seq, "ret5")
                action_r5 = "switch_seq"
            else:
                # Fallback: pure local
                ds_current["ret5_w_local"] = 1.0
                ds_current["ret5_w_global"] = 0.0
                ds_current["ret5_w_gru_ret5"] = 0.0
                ds_current["ret5_w_tf_ret5"] = 0.0
                action_r5 = "switch_local"
        else:
            # Medium: small perturbation
            perturb_weights(ds_current, RET5_KEYS, rng, magnitude=0.1)
            action_r5 = "perturb"

        # --- Ret60 decision ---
        if aggressive:
            perturb_weights(ds_current, RET60_KEYS, rng, magnitude=0.3)
            action_r60 = "explore"
        elif avg_r60 >= IC_GOOD_THRESHOLD:
            # Good IC: worth big exploration
            perturb_weights(ds_current, RET60_KEYS, rng, magnitude=0.3)
            action_r60 = "big_jump"
        elif avg_r60 < IC_BAD_THRESHOLD:
            if ds_seq:
                switch_to_seq(ds_current, ds_seq, "ret60")
                action_r60 = "switch_seq"
            else:
                ds_current["ret60_w_local"] = 1.0
                ds_current["ret60_w_global"] = 0.0
                ds_current["ret60_w_gru_ret60"] = 0.0
                ds_current["ret60_w_tf_ret60"] = 0.0
                action_r60 = "switch_local"
        else:
            # Medium: small perturbation
            perturb_weights(ds_current, RET60_KEYS, rng, magnitude=0.1)
            action_r60 = "perturb"

        # Ensure normalization
        normalize_weights(ds_current, RET5_KEYS)
        normalize_weights(ds_current, RET60_KEYS)

        next_weights[ds] = ds_current
        actions.append(f"  {ds}: R5={action_r5}(avg={avg_r5:.3f}) R60={action_r60}(avg={avg_r60:.3f})")

    # Save
    with open(args.output, "w") as f:
        json.dump(next_weights, f, indent=2)

    print(f"Next-round weights saved: {args.output}")
    print(f"\nActions taken:")
    for a in actions:
        print(a)

    # Summary
    n_keep = sum(1 for a in actions if "keep" in a)
    n_switch = sum(1 for a in actions if "switch" in a)
    n_perturb = sum(1 for a in actions if "perturb" in a)
    print(f"\nSummary: keep={n_keep}, switch={n_switch}, perturb={n_perturb}")


if __name__ == "__main__":
    main()
