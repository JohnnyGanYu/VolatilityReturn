#!/usr/bin/env python3
"""
v6 weight optimization — compatibility wrapper.

This module re-exports from scripts/archive/reoptimize_v6.py for backward compatibility
and from reoptimize_weights.py for the full implementation.

For actual usage, prefer: python reoptimize_weights.py
"""

# Re-export from the archive version for test compatibility
from scripts.archive.reoptimize_v6 import (
    validate_weights,
    RET5_WEIGHT_KEYS,
    RET60_WEIGHT_KEYS,
    WEIGHT_TO_FILES,
    prune_models,
    get_available_models,
    pearson_ic,
    NUM_DATASETS,
)
