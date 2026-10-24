"""Labelled dataset generation from the simulator.

Splits are made by independent simulation run, never by row: every run gets a seed from a range
that belongs to exactly one split, so near-identical samples of one run can never end up on both
sides of a train/test boundary.
"""
import numpy as np
import pandas as pd

import nuclearrad as nr
from feature_engineering import FEATURE_COLUMNS, compute_features

# seed ranges per split (disjoint). Batch testing uses its own range far away from these.
SPLIT_SEED_BASE = {"train": 10_000, "validation": 20_000, "test": 30_000}
BATCH_SEED_BASE = 900_000
DEFAULT_RUNS = {"train": 12, "validation": 4, "test": 5}     # runs per scenario


def run_seed(split, scenario_index, rep, base=None):
    """Deterministic seed for the rep-th run of a scenario in a split."""
    base = SPLIT_SEED_BASE[split] if base is None else base
    return base + scenario_index * 1000 + rep


def generate_runs(split, runs_per_scenario, scenarios=None, base=None):
    """Generate runs_per_scenario random parameter variations of every scenario for one split."""
    scenarios = list(nr.SIM_SCENARIOS) if scenarios is None else list(scenarios)
    runs = []
    for si, scenario in enumerate(nr.SIM_SCENARIOS):
        if scenario not in scenarios:
            continue
        for rep in range(runs_per_scenario):
            runs.append(nr.simulate(nr.random_config(scenario, run_seed(split, si, rep, base))))
    return runs


def generate_splits(runs=None):
    """Train / validation / held-out test runs, keyed by split name."""
    runs = DEFAULT_RUNS if runs is None else runs
    return {split: generate_runs(split, n) for split, n in runs.items()}


def build_dataset(runs, observable_only=False):
    """Feature matrix and labels for every post-warm-up (sample, sensor) of every run.

    observable_only drops samples whose fault exists but is still smaller than the healthy noise
    (e.g. the first minutes of a slow drift); used for training only, never for evaluation.
    Returns X (rows, features), y (labels), and a meta DataFrame (run_id, scenario, time_step, sensor).
    """
    Xs, ys, metas = [], [], []
    for run in runs:
        w = run.config.warmup
        feats = compute_features(run.readings, w)[w:]
        labels = run.labels[w:]
        keep = run.observable[w:] if observable_only else np.ones(labels.shape, bool)
        T, S, F = feats.shape
        t_idx, s_idx = np.nonzero(keep)
        Xs.append(feats[t_idx, s_idx])
        ys.append(labels[t_idx, s_idx].astype(str))
        metas.append(pd.DataFrame({"run_id": run.run_id, "scenario": run.config.scenario,
                                   "time_step": t_idx + w, "sensor": s_idx}))
    if not Xs:
        raise ValueError("no runs given")
    return np.concatenate(Xs), np.concatenate(ys), pd.concat(metas, ignore_index=True)


def observations_frame(runs):
    """Long table of every observation with its ground truth (for export and inspection)."""
    return pd.concat([run.to_long_frame() for run in runs], ignore_index=True)


def feature_names():
    return list(FEATURE_COLUMNS)
