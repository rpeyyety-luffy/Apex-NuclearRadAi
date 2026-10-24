"""Per-sensor, per-sample features for the fault model.

Every feature at sample t uses only readings at or before t (trailing windows, no centring,
no backward fill), so the model sees exactly what a live monitor would have seen.
The warm-up period (the first `warmup` samples) is treated as a healthy calibration period:
each sensor's normal level and normal scatter are learnt from it.
"""
import numpy as np
import pandas as pd

FEATURE_VERSION = 1
FEATURE_COLUMNS = [
    "reading_rel",        # reading / own warm-up level - 1
    "dev_peer",           # reading vs median of the other sensors (both scaled to their warm-up level)
    "dev_peer_z",         # dev_peer in units of its warm-up scatter
    "dev_peer_mean5",     # rolling mean of dev_peer, last 5 samples
    "dev_peer_mean20",    # rolling mean of dev_peer, last 20 samples
    "trend20",            # change of dev_peer over the last ~20 samples (rate of change)
    "step",               # change from the previous reading (scaled)
    "step_z",             # step in units of the sensor's warm-up step scatter
    "roll_std10_ratio",   # scatter of the last 10 steps vs warm-up step scatter
    "flat_run",           # identical readings in a row (a frozen sensor never changes)
    "missing",            # 1 when this reading is missing
    "missing_frac10",     # share of the last 10 readings that were missing
    "network_level",      # median over all sensors of reading_rel (a plant-wide change moves this)
    "peers_moved_share",  # share of the other sensors that are far from their own warm-up level
    "peer_spread",        # spread of reading_rel across sensors at this sample
]


def _robust_sd(a):
    a = a[np.isfinite(a)]
    if a.size < 3:
        return np.nan
    return 1.4826 * float(np.median(np.abs(a - np.median(a))))


def _trailing(frame, window, func):
    return getattr(frame.rolling(window, min_periods=1), func)().to_numpy()


def compute_features(readings, warmup):
    """Return a (samples, sensors, features) array for a (samples, sensors) reading matrix.

    Missing readings (NaN) are allowed: the 'missing' features flag them and the other features
    fall back to neutral values instead of inventing a reading.
    """
    x = np.asarray(readings, dtype=float)
    if x.ndim != 2 or x.shape[1] < 3:
        raise ValueError("readings must be a (samples, sensors>=3) array")
    T, S = x.shape
    if not 5 <= warmup < T:
        raise ValueError("warmup must be at least 5 samples and shorter than the run")

    level = np.nanmedian(x[:warmup], axis=0)
    if not np.all(np.isfinite(level)) or np.any(level == 0):
        raise ValueError("every sensor needs valid, non-zero readings during the warm-up")
    n = x / level                                    # 1.0 = sensor's own healthy level
    rel = n - 1
    missing = np.isnan(x)

    # peer consensus: median of the OTHER sensors at the same sample (available in real time)
    peer = np.full((T, S), np.nan)
    for i in range(S):
        others = np.delete(n, i, axis=1)
        valid = np.isfinite(others).any(axis=1)
        peer[valid, i] = np.nanmedian(others[valid], axis=1)
    dev_peer = n / peer - 1

    step = pd.DataFrame(n).diff().to_numpy()
    peer_sd = np.array([_robust_sd(dev_peer[:warmup, i]) for i in range(S)])
    step_sd = np.array([_robust_sd(step[1:warmup, i]) for i in range(S)])
    rel_sd = np.array([_robust_sd(rel[:warmup, i]) for i in range(S)])
    floor = lambda v: np.where(np.isfinite(v) & (v > 1e-6), v, 1e-3)
    peer_sd, step_sd, rel_sd = floor(peer_sd), floor(step_sd), floor(rel_sd)

    dp = pd.DataFrame(dev_peer)
    mean5 = _trailing(dp, 5, "mean")
    mean20 = _trailing(dp, 20, "mean")
    trend = mean5 - pd.DataFrame(mean5).shift(15).to_numpy()
    roll_std = _trailing(pd.DataFrame(step), 10, "std") / step_sd

    flat = np.zeros((T, S))
    for t in range(1, T):
        same = np.abs(x[t] - x[t - 1]) < 1e-12          # NaN compares False
        flat[t] = np.where(same, flat[t - 1] + 1, 0)

    miss_frac = _trailing(pd.DataFrame(missing.astype(float)), 10, "mean")
    with np.errstate(all="ignore"):
        network = np.nanmedian(rel, axis=1, keepdims=True)      # NaN only if every sensor is missing
    network = np.where(np.isfinite(network), network, 0.0)
    moved = np.abs(rel) > 4 * rel_sd
    moved_share = np.zeros((T, S))
    live = ~missing
    for i in range(S):
        others = [j for j in range(S) if j != i]
        count = live[:, others].sum(axis=1)
        moved_share[:, i] = np.where(count > 0, (moved[:, others] & live[:, others]).sum(axis=1) / np.maximum(count, 1), 0)
    spread = np.nanstd(np.where(missing, np.nan, rel), axis=1, keepdims=True)

    feats = np.stack([
        rel, dev_peer, dev_peer / peer_sd, mean5, mean20, trend, step, step / step_sd, roll_std,
        np.minimum(flat, 30), missing.astype(float), miss_frac,
        np.repeat(network, S, axis=1), moved_share, np.repeat(np.nan_to_num(spread), S, axis=1),
    ], axis=2)
    # neutral values where a feature cannot be computed (missing reading, start of a window)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(feats, -1e3, 1e3, out=feats)
    return feats


def feature_frame(readings, warmup):
    """Features as a DataFrame with one row per (time_step, sensor)."""
    feats = compute_features(readings, warmup)
    T, S, F = feats.shape
    frame = pd.DataFrame(feats.reshape(T * S, F), columns=FEATURE_COLUMNS)
    frame.insert(0, "sensor", np.tile(np.arange(S), T))
    frame.insert(0, "time_step", np.repeat(np.arange(T), S))
    return frame
