"""Evaluation formulas. Every function returns None where a metric is undefined (zero denominator,
empty set) so the dashboard can show "N/A" instead of a misleading number.

Positive class for fault detection = an INSTRUMENT fault (spike, drift, stuck, noise, dropout).
"Normal" and "Process change" (a real plant change reported correctly) are negatives.

Two evaluation units are used and never mixed:
  * observation: one (sample, sensor) pair after the warm-up
  * run (event):  one whole simulated scenario
"""
import numpy as np
import pandas as pd

NA = "N/A"


def safe_div(num, den):
    """num / den, or None when den is zero."""
    return None if den == 0 else num / den


def confusion_counts(actual, predicted):
    """TP, TN, FP, FN for boolean arrays (True = fault)."""
    a = np.asarray(actual, bool)
    p = np.asarray(predicted, bool)
    if a.shape != p.shape:
        raise ValueError("actual and predicted must have the same shape")
    return dict(TP=int(np.sum(a & p)), TN=int(np.sum(~a & ~p)),
                FP=int(np.sum(~a & p)), FN=int(np.sum(a & ~p)))


def accuracy(c):
    """(TP + TN) / (TP + TN + FP + FN)"""
    return safe_div(c["TP"] + c["TN"], c["TP"] + c["TN"] + c["FP"] + c["FN"])


def precision(c):
    """TP / (TP + FP)"""
    return safe_div(c["TP"], c["TP"] + c["FP"])


def recall(c):
    """TP / (TP + FN)  (sensitivity)"""
    return safe_div(c["TP"], c["TP"] + c["FN"])


def specificity(c):
    """TN / (TN + FP)"""
    return safe_div(c["TN"], c["TN"] + c["FP"])


def false_positive_rate(c):
    """FP / (FP + TN)"""
    return safe_div(c["FP"], c["FP"] + c["TN"])


def detection_metrics(actual, predicted):
    """All binary detection metrics plus the counts they were computed from."""
    c = confusion_counts(actual, predicted)
    return dict(**c, n=sum(c.values()), accuracy=accuracy(c), precision=precision(c), recall=recall(c),
                specificity=specificity(c), false_positive_rate=false_positive_rate(c))


def multiclass_confusion(actual, predicted, labels):
    """Confusion matrix as a DataFrame (rows = actual, columns = predicted)."""
    m = pd.crosstab(pd.Series(list(actual), name="actual"), pd.Series(list(predicted), name="predicted"))
    return m.reindex(index=labels, columns=labels, fill_value=0)


def per_class_report(actual, predicted, labels):
    """Precision / recall / support per class (one-vs-rest)."""
    a, p = np.asarray(actual, object), np.asarray(predicted, object)
    rows = []
    for lab in labels:
        c = confusion_counts(a == lab, p == lab)
        rows.append(dict(cls=lab, support=int(np.sum(a == lab)), precision=precision(c), recall=recall(c)))
    return pd.DataFrame(rows)


def mae(reference, estimate):
    """(1/N) * sum |reference_i - estimate_i| over pairs where both are finite; None if there are none."""
    r = np.asarray(reference, float).ravel()
    e = np.asarray(estimate, float).ravel()
    if r.shape != e.shape:
        raise ValueError("reference and estimate must be aligned")
    ok = np.isfinite(r) & np.isfinite(e)
    return float(np.mean(np.abs(r[ok] - e[ok]))) if ok.any() else None


def improvement_pct(mae_raw, mae_corrected):
    """(MAE_raw - MAE_corrected) / MAE_raw * 100. Positive = error went down. None if undefined."""
    if mae_raw is None or mae_corrected is None or mae_raw == 0:
        return None
    return (mae_raw - mae_corrected) / mae_raw * 100


def detection_delay(first_detection, fault_start):
    """first valid detection time - fault start time, in samples. None = missed (never invented)."""
    if first_detection is None or fault_start is None or first_detection < fault_start:
        return None
    return int(first_detection - fault_start)


def fmt(value, digits=3, pct=False, suffix=""):
    """Format a metric for display, using N/A for undefined values."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return NA
    if pct:
        return f"{100 * value:.{max(digits - 2, 1)}f}%"
    if isinstance(value, (int, np.integer)):
        return f"{value}{suffix}"
    return f"{value:.{digits}f}{suffix}"


def correction_metrics(run, corrected, sensors=None, window=None):
    """Raw vs corrected error against the healthy reference, for the instrument-faulty sensors.

    Comparison set: samples inside the fault window (or `window`) of the given sensors (default: the
    truly faulty ones) where BOTH the raw reading and the corrected estimate exist, so raw and
    corrected MAE use the same observations. Gaps (raw reading missing) cannot have a raw error;
    estimates filled into them are scored separately (gap_estimate_mae, gap_coverage).
    """
    cfg = run.config
    sensors = list(run.fault_sensors if sensors is None else sensors)
    if not sensors:
        return dict(sensors=[], n_compared=0, raw_mae=None, corrected_mae=None, improvement_pct=None,
                    gap_samples=0, gap_coverage=None, gap_estimate_mae=None)
    window = slice(cfg.fault_start, cfg.fault_end) if window is None else window
    truth = run.truth[window][:, sensors]
    raw = run.readings[window][:, sensors]
    corr = np.asarray(corrected, float)[window][:, sensors]
    both = np.isfinite(raw) & np.isfinite(corr)
    raw_mae = mae(truth[both], raw[both]) if both.any() else None
    corr_mae = mae(truth[both], corr[both]) if both.any() else None
    gaps = ~np.isfinite(raw)
    filled = gaps & np.isfinite(corr)
    return dict(sensors=sensors, n_compared=int(both.sum()), raw_mae=raw_mae, corrected_mae=corr_mae,
                improvement_pct=improvement_pct(raw_mae, corr_mae),
                gap_samples=int(gaps.sum()), gap_coverage=safe_div(int(filled.sum()), int(gaps.sum())),
                gap_estimate_mae=mae(truth[filled], corr[filled]) if filled.any() else None)
