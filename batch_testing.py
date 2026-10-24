"""Reproducible batch evaluation: detection, response and correction for many independent runs.

Usage (Windows Command Prompt or any shell):
    python batch_testing.py --runs-per-scenario 5 --seed-base 900000 --out results/batch_results.csv

Batch runs use seeds from BATCH_SEED_BASE upward, a range that the training, validation and test
splits never use, so these runs are independent of everything the model was trained or tuned on.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import nuclearrad as nr
import evaluation as ev
import fault_detection as fd
import fault_response as fr
from dataset_generator import BATCH_SEED_BASE, generate_runs


def _names(idx):
    return ";".join(nr.NAMES[i] for i in sorted(idx)) or "none"


def run_test_case(bundle, run, split="batch"):
    """Detect, respond and correct one run; compare everything against its ground truth."""
    cfg, w = run.config, run.config.warmup
    det = fd.predict_run(bundle, run.readings, w, run.names)
    rules = fd.rule_based_run(run.readings, w, bundle["min_flagged"], run.names, bundle.get("min_spike"))
    rec = fr.recommend(det)
    corr = fr.apply_correction(run.readings, det, w)

    expected_fault = bool(run.fault_sensors)
    expected_process = "Process change" in run.expected_category
    predicted = {s["sensor"] for s in det["sensors"]}
    expected = set(run.fault_sensors)
    outcome = ("TP" if det["fault_present"] else "FN") if expected_fault else ("FP" if det["fault_present"] else "TN")
    delay = ev.detection_delay(fd.detection_time_for(det, expected), cfg.fault_start) if expected_fault else None
    rule_delay = ev.detection_delay(fd.detection_time_for(rules, expected), cfg.fault_start) if expected_fault else None

    # correction is scored on the truly faulty sensors; a false alarm is scored on the sensors that were
    # wrongly "corrected" over the whole post-warm-up period
    if expected_fault:
        cm = ev.correction_metrics(run, corr["corrected"])
    elif predicted:
        cm = ev.correction_metrics(run, corr["corrected"], sensors=sorted(predicted), window=slice(w, None))
    else:
        cm = ev.correction_metrics(run, corr["corrected"], sensors=[])
    applied = bool(corr["estimated"].any())
    if not expected_fault and not predicted:
        status = "not needed (process change, not corrected)" if expected_process else "not needed"
    elif expected_fault and not (predicted & expected):
        status = "not applied (fault on this sensor missed)"
    elif not applied:
        status = "unavailable (not enough trusted peers)"
    elif not expected_fault:
        status = "applied to a healthy sensor (false alarm)"
    elif cm["improvement_pct"] is None:
        status = "gap estimate only (no raw reading to compare)" if cm["gap_samples"] else "applied (no comparable samples)"
    else:
        status = "applied, error reduced" if cm["improvement_pct"] > 0 else "applied, error increased"

    response_ok = (det["fault_present"] == expected_fault and det["process_change"] == expected_process
                   and (not expected_fault or bool(predicted & expected)))
    return dict(
        run_id=run.run_id, split=split, scenario=cfg.scenario, seed=cfg.seed, fault_start=cfg.fault_start,
        fault_end=cfg.fault_end, magnitude=cfg.mag, noise=cfg.noise,
        expected_category=run.expected_category, predicted_category=det["category"],
        category_correct=det["category"] == run.expected_category,
        expected_sensor=_names(expected), predicted_sensor=_names(predicted),
        sensor_correct=predicted == expected,
        fault_expected=expected_fault, fault_detected=det["fault_present"], detection_outcome=outcome,
        false_positive=outcome == "FP", missed_detection=outcome == "FN",
        detection_delay_samples=delay, model_confidence_uncalibrated=det["primary_confidence"],
        recommended_response=" | ".join(r["action"] for r in rec["items"]), response_appropriate=response_ok,
        correction_status=status, raw_mae=cm["raw_mae"], corrected_mae=cm["corrected_mae"],
        correction_improvement_pct=cm["improvement_pct"], compared_samples=cm["n_compared"],
        gap_samples=cm["gap_samples"], gap_coverage=cm["gap_coverage"], gap_estimate_mae=cm["gap_estimate_mae"],
        rule_category=rules["category"], rule_category_correct=rules["category"] == run.expected_category,
        rule_fault_detected=rules["fault_present"], rule_detection_delay_samples=rule_delay,
    ), det, rules


def observation_arrays(runs, detections):
    """Actual vs predicted labels per (sample, sensor) after the warm-up, pooled over runs."""
    actual, predicted = [], []
    for run, det in zip(runs, detections):
        w = run.config.warmup
        actual.append(run.labels[w:].ravel().astype(str))
        predicted.append(det["sample_labels"][w:].ravel().astype(str))
    return np.concatenate(actual), np.concatenate(predicted)


def evaluate_runs(bundle, runs, split):
    """Run every test case and compute observation-level and run-level metrics."""
    rows, dets, rule_dets = [], [], []
    for run in runs:
        row, det, rules = run_test_case(bundle, run, split)
        rows.append(row)
        dets.append(det)
        rule_dets.append(rules)
    table = pd.DataFrame(rows)
    actual, pred = observation_arrays(runs, dets)
    _, rule_pred = observation_arrays(runs, rule_dets)
    is_fault = lambda a: np.isin(a, nr.INSTRUMENT_FAULTS)
    return dict(
        split=split, table=table,
        observation=ev.detection_metrics(is_fault(actual), is_fault(pred)),
        observation_rules=ev.detection_metrics(is_fault(actual), is_fault(rule_pred)),
        observation_confusion=ev.multiclass_confusion(actual, pred, nr.CLASSES),
        observation_per_class=ev.per_class_report(actual, pred, nr.CLASSES),
        summary=summarise(table),
    )


def summarise(table):
    """Run-level metrics from a results table (one row per run)."""
    if table.empty:
        return {}
    run_m = ev.detection_metrics(table["fault_expected"], table["fault_detected"])
    rule_m = ev.detection_metrics(table["fault_expected"], table["rule_fault_detected"])
    delays = table["detection_delay_samples"].dropna()
    corrected = table.dropna(subset=["raw_mae", "corrected_mae"])
    corrected = corrected[corrected["fault_expected"]]
    pooled_raw = corrected["raw_mae"].mean() if len(corrected) else None
    pooled_corr = corrected["corrected_mae"].mean() if len(corrected) else None
    return dict(
        runs=len(table), run_detection=run_m, rule_run_detection=rule_m,
        category_accuracy=float(table["category_correct"].mean()),
        rule_category_accuracy=float(table["rule_category_correct"].mean()),
        sensor_accuracy_on_faults=ev.safe_div(int((table["sensor_correct"] & table["fault_expected"]).sum()),
                                              int(table["fault_expected"].sum())),
        response_appropriate=float(table["response_appropriate"].mean()),
        mean_detection_delay=float(delays.mean()) if len(delays) else None,
        median_detection_delay=float(delays.median()) if len(delays) else None,
        missed_faults=int(table["missed_detection"].sum()), false_alarms=int(table["false_positive"].sum()),
        mean_raw_mae=None if pooled_raw is None or np.isnan(pooled_raw) else float(pooled_raw),
        mean_corrected_mae=None if pooled_corr is None or np.isnan(pooled_corr) else float(pooled_corr),
        mean_improvement_pct=ev.improvement_pct(pooled_raw, pooled_corr) if pooled_raw is not None else None,
        correction_status_counts=table["correction_status"].value_counts().to_dict(),
    )


def by_category(table):
    """Results grouped by the expected run category."""
    if table.empty:
        return table
    g = table.groupby("expected_category")
    out = pd.DataFrame({
        "runs": g.size(),
        "category_accuracy": g["category_correct"].mean(),
        "detected": g["fault_detected"].mean(),
        "sensor_correct": g["sensor_correct"].mean(),
        "mean_delay_samples": g["detection_delay_samples"].mean(),
        "mean_raw_mae": g["raw_mae"].mean(),
        "mean_corrected_mae": g["corrected_mae"].mean(),
        "rule_category_accuracy": g["rule_category_correct"].mean(),
    })
    out["improvement_pct"] = [ev.improvement_pct(r, c) if np.isfinite(r) and np.isfinite(c) else None
                              for r, c in zip(out["mean_raw_mae"], out["mean_corrected_mae"])]
    return out.reset_index()


def run_batch(bundle, runs_per_scenario=5, seed_base=BATCH_SEED_BASE, scenarios=None):
    """Generate fresh independent runs and evaluate them. Returns the evaluate_runs dict."""
    if seed_base < 100_000:
        raise ValueError("batch seeds must stay outside the training/validation/test seed ranges")
    runs = generate_runs("batch", runs_per_scenario, scenarios=scenarios, base=seed_base)
    return evaluate_runs(bundle, runs, f"batch (seeds from {seed_base})")


def main():
    import model_training     # imported here: model_training imports this module
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-per-scenario", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=BATCH_SEED_BASE)
    parser.add_argument("--out", default="results/batch_results.csv")
    parser.add_argument("--retrain", action="store_true", help="retrain the model before testing")
    args = parser.parse_args()

    bundle = model_training.load_or_train(retrain=args.retrain, verbose=True)
    result = run_batch(bundle, args.runs_per_scenario, args.seed_base)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result["table"].to_csv(out, index=False)
    by_category(result["table"]).to_csv(out.with_name(out.stem + "_by_category.csv"), index=False)
    s = result["summary"]
    print(f"{s['runs']} independent runs, written to {out}")
    print(f"run-level detection: accuracy {ev.fmt(s['run_detection']['accuracy'])}, "
          f"recall {ev.fmt(s['run_detection']['recall'])}, FPR {ev.fmt(s['run_detection']['false_positive_rate'])}")
    print(f"category accuracy {ev.fmt(s['category_accuracy'])} (rules {ev.fmt(s['rule_category_accuracy'])}), "
          f"missed {s['missed_faults']}, false alarms {s['false_alarms']}")
    print(f"mean detection delay {ev.fmt(s['mean_detection_delay'], 1)} samples; "
          f"MAE raw {ev.fmt(s['mean_raw_mae'])} -> corrected {ev.fmt(s['mean_corrected_mae'])} "
          f"({ev.fmt(s['mean_improvement_pct'], 1)}%)")


if __name__ == "__main__":
    main()
