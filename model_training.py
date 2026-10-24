"""Train, validate, test, save and load the fault model.

    python model_training.py            # train if no compatible saved model, then print the report
    python model_training.py --retrain  # always retrain

Data flow:
  train split      -> fit the Random Forest (observation level)
  validation split -> choose `min_flagged` and `min_spike` (how many flagged samples make a sensor "faulty")
  test split       -> evaluated ONCE after everything is fixed; never used for fitting or tuning
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

import nuclearrad as nr
import batch_testing as bt
import evaluation as ev
import fault_detection as fd
from dataset_generator import DEFAULT_RUNS, build_dataset, generate_runs
from feature_engineering import FEATURE_COLUMNS, FEATURE_VERSION

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_FILE = "nuclearrad_rf.joblib"
META_FILE = "model_meta.json"
SIM_VERSION = 2
AGGREGATION_VERSION = 2
MIN_FLAGGED_CANDIDATES = range(1, 13)


def model_version():
    """Changes whenever the features, simulator or scikit-learn version change (forces retraining)."""
    return (f"rf-features{FEATURE_VERSION}-sim{SIM_VERSION}-agg{AGGREGATION_VERSION}"
            f"-sklearn{sklearn.__version__}")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.reset_index().to_dict(orient="list"))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def tune_thresholds(bundle, runs):
    """Pick min_flagged and min_spike on the validation runs: best balanced run-level detection
    accuracy; ties go to the smaller (faster) min_flagged and then the larger (stricter) min_spike."""
    totals, spikes, expected = [], [], []
    for run in runs:
        det = fd.predict_run({**bundle, "min_flagged": 1, "min_spike": 1}, run.readings, run.config.warmup,
                             run.names)
        totals.append(det["sample_flags"].sum(axis=0))
        spikes.append((det["sample_flags"] & (det["sample_labels"] == "Spike")).sum(axis=0))
        expected.append(bool(run.fault_sensors))
    totals, spikes, expected = np.array(totals), np.array(spikes), np.array(expected)
    scores = []
    for m in MIN_FLAGGED_CANDIDATES:
        for ms in range(1, m + 1):
            detected = ((totals >= m) | (spikes >= ms)).any(axis=1)
            c = ev.confusion_counts(expected, detected)
            rec, spec = ev.recall(c) or 0.0, ev.specificity(c) or 0.0
            scores.append(((rec + spec) / 2, -m, ms))
    best = max(scores)
    table = [dict(min_flagged=-m, min_spike=ms, balanced_accuracy=s) for s, m, ms in sorted(scores, key=lambda r: (-r[1], r[2]))]
    return -best[1], best[2], table


def train(runs=None, n_estimators=150, random_state=0, verbose=False):
    """Generate the splits, fit, tune on validation, evaluate on test. Returns the model bundle."""
    runs = dict(DEFAULT_RUNS if runs is None else runs)
    t0 = time.time()
    log = print if verbose else (lambda *a, **k: None)
    splits = {name: generate_runs(name, n) for name, n in runs.items()}
    ids = {name: {r.run_id for r in rs} for name, rs in splits.items()}
    if ids["train"] & ids["validation"] or ids["train"] & ids["test"] or ids["validation"] & ids["test"]:
        raise RuntimeError("a run appears in more than one split")

    X, y, meta = build_dataset(splits["train"], observable_only=True)
    log(f"training on {len(splits['train'])} runs, {len(y)} observations")
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=18, min_samples_leaf=2,
                                   class_weight="balanced_subsample", n_jobs=-1, random_state=random_state)
    model.fit(X, y)
    bundle = dict(model=model, classes=list(model.classes_), features=list(FEATURE_COLUMNS))

    min_flagged, min_spike, tuning = tune_thresholds(bundle, splits["validation"])
    bundle.update(min_flagged=int(min_flagged), min_spike=int(min_spike))
    log(f"tuned on validation: min_flagged={min_flagged}, min_spike={min_spike}")

    val = bt.evaluate_runs(bundle, splits["validation"], "validation")
    test = bt.evaluate_runs(bundle, splits["test"], "test")      # first and only use of the test split
    train_obs_pred = model.predict(X)
    importance = sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda p: -p[1])

    bundle["meta"] = _jsonable(dict(
        version=model_version(), trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        training_seconds=round(time.time() - t0, 1), n_estimators=n_estimators, random_state=random_state,
        runs_per_scenario=runs, scenarios=list(nr.SIM_SCENARIOS),
        n_runs={k: len(v) for k, v in splits.items()},
        n_observations=dict(train=int(len(y)),
                            validation=int(sum((r.config.n_samples - r.config.warmup) * r.readings.shape[1]
                                               for r in splits["validation"])),
                            test=int(sum((r.config.n_samples - r.config.warmup) * r.readings.shape[1]
                                         for r in splits["test"]))),
        train_class_counts=pd.Series(y).value_counts().to_dict(),
        min_flagged=int(min_flagged), min_spike=int(min_spike), threshold_tuning=tuning,
        train_observation_accuracy=float(np.mean(train_obs_pred == y)),
        feature_importance=[dict(feature=f, importance=float(v)) for f, v in importance],
        validation=dict(observation=val["observation"], observation_rules=val["observation_rules"],
                        summary=val["summary"]),
        test=dict(observation=test["observation"], observation_rules=test["observation_rules"],
                  summary=test["summary"], observation_confusion=test["observation_confusion"],
                  observation_per_class=test["observation_per_class"]),
    ))
    bundle["test_table"] = test["table"]
    bundle["validation_table"] = val["table"]
    log(f"done in {time.time() - t0:.0f} s")
    return bundle


def save(bundle, directory=None):
    """Persist the model (joblib) and a human-readable report (JSON + CSV of the test runs)."""
    directory = Path(directory or MODEL_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, directory / MODEL_FILE, compress=3)
    (directory / META_FILE).write_text(json.dumps(bundle["meta"], indent=2))
    bundle["test_table"].to_csv(directory / "test_results.csv", index=False)
    bundle["validation_table"].to_csv(directory / "validation_results.csv", index=False)


def load(directory=None):
    """Load a saved bundle; None when missing, unreadable or trained with a different version."""
    path = Path(directory or MODEL_DIR) / MODEL_FILE
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
    except Exception:
        return None
    if bundle.get("meta", {}).get("version") != model_version():
        return None
    return bundle


def load_or_train(retrain=False, directory=None, verbose=False, **train_kwargs):
    """Load the saved model, or train and save one when there is none (or retrain=True)."""
    bundle = None if retrain else load(directory)
    if bundle is None:
        bundle = train(verbose=verbose, **train_kwargs)
        save(bundle, directory)
    return bundle


def print_report(bundle):
    m = bundle["meta"]
    print(f"model {m['version']} trained {m['trained_at']} on {m['n_runs']['train']} runs "
          f"({m['n_observations']['train']} observations); min_flagged={m['min_flagged']}, "
          f"min_spike={m['min_spike']}")
    for split in ("validation", "test"):
        o, s = m[split]["observation"], m[split]["summary"]
        r = s["run_detection"]
        print(f"[{split}] observations: acc {ev.fmt(o['accuracy'])} prec {ev.fmt(o['precision'])} "
              f"rec {ev.fmt(o['recall'])} FPR {ev.fmt(o['false_positive_rate'])} | runs ({s['runs']}): "
              f"detect acc {ev.fmt(r['accuracy'])} rec {ev.fmt(r['recall'])} FPR {ev.fmt(r['false_positive_rate'])}, "
              f"category acc {ev.fmt(s['category_accuracy'])} (rules {ev.fmt(s['rule_category_accuracy'])}), "
              f"delay {ev.fmt(s['mean_detection_delay'], 1)} samples, MAE {ev.fmt(s['mean_raw_mae'])} -> "
              f"{ev.fmt(s['mean_corrected_mae'])}, maintenance correct {ev.fmt(s['maintenance_accuracy'])}, "
              f"unneeded maintenance in {s['runs_with_unneeded_maintenance']} runs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load the NuclearRad AI fault model")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    print_report(load_or_train(retrain=args.retrain, verbose=True))
