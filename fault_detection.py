"""Fault detection and classification.

Stage 1 (observation): the trained model gives class probabilities for every (sample, sensor)
          after the warm-up. A sample is flagged as an instrument fault when the summed probability
          of the instrument-fault classes is at least SAMPLE_THRESHOLD.
Stage 2 (run):       a sensor is reported faulty once it has collected `min_flagged` flagged samples,
          or `min_spike` samples flagged as Spike (spikes are short, so they never add up to
          many samples); both are tuned on the validation split. A plant-wide "Process change" is reported when most
          sensors are classified as following a process change for `min_flagged` samples.

The same stage-2 aggregation is applied to the existing rule-based detector (nuclearrad.assess)
so the two can be compared on equal terms. Their outputs are kept separate.

Probabilities come from a Random Forest (share of trees voting for a class). They are NOT
calibrated and must not be read as the chance that the prediction is right.
"""
import numpy as np

import nuclearrad as nr
from feature_engineering import compute_features

SAMPLE_THRESHOLD = 0.5
LOW_CONFIDENCE = 0.6


def _first_reaching(flags, count):
    """First index at which the running count of True values reaches `count` (None if never)."""
    cum = np.cumsum(flags)
    hit = np.flatnonzero(cum >= count)
    return int(hit[0]) if hit.size else None


def aggregate(flags, fault_scores, process_flags, min_flagged, warmup, names, source,
              sample_labels, confidence=None, min_spike=None):
    """Turn per-sample decisions into a run-level verdict.

    flags         (T, S) bool   instrument fault flagged at that sample
    fault_scores  (T, S, F)     score per instrument-fault class (to pick the fault type)
    process_flags (T,)  bool    plant-wide process change seen at that sample
    confidence    (T, S) float  model vote share for the chosen class (None for rules)
    min_spike     int           samples labelled Spike that are enough on their own (None = same as min_flagged)
    """
    min_spike = min_flagged if min_spike is None else min_spike
    T, S = flags.shape
    sensors = []
    for i in range(S):
        f = flags[:, i].copy()
        f[:warmup] = False
        count = int(f.sum())
        times = [_first_reaching(f, min_flagged),
                 _first_reaching(f & (sample_labels[:, i] == "Spike"), min_spike)]
        times = [t for t in times if t is not None]
        if not times:
            continue
        first = min(times)
        scores = fault_scores[f, i].sum(axis=0)
        cls = nr.INSTRUMENT_FAULTS[int(np.argmax(scores))]
        conf = float(np.mean(confidence[f, i])) if confidence is not None else None
        sensors.append(dict(sensor=i, name=names[i], fault=cls, flagged_samples=count,
                            first_detection=first, confidence=conf))
    sensors.sort(key=lambda d: -d["flagged_samples"])

    pf = process_flags.copy()
    pf[:warmup] = False
    process_first = _first_reaching(pf, min_flagged)
    process = process_first is not None

    if len(sensors) >= 2:
        fault_part = "Multiple faults"
    elif sensors:
        fault_part = sensors[0]["fault"]
    else:
        fault_part = None
    if process and fault_part:
        category = f"Process change + {fault_part}"
    elif process:
        category = "Process change"
    else:
        category = fault_part or "Normal"

    notes = []
    if len(sensors) >= S - 2:
        notes.append(f"{len(sensors)} of {S} sensors disagree with their peers: peer comparison is unreliable, "
                     "so the 'faulty' sensors may in fact be the healthy ones.")
    if process and sensors:
        notes.append("A plant-wide change and a sensor fault are both present; the fault reading should be "
                     "checked by an independent instrument before any correction is trusted.")
    if confidence is not None and sensors and sensors[0]["confidence"] < LOW_CONFIDENCE:
        notes.append(f"Low model vote share ({sensors[0]['confidence']:.0%}) for the fault type: treat the "
                     "category as uncertain.")
    return dict(source=source, category=category, fault_present=bool(sensors), process_change=process,
                process_first_detection=process_first, sensors=sensors,
                primary_sensor=sensors[0]["sensor"] if sensors else None,
                primary_confidence=sensors[0]["confidence"] if sensors else None,
                min_flagged=min_flagged, min_spike=min_spike, sample_labels=sample_labels, sample_flags=flags,
                notes=notes)


def predict_observations(bundle, readings, warmup):
    """Class probabilities (T, S, C) for every sample after the warm-up (zeros before it)."""
    feats = compute_features(readings, warmup)
    T, S, F = feats.shape
    model = bundle["model"]
    proba = np.zeros((T, S, len(nr.CLASSES)))
    p = model.predict_proba(feats[warmup:].reshape(-1, F)).reshape(T - warmup, S, -1)
    for j, cls in enumerate(model.classes_):
        proba[warmup:, :, nr.CLASSES.index(cls)] = p[:, :, j]
    return proba


def predict_run(bundle, readings, warmup, names=None):
    """Machine-learning detection for one run. Uses readings only (never truth or labels)."""
    readings = np.asarray(readings, float)
    T, S = readings.shape
    names = list(names) if names is not None else nr.NAMES[:S]
    proba = predict_observations(bundle, readings, warmup)
    fault_idx = [nr.CLASSES.index(c) for c in nr.INSTRUMENT_FAULTS]
    fault_scores = proba[:, :, fault_idx]
    fault_prob = fault_scores.sum(axis=2)
    flags = fault_prob >= SAMPLE_THRESHOLD
    flags[:warmup] = False

    labels = np.full((T, S), "Warm-up", dtype=object)
    best_fault = np.array(nr.INSTRUMENT_FAULTS, dtype=object)[np.argmax(fault_scores, axis=2)]
    other = np.where(proba[:, :, nr.CLASSES.index("Process change")] > proba[:, :, 0], "Process change", "Normal")
    labels[warmup:] = np.where(flags, best_fault, other)[warmup:]
    conf = np.where(flags, fault_scores.max(axis=2), np.maximum(proba[:, :, 0], proba[:, :, -1]))

    live = np.isfinite(readings).sum(axis=1)
    process_votes = (labels == "Process change").sum(axis=1)
    process_flags = (process_votes >= np.ceil(live / 2)) & (live >= 3)
    result = aggregate(flags, fault_scores, process_flags, bundle["min_flagged"], warmup, names,
                       "Machine learning (Random Forest)", labels, confidence=conf,
                       min_spike=bundle.get("min_spike"))
    result["proba"] = proba
    return result


def rule_based_run(readings, warmup, min_flagged, names=None, min_spike=None):
    """The existing rule-based detector (nuclearrad.assess) passed through the same aggregation."""
    readings = np.asarray(readings, float)
    T, S = readings.shape
    names = list(names) if names is not None else nr.NAMES[:S]
    res = nr.assess(readings, names=names, warmup=warmup)
    rule_labels = np.where(res["fault"] == "Healthy", "Normal", res["fault"]).astype(object)
    flags = np.isin(rule_labels, nr.INSTRUMENT_FAULTS)
    flags[:warmup] = False
    scores = np.stack([(rule_labels == c).astype(float) for c in nr.INSTRUMENT_FAULTS], axis=2)
    process_flags = np.array([v.startswith("Possible real leak") for v in res["verdict"]])   # not "..., not a leak"
    labels = rule_labels.copy()
    labels[:warmup] = "Warm-up"
    labels[process_flags[:, None] & ~flags] = "Process change"
    result = aggregate(flags, scores, process_flags, min_flagged, warmup, names,
                       "Rule-based (nuclearrad.assess)", labels, min_spike=min_spike)
    result["assessment"] = res
    return result


def detection_time_for(result, sensors):
    """First detection of any truly faulty sensor (None if none of them was detected)."""
    times = [s["first_detection"] for s in result["sensors"] if s["sensor"] in sensors]
    return min(times) if times else None
