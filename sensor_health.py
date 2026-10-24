"""Sensor health and maintenance recommendations (simulation only).

For every sensor:
  health score  = 100 x (1 - mean model probability of an instrument fault over the last
                  HEALTH_WINDOW samples). It comes straight from the Random Forest and is as
                  uncalibrated as the model's vote share.
  condition     = Healthy / Watch / Degraded / Faulty / Failed, from the detection and the score.
  severity      = measured from the readings since the sensor was first detected (drift offset,
                  noise ratio, spike events, missing share), never from the healthy reference.
  maintenance   = REPLACE / REPAIR / RECALIBRATE / INSPECT / MONITOR / NONE, from the fault type
                  the model predicted and the measured severity (table MAINTENANCE below).

The severity limits are illustrative values for this simulator, not limits of any real
instrument. Recommendations are suggestions for a simulated maintenance team; nothing here
acts on real equipment.
"""
import numpy as np

import nuclearrad as nr

HEALTH_WINDOW = 20         # samples used for the current health score
HEALTHY_SCORE = 80         # at or above: healthy (if not detected)
FAULTY_SCORE = 50          # below: faulty
DRIFT_REPLACE = 0.30       # drift offset (fraction of the reading) beyond which recalibration is not enough
NOISE_REPLACE = 8.0        # scatter this many times the warm-up scatter: the detector itself is failing
SPIKE_REPAIR_EVENTS = 4    # this many separate spike events: a loose connector or cable, not a one-off
DROPOUT_FAILED = 0.5       # share of recent samples missing: the channel has effectively lost its signal

ACTIONS = ["NONE", "MONITOR", "INSPECT", "RECALIBRATE", "REPAIR", "REPLACE"]   # increasing intervention

MAINTENANCE = {
    # fault: (condition for the stronger action, stronger action, weaker action)
    "Stuck": "REPLACE always: a frozen output means the detector or its electronics stopped responding.",
    "Dropout": "REPAIR always: missing data points to cabling, connector, power or logger problems.",
    "Drift": f"RECALIBRATE below a {DRIFT_REPLACE:.0%} offset; REPLACE at or above it.",
    "Noise": f"REPAIR (shielding, grounding, supply) below {NOISE_REPLACE:g}x normal scatter; REPLACE above it.",
    "Spike": f"INSPECT connectors for fewer than {SPIKE_REPAIR_EVENTS} spike events; REPAIR from {SPIKE_REPAIR_EVENTS} on.",
}


def _peer_deviation(readings, warmup):
    """Each sensor's reading vs the median of the others, both scaled to their warm-up level."""
    x = np.asarray(readings, float)
    rel = x / np.nanmedian(x[:warmup], axis=0)
    S = x.shape[1]
    dev = np.full(x.shape, np.nan)
    for i in range(S):
        peers = np.delete(rel, i, axis=1)
        with np.errstate(all="ignore"):
            dev[:, i] = rel[:, i] / np.nanmedian(peers, axis=1) - 1
    return rel, dev


def _events(mask):
    """Number of separate runs of True values."""
    m = np.asarray(mask, bool).astype(int)
    return int(np.sum(np.diff(np.concatenate([[0], m])) == 1))


def _longest(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def health_timeline(detection, window=HEALTH_WINDOW):
    """Health score (0-100) for every sample and sensor: 100 x (1 - trailing mean fault probability)."""
    proba = detection["proba"]
    fault_idx = [nr.CLASSES.index(c) for c in nr.INSTRUMENT_FAULTS]
    p = proba[:, :, fault_idx].sum(axis=2)
    cum = np.cumsum(np.vstack([np.zeros((1, p.shape[1])), p]), axis=0)
    T = len(p)
    t = np.arange(T)
    lo = np.maximum(t + 1 - window, 0)
    mean = (cum[t + 1] - cum[lo]) / (t + 1 - lo)[:, None]
    return 100 * (1 - mean)


def severity(fault, readings, detection, sensor, warmup, since):
    """Measured severity of a detected fault, from readings and model labels only."""
    rel, dev = _peer_deviation(readings, warmup)
    x = np.asarray(readings, float)[:, sensor]
    labels = detection["sample_labels"][:, sensor]
    after = slice(since, None)
    out = {}
    if fault == "Drift":
        d = np.abs(dev[after, sensor])
        out["offset"] = float(np.nanpercentile(d, 95)) if np.isfinite(d).any() else 0.0
    elif fault == "Noise":
        base = np.nanstd(np.diff(rel[:warmup, sensor]))
        flagged = np.diff(rel[since:, sensor])[(labels[since + 1:] == "Noise")]
        flagged = flagged[np.isfinite(flagged)]
        out["noise_ratio"] = float(np.std(flagged) / base) if flagged.size >= 3 and base > 0 else 0.0
    elif fault == "Spike":
        out["spike_events"] = _events(labels[warmup:] == "Spike")
    elif fault == "Dropout":
        missing = np.isnan(x[after])
        recent = np.isnan(x[-HEALTH_WINDOW:])
        out.update(missing_share=float(missing.mean()), recent_missing_share=float(recent.mean()),
                   longest_gap=_longest(missing))
    elif fault == "Stuck":
        out["frozen_samples"] = _longest(np.concatenate([[False], np.diff(x[after]) == 0]))
    return out


def maintenance_action(fault, sev):
    """(action, urgency, reason) for a detected fault type and its measured severity."""
    if fault == "Stuck":
        return ("REPLACE", "Immediate",
                f"Output frozen for {sev.get('frozen_samples', 0)} samples: the detector or its electronics "
                "stopped responding. Replace it and confirm the new sensor tracks its peers.")
    if fault == "Dropout":
        urgent = sev.get("recent_missing_share", 0) >= DROPOUT_FAILED
        return ("REPAIR", "Immediate" if urgent else "Planned",
                f"{sev.get('missing_share', 0):.0%} of readings missing since detection (longest gap "
                f"{sev.get('longest_gap', 0)} samples). Check cable, connector, power and data logger; replace "
                "the sensor if the signal does not return after repair.")
    if fault == "Drift":
        off = sev.get("offset", 0.0)
        if off >= DRIFT_REPLACE:
            return ("REPLACE", "Planned",
                    f"Drifted {off:.0%} away from its peers, beyond the {DRIFT_REPLACE:.0%} recalibration limit: "
                    "likely detector ageing. Replace, and recalibrate the replacement.")
        return ("RECALIBRATE", "Planned",
                f"Drifted {off:.0%} away from its peers. Recalibrate against a reference source; replace if the "
                "drift returns soon after.")
    if fault == "Noise":
        ratio = sev.get("noise_ratio", 0.0)
        if ratio >= NOISE_REPLACE:
            return ("REPLACE", "Planned",
                    f"Scatter {ratio:.1f}x its normal level: the detector itself is likely failing. Replace it.")
        return ("REPAIR", "Planned",
                f"Scatter {ratio:.1f}x its normal level. Repair shielding, grounding or the power supply; "
                "replace if the noise persists.")
    if fault == "Spike":
        n = sev.get("spike_events", 0)
        if n >= SPIKE_REPAIR_EVENTS:
            return ("REPAIR", "Planned",
                    f"{n} separate spike events: repeated glitches point to a loose connector, damaged cable or "
                    "interference. Repair the connection.")
        return ("INSPECT", "Routine",
                f"{n} spike event(s). Inspect connectors and cabling at the next routine check.")
    return ("INSPECT", "Routine", f"No predefined maintenance for {fault}.")


def assess_health(detection, readings, warmup, t=None):
    """Health, condition and maintenance recommendation for every sensor at sample t (default: last)."""
    readings = np.asarray(readings, float)
    T, S = readings.shape
    t = T - 1 if t is None else int(t)
    scores = health_timeline(detection)
    detected = {d["sensor"]: d for d in detection["sensors"] if d["first_detection"] <= t}
    out = []
    for i in range(S):
        name = nr.NAMES[i] if i < len(nr.NAMES) else str(i)
        score = float(scores[t, i])
        row = dict(sensor=i, name=name, health_score=round(score, 1), fault="None", severity={})
        if i in detected:
            d = detected[i]
            sev = severity(d["fault"], readings[: t + 1], _upto(detection, t), i, warmup, d["first_detection"])
            action, urgency, reason = maintenance_action(d["fault"], sev)
            if d["fault"] == "Stuck" or (d["fault"] == "Dropout"
                                         and sev.get("recent_missing_share", 0) >= DROPOUT_FAILED):
                condition = "Failed"
            elif score < FAULTY_SCORE:
                condition = "Faulty"
            else:
                condition = "Degraded"
            row.update(fault=d["fault"], severity=sev, condition=condition, action=action, urgency=urgency,
                       reason=reason, first_detection=d["first_detection"], confidence=d["confidence"])
        elif score < HEALTHY_SCORE:
            row.update(condition="Watch", action="MONITOR", urgency="Routine",
                       reason=f"The model sees some fault-like behaviour (health {score:.0f}/100) but not enough "
                              "to confirm a fault. Keep monitoring.")
        else:
            row.update(condition="Healthy", action="NONE", urgency="-", reason="Agrees with its peers.")
        out.append(row)
    return out


def _upto(detection, t):
    """The detection's per-sample arrays cut at sample t (so severity never looks past t)."""
    return dict(detection, sample_labels=detection["sample_labels"][: t + 1])


def expected_actions(run):
    """The maintenance action each truly faulty sensor should get, from the simulation's ground truth.

    Used for evaluation only. Applies the same MAINTENANCE table to the configured fault and magnitude.
    """
    cfg = run.config
    out = {}
    if cfg.scenario.startswith("Spike"):
        out[cfg.fault_sensor] = ("REPAIR" if _events(run.labels[:, cfg.fault_sensor] == "Spike")
                                 >= SPIKE_REPAIR_EVENTS else "INSPECT")
    elif cfg.scenario.startswith("Drift") or cfg.scenario == "Two faulty sensors":
        out[cfg.fault_sensor] = "REPLACE" if abs(cfg.mag) >= DRIFT_REPLACE else "RECALIBRATE"
        if cfg.scenario == "Two faulty sensors":
            out[cfg.second_sensor] = "REPLACE"
    elif cfg.scenario == "Real leak + Drift":
        out[cfg.fault_sensor] = "RECALIBRATE"          # the simulator's leak drift is +20%
    elif cfg.scenario == "Stuck":
        out[cfg.fault_sensor] = "REPLACE"
    elif cfg.scenario == "Noise":
        out[cfg.fault_sensor] = "REPLACE" if cfg.mag >= NOISE_REPLACE else "REPAIR"
    elif cfg.scenario == "Dropout":
        out[cfg.fault_sensor] = "REPAIR"
    return out
