"""Recommended responses and the simulated correction stage.

These are responses inside a simulation. Nothing here controls, or should be used to control,
real plant equipment.

Correction method (illustrative): a detected faulty sensor's reading is replaced by a robust
estimate from its TRUSTED peers, i.e. the median of the peers' readings, each scaled by its own
warm-up level, re-scaled to the faulty sensor's warm-up level. Only readings are used. The
healthy reference ("truth") is used afterwards for evaluation only.
"""
import numpy as np

import nuclearrad as nr

MIN_TRUSTED_PEERS = 2

RECOMMENDATIONS = {
    "Normal": dict(action="No action", correct=False,
                   text="All sensors agree with each other. Keep monitoring."),
    "Spike": dict(action="Flag and replace the spike samples", correct=True,
                  text="Flag the isolated samples and evaluate a robust peer-based replacement for them only. "
                       "Inspect cabling, connectors and grounding if spikes repeat."),
    "Drift": dict(action="Flag the trend and use a peer estimate", correct=True,
                  text="The sensor is moving away from its peers. Use a peer-based estimate from detection onward "
                       "and schedule recalibration against a reference instrument."),
    "Stuck": dict(action="Mark frozen and use a peer estimate", correct=True,
                  text="The sensor stopped changing while its peers keep moving. Compare against independent "
                       "sensors, check that the channel is still polled, and plan a power-cycle or replacement."),
    "Noise": dict(action="Flag instability and use a robust estimate", correct=True,
                  text="Scatter is far above the sensor's normal level. Use a robust peer estimate and check "
                       "shielding, grounding and supply ripple."),
    "Dropout": dict(action="Mark unavailable; estimate for display", correct=True,
                    text="Readings are missing. Mark them unavailable; a peer-based estimate can be shown for "
                         "visualisation but is not a measurement. Check the link and logger."),
    "Multiple faults": dict(action="Warn: peer correction may be unreliable", correct=True,
                            text="More than one sensor disagrees with its peers. Peer-based correction needs enough "
                                 "healthy sensors and may be unreliable; verify with an independent instrument."),
    "Process change": dict(action="Do not correct; escalate for review", correct=False,
                           text="Several sensors changed together, which looks like a real process change rather "
                                "than a sensor failure. Readings are NOT corrected. Escalate to the operating team "
                                "in the simulation and confirm with independent evidence."),
}
PERSISTENT = {"Drift", "Stuck", "Noise"}       # corrected on every sample after detection
TRANSIENT = {"Spike", "Dropout"}               # corrected only on the flagged samples


def recommend(detection):
    """Recommended response for a run-level detection result (independent of what the user simulates)."""
    category = detection["category"]
    parts = [p.strip() for p in category.split("+")]
    recs = []
    for part in parts:
        rec = RECOMMENDATIONS.get(part)
        if rec is None:
            rec = dict(action="Review manually", correct=False, text=f"No predefined response for {part}.")
        recs.append(dict(category=part, **rec))
    sensors = ", ".join(f"{s['name']} ({s['fault'].lower()})" for s in detection["sensors"]) or "none"
    return dict(category=category, items=recs, sensors=sensors, notes=list(detection.get("notes", [])),
                correction_recommended=any(r["correct"] for r in recs) and detection["fault_present"])


def apply_correction(readings, detection, warmup, min_peers=MIN_TRUSTED_PEERS, sensors=None):
    """Simulate the correction for the sensors the detector reported (or `sensors` if given).

    Returns corrected readings (raw readings everywhere no correction was made), masks of where an
    estimate was made or was unavailable, a per-sensor status and warnings. Raw readings and the
    detection result are not modified.
    """
    x = np.asarray(readings, float)
    T, S = x.shape
    corrected = x.copy()
    estimated = np.zeros((T, S), bool)
    unavailable = np.zeros((T, S), bool)
    warnings = []

    detected = {d["sensor"]: d for d in detection["sensors"]}
    targets = list(detected) if sensors is None else [s for s in sensors if s in detected]
    if detection.get("process_change") and not detection["fault_present"]:
        warnings.append("Plant-wide process change: readings are reported as they are and not corrected.")
        return dict(corrected=corrected, estimated=estimated, unavailable=unavailable, status={},
                    warnings=warnings, method="none (process change)")
    if not targets:
        return dict(corrected=corrected, estimated=estimated, unavailable=unavailable, status={},
                    warnings=["No faulty sensor detected: nothing to correct."], method="none")

    level = np.nanmedian(x[:warmup], axis=0)        # calibration from the healthy warm-up (readings only)
    flags = np.asarray(detection["sample_flags"], bool)
    # a sensor is not trusted as a reference from the moment it was detected as faulty
    distrust_from = {i: d["first_detection"] for i, d in detected.items()}
    status = {}
    for i in targets:
        d = detected[i]
        start = d["first_detection"]
        if d["fault"] in PERSISTENT:
            todo = np.arange(start, T)
        else:
            todo = np.flatnonzero(flags[:, i] | (np.isnan(x[:, i]) & (np.arange(T) >= warmup)))
            todo = todo[todo >= warmup]
        made = 0
        for t in todo:
            peers = [j for j in range(S) if j != i and np.isfinite(x[t, j]) and not flags[t, j]
                     and not (j in distrust_from and t >= distrust_from[j])]
            if len(peers) >= min_peers:
                corrected[t, i] = np.median(x[t, peers] / level[peers]) * level[i]
                estimated[t, i] = True
                made += 1
            else:
                corrected[t, i] = x[t, i]           # keep the raw reading (NaN stays NaN), do not invent one
                unavailable[t, i] = True
        missed = len(todo) - made
        status[i] = dict(name=nr.NAMES[i] if i < len(nr.NAMES) else str(i), fault=d["fault"],
                         samples_to_correct=int(len(todo)), estimated=int(made), unavailable=int(missed),
                         possible=made > 0)
        if missed:
            warnings.append(f"{status[i]['name']}: no reliable estimate for {missed} of {len(todo)} samples "
                            f"(fewer than {min_peers} trusted peers); raw readings kept there.")
    if len(detected) >= 2:
        warnings.append("Several sensors were corrected from the same peer group; estimates rely on the "
                        "remaining sensors being healthy.")
    if detection.get("process_change"):
        warnings.append("A process change is also present: peers follow the real change, so the estimate "
                        "tracks the plant, but it is not an independent measurement.")
    return dict(corrected=corrected, estimated=estimated, unavailable=unavailable, status=status,
                warnings=warnings, method=f"median of trusted peers (min {min_peers}), scaled by warm-up level")
