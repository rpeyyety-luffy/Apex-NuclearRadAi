"""NuclearRad AI - core logic (no UI).

1. generate()  : synthetic plant with 5 sensors where we know the truth
2. assess()    : peer cross-check -> fault rules -> trust score -> verdict -> fix
3. forecast()  : rank sensors by failure risk and suggest fixes
4. summarise() : numbers for the demo (alarm vs AI, error before/after)

Everything is plain numpy/pandas so you can explain it line by line.
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- settings
N_SENSORS, N_SAMPLES = 5, 180
BASE, NOISE = 100.0, 0.012          # baseline reading, 1.2% noise
WARMUP = 40                         # first 40 samples assumed healthy (calibration)
FAULT_START, FAULT_SENSOR = 80, 2   # RAD-03 breaks at sample 80
NAMES = [f"RAD-{i + 1:02d}" for i in range(N_SENSORS)]

SCENARIOS = ["Normal", "Real leak", "Spike", "Drift", "Stuck",
             "Noise", "Dropout", "Real leak + Drift"]

ACTIONS = {"Spike": "INSPECT", "Drift": "RECALIBRATE", "Stuck": "REPLACE",
           "Noise": "INSPECT", "Dropout": "INSPECT"}

# detection thresholds (tune these)
SPIKE_R = 0.20      # one sample this far from peers = spike candidate
DRIFT_R = 0.04      # persistent 4% offset from peers = drift
FLAT_RUN = 5        # identical readings in a row = stuck
NOISE_RATIO = 3.5   # short-term scatter vs its own healthy scatter
EVENT_RISE = 1.08   # a sensor "rises" when 8% above its baseline
EVENT_SHARE = 0.60  # share of healthy sensors rising = real leak
UNCERTAIN_TRUST = 75


# --------------------------------------------------------- 1. synthetic data
def generate(scenario, seed=7, noise=NOISE):
    """Return (readings, truth), both shaped (samples, sensors).
    Only RAD-03 is faulty. 'Real leak' lifts every sensor together."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES)
    k = FAULT_SENSOR
    truth = np.full((N_SAMPLES, N_SENSORS), BASE)

    if "Real leak" in scenario:
        ramp = np.clip((t - FAULT_START) / 30, 0, 1)          # +90% over 30 samples
        truth = truth * (1 + 0.9 * ramp)[:, None]

    x = truth * (1 + noise * rng.standard_normal(truth.shape))

    if scenario == "Spike":
        for s in (FAULT_START, 120):                          # 1-2 sample jumps, 2.5-5x
            n = int(rng.integers(1, 3))
            x[s:s + n, k] = truth[s:s + n, k] * rng.uniform(2.5, 5)
    elif scenario in ("Drift", "Real leak + Drift"):
        x[:, k] *= 1 + 0.007 * np.clip(t - FAULT_START, 0, None)   # bias grows 0.7%/sample
    elif scenario == "Stuck":
        x[FAULT_START:, k] = x[FAULT_START, k]                # freezes at last value
    elif scenario == "Noise":
        after = t >= FAULT_START
        x[after, k] = truth[after, k] * (1 + noise * 8 * rng.standard_normal(after.sum()))
    elif scenario == "Dropout":
        for a, b in ((80, 95), (130, 140)):                   # gaps with no data
            x[a:b, k] = np.nan
    return x, truth


# ------------------------------------------------------------ 2. the AI
def assess(x, names=None, warmup=WARMUP):
    """Run the whole pipeline on readings x (samples, sensors).
    names/warmup default to the synthetic plant; uploaded data passes its own."""
    names = list(names) if names is not None else NAMES[:x.shape[1]]
    T, S = x.shape

    # (a) calibrate every sensor against its peers during the healthy warm-up
    peer0 = np.array([np.nanmedian(np.delete(x[:warmup], i, axis=1), axis=1)
                      for i in range(S)]).T
    cal = np.nanmedian(x[:warmup] / peer0, axis=0)
    y = x / cal

    # (b) peer cross-check: r = distance from the consensus of the OTHER sensors.
    #     A real event moves everyone, so r stays ~0. A fault moves one sensor.
    peer = np.array([np.nanmedian(np.delete(y, i, axis=1), axis=1) for i in range(S)]).T
    r = y / peer - 1
    r_med = pd.DataFrame(r).rolling(5, min_periods=1).median().to_numpy()   # robust 5-sample view
    dr = pd.DataFrame(r).diff().to_numpy()                                   # sample-to-sample change

    def mad(a):
        a = a[~np.isnan(a)]
        return np.median(np.abs(a - np.median(a))) if a.size >= 6 else 0.0

    base_mad = np.array([mad(dr[1:warmup, i]) for i in range(S)])
    base_mad = np.maximum(base_mad, 0.7 * np.median(base_mad))   # no sensor gets an unluckily tiny baseline

    # short-term scatter vs healthy scatter; "noisy" needs two samples in a row
    ratio = np.zeros((T, S))
    for t in range(warmup, T):
        for i in range(S):
            ratio[t, i] = mad(dr[max(1, t - 11):t + 1, i]) / base_mad[i]
    hit = (ratio > NOISE_RATIO) & (np.roll(ratio, 1, axis=0) > NOISE_RATIO)
    noisy = pd.DataFrame(hit).rolling(4, min_periods=1).max().to_numpy().astype(bool)   # keep the label steady

    # run length of identical readings (a frozen sensor never changes at all)
    flat = np.zeros((T, S), int)
    for t in range(1, T):
        same = np.abs(x[t] - x[t - 1]) < 1e-12        # NaN compares False
        flat[t] = np.where(same, flat[t - 1] + 1, 0)

    # (c) fault rules, one sample at a time, using only past data
    fault = np.full((T, S), "Healthy", dtype=object)
    for t in range(warmup, T):
        for i in range(S):
            if np.isnan(x[t, i]):
                fault[t, i] = "Dropout"
            elif flat[t, i] >= FLAT_RUN:
                fault[t, i] = "Stuck"
            elif noisy[t, i]:
                fault[t, i] = "Noise"
            elif abs(r[t, i]) > SPIKE_R and abs(r_med[t, i]) < DRIFT_R:
                fault[t, i] = "Spike"
            elif abs(r_med[t, i]) > DRIFT_R and ratio[t, i] < NOISE_RATIO:
                fault[t, i] = "Drift"

    flagged = fault != "Healthy"

    # (d) trust = 1 - smoothed fault probability. A small early-warning penalty
    #     grows with the distance from peers even before a rule fires.
    soft = np.nan_to_num(np.clip(np.abs(r_med) / 0.10, 0, 0.5))
    penalty = np.where(flagged, 0.9, soft)
    penalty[:warmup] = 0
    trust = 100 * (1 - pd.DataFrame(penalty).ewm(alpha=0.3).mean().to_numpy())

    # (e) fix: a flagged sensor is replaced by a virtual reading from its peers
    corrected = np.where(flagged, peer, y)

    # (f) network verdict at every sample
    level = pd.DataFrame(y).rolling(3, min_periods=1).mean().to_numpy() / np.nanmean(y[:warmup], axis=0)
    rising = level > EVENT_RISE
    verdict, action, reason = [], [], []
    for t in range(T):
        ok = [i for i in range(S) if not flagged[t, i]]
        bad = [i for i in range(S) if flagged[t, i]]
        unsure = [i for i in ok if trust[t, i] < UNCERTAIN_TRUST]
        event = len(ok) >= 3 and rising[t, ok].mean() >= EVENT_SHARE
        if t < warmup:
            v, a, why = "Calibrating", "WAIT", "Learning each sensor's normal behaviour."
        elif event and bad:
            who = ", ".join(names[i] for i in bad)
            v = "Possible real leak + sensor fault"
            a = "ESCALATE to operators; " + "; ".join(f"{ACTIONS[fault[t, i]]} {names[i]}" for i in bad)
            why = (f"{len(ok)} healthy sensors rise together (a leak is never suppressed). "
                   f"{who} also disagrees with its peers ({fault[t, bad[0]].lower()}).")
        elif event:
            v, a = "Possible real leak", "ESCALATE to operators"
            why = "Several sensors rose together with similar timing. Never treated as a fault."
        elif bad:
            who = ", ".join(names[i] for i in bad)
            kinds = ", ".join(sorted({fault[t, i].lower() for i in bad}))
            others = ", ".join(names[i] for i in ok)
            v = "Sensor fault, not a leak"
            a = "; ".join(f"{ACTIONS[fault[t, i]]} {names[i]}" for i in bad)
            why = f"{who}: {kinds}. Peers ({others}) stay normal, so the sensor is the suspect."
        elif unsure:
            v, a = "Uncertain", "CHECK " + ", ".join(names[i] for i in unsure)
            why = "Evidence is thin. Treated as a possible leak until checked."
        else:
            v, a, why = "Normal", "NONE", "All sensors agree."
        verdict.append(v); action.append(a); reason.append(why)

    return dict(x=x, y=y, peer=peer, r=r, r_med=r_med, ratio=ratio, flat=flat, fault=fault,
                trust=trust, corrected=corrected, verdict=verdict, action=action, reason=reason,
                names=names, warmup=warmup)


# ----------------------------------------------- 3. failure prediction
RISK_WINDOW = 20    # recent samples used to measure each sensor's trend
HEALTHY_DEV = 0.01  # healthy sensors sit within ~1% of their peers
HEALTHY_NOISE = 2.0 # and within 2x their warm-up scatter

REMEDIES = {
    "Drift": ["Recalibrate against a check source or a reference instrument.",
              "Check temperature compensation and the high-voltage supply for slow change.",
              "If drift returns soon after recalibration, plan to replace the detector."],
    "Spike": ["Inspect cables, connectors and grounding for loose contacts or interference.",
              "Look for nearby equipment switching on at the spike times.",
              "Add a short median filter to the data logger if spikes keep coming back."],
    "Stuck": ["Check that the data logger is still polling this channel (a frozen value often means no new data).",
              "Power-cycle the sensor and its signal conditioner.",
              "Replace the sensor if it stays frozen after a restart."],
    "Noise": ["Check cable shielding and grounding; noise often comes from interference.",
              "Measure high-voltage supply ripple.",
              "Rising noise over weeks points to detector ageing; schedule a replacement."],
    "Dropout": ["Check the communication link, power and connector for this channel.",
                "Review logger buffer and network errors around the gaps.",
                "Use the AI-corrected peer estimate until the data stream is stable."],
}


def forecast(res, horizon=30):
    """Rank sensors by how likely they are to fail next.

    Looks at the last RISK_WINDOW samples of each sensor: how fast it is moving away
    from its peers, how noisy it is getting, how long it has been frozen, how often it
    was flagged and how far its trust has fallen. Returns one dict per sensor, riskiest first.
    """
    x, r_med, ratio, flat = res["x"], res["r_med"], res["ratio"], res["flat"]
    fault, trust, names, warmup = res["fault"], res["trust"], res["names"], res["warmup"]
    T, S = x.shape
    lo = max(warmup, T - RISK_WINDOW)
    t = np.arange(lo, T)
    rows = []
    for i in range(S):
        now = fault[-1, i]
        dev = r_med[lo:, i]
        ok = ~np.isnan(dev)
        slope = np.polyfit(t[ok], dev[ok], 1)[0] if ok.sum() >= 5 else 0.0
        dev_now = float(dev[ok][-1]) if ok.any() else 0.0
        moving_away = np.sign(slope) == np.sign(dev_now) or abs(dev_now) < 1e-9
        to_drift = ((DRIFT_R - abs(dev_now)) / abs(slope)
                    if moving_away and abs(slope) > 1e-6 and abs(dev_now) < DRIFT_R else None)
        noise_now = float(np.nanmean(ratio[lo:, i])) if T > warmup else 0.0
        flags = float(np.mean(fault[lo:, i] != "Healthy")) if T > warmup else 0.0
        trust_now = float(trust[-1, i])

        # each signal scaled 0..1: 0 = inside normal healthy variation, 1 = at the fault threshold
        signals = {
            "Drift": max((abs(dev_now) - HEALTHY_DEV) / (DRIFT_R - HEALTHY_DEV),
                         1 - to_drift / horizon if to_drift is not None and to_drift < horizon else 0),
            "Noise": (noise_now - HEALTHY_NOISE) / (NOISE_RATIO - HEALTHY_NOISE),
            "Stuck": (flat[-1, i] - 1) / (FLAT_RUN - 1),
            "Dropout": float(np.mean(np.isnan(x[lo:, i]))) * 2,
            "Spike": float(np.mean(np.abs(res["r"][lo:, i]) > SPIKE_R)) * 4,
        }
        signals = {k: float(np.clip(v, 0, 1)) for k, v in signals.items()}
        likely = max(signals, key=signals.get)
        trust_loss = np.clip((90 - trust_now) / 40, 0, 1)
        score = 100 * (0.6 * signals[likely] + 0.2 * flags + 0.2 * trust_loss)

        if now != "Healthy":
            status, mode, score = "Faulty now", now, max(score, 90)
            outlook = f"Already showing {now.lower()}. Act now."
        elif score >= 50:
            status, mode = "High risk", likely
            outlook = (f"Moving away from its peers; likely to reach the drift limit in ~{to_drift:.0f} readings."
                       if likely == "Drift" and to_drift is not None
                       else f"Early signs of {likely.lower()}; failure likely within {horizon} readings.")
        elif score >= 30:
            status, mode = "Watch", likely
            outlook = f"Weak signs of {likely.lower()}. Recheck at the next data upload."
        elif (past := fault[warmup:, i][fault[warmup:, i] != "Healthy"]).size >= 2:
            # recovered now, but intermittent faults tend to come back
            kinds, counts = np.unique(past, return_counts=True)
            status, mode, score = "Watch", kinds[counts.argmax()], max(score, 30)
            outlook = (f"Healthy now, but showed {mode.lower()} in {past.size} earlier readings. "
                       "Intermittent faults often return.")
        else:
            status, mode, outlook = "Healthy", None, "Agrees with its peers. No action needed."

        rows.append(dict(sensor=names[i], status=status, risk=round(score), likely_fault=mode,
                         trust=round(trust_now), deviation_pct=round(100 * dev_now, 2),
                         samples_to_drift_limit=int(to_drift) if to_drift is not None and to_drift < horizon else None,
                         flagged_recent_pct=round(100 * flags), outlook=outlook,
                         remedies=REMEDIES.get(mode, []) if status != "Healthy" else []))
    return sorted(rows, key=lambda row: -row["risk"])


# ------------------------------------------------------- 4. demo numbers
def summarise(res, truth, k=FAULT_SENSOR):
    """Threshold alarm vs NuclearRad AI, and reading error before/after the fix."""
    x, fault = res["x"], res["fault"]
    base = np.nanmean(x[:WARMUP], axis=0)
    alarm = np.where((np.nan_to_num(x) > 1.3 * base).any(axis=1))[0]
    alarm = alarm[alarm >= WARMUP]
    ai = np.where((fault[:, k] != "Healthy") & (np.arange(len(x)) >= FAULT_START))[0]

    seg = slice(FAULT_START, None)

    def rms(a):
        e = a[seg, k] / truth[seg, k] - 1
        e = e[~np.isnan(e)]
        return float(np.sqrt(np.mean(e ** 2)) * 100) if e.size else float("nan")

    return dict(first_alarm=int(alarm[0]) if alarm.size else None,
                first_ai=int(ai[0]) if ai.size else None,
                err_before=rms(x), err_after=rms(res["corrected"]))


def load_or_train():
    return None