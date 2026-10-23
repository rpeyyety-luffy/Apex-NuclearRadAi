"""NuclearRad AI - core logic (no UI).

1. generate()  : synthetic plant with 5 sensors where we know the truth
2. assess()    : peer cross-check -> fault rules -> trust score -> verdict -> fix
3. summarise() : numbers for the demo (alarm vs AI, error before/after)

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
def assess(x):
    """Run the whole pipeline on readings x (samples, sensors)."""
    T, S = x.shape

    # (a) calibrate every sensor against its peers during the healthy warm-up
    peer0 = np.array([np.nanmedian(np.delete(x[:WARMUP], i, axis=1), axis=1)
                      for i in range(S)]).T
    cal = np.nanmedian(x[:WARMUP] / peer0, axis=0)
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

    base_mad = np.array([mad(dr[1:WARMUP, i]) for i in range(S)])
    base_mad = np.maximum(base_mad, 0.7 * np.median(base_mad))   # no sensor gets an unluckily tiny baseline

    # short-term scatter vs healthy scatter; "noisy" needs two samples in a row
    ratio = np.zeros((T, S))
    for t in range(WARMUP, T):
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
    for t in range(WARMUP, T):
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
    penalty[:WARMUP] = 0
    trust = 100 * (1 - pd.DataFrame(penalty).ewm(alpha=0.3).mean().to_numpy())

    # (e) fix: a flagged sensor is replaced by a virtual reading from its peers
    corrected = np.where(flagged, peer, y)

    # (f) network verdict at every sample
    level = pd.DataFrame(y).rolling(3, min_periods=1).mean().to_numpy() / np.nanmean(y[:WARMUP], axis=0)
    rising = level > EVENT_RISE
    verdict, action, reason = [], [], []
    for t in range(T):
        ok = [i for i in range(S) if not flagged[t, i]]
        bad = [i for i in range(S) if flagged[t, i]]
        unsure = [i for i in ok if trust[t, i] < UNCERTAIN_TRUST]
        event = len(ok) >= 3 and rising[t, ok].mean() >= EVENT_SHARE
        if t < WARMUP:
            v, a, why = "Calibrating", "WAIT", "Learning each sensor's normal behaviour."
        elif event and bad:
            names = ", ".join(NAMES[i] for i in bad)
            v = "Possible real leak + sensor fault"
            a = "ESCALATE to operators; " + "; ".join(f"{ACTIONS[fault[t, i]]} {NAMES[i]}" for i in bad)
            why = (f"{len(ok)} healthy sensors rise together (a leak is never suppressed). "
                   f"{names} also disagrees with its peers ({fault[t, bad[0]].lower()}).")
        elif event:
            v, a = "Possible real leak", "ESCALATE to operators"
            why = "Several sensors rose together with similar timing. Never treated as a fault."
        elif bad:
            names = ", ".join(NAMES[i] for i in bad)
            kinds = ", ".join(sorted({fault[t, i].lower() for i in bad}))
            others = ", ".join(NAMES[i] for i in ok)
            v = "Sensor fault, not a leak"
            a = "; ".join(f"{ACTIONS[fault[t, i]]} {NAMES[i]}" for i in bad)
            why = f"{names}: {kinds}. Peers ({others}) stay normal, so the sensor is the suspect."
        elif unsure:
            v, a = "Uncertain", "CHECK " + ", ".join(NAMES[i] for i in unsure)
            why = "Evidence is thin. Treated as a possible leak until checked."
        else:
            v, a, why = "Normal", "NONE", "All sensors agree."
        verdict.append(v); action.append(a); reason.append(why)

    return dict(x=x, y=y, peer=peer, r=r, fault=fault, trust=trust, corrected=corrected,
                verdict=verdict, action=action, reason=reason)


# ------------------------------------------------------- 3. demo numbers
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