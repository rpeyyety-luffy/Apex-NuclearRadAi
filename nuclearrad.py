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


def load_or_train(retrain=False):
    """Load the trained fault model, training it first if no compatible model is saved.

    Previously this returned None. The model lives in model_training.py; it is imported here
    (not at the top) because model_training itself imports this module.
    """
    import model_training
    return model_training.load_or_train(retrain=retrain)


# ===================================================== 5. configurable simulator
# The functions above keep the original fixed demo (RAD-03 breaks at sample 80). The simulator
# below adds what the machine-learning pipeline needs: every fault parameter is configurable,
# the healthy reference ("truth") is kept apart from the observed readings, and every sample of
# every sensor carries a ground-truth label. Values are illustrative normalized units, not
# validated physical limits of a real facility.
from dataclasses import asdict, dataclass, field
import hashlib

BASELINES = (100.0, 102.0, 98.0, 101.0, 99.0)   # healthy level of each sensor (normalized units)

# fault categories a sensor-level observation can have. "Process change" is NOT an instrument fault:
# the plant itself changed and the sensors report it correctly.
CLASSES = ["Normal", "Spike", "Drift", "Stuck", "Noise", "Dropout", "Process change"]
INSTRUMENT_FAULTS = ["Spike", "Drift", "Stuck", "Noise", "Dropout"]

# scenario -> run-level category the system should report for it
SIM_SCENARIOS = {
    "Normal": "Normal",
    "Spike (positive)": "Spike",
    "Spike (negative)": "Spike",
    "Drift (upward)": "Drift",
    "Drift (downward)": "Drift",
    "Stuck": "Stuck",
    "Noise": "Noise",
    "Dropout": "Dropout",
    "Two faulty sensors": "Multiple faults",
    "Real leak": "Process change",
    "Real leak + Drift": "Process change + Drift",
}
RUN_CATEGORIES = sorted(set(SIM_SCENARIOS.values()))

# magnitude meaning and sensible bounds per scenario: (default, low, high, meaning)
MAGNITUDES = {
    "Normal": (0.0, 0.0, 0.0, "not used"),
    "Spike (positive)": (1.0, 0.3, 3.0, "spike height as a fraction of the reading (1.0 = doubles)"),
    "Spike (negative)": (-0.5, -0.9, -0.2, "spike depth as a fraction of the reading"),
    "Drift (upward)": (0.25, 0.05, 0.6, "offset reached at the end of the fault window (0.25 = +25%)"),
    "Drift (downward)": (-0.25, -0.6, -0.05, "offset reached at the end of the fault window"),
    "Stuck": (0.0, 0.0, 0.0, "not used: the sensor freezes at its reading at fault start"),
    "Noise": (6.0, 3.0, 15.0, "noise multiplier on the faulty sensor"),
    "Dropout": (0.3, 0.1, 0.6, "share of the fault window with no data"),
    "Two faulty sensors": (0.25, 0.05, 0.6, "drift offset of the first sensor (the second one freezes)"),
    "Real leak": (0.6, -0.5, 1.5, "plant-wide change in the true level (0.6 = +60%)"),
    "Real leak + Drift": (0.6, 0.2, 1.5, "plant-wide rise; the faulty sensor also drifts +20%"),
}


@dataclass
class ScenarioConfig:
    """Everything needed to reproduce one simulated run."""
    scenario: str = "Drift (upward)"
    seed: int = 7
    n_samples: int = N_SAMPLES
    warmup: int = WARMUP
    noise: float = NOISE
    fault_sensor: int = FAULT_SENSOR
    second_sensor: int = 4            # only used by "Two faulty sensors"
    fault_start: int = FAULT_START
    duration: int = None              # samples; None = until the end of the run
    magnitude: float = None           # None = scenario default (see MAGNITUDES)
    n_affected: int = N_SENSORS       # sensors moved by a shared process change
    baselines: tuple = BASELINES

    def validate(self):
        """Raise ValueError for settings the simulator cannot honour."""
        if self.scenario not in SIM_SCENARIOS:
            raise ValueError(f"Unknown scenario {self.scenario!r}")
        S = len(self.baselines)
        if S < 3:
            raise ValueError("At least three sensors are needed for peer comparison")
        if not 60 <= self.n_samples <= 2000:
            raise ValueError("n_samples must be between 60 and 2000")
        if not 20 <= self.warmup < self.n_samples - 10:
            raise ValueError("warmup must be at least 20 samples and leave room for the fault")
        if not 0 < self.noise <= 0.1:
            raise ValueError("noise must be between 0 and 10%")
        if not 0 <= self.fault_sensor < S or not 0 <= self.second_sensor < S:
            raise ValueError("sensor index out of range")
        if self.scenario == "Two faulty sensors" and self.second_sensor == self.fault_sensor:
            raise ValueError("the two faulty sensors must differ")
        if not self.warmup <= self.fault_start < self.n_samples - 5:
            raise ValueError("fault_start must be after the warm-up and at least 5 samples before the end")
        if self.duration is not None and self.duration < 5:
            raise ValueError("duration must be at least 5 samples")
        if not 3 <= self.n_affected <= S:
            raise ValueError("a shared change must affect at least three sensors")
        lo, hi = MAGNITUDES[self.scenario][1:3]
        if self.magnitude is not None and not lo <= self.magnitude <= hi:
            raise ValueError(f"magnitude for {self.scenario} must be between {lo} and {hi}")

    @property
    def fault_end(self):
        """Exclusive end index of the fault window."""
        end = self.n_samples if self.duration is None else self.fault_start + self.duration
        return min(end, self.n_samples)

    @property
    def mag(self):
        return MAGNITUDES[self.scenario][0] if self.magnitude is None else self.magnitude

    def run_id(self):
        """Short stable identifier derived from every parameter."""
        text = repr(sorted(asdict(self).items()))
        return hashlib.sha1(text.encode()).hexdigest()[:10]


@dataclass
class SimulationRun:
    """One simulated run. readings is what the instruments report; truth is the healthy reference."""
    config: ScenarioConfig
    readings: np.ndarray          # (samples, sensors); NaN = no data
    truth: np.ndarray             # (samples, sensors); never altered by an instrument fault
    labels: np.ndarray            # (samples, sensors); one of CLASSES
    observable: np.ndarray        # (samples, sensors); False where a fault is too small to be visible yet
    fault_sensors: list = field(default_factory=list)   # instrument-faulty sensors
    names: list = field(default_factory=lambda: list(NAMES))

    @property
    def run_id(self):
        return self.config.run_id()

    @property
    def expected_category(self):
        return SIM_SCENARIOS[self.config.scenario]

    def to_long_frame(self):
        """One row per (sample, sensor) with the ground truth and the scenario parameters."""
        cfg = self.config
        T, S = self.readings.shape
        t = np.repeat(np.arange(T), S)
        s = np.tile(np.arange(S), T)
        in_window = (t >= cfg.fault_start) & (t < cfg.fault_end)
        return pd.DataFrame({
            "run_id": self.run_id,
            "scenario_id": cfg.scenario,
            "time_step": t,
            "sensor_id": [self.names[i] for i in s],
            "healthy_reference": self.truth.ravel(),
            "observed_reading": self.readings.ravel(),
            "fault_type": self.labels.ravel(),
            "fault_present": np.isin(self.labels.ravel(), INSTRUMENT_FAULTS),
            "affected_sensor": np.isin(s, self.fault_sensors),
            "fault_start": cfg.fault_start if cfg.scenario != "Normal" else -1,
            "fault_end": cfg.fault_end if cfg.scenario != "Normal" else -1,
            "in_fault_window": in_window & (cfg.scenario != "Normal"),
            "magnitude": cfg.mag,
            "noise": cfg.noise,
            "seed": cfg.seed,
        })


def simulate(cfg):
    """Generate one reproducible run from a ScenarioConfig (same config + seed = same run)."""
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    T, S = cfg.n_samples, len(cfg.baselines)
    base = np.asarray(cfg.baselines, dtype=float)
    t = np.arange(T)
    a, b = cfg.fault_start, cfg.fault_end
    k, mag = cfg.fault_sensor, cfg.mag
    truth = np.tile(base, (T, 1))
    labels = np.full((T, S), "Normal", dtype=object)
    observable = np.ones((T, S), bool)
    faulty = []

    # (1) the plant: a shared process change moves the TRUE level of several sensors together
    if cfg.scenario in ("Real leak", "Real leak + Drift"):
        ramp = np.clip((t - a) / 30, 0, 1) * ((t >= a) & (t < b))
        affected = rng.choice(S, size=cfg.n_affected, replace=False) if cfg.n_affected < S else np.arange(S)
        if cfg.scenario == "Real leak + Drift" and k not in affected:
            affected[0] = k
        truth[:, affected] *= (1 + mag * ramp)[:, None]
        changed = np.abs(truth / base - 1) >= 2 * cfg.noise
        labels[changed] = "Process change"

    # (2) the instruments: healthy noise on top of the truth
    x = truth * (1 + cfg.noise * rng.standard_normal((T, S)))

    # (3) instrument faults change the readings only; truth is never touched
    def drift(sensor, offset):
        span = max(b - a - 1, 1)
        bias = np.zeros(T)
        bias[a:b] = offset * (np.arange(b - a) / span)
        x[:, sensor] *= 1 + bias
        labels[a:b, sensor] = "Drift"
        observable[a:b, sensor] = np.abs(bias[a:b]) >= 1.5 * cfg.noise   # too small to see yet
        faulty.append(sensor)

    def stuck(sensor):
        x[a:b, sensor] = x[a, sensor]
        labels[a:b, sensor] = "Stuck"
        observable[a:min(a + 2, b), sensor] = False                       # needs a few repeats
        faulty.append(sensor)

    if cfg.scenario.startswith("Spike"):
        events = int(rng.integers(3, 6))
        starts = np.sort(rng.choice(np.arange(a, max(b - 2, a + 1)), size=min(events, max(b - 2 - a, 1)),
                                    replace=False))
        for s0 in starts:
            n = int(rng.integers(1, 3))
            x[s0:s0 + n, k] = truth[s0:s0 + n, k] * (1 + mag * rng.uniform(0.8, 1.2))
            labels[s0:min(s0 + n, b), k] = "Spike"
        faulty.append(k)
    elif cfg.scenario.startswith("Drift"):
        drift(k, mag)
    elif cfg.scenario == "Stuck":
        stuck(k)
    elif cfg.scenario == "Noise":
        x[a:b, k] = truth[a:b, k] * (1 + cfg.noise * mag * rng.standard_normal(b - a))
        labels[a:b, k] = "Noise"
        observable[a:min(a + 3, b), k] = False
        faulty.append(k)
    elif cfg.scenario == "Dropout":
        missing = max(1, int(round(mag * (b - a))))
        gaps = max(1, min(3, missing // 5))
        lengths = np.full(gaps, missing // gaps)
        lengths[: missing % gaps] += 1
        cursor = a
        for g, length in enumerate(lengths):
            room = (b - cursor) - int(lengths[g + 1:].sum()) - length
            start = cursor + int(rng.integers(0, max(room // max(gaps - g, 1), 0) + 1))
            x[start:start + length, k] = np.nan
            labels[start:start + length, k] = "Dropout"
            cursor = start + length + 1
        faulty.append(k)
    elif cfg.scenario == "Two faulty sensors":
        drift(k, mag)
        stuck(cfg.second_sensor)
    elif cfg.scenario == "Real leak + Drift":
        drift(k, 0.2)

    return SimulationRun(config=cfg, readings=x, truth=truth, labels=labels, observable=observable,
                         fault_sensors=sorted(set(faulty)), names=NAMES[:S] if S <= len(NAMES)
                         else [f"RAD-{i + 1:02d}" for i in range(S)])


def random_config(scenario, seed):
    """A random but reproducible parameter variation of one scenario (used to build datasets).

    The same scenario category with different magnitudes, start times, durations, sensors and
    noise gives many independent runs; a category alone is not a training example.
    """
    rng = np.random.default_rng(seed)
    S = len(BASELINES)
    sensor = int(rng.integers(0, S))
    second = int((sensor + rng.integers(1, S)) % S)
    start = int(rng.integers(55, 115))
    duration = None if rng.random() < 0.5 else int(rng.integers(40, N_SAMPLES - start + 1))
    default, lo, hi, _ = MAGNITUDES[scenario]
    magnitude = None if lo == hi else float(np.round(rng.uniform(lo, hi), 3))
    if scenario == "Real leak" and magnitude is not None and abs(magnitude) < 0.1:
        magnitude = 0.1 if magnitude >= 0 else -0.1     # a change too small to see is not a test
    return ScenarioConfig(scenario=scenario, seed=int(seed), noise=float(np.round(rng.uniform(0.008, 0.02), 4)),
                          fault_sensor=sensor, second_sensor=second, fault_start=start, duration=duration,
                          magnitude=magnitude, n_affected=int(rng.choice([4, 5, 5])))