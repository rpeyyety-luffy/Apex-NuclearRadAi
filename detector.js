/* NuclearRad fault detector: compares uploaded sensor readings with an ideal reference run
   and labels every reading as normal or as a specific anomaly.
   Runs in the browser (window.Detector) and in Node for tests (module.exports). */
(function (root) {
  "use strict";

  // Ideal reference: reference_ideal.xlsx (a2.xlsx), 600 one-minute readings over 10 h of
  // steady full-power operation with every sensor healthy. Per sensor: mean level, spread of a
  // single reading (sd) and reading-to-reading scatter (noise).
  const REFERENCE = {
    name: "Ideal run: 10 h steady full power, all sensors healthy",
    units: "µSv/h",
    sensors: {
      1: { mean: 2.5093, sd: 0.0544, noise: 0.0545 },
      2: { mean: 7.9871, sd: 0.0962, noise: 0.0902 },
      3: { mean: 0.8044, sd: 0.0293, noise: 0.0315 },
      4: { mean: 0.2499, sd: 0.0174, noise: 0.0178 },
      5: { mean: 1.4987, sd: 0.0408, noise: 0.0388 },
    },
  };

  // detection thresholds (tune these)
  const REF_MATCH = 0.30;     // use the reference only if the sensor's normal level is within 30% of it
  const BASELINE_FRAC = 0.15; // otherwise learn "normal" from the first 15% of the file
  const LEVEL_SIGMA = 4;      // trend this many standard errors away from its peers = drift
  const MIN_SHIFT = 0.01;     // ... and at least 1% away, so tiny calibration differences do not alarm
  const SPIKE_SIGMA = 5;      // one reading this far from the local trend = spike
  const NOISE_RATIO = 3;      // scatter this many times the normal scatter = noisy
  const NOISE_RUN = 5;        // readings in a row the noise model must flag
  const STUCK_RUN = 8;        // identical readings in a row = stuck
  const EVENT_RISE = 0.20;    // most sensors 20%+ above normal together = real radiation rise, not a fault
  const EVENT_SHARE = 0.6;    // "most" = this share of the sensors reporting
  const EVENT_FALL = 0.05;    // most sensors 5%+ below normal together = real pressure loss (leak)
  const FULL_LEAK = 0.25;     // plant level below 25% of normal = full leak: pressure has collapsed to near zero
  const LEAK_STABLE = 0.02;   // plant level moving less than 2% of normal over the recent readings = stabilised

  const TYPES = ["Normal", "Drift high", "Drift low", "Spike", "Noisy", "Stuck", "No signal", "Real rise"];

  const ADVICE = {
    "Drift high": [
      "Recalibrate the sensor against a check source or a reference instrument.",
      "Check the high-voltage supply and temperature compensation for slow change.",
      "If it drifts again soon after recalibration, plan to replace the detector.",
    ],
    "Drift low": [
      "Check for lost detector efficiency: a failing tube or crystal reads low.",
      "Look for new shielding, equipment or debris between the detector and the source.",
      "Recalibrate; replace the detector if the response keeps falling.",
    ],
    "Spike": [
      "Inspect cables, connectors and grounding for loose contacts or interference.",
      "Check whether nearby equipment switched on at the spike times.",
      "If spikes keep coming back, add a short median filter in the data logger.",
    ],
    "Noisy": [
      "Check cable shielding and grounding; interference is the usual cause.",
      "Measure ripple on the high-voltage supply.",
      "Noise that grows over weeks points to an ageing detector; schedule a replacement.",
    ],
    "Stuck": [
      "Check that the data logger is still polling this channel; a frozen value often means no new data.",
      "Power-cycle the sensor and its signal conditioner.",
      "Replace the sensor if it stays frozen after a restart.",
    ],
    "No signal": [
      "Check the power supply, cable and connector for this channel.",
      "Review data-logger and network errors around the gaps.",
      "A reading of zero from a gamma monitor is not physical; treat it as a dead channel.",
    ],
    "Real rise": [
      "This is not a sensor fault: several sensors rose together.",
      "Treat it as a possible real increase in radiation and escalate to operators.",
      "Follow radiation-protection procedures before any sensor maintenance.",
    ],
  };

  // ------------------------------------------------------------------ helpers
  const isReading = (v) => Number.isFinite(v) && v > 0;

  function median(a) {
    const s = a.filter(Number.isFinite).sort((x, y) => x - y);
    if (!s.length) return NaN;
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  function stepNoise(a) {
    // robust reading-to-reading scatter, same units as one reading's standard deviation
    const d = [];
    for (let i = 1; i < a.length; i++) if (isReading(a[i]) && isReading(a[i - 1])) d.push(Math.abs(a[i] - a[i - 1]));
    return median(d) / (0.6745 * Math.SQRT2);
  }

  function robustSd(a) {
    const m = median(a);
    return 1.4826 * median(a.map((v) => Math.abs(v - m)));
  }

  function rolling(a, w, fn) {
    const h = w >> 1, out = new Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = fn(a.slice(Math.max(0, i - h), i + h + 1));
    return out;
  }

  function slope(ys) {
    const pts = ys.map((y, x) => [x, y]).filter((p) => Number.isFinite(p[1]));
    if (pts.length < 5) return 0;
    const mx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
    const my = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    let num = 0, den = 0;
    for (const [x, y] of pts) { num += (x - mx) * (y - my); den += (x - mx) ** 2; }
    return den ? num / den : 0;
  }

  function mostCommon(list) {
    const c = {};
    for (const v of list) c[v] = (c[v] || 0) + 1;
    return Object.keys(c).sort((a, b) => c[b] - c[a])[0];
  }

  function sensorId(name) {
    // "RAD 1", "RAD-01", "Sensor 1 (uSv/h)" all map to sensor 1
    const m = String(name).replace(/\(.*?\)/g, "").match(/(\d+)\s*$/);
    return m ? parseInt(m[1], 10) : null;
  }

  // Build a reference profile from any healthy recording (columns of readings).
  function profileFrom(names, columns, label) {
    const sensors = {};
    names.forEach((name, i) => {
      const v = columns[i].filter(isReading);
      const mean = v.reduce((s, x) => s + x, 0) / v.length;
      const sd = Math.sqrt(v.reduce((s, x) => s + (x - mean) ** 2, 0) / (v.length - 1));
      sensors[sensorId(name) ?? name] = { mean, sd, noise: stepNoise(columns[i]) };
    });
    return { name: label, units: "", sensors };
  }

  // ------------------------------------------------------------------ analysis
  function analyse(names, columns, opts = {}) {
    const reference = opts.reference || REFERENCE;
    const horizon = opts.horizon || 60;
    const n = columns[0].length;
    let w = Math.round(n / 24);
    w = Math.min(61, Math.max(7, w % 2 ? w : w + 1));        // trend window: odd, 7..61 readings
    const baseN = Math.max(20, Math.round(n * BASELINE_FRAC));

    const sensors = names.map((name, i) => {
      const x = columns[i].map((v) => (isReading(v) ? v : NaN));
      const early = x.slice(0, baseN).filter(Number.isFinite);
      const ownLevel = median(early);
      const ref = reference.sensors[sensorId(name) ?? name];

      let mode, mean, sd, noise, note;
      if (ref && Math.abs(ownLevel / ref.mean - 1) <= REF_MATCH) {
        mode = "reference";
        ({ mean, sd, noise } = ref);
        note = `Compared with the ideal reference (normal level ${fmt(mean)}).`;
      } else {
        mode = "own";
        mean = ownLevel;
        sd = robustSd(early);
        noise = stepNoise(columns[i].slice(0, baseN));
        note = ref
          ? `Its normal level here (${fmt(ownLevel)}) is far from the ideal (${fmt(ref.mean)}), probably a different monitor location, so it is compared with its own first ${baseN} readings.`
          : `No matching sensor in the ideal reference, so it is compared with its own first ${baseN} readings.`;
      }
      sd = Math.max(sd || 0, noise || 0, Math.abs(mean) * 1e-3, 1e-12);
      noise = Math.max(noise || 0, sd * 0.5);

      const level = rolling(x, w, median);
      const scatter = rolling(
        x.map((v, t) => (t && Number.isFinite(v) && Number.isFinite(x[t - 1]) ? Math.abs(v - x[t - 1]) : NaN)),
        Math.max(w, 21), median,
      ).map((m) => m / (0.6745 * Math.SQRT2));
      // standard error of the trend line, as a fraction of normal
      const se = 1.2533 * sd / Math.sqrt(w) / Math.abs(mean);

      // identical readings in a row
      const run = new Array(n).fill(0);
      for (let t = 1; t < n; t++) run[t] = Number.isFinite(x[t]) && x[t] === x[t - 1] ? run[t - 1] + 1 : 0;
      const stuck = new Array(n).fill(false);
      for (let t = n - 1; t >= 0; t--) {
        if (run[t] + 1 >= STUCK_RUN) for (let k = t - run[t]; k <= t; k++) stuck[k] = true;
      }
      return { name, mode, note, mean, sd, noise, se, x, level, scatter, stuck, norm: level.map((v) => v / mean) };
    });

    // Drift is judged against the other sensors at the same moment: a change that moves every
    // sensor together (daily background cycle, a real leak) is the plant, not a sensor fault.
    const peerMode = sensors.length >= 3;
    // how far a sensor may move from its peers (as a fraction of normal) before it counts as drift;
    // the comparison is only as steady as the noisier of the sensor and its peers
    for (const s of sensors) {
      const peerSe = peerMode ? median(sensors.filter((o) => o !== s).map((o) => o.se)) : 0;
      s.thr = Math.max(LEVEL_SIGMA * Math.hypot(s.se, peerSe), MIN_SHIFT);
    }
    const events = [];
    const fallen = new Array(n).fill(false);
    const counting = (s, t) => Math.sqrt(Math.max(1, s.norm[t] || 1)); // Poisson: scatter grows with level
    // noise and spikes are judged against the peers' roughness at the same moment: when the whole
    // plant becomes rough (e.g. pressure collapsing in a leak) that is the plant, not one sensor
    const rough = (s, t) => s.scatter[t] / (s.noise * counting(s, t));
    const peerRough = (s, t) => {
      if (!peerMode) return 1;
      const r = median(sensors.filter((o) => o !== s).map((o) => rough(o, t)));
      return Number.isFinite(r) ? Math.max(1, r) : 1;
    };
    const noisy = (s, t) => rough(s, t) > NOISE_RATIO * peerRough(s, t);
    const spiky = (s, t) => Math.abs(s.x[t] - s.level[t]) > SPIKE_SIGMA * s.sd * counting(s, t) * peerRough(s, t);
    // offset from the peers as a fraction of normal level; a ratio to the peers would blow up when
    // every sensor falls towards zero together
    const relTo = (s, t, pool) => {
      const peers = peerMode ? median(pool.filter((o) => o !== s).map((o) => o.norm[t])) : 1;
      const p = Number.isFinite(peers) ? peers : 1;
      return (s.norm[t] - p) / Math.max(1, p);
    };
    // drift: the machine-learned model (driftml.js) catches drift early; the trend rule stays as a
    // backstop for small shifts on very noisy sensors, which the training data did not cover
    // dropout: the dropout model rates how likely each sensor is to keep losing its signal
    // noise: the noise model and the scatter rule each flag a noisy sensor (either one is enough)
    // stuck: the stuck model and the identical-readings rule each flag a frozen sensor (either one is enough)
    // spike: the spike model rates how likely each sensor is to keep producing spikes
    // leak: the leak model rates the whole plant; sensors falling together is a real pressure loss
    const FML = typeof FaultML !== "undefined" ? FaultML : typeof require === "function" ? require("./faultml.js") : null;
    const models = opts.ml !== false && FML ? FML.MODELS : {};
    const ml = peerMode && models.drift ? FML.predict("drift", columns, baseN) : null;
    const dropML = models.dropout ? FML.predict("dropout", columns, baseN) : null;
    const noiseML = peerMode && models.noise ? FML.predict("noise", columns, baseN) : null;
    const stuckML = peerMode && models.stuck ? FML.predict("stuck", columns, baseN) : null;
    const spikeML = peerMode && models.spike ? FML.predict("spike", columns, baseN) : null;
    const leakProb = peerMode && models.leak ? FML.predict("leak", columns, baseN)[0].map((r) => r.p) : null;
    // like noise, a leak lasts: it counts once the model has flagged NOISE_RUN readings in a row
    const leakReadings = leakProb ? leakProb.map((_, t) => t).filter((t) => t >= NOISE_RUN - 1
      && leakProb.slice(t - NOISE_RUN + 1, t + 1).every((p) => p >= models.leak.threshold)) : [];
    sensors.forEach((s, i) => {
      s.driftProb = ml ? ml[i].map((r) => r.p) : null;
      s.dropoutProb = dropML ? dropML[i].map((r) => r.p) : null;
      s.noiseProb = noiseML ? noiseML[i].map((r) => r.p) : null;
      s.stuckProb = stuckML ? stuckML[i].map((r) => r.p) : null;
      s.spikeProb = spikeML ? spikeML[i].map((r) => r.p) : null;
      // excess noise lasts, so the model must flag NOISE_RUN readings in a row before it counts
      s.noiseML = s.noiseProb ? s.noiseProb.map((_, t) => t >= NOISE_RUN - 1
        && s.noiseProb.slice(t - NOISE_RUN + 1, t + 1).every((p) => p >= models.noise.threshold)) : null;
    });
    const stuckNow = (s, t) => s.stuck[t] || !!(s.stuckProb && s.stuckProb[t] >= models.stuck.threshold);
    const drift = (s, i, t) => {
      if (ml && ml[i][t].p >= models.drift.threshold) return ml[i][t].direction > 0 ? "Drift high" : "Drift low";
      return s.rel[t] > s.thr ? "Drift high" : s.rel[t] < -s.thr ? "Drift low" : null;
    };
    for (const s of sensors) s.rel = new Array(n);
    for (let t = 0; t < n; t++) {
      // two passes: sensors that look faulty at this moment are left out of everyone else's peer group
      const suspect = new Set(sensors.filter((s) => !Number.isFinite(s.x[t]) || stuckNow(s, t) || noisy(s, t)
        || Math.abs(relTo(s, t, sensors)) > s.thr));
      const trusted = sensors.filter((s) => !suspect.has(s));
      for (const s of sensors) s.rel[t] = relTo(s, t, trusted.length - (suspect.has(s) ? 0 : 1) >= 2 ? trusted : sensors);
      // only sensors that agree with their peers count towards a real rise
      const live = sensors.filter((s) => Number.isFinite(s.norm[t]));
      const high = live.filter((s) => s.norm[t] - 1 > Math.max(EVENT_RISE, s.thr) && Math.abs(s.rel[t]) <= s.thr);
      const event = live.length >= 3 && high.length >= EVENT_SHARE * live.length;
      if (event) events.push(t);
      // a leak is the mirror image: most sensors that still agree with each other fall together
      const low = live.filter((s) => 1 - s.norm[t] > Math.max(EVENT_FALL, s.thr) && Math.abs(s.rel[t]) <= s.thr);
      fallen[t] = live.length >= 3 && low.length >= EVENT_SHARE * live.length;
      sensors.forEach((s, i) => {
        let type;
        if (!Number.isFinite(s.x[t])) type = "No signal";
        else if (stuckNow(s, t)) type = "Stuck";
        else if (noisy(s, t) || (s.noiseML && s.noiseML[t])) type = "Noisy";
        else if (spiky(s, t)) type = "Spike";
        else if (drift(s, i, t)) type = drift(s, i, t);
        else if (event && high.includes(s)) type = "Real rise";
        else type = "Normal";
        (s.type = s.type || new Array(n))[t] = type;
      });
    }

    // ----------------------------------------------------------- per-sensor verdict
    const tail = Math.max(5, Math.round(n * 0.05));
    const recent = Math.max(30, Math.round(n * 0.1));
    const minWatch = Math.max(3, Math.round(n * 0.01));  // fewer abnormal readings than this = a blip
    for (const s of sensors) {
      const faults = s.type.filter((t) => t !== "Normal" && t !== "Real rise");
      const lastFaults = s.type.slice(-tail).filter((t) => t !== "Normal" && t !== "Real rise");
      s.counts = Object.fromEntries(TYPES.map((k) => [k, s.type.filter((t) => t === k).length]));
      s.anomalies = faults.length;
      s.anomalyPct = 100 * faults.length / n;
      s.firstIssue = s.type.findIndex((t) => t !== "Normal" && t !== "Real rise");
      s.devPct = 100 * ([...s.rel].reverse().find(Number.isFinite) || 0);
      s.eta = null;

      if (lastFaults.length >= tail / 2) {
        s.status = "Faulty";
        s.fault = mostCommon(lastFaults);
        s.summary = `Showing ${s.fault.toLowerCase()} right now (${s.anomalies} abnormal readings in total).`;
      } else if (faults.length >= minWatch || s.counts["Spike"] >= 2) {
        s.status = "Watch";
        s.fault = mostCommon(faults);
        s.summary = `Normal at the end of the file, but showed ${s.fault.toLowerCase()} in ${faults.length} earlier readings. Intermittent faults often return.`;
      } else {
        // healthy now: is its trend heading for the drift limit?
        const k = slope(s.rel.slice(-recent)), kShort = slope(s.rel.slice(-Math.round(recent / 2)));
        const dev = [...s.rel].reverse().find(Number.isFinite) || 0;
        // at risk = already 60% of the way to the drift limit and steadily moving away from the peers
        const moving = k * Math.sign(dev || k) > 0 && kShort * k > 0
          && Math.abs(dev) > 0.6 * s.thr && Math.abs(k) * recent > s.thr;
        const eta = moving ? (s.thr - Math.abs(dev)) / Math.abs(k) : Infinity;
        if (eta < horizon) {
          s.status = "At risk";
          s.fault = k > 0 ? "Drift high" : "Drift low";
          s.eta = Math.max(0, Math.round(eta));
          s.summary = `Trending ${k > 0 ? "up" : "down"}; expected to leave the normal range in about ${s.eta} readings.`;
        } else {
          s.status = "Healthy";
          s.fault = null;
          s.summary = faults.length
            ? `Matches normal behaviour apart from ${faults.length} isolated reading${faults.length > 1 ? "s" : ""}.`
            : "Matches normal behaviour.";
        }
      }
      const dropNow = s.dropoutProb ? s.dropoutProb[n - 1] : 0;
      if (s.dropoutProb && dropNow >= models.dropout.threshold) {
        const gaps = s.type.filter((t, k) => t === "No signal" && s.type[k - 1] !== "No signal").length;
        if (s.status === "Healthy" || s.status === "At risk") {
          s.status = "Watch";
          s.fault = "No signal";
          s.summary = `Lost its signal ${gaps} time${gaps === 1 ? "" : "s"}.`;
        }
        s.summary += ` The dropout model rates further signal loss ${Math.round(100 * dropNow)}% likely.`;
      }
      const spikeNow = s.spikeProb ? s.spikeProb[n - 1] : 0;
      if (s.spikeProb && spikeNow >= models.spike.threshold && s.counts["Spike"]) {
        if (s.status === "Healthy" || s.status === "At risk") {
          s.status = "Watch";
          s.fault = "Spike";
          s.summary = `Showed ${s.counts["Spike"]} spike reading${s.counts["Spike"] === 1 ? "" : "s"}.`;
        }
        s.summary += ` The spike model rates further spikes ${Math.round(100 * spikeNow)}% likely.`;
      }
      s.advice = s.fault ? ADVICE[s.fault] : [];
    }

    // ----------------------------------------------------------- sensor health and maintenance timing
    const spikeStart = (s, t) => s.type[t] === "Spike" && (t === 0 || s.type[t - 1] !== "Spike");
    for (const s of sensors) {
      const series = {
        drift: s.rel.map((v) => (Number.isFinite(v) ? Math.abs(v) : NaN)),
        noise: s.x.map((_, t) => (Number.isFinite(s.scatter[t]) ? rough(s, t) / peerRough(s, t) : NaN)),
        dropout: trailingShare(s.x.map((v) => !Number.isFinite(v)), MAINT_WINDOW),
        spike: trailingShare(s.x.map((_, t) => spikeStart(s, t)), MAINT_WINDOW).map((v) => 100 * v),
      };
      s.maintenance = maintenance(s, series, recent);
    }

    // ----------------------------------------------------------- leak: full or partial
    // A leak is reported when the leak model flags it at the end of the file, or when most sensors
    // that agree with each other have fallen together over the last readings (rule backstop).
    const endN = Math.max(5, Math.round(n * 0.05));
    const modelLeak = leakReadings.length > 0 && leakReadings[leakReadings.length - 1] >= n - endN;
    const ruleLeak = fallen.slice(-endN).every(Boolean);
    let leak = null;
    if (modelLeak || ruleLeak) {
      // plant level = median of the sensors not judged faulty, so a faulty sensor cannot hide or fake the drop
      const healthy = sensors.filter((s) => s.status !== "Faulty");
      const pool = healthy.length >= 3 ? healthy : sensors;
      const plant = Array.from({ length: n }, (_, t) => median(pool.map((s) => s.norm[t])));
      const endLevel = median(plant.slice(-endN));
      const recentLevels = plant.slice(-recent);
      const change = slope(recentLevels) * recentLevels.length;     // change across the recent readings
      let start = n - 1;
      while (start > 0 && (fallen[start - 1] || (leakProb && leakProb[start - 1] >= models.leak.threshold))) start--;
      if (modelLeak) start = Math.min(start, leakReadings[0]);
      const normal = median(pool.map((s) => s.mean));
      leak = {
        kind: endLevel < FULL_LEAK ? "full" : "partial", start, endLevel, normal, endValue: endLevel * normal,
        stable: Math.abs(change) < LEAK_STABLE, source: modelLeak && ruleLeak ? "model and rule" : modelLeak ? "leak model" : "rule",
        prob: leakProb ? leakProb[n - 1] : null,
      };
    }

    let verdict;
    if (leak) {
      const flagged = sensors.filter((s) => s.status !== "Healthy");
      const others = flagged.length
        ? ` On top of the leak, ${flagged.map((s) => `${s.name} (${s.status === "Faulty" ? "" : `${s.status.toLowerCase()}: `}${s.fault.toLowerCase()})`).join(", ")} ${flagged.length > 1 ? "disagree" : "disagrees"} with the other sensors: a sensor fault, not part of the leak.`
        : " No sensor shows its own fault: every sensor follows the same drop.";
      const how = leak.prob !== null && leak.source !== "rule" ? ` The leak model rates it ${Math.round(100 * leak.prob)}% likely.` : "";
      if (leak.kind === "full") {
        verdict = { level: "event", kind: "full", text: `Full leak: every sensor fell together from about ${fmt(leak.normal)} to near zero`
          + ` (${fmt(leak.endValue)}, ${Math.round(100 * leak.endLevel)}% of normal) from about reading ${leak.start + 1}.`
          + ` The sensors agree with each other, so this is not peer drift or a sensor fault: the plant has lost its pressure.${how}`
          + ` Maximum danger: escalate to operators immediately.${others}` };
      } else {
        verdict = { level: "event", kind: "partial", text: `Real leak: all sensors dropped together from about ${fmt(leak.normal)}, starting near reading ${leak.start + 1},`
          + (leak.stable ? ` and stabilised midway at about ${fmt(leak.endValue)} (${Math.round(100 * leak.endLevel)}% of normal).`
            : ` and are still falling (now ${fmt(leak.endValue)}, ${Math.round(100 * leak.endLevel)}% of normal).`)
          + ` This is the plant, not a sensor fault.${how} Escalate to operators.${others}` };
      }
    } else if (events.length) verdict = { level: "event", text: `Possible real radiation rise: most sensors rose together in ${events.length} readings. Escalate to operators.` };
    else if (sensors.some((s) => s.status === "Faulty")) verdict = { level: "fault", text: "Sensor fault detected. The other sensors stay normal, so the faulty sensor is the suspect, not the plant." };
    else if (sensors.some((s) => s.status !== "Healthy")) verdict = { level: "watch", text: "No sensor is failing right now, but some need attention." };
    else verdict = { level: "ok", text: "All sensors match normal behaviour." };

    // overall plant level against the ideal, from the sensors that could be compared with it
    const refSensors = sensors.filter((s) => s.mode === "reference");
    const plantVsIdeal = refSensors.length ? 100 * (median(refSensors.map((s) => median(s.norm))) - 1) : null;

    // readings to shade as the leak: the model's flags plus the rule's run up to the end of the file
    const leakSpan = leak ? [...new Set([...leakReadings, ...Array.from({ length: n - leak.start }, (_, k) => leak.start + k)])].sort((a, b) => a - b) : [];
    return { sensors, events, leak: leak ? { ...leak, prob: leakProb, readings: leakSpan } : null, verdict, plantVsIdeal, peerMode, window: w, baseN, n, reference, advice: ADVICE };
  }

  // ------------------------------------------------------------------ maintenance prediction
  // Each fault type has a degradation measure with two limits: SERVICE (repair or recalibrate due) and
  // REPLACE (the sensor can no longer be trusted). Health = 100 x (1 - how far the measure has gone from
  // normal towards the replace limit). The readings left until each limit come from extrapolating the
  // recent trend of the measure in a straight line: a forecast that assumes the fault keeps developing
  // as it has, not a guarantee. Limits are illustrative values for these recordings, not instrument specs.
  const MAINT_WINDOW = 100;   // readings used for the dropout and spike rates
  const MAINT = {
    drift: { normal: 0, service: 0.02, replace: 0.10, action: "Recalibrate",
      measure: (v) => `${(100 * v).toFixed(1)}% off its peers`, limit: (v) => `${(100 * v).toFixed(0)}%` },
    noise: { normal: 1, service: NOISE_RATIO, replace: 10, action: "Repair",
      measure: (v) => `scatter ${v.toFixed(1)}x normal`, limit: (v) => `${v}x` },
    dropout: { normal: 0, service: 0.05, replace: 0.5, action: "Repair",
      measure: (v) => `${(100 * v).toFixed(0)}% of the last ${MAINT_WINDOW} readings missing`, limit: (v) => `${(100 * v).toFixed(0)}% missing` },
    spike: { normal: 0, service: 1, replace: 5, action: "Repair",
      measure: (v) => `${v.toFixed(1)} spikes per 100 readings`, limit: (v) => `${v} per 100 readings` },
  };
  const NOUN = { Recalibrate: "recalibration", Repair: "repair" };
  const KIND_OF = { "Drift high": "drift", "Drift low": "drift", Noisy: "noise", "No signal": "dropout", Spike: "spike", Stuck: "stuck" };

  function trailingShare(mask, w) {
    const out = new Array(mask.length);
    let sum = 0;
    for (let t = 0; t < mask.length; t++) {
      sum += mask[t] ? 1 : 0;
      if (t >= w) sum -= mask[t - w] ? 1 : 0;
      out[t] = sum / Math.min(t + 1, w);
    }
    return out;
  }

  // readings until `limit` at the recent rate (0 = already there, Infinity = not getting worse)
  function readingsTo(values, limit, span, cfg) {
    const now = median(values.slice(-5));
    if (!Number.isFinite(now)) return { now, rate: 0, left: Infinity };
    if (now >= limit) return { now, rate: 0, left: 0 };
    const rate = slope(values.slice(-span));
    // a trend must move at least 5% of the normal-to-replace range across the window to count
    if (!(rate > 0) || rate * span < 0.05 * (cfg.replace - cfg.normal)) return { now, rate, left: Infinity };
    return { now, rate, left: Math.round((limit - now) / rate) };
  }

  // the trend is fitted over the fault's history since it was first seen (at least the recent window,
  // at most 300 readings): a slow drift looks flat over a few dozen readings
  function trendSpan(s, recent) {
    const since = s.firstIssue >= 0 ? s.x.length - s.firstIssue : 0;
    return Math.max(recent, Math.min(since, 300));
  }

  function maintenance(s, series, recent) {
    // drift limits can never be tighter than what this sensor can resolve against its peers (s.thr)
    const cfgFor = (k) => (k === "drift"
      ? { ...MAINT.drift, service: Math.max(MAINT.drift.service, s.thr), replace: Math.max(MAINT.drift.replace, 3 * s.thr) }
      : MAINT[k]);
    const health = (kind, v) => {
      const c = cfgFor(kind);
      return Math.round(100 * Math.min(1, Math.max(0, 1 - (v - c.normal) / (c.replace - c.normal))));
    };
    const kind = KIND_OF[s.fault] || null;
    const now = Object.fromEntries(Object.keys(MAINT).map((k) => [k, median(series[k].slice(-5))]));
    const valueOf = (k) => (Number.isFinite(now[k]) ? now[k] : MAINT[k].normal);
    if (s.status === "Healthy" || !kind) {
      // healthy: report how much margin is left, but no maintenance date. Short-term jitter in a healthy
      // sensor is not a trend; a real trend shows up as "At risk" first.
      const k = Object.keys(MAINT).sort((a, b) => health(a, valueOf(a)) - health(b, valueOf(b)))[0];
      return { kind: k, health: health(k, valueOf(k)), value: valueOf(k), action: "None", urgency: "-",
        serviceIn: Infinity, replaceIn: Infinity, measure: cfgFor(k).measure(valueOf(k)),
        text: "No maintenance due: it agrees with its peers." };
    }
    if (kind === "stuck") {
      return { kind, health: 0, action: "Replace", urgency: "Immediate", serviceIn: 0, replaceIn: 0,
        text: "Replace now: the output is frozen, so the sensor no longer measures anything." };
    }
    const k = kind;
    const c = cfgFor(k);
    const span = trendSpan(s, recent);
    const svc = readingsTo(series[k], c.service, span, c);
    const rep = readingsTo(series[k], c.replace, span, c);
    const value = valueOf(k);
    const m = { kind: k, health: health(k, value), value, rate: svc.rate, serviceIn: svc.left, replaceIn: rep.left,
      serviceLimit: c.service, replaceLimit: c.replace, measure: c.measure(value) };
    const replaceLater = Number.isFinite(rep.left)
      ? ` At the current rate it reaches the replacement limit (${c.limit(c.replace)}) in about ${rep.left} readings: replace by then if the ${NOUN[c.action]} does not hold.`
      : ` It is not getting worse, so replacement is not expected on the current trend; replace only if the ${NOUN[c.action]} does not fix it.`;
    if (rep.left === 0) {
      Object.assign(m, { action: "Replace", urgency: "Immediate",
        text: `Replace now: ${m.measure}, beyond the replacement limit (${c.limit(c.replace)}).` });
    } else if (svc.left === 0) {
      Object.assign(m, { action: c.action, urgency: "Now", text: `${c.action} now: ${m.measure}, past the service limit (${c.limit(c.service)}).${replaceLater}` });
    } else if (Number.isFinite(svc.left)) {
      Object.assign(m, { action: c.action, urgency: "Planned",
        text: `${c.action} in about ${svc.left} readings: ${m.measure} and rising towards the service limit (${c.limit(c.service)}).` });
    } else {
      Object.assign(m, { action: "Inspect", urgency: "Routine",
        text: `Inspect at the next routine check: ${m.measure}, below the service limit (${c.limit(c.service)}) and not rising.` });
    }
    return m;
  }

  // "about 240 readings (about 4.0 min)" when the time between readings is known
  function whenText(readings, secondsPerReading) {
    if (readings === 0) return "now";
    if (!Number.isFinite(readings)) return "not expected on the current trend";
    if (!secondsPerReading) return `in about ${readings} readings`;
    const sec = readings * secondsPerReading;
    const span = sec < 120 ? `${Math.round(sec)} s` : sec < 7200 ? `${(sec / 60).toFixed(1)} min`
      : sec < 172800 ? `${(sec / 3600).toFixed(1)} h` : `${(sec / 86400).toFixed(1)} days`;
    return `in about ${readings} readings (about ${span})`;
  }

  // seconds between readings from a time column: dates, or numbers in a column named like "Time_s"
  function secondsPerReading(times, header = "") {
    if (!times || times.length < 3) return null;
    let t = times.map((v) => (typeof v === "number" ? v : Date.parse(String(v).replace(" ", "T")) / 1000));
    if (typeof times[0] === "number" || /^\s*-?\d+(\.\d+)?\s*$/.test(String(times[0]))) {
      t = times.map(Number);
      const unit = /(^|_|\s|\()(s|sec|secs|seconds)\)?$/i.test(header) ? 1 : /(min|minutes)\)?$/i.test(header) ? 60
        : /(h|hr|hours)\)?$/i.test(header) ? 3600 : null;
      if (!unit) return null;
      t = t.map((v) => v * unit);
    }
    const d = median(t.slice(1).map((v, i) => v - t[i]).filter((x) => Number.isFinite(x) && x > 0));
    return Number.isFinite(d) ? d : null;
  }

  function fmt(v) {
    return Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2);
  }

  const api = { REFERENCE, TYPES, ADVICE, MAINT, analyse, profileFrom, sensorId, median, fmt, whenText, secondsPerReading, readingsTo, trendSpan };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Detector = api;
})(typeof window !== "undefined" ? window : globalThis);
