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
    const counting = (s, t) => Math.sqrt(Math.max(1, s.norm[t] || 1)); // Poisson: scatter grows with level
    const noisy = (s, t) => s.scatter[t] > NOISE_RATIO * s.noise * counting(s, t);
    const relTo = (s, t, pool) => {
      const peers = peerMode ? median(pool.filter((o) => o !== s).map((o) => o.norm[t])) : 1;
      return s.norm[t] / (Number.isFinite(peers) ? peers : 1) - 1;
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
      sensors.forEach((s, i) => {
        let type;
        if (!Number.isFinite(s.x[t])) type = "No signal";
        else if (stuckNow(s, t)) type = "Stuck";
        else if (noisy(s, t) || (s.noiseML && s.noiseML[t])) type = "Noisy";
        else if (Math.abs(s.x[t] - s.level[t]) > SPIKE_SIGMA * s.sd * counting(s, t)) type = "Spike";
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

    let verdict;
    const leakNow = leakReadings.length && leakReadings[leakReadings.length - 1] >= n - Math.max(5, Math.round(n * 0.05));
    if (leakNow) {
      const faulty = sensors.filter((s) => s.status === "Faulty");
      verdict = { level: "event", text: `Real pressure loss: all sensors are falling together from about reading ${leakReadings[0] + 1}`
        + ` (the leak model rates it ${Math.round(100 * leakProb[n - 1])}% likely). This is the plant, not a sensor fault. Escalate to operators.`
        + (faulty.length ? ` Separately, ${faulty.map((s) => `${s.name} (${s.fault.toLowerCase()})`).join(", ")} ${faulty.length > 1 ? "show their" : "shows its"} own fault.` : "") };
    } else if (events.length) verdict = { level: "event", text: `Possible real radiation rise: most sensors rose together in ${events.length} readings. Escalate to operators.` };
    else if (sensors.some((s) => s.status === "Faulty")) verdict = { level: "fault", text: "Sensor fault detected. The other sensors stay normal, so the faulty sensor is the suspect, not the plant." };
    else if (sensors.some((s) => s.status !== "Healthy")) verdict = { level: "watch", text: "No sensor is failing right now, but some need attention." };
    else verdict = { level: "ok", text: "All sensors match normal behaviour." };

    // overall plant level against the ideal, from the sensors that could be compared with it
    const refSensors = sensors.filter((s) => s.mode === "reference");
    const plantVsIdeal = refSensors.length ? 100 * (median(refSensors.map((s) => median(s.norm))) - 1) : null;

    return { sensors, events, leak: leakProb ? { prob: leakProb, readings: leakReadings } : null, verdict, plantVsIdeal, peerMode, window: w, baseN, n, reference, advice: ADVICE };
  }

  function fmt(v) {
    return Math.abs(v) >= 10 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2);
  }

  const api = { REFERENCE, TYPES, ADVICE, analyse, profileFrom, sensorId, median, fmt };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Detector = api;
})(typeof window !== "undefined" ? window : globalThis);
