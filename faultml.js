/* Machine-learned sensor fault models, trained on labelled recordings in ml/data
   (see ml/train_models.js). Each model is a logistic regression over a few features
   computed from past readings only, so it works the same on a live feed.

   drift   - each sensor is compared with how it related to its peers in normal operation
             (the healthy baseline at the start of the file); six features describe how far,
             how fast and how steadily it is moving away.
   dropout - how often, how recently and for how long a sensor has lost its signal; it rates
             whether the sensor is dropping out intermittently, also between the gaps.
   noise   - how rough a sensor's readings are from one reading to the next, compared with its own
             normal roughness and with its peers'; slow drift and plant-wide changes do not count.
   spike   - how far each reading jumps from the sensor's recent level while its peers stay put, and
             how often that has happened lately; it rates whether the sensor keeps producing spikes.
   stuck   - how long a sensor has repeated exactly the same value and how flat its recent readings are,
             while its peers keep moving.
   leak    - one rating for the whole plant: whether the sensors are falling together, as they do
             in a real coolant leak (a fault of one sensor moves only that sensor).

   Runs in the browser (window.FaultML) and in Node (module.exports). */
(function (root) {
  "use strict";

  // BEGIN MODELS (written by ml/train_models.js - do not edit by hand)
  const MODELS = {"drift":{"features":["|z15|","|z60|","away60","beyond2","persist","scatter"],"mean":[0.82743,0.94961,0.08403,0.14739,0.22869,0.79561],"std":[0.80544,0.8985,0.64356,0.23356,0.2256,0.36265],"weights":[1.08704,-0.25565,1.90705,1.16682,0.62206,-1.9589],"bias":-9.37028,"threshold":0.9515},"dropout":{"features":["missing now","missing 30","missing 120","gaps 300","longest gap 300","since gap"],"mean":[0.0057,0.00568,0.00518,0.07989,0.12219,5.5322],"std":[0.07527,0.0545,0.03631,0.25521,0.4944,0.69833],"weights":[0.43562,0.1281,0.56674,0.20337,2.12168,-0.08882],"bias":-7.66123,"threshold":0.5},"noise":{"features":["rough8","rough30","rough60","jumps30","rough30 vs peers"],"mean":[-0.01155,0.08957,0.10143,0.0498,0.04926],"std":[0.82955,0.7409,0.70999,0.16538,0.73765],"weights":[1.78908,-0.40559,-0.19119,1.10703,2.27607],"bias":-7.81374,"threshold":0.882},"stuck":{"features":["same run","flat30","rough15","peers rough15"],"mean":[0.13071,0.02862,0.09925,0.08372],"std":[0.80263,0.15447,0.77856,0.28013],"weights":[3.03476,1.06374,-0.37868,-0.15214],"bias":-4.99379,"threshold":0.5},"spike":{"features":["jump now","peers jump now","spikes 60","spikes 300","since spike","rough30"],"mean":[0.64646,0.6135,0.1236,0.19284,5.41102,0.09948],"std":[0.53839,0.33662,0.51584,0.73478,1.09778,0.74089],"weights":[-1.26151,-0.44201,-2.87321,2.04626,-3.39144,-0.80159],"bias":-3.32246,"threshold":0.999},"leak":{"features":["fall15","fall60","slope60","sensors falling"],"mean":[0.53973,0.50686,0.33508,0.13088],"std":[1.72469,1.663,1.33888,0.31403],"weights":[2.69737,-0.90406,1.53284,1.73799],"bias":-4.45347,"threshold":0.7597}};
  // END MODELS

  const FEATURES = {
    drift: ["|z15|", "|z60|", "away60", "beyond2", "persist", "scatter"],
    dropout: ["missing now", "missing 30", "missing 120", "gaps 300", "longest gap 300", "since gap"],
    noise: ["rough8", "rough30", "rough60", "jumps30", "rough30 vs peers"],
    leak: ["fall15", "fall60", "slope60", "sensors falling"],
    stuck: ["same run", "flat30", "rough15", "peers rough15"],
    spike: ["jump now", "peers jump now", "spikes 60", "spikes 300", "since spike", "rough30"],
  };

  function median(a) {
    const s = a.filter(Number.isFinite).sort((x, y) => x - y);
    if (!s.length) return NaN;
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  function robustSd(a) {
    const m = median(a);
    return 1.4826 * median(a.map((v) => Math.abs(v - m)));
  }

  function tail(a, t, w) {
    return a.slice(Math.max(0, t - w + 1), t + 1).filter(Number.isFinite);
  }

  function slope(ys) {
    const n = ys.length;
    if (n < 5) return 0;
    const mx = (n - 1) / 2, my = ys.reduce((s, y) => s + y, 0) / n;
    let num = 0, den = 0;
    ys.forEach((y, x) => { num += (x - mx) * (y - my); den += (x - mx) ** 2; });
    return num / den;
  }

  const present = (v) => Number.isFinite(v) && v > 0;

  // d[i][t]: sensor i's gap to its peers, minus its normal gap, in units of normal scatter.
  // Readings are first scaled by each sensor's normal level, so sensors at different levels
  // (and a real change that scales every sensor together) compare fairly.
  function deviations(columns, baseN) {
    const n = columns[0].length;
    const scaled = columns.map((col) => {
      const level = median(col.slice(0, baseN).filter((v) => v > 0));
      return col.map((v) => (present(v) ? v / level : NaN));
    });
    return scaled.map((col, i) => {
      const gap = col.map((v, t) => v - median(scaled.filter((_, j) => j !== i).map((c) => c[t])));
      const base = gap.slice(0, baseN);
      const offset = median(base);
      const scale = Math.max(robustSd(base), 1e-9);
      return Array.from({ length: n }, (_, t) => (gap[t] - offset) / scale);
    });
  }

  function driftFeatures(columns, baseN) {
    return deviations(columns, baseN).map((d) => d.map((_, t) => {
      const w15 = tail(d, t, 15), w30 = tail(d, t, 30), w60 = tail(d, t, 60);
      if (w15.length < 5) return null;
      const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
      // medians, so a single spike does not look like a shift (1.2533 = efficiency of the median)
      const z15 = median(w15) * Math.sqrt(w15.length) / 1.2533;
      const z60 = median(w60) * Math.sqrt(w60.length) / 1.2533;
      const away = slope(w60) * w60.length * Math.sign(z60 || 1);   // change across the window, away from normal
      const beyond2 = w30.filter((v) => Math.abs(v) > 2).length / w30.length;
      const persist = Math.abs(mean(w30.map(Math.sign)));
      const scatter = robustSd(w30);
      return [Math.abs(z15), Math.abs(z60), away, beyond2, persist, scatter];
    }));
  }

  function dropoutFeatures(columns) {
    return columns.map((col) => {
      const missing = col.map((v) => !present(v));
      let lastMissing = -Infinity;
      const run = [];   // length of the gap ending at each reading
      return missing.map((m, t) => {
        if (m) lastMissing = t;
        run[t] = m ? (run[t - 1] || 0) + 1 : 0;
        const frac = (w) => { let c = 0; for (let k = Math.max(0, t - w + 1); k <= t; k++) c += missing[k]; return c / Math.min(w, t + 1); };
        let gaps = 0;
        for (let k = Math.max(1, t - 299); k <= t; k++) if (missing[k] && !missing[k - 1]) gaps++;
        if (t < 300 && missing[0]) gaps++;
        let longest = 0;
        for (let k = Math.max(0, t - 299); k <= t; k++) longest = Math.max(longest, run[k]);
        return [+m, frac(30), frac(120), gaps, longest, Math.min(300, t - lastMissing)];
      });
    });
  }

  // roughness = scatter of the step from one reading to the next, in units of the sensor's own
  // normal step scatter; a step-to-step measure, so slow drift and slow plant swings barely count.
  // "vs peers" compares it with the other sensors' roughness, so noise across the whole plant does not.
  function noiseFeatures(columns, baseN) {
    const steps = columns.map((col) => {
      const level = median(col.slice(0, baseN).filter((v) => v > 0));
      const st = col.map((v, t) => (t > 0 && present(v) && present(col[t - 1]) ? (v - col[t - 1]) / level : NaN));
      const scale = Math.max(robustSd(st.slice(1, baseN)), 1e-9);
      return st.map((v) => v / scale);
    });
    const rough = steps.map((st) => st.map((_, t) => (t >= 4 ? robustSd(tail(st, t, 30)) : NaN)));
    return steps.map((st, i) => st.map((_, t) => {
      const s8 = tail(st, t, 8), s30 = tail(st, t, 30), s60 = tail(st, t, 60);
      if (s8.length < 5) return null;
      const peers = median(rough.filter((_, j) => j !== i).map((r) => r[t]));
      const r30 = robustSd(s30);
      return [robustSd(s8), r30, robustSd(s60), s30.filter((v) => Math.abs(v) > 3).length / s30.length,
        Number.isFinite(peers) ? r30 / Math.max(peers, 0.05) : 1];
    }));
  }

  // one row for the whole plant: the median of the sensors (each scaled by its normal level), how far
  // and how steadily it has fallen below normal in units of its normal scatter, and the share of
  // sensors that have each fallen clearly below their own normal level
  function leakFeatures(columns, baseN) {
    const scaled = columns.map((col) => {
      const level = median(col.slice(0, baseN).filter((v) => v > 0));
      return col.map((v) => (present(v) ? v / level : NaN));
    });
    const n = columns[0].length;
    const plant = Array.from({ length: n }, (_, t) => median(scaled.map((c) => c[t])));
    const base = plant.slice(0, baseN), normal = median(base);
    const sigma = Math.max(robustSd(base), 1e-9);
    const own = scaled.map((c) => ({ mid: median(c.slice(0, baseN)), sd: Math.max(robustSd(c.slice(0, baseN)), 1e-9) }));
    return [plant.map((_, t) => {
      const w15 = tail(plant, t, 15), w60 = tail(plant, t, 60);
      if (w15.length < 5) return null;
      const live = scaled.filter((c) => Number.isFinite(c[t]));
      const falling = scaled.filter((c, i) => Number.isFinite(c[t]) && (own[i].mid - median(tail(c, t, 15))) / own[i].sd > 3).length;
      return [(normal - median(w15)) / sigma, (normal - median(w60)) / sigma, -slope(w60) * w60.length / sigma,
        live.length ? falling / live.length : 0];
    })];
  }

  // jump = a reading's distance from the median of the sensor's previous 10 readings, in units of its
  // normal scatter; a spike is a jump of over SPIKE_JUMP while the peers' typical jump stays small, so a
  // plant-wide step does not count. rough30 (robust, so a few spikes barely move it) tells spikes from noise.
  const SPIKE_JUMP = 8;
  function spikeFeatures(columns, baseN) {
    const jumps = columns.map((col) => {
      const level = median(col.slice(0, baseN).filter((v) => v > 0));
      const r = col.map((v, t) => (t >= 3 && present(v) ? (v - median(col.slice(Math.max(0, t - 10), t).filter(present))) / level : NaN));
      const scale = Math.max(robustSd(r.slice(0, baseN)), 1e-9);
      return r.map((v) => v / scale);
    });
    return jumps.map((z, i) => {
      const peers = z.map((_, t) => median(jumps.filter((_, j) => j !== i).map((o) => Math.abs(o[t]))));
      const spike = z.map((v, t) => Math.abs(v) > SPIKE_JUMP && !(peers[t] > 3));
      let last = -Infinity;
      return z.map((v, t) => {
        if (spike[t]) last = t;
        if (t < 8) return null;
        let s60 = 0, s300 = 0;   // spikes counted once, also when they last two readings
        for (let k = Math.max(1, t - 299); k <= t; k++) if (spike[k] && !spike[k - 1]) { s300++; if (k > t - 60) s60++; }
        return [Number.isFinite(v) ? Math.abs(v) : 0, Number.isFinite(peers[t]) ? peers[t] : 0, s60, s300,
          Math.min(300, t - last), tail(z, t, 30).length >= 5 ? robustSd(tail(z, t, 30)) : 1];   // 1 = normal roughness inside a long gap
      });
    });
  }

  // a frozen sensor repeats one value exactly; rough15 is its step scatter over the last 15 readings in units
  // of its normal step scatter (0 when frozen), and the peers' rough15 shows whether the plant itself is still
  function stuckFeatures(columns, baseN) {
    const steps = columns.map((col) => {
      const st = col.map((v, t) => (t > 0 && present(v) && present(col[t - 1]) ? v - col[t - 1] : NaN));
      const scale = Math.max(robustSd(st.slice(1, baseN)), 1e-12);
      return st.map((v) => v / scale);
    });
    const rough = steps.map((st) => st.map((_, t) => {
      const w = tail(st, t, 15);
      return w.length >= 5 ? Math.sqrt(w.reduce((a, v) => a + v * v, 0) / w.length) : NaN;
    }));
    return columns.map((col, i) => {
      let run = 0;
      return col.map((v, t) => {
        run = t > 0 && present(v) && v === col[t - 1] ? run + 1 : 0;
        if (t < 8 || !present(v)) return null;
        const w30 = tail(steps[i], t, 30);
        const peers = median(rough.filter((_, j) => j !== i).map((r) => r[t]));
        return [run, w30.length ? w30.filter((x) => x === 0).length / w30.length : 0, Number.isFinite(rough[i][t]) ? rough[i][t] : 1,
          Number.isFinite(peers) ? peers : 1];
      });
    });
  }

  function features(kind, columns, baseN) {
    if (kind === "stuck") return stuckFeatures(columns, baseN);
    if (kind === "spike") return spikeFeatures(columns, baseN);
    if (kind === "drift") return driftFeatures(columns, baseN);
    if (kind === "noise") return noiseFeatures(columns, baseN);
    if (kind === "leak") return leakFeatures(columns, baseN);
    return dropoutFeatures(columns);
  }

  function transform(kind, f) {
    // log-compress unbounded counts so one extreme value does not swamp the other features
    if (kind === "drift") return [Math.log1p(f[0]), Math.log1p(f[1]), Math.sign(f[2]) * Math.log1p(Math.abs(f[2])), f[3], f[4], Math.log1p(f[5])];
    if (kind === "noise") return [Math.log(f[0] + 0.05), Math.log(f[1] + 0.05), Math.log(f[2] + 0.05), f[3], Math.log(f[4] + 0.05)];
    const signedLog = (v) => Math.sign(v) * Math.log1p(Math.abs(v));
    if (kind === "leak") return [signedLog(f[0]), signedLog(f[1]), signedLog(f[2]), f[3]];
    if (kind === "stuck") return [Math.log1p(f[0]), f[1], Math.log(f[2] + 0.05), Math.log(f[3] + 0.05)];
    if (kind === "spike") return [Math.log1p(f[0]), Math.log1p(f[1]), Math.log1p(f[2]), Math.log1p(f[3]), Math.log1p(f[4]), Math.log(f[5] + 0.05)];
    return [f[0], f[1], f[2], Math.log1p(f[3]), Math.log1p(f[4]), Math.log1p(f[5])];
  }

  function probability(kind, model, f) {
    if (!f) return 0;
    const x = transform(kind, f);
    let s = model.bias;
    for (let k = 0; k < x.length; k++) s += model.weights[k] * (x[k] - model.mean[k]) / model.std[k];
    return 1 / (1 + Math.exp(-s));
  }

  // probability per sensor per reading (leak: a single row for the plant); null when that model is not bundled
  function predict(kind, columns, baseN, model = MODELS[kind]) {
    if (!model) return null;
    const d = kind === "drift" ? deviations(columns, baseN) : null;
    return features(kind, columns, baseN).map((fs, i) => fs.map((f, t) => ({
      p: probability(kind, model, f),
      direction: d ? Math.sign(d[i][t]) || 1 : 0,
    })));
  }

  const api = { MODELS, FEATURES, features, deviations, transform, probability, predict };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FaultML = api;
})(typeof window !== "undefined" ? window : globalThis);
