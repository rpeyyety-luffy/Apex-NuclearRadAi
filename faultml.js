/* Machine-learned sensor fault models, trained on labelled recordings in ml/data
   (see ml/train_models.js). Each model is a logistic regression over a few features
   computed from past readings only, so it works the same on a live feed.

   drift   - each sensor is compared with how it related to its peers in normal operation
             (the healthy baseline at the start of the file); six features describe how far,
             how fast and how steadily it is moving away.
   dropout - how often, how recently and for how long a sensor has lost its signal; it rates
             whether the sensor is dropping out intermittently, also between the gaps.

   Runs in the browser (window.FaultML) and in Node (module.exports). */
(function (root) {
  "use strict";

  // BEGIN MODELS (written by ml/train_models.js - do not edit by hand)
  const MODELS = {"drift":{"features":["|z15|","|z60|","away60","beyond2","persist","scatter"],"mean":[0.89827,1.0749,0.16945,0.1357,0.26662,0.71481],"std":[0.98521,1.10355,0.64443,0.24451,0.25216,0.13793],"weights":[0.75259,0.71443,0.62006,1.29259,0.21704,0.04973],"bias":-4.03694,"threshold":0.5},"dropout":{"features":["missing now","missing 30","missing 120","gaps 300","longest gap 300","since gap"],"mean":[0.00899,0.00895,0.00799,0.12157,0.19231,5.43946],"std":[0.09437,0.06779,0.04404,0.30394,0.61205,0.85212],"weights":[0.52148,0.68396,1.20484,-0.14896,1.89205,0.24259],"bias":-4.01538,"threshold":0.5}};
  // END MODELS

  const FEATURES = {
    drift: ["|z15|", "|z60|", "away60", "beyond2", "persist", "scatter"],
    dropout: ["missing now", "missing 30", "missing 120", "gaps 300", "longest gap 300", "since gap"],
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

  function features(kind, columns, baseN) {
    return kind === "drift" ? driftFeatures(columns, baseN) : dropoutFeatures(columns);
  }

  function transform(kind, f) {
    // log-compress unbounded counts so one extreme value does not swamp the other features
    if (kind === "drift") return [Math.log1p(f[0]), Math.log1p(f[1]), Math.sign(f[2]) * Math.log1p(Math.abs(f[2])), f[3], f[4], Math.log1p(f[5])];
    return [f[0], f[1], f[2], Math.log1p(f[3]), Math.log1p(f[4]), Math.log1p(f[5])];
  }

  function probability(kind, model, f) {
    if (!f) return 0;
    const x = transform(kind, f);
    let s = model.bias;
    for (let k = 0; k < x.length; k++) s += model.weights[k] * (x[k] - model.mean[k]) / model.std[k];
    return 1 / (1 + Math.exp(-s));
  }

  // probability per sensor per reading; null when that model is not bundled
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
