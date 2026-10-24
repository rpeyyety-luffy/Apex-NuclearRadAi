/* Machine-learned drift detector.
   Each sensor is compared with how it related to its peers in normal operation (the healthy
   baseline at the start of the file). Six features describe how far, how fast and how steadily
   it is moving away; a logistic-regression model trained on labelled drift recordings
   (ml/data, see ml/train_drift_model.js) turns them into a drift probability per reading.
   Only past readings are used, so it works the same on a live feed.
   Runs in the browser (window.DriftML) and in Node (module.exports). */
(function (root) {
  "use strict";

  // BEGIN MODEL (written by ml/train_drift_model.js - do not edit by hand)
  const MODEL = {"features":["|z15|","|z60|","away60","beyond2","persist","scatter"],"mean":[1.01688,1.22067,0.23183,0.16255,0.29916,0.72447],"std":[1.11511,1.24179,0.7134,0.28044,0.27674,0.14031],"weights":[0.86424,0.82609,0.75213,1.35311,0.32329,0.11733],"bias":-3.7804,"threshold":0.5,"trainedOn":["drift1.csv","drift2.csv","drift3.csv","drift4.csv","drift5.csv","drift6.csv","drift7.csv","drift8.csv","normal_radiation.csv"]};
  // END MODEL

  const FEATURES = ["|z15|", "|z60|", "away60", "beyond2", "persist", "scatter"];

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

  // d[i][t]: sensor i's gap to its peers, minus its normal gap, in units of normal scatter.
  // Readings are first scaled by each sensor's normal level, so sensors at different levels
  // (and a real change that scales every sensor together) compare fairly.
  function deviations(columns, baseN) {
    const n = columns[0].length;
    const scaled = columns.map((col) => {
      const level = median(col.slice(0, baseN).filter((v) => v > 0));
      return col.map((v) => (Number.isFinite(v) && v > 0 ? v / level : NaN));
    });
    return scaled.map((col, i) => {
      const gap = col.map((v, t) => v - median(scaled.filter((_, j) => j !== i).map((c) => c[t])));
      const base = gap.slice(0, baseN);
      const offset = median(base);
      const scale = Math.max(robustSd(base), 1e-9);
      return Array.from({ length: n }, (_, t) => (gap[t] - offset) / scale);
    });
  }

  function features(columns, baseN) {
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

  function transform(f) {
    // log-compress the unbounded z-scores so a huge drift does not swamp the other features
    return [Math.log1p(f[0]), Math.log1p(f[1]), Math.sign(f[2]) * Math.log1p(Math.abs(f[2])), f[3], f[4], Math.log1p(f[5])];
  }

  function probability(model, f) {
    if (!f) return 0;
    const x = transform(f);
    let s = model.bias;
    for (let k = 0; k < x.length; k++) s += model.weights[k] * (x[k] - model.mean[k]) / model.std[k];
    return 1 / (1 + Math.exp(-s));
  }

  // drift probability per sensor per reading; null when no model is bundled
  function predict(columns, baseN, model = MODEL) {
    if (!model) return null;
    const d = deviations(columns, baseN);
    return features(columns, baseN).map((fs, i) => fs.map((f, t) => ({
      p: probability(model, f),
      direction: Math.sign(d[i][t]) || 1,
    })));
  }

  const api = { MODEL, FEATURES, features, deviations, transform, probability, predict };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.DriftML = api;
})(typeof window !== "undefined" ? window : globalThis);
