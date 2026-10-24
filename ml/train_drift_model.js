/* Train the drift detector in driftml.js.  Run: node ml/train_drift_model.js

   Data (ml/data):
     drift*.csv            labelled drift recordings: Anomaly_Active = 1 while the fault is
                           active, Faulty_Sensor = index of the drifting sensor
     normal_radiation.csv  healthy recording, used only as normal (negative) examples
   Each held-out drift file is scored by a model trained on the others (leave-one-file-out),
   then the final model is trained on everything and written into driftml.js. */
const fs = require("fs");
const path = require("path");
const DriftML = require("../driftml.js");
const Detector = require("../detector.js");

const DATA = path.join(__dirname, "data");
const GRACE = 40;          // first readings after drift onset: too small to see yet, left out of training
const MAX_FALSE_RATE = 0.001;
const EPOCHS = 400, RATE = 0.5, L2 = 1e-3;

function loadCsv(file) {
  const [head, ...lines] = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const cols = head.split(",");
  const rows = lines.map((l) => l.split(","));
  const sensorIdx = cols.map((c, i) => (/sensor|rad/i.test(c) && !/anomal|leak|faulty/i.test(c) ? i : -1)).filter((i) => i >= 0);
  const get = (name) => rows.map((r) => +r[cols.indexOf(name)]);
  const n = rows.length;
  return {
    name: path.basename(file),
    names: sensorIdx.map((i) => cols[i]),
    columns: sensorIdx.map((i) => rows.map((r) => (r[i] === "" ? NaN : +r[i]))),
    active: cols.includes("Anomaly_Active") ? get("Anomaly_Active") : new Array(n).fill(0),
    faulty: cols.includes("Faulty_Sensor") ? get("Faulty_Sensor")[0] : 0,
    n,
    baseN: Math.max(20, Math.round(n * 0.15)),
  };
}

function examples(ds) {
  const feats = DriftML.features(ds.columns, ds.baseN);
  const onset = ds.active.indexOf(1);
  const out = [];
  feats.forEach((fs_, i) => fs_.forEach((f, t) => {
    if (!f || t < ds.baseN) return;
    const drifting = ds.active[t] === 1 && ds.faulty === i + 1;
    if (drifting && t < onset + GRACE) return;
    out.push({ x: DriftML.transform(f), y: drifting ? 1 : 0 });
  }));
  return out;
}

function train(sets) {
  const data = sets.flatMap(examples);
  const k = data[0].x.length;
  const mean = Array.from({ length: k }, (_, j) => data.reduce((s, d) => s + d.x[j], 0) / data.length);
  const std = Array.from({ length: k }, (_, j) => Math.sqrt(data.reduce((s, d) => s + (d.x[j] - mean[j]) ** 2, 0) / data.length) || 1);
  const X = data.map((d) => d.x.map((v, j) => (v - mean[j]) / std[j]));
  const pos = data.filter((d) => d.y).length, neg = data.length - pos;
  const cw = (y) => (y ? data.length / (2 * pos) : data.length / (2 * neg));   // balance the classes
  let w = new Array(k).fill(0), b = 0;
  for (let e = 0; e < EPOCHS; e++) {
    const gw = new Array(k).fill(0);
    let gb = 0;
    X.forEach((x, n) => {
      const z = b + x.reduce((s, v, j) => s + v * w[j], 0);
      const err = (1 / (1 + Math.exp(-z)) - data[n].y) * cw(data[n].y);
      x.forEach((v, j) => (gw[j] += err * v));
      gb += err;
    });
    w = w.map((wj, j) => wj - RATE * (gw[j] / data.length + L2 * wj));
    b -= RATE * gb / data.length;
  }
  const model = { features: DriftML.FEATURES, mean, std, weights: w, bias: b };
  // decision threshold: lowest probability that keeps healthy readings flagged under MAX_FALSE_RATE
  const healthy = data.filter((d) => !d.y).map((d) => prob(model, d.x)).sort((a, c) => c - a);
  model.threshold = Math.max(0.5, healthy[Math.floor(healthy.length * MAX_FALSE_RATE)] + 1e-6);
  return model;
}

function prob(model, x) {
  let s = model.bias;
  x.forEach((v, j) => (s += model.weights[j] * (v - model.mean[j]) / model.std[j]));
  return 1 / (1 + Math.exp(-s));
}

// detection = first reading after onset from which the faulty sensor stays flagged for 10 readings
function firstSustained(flags, from) {
  for (let t = from; t < flags.length - 10; t++) if (flags.slice(t, t + 10).every(Boolean)) return t;
  return -1;
}

function score(ds, flagged) {
  const onset = ds.active.indexOf(1);
  const faulty = ds.faulty - 1;
  let falseFlags = 0, healthyReadings = 0;
  flagged.forEach((f, i) => f.forEach((v, t) => {
    if (t < ds.baseN) return;
    const isDrift = i === faulty && ds.active[t] === 1;
    if (!isDrift) { healthyReadings++; if (v) falseFlags++; }
  }));
  const hit = faulty >= 0 && onset >= 0 ? firstSustained(flagged[faulty], onset) : -1;
  return { delay: hit >= 0 ? hit - onset : null, falseFlags, healthyReadings };
}

function mlFlags(ds, model) {
  return DriftML.predict(ds.columns, ds.baseN, model).map((s) => s.map((r) => r.p >= model.threshold));
}

function ruleFlags(ds) {
  return Detector.analyse(ds.names, ds.columns, { drift: "rules" }).sensors.map((s) => s.type.map((t) => t === "Drift high" || t === "Drift low"));
}

const files = fs.readdirSync(DATA).filter((f) => f.endsWith(".csv")).sort();
const sets = files.map((f) => loadCsv(path.join(DATA, f)));
const drifts = sets.filter((s) => s.faulty > 0);
const normals = sets.filter((s) => s.faulty === 0);

console.log("Leave-one-file-out evaluation (drift onset -> sustained detection, false drift flags on healthy readings)\n");
const rows = [];
for (const held of sets) {
  const model = train(sets.filter((s) => s !== held));
  const ml = score(held, mlFlags(held, model));
  const rule = score(held, ruleFlags(held));
  rows.push({ file: held.name, faulty: held.faulty || "-", onset: held.active.indexOf(1) + 1 || "-", ml, rule });
}
const fmt = (r) => (r.delay === null ? "  n/a" : `${String(r.delay).padStart(4)} s`);
const lines = ["| File | Faulty sensor | Drift starts | ML: detected after | ML: false flags | Rules: detected after | Rules: false flags |",
  "|---|---|---|---|---|---|---|",
  ...rows.map((r) => `| ${r.file} | ${r.faulty} | ${r.onset} s | ${fmt(r.ml).trim()} | ${r.ml.falseFlags} / ${r.ml.healthyReadings} | ${fmt(r.rule).trim()} | ${r.rule.falseFlags} / ${r.rule.healthyReadings} |`)];
console.log(lines.join("\n"));

const final = train(sets);
const block = `  // BEGIN MODEL (written by ml/train_drift_model.js - do not edit by hand)\n  const MODEL = ${JSON.stringify({
  ...final,
  mean: final.mean.map((v) => +v.toFixed(5)),
  std: final.std.map((v) => +v.toFixed(5)),
  weights: final.weights.map((v) => +v.toFixed(5)),
  bias: +final.bias.toFixed(5),
  threshold: +final.threshold.toFixed(4),
  trainedOn: files,
})};\n  // END MODEL`;
const target = path.join(__dirname, "..", "driftml.js");
const src = fs.readFileSync(target, "utf8").replace(/  \/\/ BEGIN MODEL[\s\S]*?\/\/ END MODEL/, block);
fs.writeFileSync(target, src);
fs.writeFileSync(path.join(__dirname, "model_report.md"),
  `# Drift model evaluation\n\nEach file was scored by a model trained on the other files (leave-one-file-out).\n` +
  `"Detected after" = seconds from the labelled drift start until the faulty sensor stays flagged for 10 readings in a row.\n` +
  `"False flags" = drift flags on readings that were not drifting.\n\n${lines.join("\n")}\n\n` +
  `Final model trained on: ${files.join(", ")}\n\nWeights (standardised features ${DriftML.FEATURES.join(", ")}): ${final.weights.map((v) => v.toFixed(3)).join(", ")}; bias ${final.bias.toFixed(3)}; threshold ${final.threshold.toFixed(3)}\n`);
console.log(`\nFinal model written to driftml.js (threshold ${final.threshold.toFixed(3)}).`);
