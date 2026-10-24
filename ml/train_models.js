/* Train the fault models in faultml.js.  Run: node ml/train_models.js

   Data (ml/data), one CSV per recording:
     drift*.csv    one sensor slowly drifts away from the others
     dropout*.csv  one sensor loses its signal in several intervals (blank cells)
     noise*.csv    one sensor's reading-to-reading noise becomes ~12x larger
     realleak*_<fault>.csv  a real coolant leak: pressure falls on all sensors, and one sensor
                   also has its own fault (dropout, noise, drift or spike, named in the file name)
     normal_*.csv  healthy recording, used only as normal (negative) examples
   Labelled files have Anomaly_Active = 1 while the fault or leak is active, Real_Leak = 1 while
   the pressure loss is real and Faulty_Sensor = the index of the bad sensor.

   Each sensor model learns its own fault type; every other reading of every file (healthy sensors,
   healthy stretches, the other fault types) is a negative example. The leak model rates the whole
   plant at each reading and learns from the Real_Leak column. Each file is scored by a
   model trained on the other files (leave-one-file-out), then the final models are trained on
   everything and written into faultml.js. */
const fs = require("fs");
const path = require("path");
const FaultML = require("../faultml.js");
const Detector = require("../detector.js");

const DATA = path.join(__dirname, "data");
const KINDS = ["drift", "dropout", "noise", "leak"];
const GRACE = { drift: 40, dropout: 0, noise: 10, leak: 10 };   // first readings after onset that are too small to see yet
const MAX_FALSE_RATE = 0.001;
const EPOCHS = 400, RATE = 0.5, L2 = 1e-3;

function loadCsv(file) {
  const [head, ...lines] = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const cols = head.split(",");
  const rows = lines.map((l) => l.split(","));
  const sensorIdx = cols.map((c, i) => (/sensor|rad/i.test(c) && !/anomal|leak|faulty/i.test(c) ? i : -1)).filter((i) => i >= 0);
  const get = (name) => rows.map((r) => +r[cols.indexOf(name)]);
  const n = rows.length;
  const name = path.basename(file);
  return {
    name,
    kind: ["drift", "dropout", "noise", "spike"].find((k) => name.startsWith(k) || name.includes(`_${k}`)) || "normal",
    names: sensorIdx.map((i) => cols[i]),
    columns: sensorIdx.map((i) => rows.map((r) => (r[i] === "" || r[i] === undefined ? NaN : +r[i]))),
    active: cols.includes("Anomaly_Active") ? get("Anomaly_Active") : new Array(n).fill(0),
    leak: cols.includes("Real_Leak") ? get("Real_Leak") : new Array(n).fill(0),
    faulty: cols.includes("Faulty_Sensor") ? get("Faulty_Sensor")[0] : 0,
    n,
    baseN: Math.max(20, Math.round(n * 0.15)),
  };
}

// first reading of the fault this model looks for (-1: none in this file)
function onsetOf(kind, ds) {
  if (kind === "leak") return ds.leak.indexOf(1);
  if (ds.kind !== kind) return -1;
  // dropout: the first gap (in the leak files the labelled start is the leak, before the first gap)
  if (kind === "dropout") return ds.columns[ds.faulty - 1].findIndex((v) => !Number.isFinite(v));
  return ds.active.indexOf(1);
}

// what the model should say at each reading of each sensor (true / false / null = leave out)
function label(kind, ds, i, t) {
  if (kind === "leak") return ds.leak[t] ? (t < onsetOf(kind, ds) + GRACE.leak ? null : true) : false;
  if (ds.kind !== kind || ds.faulty !== i + 1) return false;
  const onset = onsetOf(kind, ds);
  if (kind === "drift" || kind === "noise") {
    if (!ds.active[t]) return false;
    return t < onset + GRACE[kind] ? null : true;
  }
  // dropout: from the first gap on, the sensor is unreliable, also between the gaps
  return t >= onset;
}

const featureCache = new Map();
function cachedFeatures(kind, ds) {
  const key = `${kind}|${ds.name}`;
  if (!featureCache.has(key)) featureCache.set(key, FaultML.features(kind, ds.columns, ds.baseN));
  return featureCache.get(key);
}

function examples(kind, ds) {
  const out = [];
  cachedFeatures(kind, ds).forEach((fs_, i) => fs_.forEach((f, t) => {
    if (!f || t < ds.baseN) return;
    const y = label(kind, ds, i, t);
    if (y !== null) out.push({ x: FaultML.transform(kind, f), y: y ? 1 : 0 });
  }));
  return out;
}

function prob(model, x) {
  let s = model.bias;
  x.forEach((v, j) => (s += model.weights[j] * (v - model.mean[j]) / model.std[j]));
  return 1 / (1 + Math.exp(-s));
}

function train(kind, sets) {
  const data = sets.flatMap((ds) => examples(kind, ds));
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
  const model = { features: FaultML.FEATURES[kind], mean, std, weights: w, bias: b };
  // decision threshold: lowest probability that keeps healthy readings flagged under MAX_FALSE_RATE
  const healthy = data.filter((d) => !d.y).map((d) => prob(model, d.x)).sort((a, c) => c - a);
  model.threshold = Math.max(0.5, healthy[Math.floor(healthy.length * MAX_FALSE_RATE)] + 1e-6);
  return model;
}

// detection = first reading after onset from which the faulty sensor stays flagged for 10 readings
function firstSustained(flags, from) {
  for (let t = from; t < flags.length - 10; t++) if (flags.slice(t, t + 10).every(Boolean)) return t;
  return -1;
}

function score(kind, ds, flagged) {
  let falseFlags = 0, negatives = 0, hits = 0, positives = 0;
  flagged.forEach((f, i) => f.forEach((v, t) => {
    if (t < ds.baseN) return;
    const y = label(kind, ds, i, t);
    if (y === null) return;
    if (y) { positives++; if (v) hits++; } else { negatives++; if (v) falseFlags++; }
  }));
  const onset = onsetOf(kind, ds);
  const hit = onset >= 0 ? firstSustained(flagged[kind === "leak" ? 0 : ds.faulty - 1], onset) : -1;
  return { delay: onset >= 0 ? (hit >= 0 ? hit - onset : "missed") : null, coverage: positives ? hits / positives : null, falseFlags, negatives };
}

function mlFlags(kind, ds, model) {
  return cachedFeatures(kind, ds).map((s) => s.map((f) => FaultML.probability(kind, model, f) >= model.threshold));
}

// the rule-based detector without the models, for comparison
function ruleFlags(kind, ds) {
  const res = Detector.analyse(ds.names, ds.columns, { ml: false });
  // the rules only know plant-wide rises (radiation); for a leak, count any plant-wide event
  if (kind === "leak") { const ev = new Set(res.events); return [res.sensors[0].type.map((_, t) => ev.has(t))]; }
  const want = { drift: ["Drift high", "Drift low"], dropout: ["No signal"], noise: ["Noisy"] }[kind];
  return res.sensors.map((s) => s.type.map((t) => want.includes(t)));
}

// A single missed sample is a data-logger hiccup, not a failing sensor, but none of the recordings
// contain one. Copies of the files without dropouts, with 1-2 isolated blanks on healthy sensors,
// teach the dropout model that difference. They are only used for training, never scored.
function withIsolatedBlanks(ds) {
  let seed = ds.name.length * 7919;
  const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const columns = ds.columns.map((col, i) => {
    if (ds.faulty === i + 1) return col;
    const copy = col.slice();
    const blanks = 1 + Math.floor(rand() * 2);
    for (let b = 0; b < blanks; b++) copy[ds.baseN + Math.floor(rand() * (ds.n - ds.baseN))] = NaN;
    return copy;
  });
  // labels stay those of the original file: a drifting sensor in a copy is still drifting
  return { ...ds, name: `${ds.name} (isolated blanks)`, columns };
}

// The noise recordings only contain ~12x noise, so the model would otherwise put the line at ~2.5x.
// The rule-based detector calls a sensor noisy from 3x its normal scatter; copies of the files where
// one healthy sensor's noise grows 2.5x halfway through, labelled normal, teach the model the same line.
function withMildNoise(ds) {
  const i = ds.faulty > 0 ? ds.faulty % ds.columns.length : 0;   // a sensor other than the faulty one
  const col = ds.columns[i], half = Math.floor(ds.n / 2);
  const smooth = col.map((_, t) => { const w = col.slice(Math.max(0, t - 7), t + 8).filter(Number.isFinite); return w.reduce((a, b) => a + b, 0) / w.length; });
  const columns = ds.columns.slice();
  columns[i] = col.map((v, t) => (t >= half && Number.isFinite(v) ? smooth[t] + 2.5 * (v - smooth[t]) : v));
  return { ...ds, name: `${ds.name} (mild noise)`, columns };
}

const files = fs.readdirSync(DATA).filter((f) => f.endsWith(".csv")).sort();
const sets = files.map((f) => loadCsv(path.join(DATA, f)));
const augmented = [...sets.filter((ds) => ds.kind !== "dropout").map(withIsolatedBlanks), ...sets.map(withMildNoise)];
const finals = {};
let report = `# Fault model evaluation\n\nEach file was scored by a model trained on the other files (leave-one-file-out).\n` +
  `- **Detected after**: seconds from the labelled fault start until the faulty sensor stays flagged for 10 readings in a row.\n` +
  `- **Covered**: share of the faulty sensor's readings from the fault start on that are flagged.\n` +
  `- **False flags**: flags on readings that are not this fault (other sensors, other files, healthy stretches).\n` +
  `- **Rules** is the detector without the models.\n`;

for (const kind of KINDS) {
  const rows = sets.map((held) => {
    const model = train(kind, [...sets.filter((s) => s !== held), ...augmented.filter((a) => !a.name.startsWith(held.name))]);
    return { ds: held, ml: score(kind, held, mlFlags(kind, held, model)), rule: score(kind, held, ruleFlags(kind, held)) };
  });
  const cell = (r) => r.delay === null ? "-" : r.delay === "missed" ? "missed" : `${r.delay} s`;
  const pct = (r) => r.coverage === null ? "-" : `${Math.round(100 * r.coverage)}%`;
  const lines = [
    `| File | Faulty sensor | ML: detected after | ML: covered | ML: false flags | Rules: detected after | Rules: covered | Rules: false flags |`,
    "|---|---|---|---|---|---|---|---|",
    ...rows.map(({ ds, ml, rule }) => `| ${ds.name} | ${kind === "leak" ? (onsetOf(kind, ds) >= 0 ? "all" : "-") : ds.kind === kind ? ds.faulty : "-"} | ${cell(ml)} | ${pct(ml)} | ${ml.falseFlags} / ${ml.negatives} | ${cell(rule)} | ${pct(rule)} | ${rule.falseFlags} / ${rule.negatives} |`),
  ];
  console.log(`\n${kind} model\n${lines.join("\n")}`);
  finals[kind] = train(kind, [...sets, ...augmented]);
  report += `\n## ${kind[0].toUpperCase() + kind.slice(1)} model\n\n${lines.join("\n")}\n\n` +
    `Weights (standardised features ${FaultML.FEATURES[kind].join(", ")}): ${finals[kind].weights.map((v) => v.toFixed(3)).join(", ")}; ` +
    `bias ${finals[kind].bias.toFixed(3)}; threshold ${finals[kind].threshold.toFixed(3)}\n`;
}

const round = (m) => ({
  ...m,
  mean: m.mean.map((v) => +v.toFixed(5)),
  std: m.std.map((v) => +v.toFixed(5)),
  weights: m.weights.map((v) => +v.toFixed(5)),
  bias: +m.bias.toFixed(5),
  threshold: +m.threshold.toFixed(4),
});
const block = `  // BEGIN MODELS (written by ml/train_models.js - do not edit by hand)\n  const MODELS = ${JSON.stringify(
  Object.fromEntries(KINDS.map((k) => [k, round(finals[k])])))};\n  // END MODELS`;
const target = path.join(__dirname, "..", "faultml.js");
fs.writeFileSync(target, fs.readFileSync(target, "utf8").replace(/  \/\/ BEGIN MODELS[\s\S]*?\/\/ END MODELS/, block));
fs.writeFileSync(path.join(__dirname, "model_report.md"), report + `\nTrained on: ${files.join(", ")}\n`);
console.log("\nFinal models written to faultml.js.");
