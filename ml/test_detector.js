/* Regression test for detector.js on the labelled recordings in ml/data.
   Run: node ml/test_detector.js   (exits with code 1 if any case fails)

   1. Every CSV: the labelled faulty sensor is flagged with the fault type its file name names,
      every other sensor is Healthy, and the verdict is a plant event exactly when Real_Leak is set.
   2. Stress cases built on realleak_maxdanger.csv (all sensors collapse together): a sensor fault
      injected into sensors 1, 3 and 5 in turn must still be found, with the leak still reported
      as a plant event and the other sensors Healthy. */
"use strict";
const fs = require("fs");
const path = require("path");
const Detector = require("../detector.js");

const DATA = path.join(__dirname, "data");
const NAMES = ["RAD-01", "RAD-02", "RAD-03", "RAD-04", "RAD-05"];
const KIND = { drift: ["Drift high", "Drift low"], dropout: ["No signal"], noise: ["Noisy"], stuck: ["Stuck"], spike: ["Spike"] };

function loadCsv(file) {
  const [header, ...lines] = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const cols = header.split(",");
  const rows = lines.map((l) => l.split(","));
  const col = (name) => rows.map((r) => (r[cols.indexOf(name)] === "" ? NaN : Number(r[cols.indexOf(name)])));
  const sensorCols = cols.filter((c) => /^Sensor[_ ]\d+/.test(c));    // "Sensor_1_MPa" or "Sensor 1 (uSv/h)"
  return {
    columns: sensorCols.map(col),
    faulty: cols.includes("Faulty_Sensor") ? Math.max(0, ...col("Faulty_Sensor").filter(Number.isFinite)) : 0,
    leak: cols.includes("Real_Leak") && col("Real_Leak").some((v) => v === 1),
  };
}

// deterministic Gaussian noise so the stress cases are reproducible
let seed = 12345;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
const gauss = () => Math.sqrt(-2 * Math.log(rand() || 1e-9)) * Math.cos(2 * Math.PI * rand());

let passed = 0, failed = 0;
function check(label, columns, faulty, expected, leak) {
  const res = Detector.analyse(NAMES, columns);
  const problems = [];
  res.sensors.forEach((s, i) => {
    const want = i + 1 === faulty ? expected : null;
    if (want) {
      if (s.status === "Healthy") problems.push(`${s.name} should be ${want.join("/")} but is Healthy`);
      else if (!want.includes(s.fault)) problems.push(`${s.name} should be ${want.join("/")} but is ${s.status}: ${s.fault}`);
    } else if (s.status !== "Healthy") problems.push(`${s.name} should be Healthy but is ${s.status}: ${s.fault}`);
  });
  if ((res.verdict.level === "event") !== leak) problems.push(`verdict ${res.verdict.level}, expected ${leak ? "event" : "not event"}`);
  if (problems.length) { failed++; console.log(`FAIL ${label}\n     ${problems.join("\n     ")}`); }
  else { passed++; console.log(`ok   ${label}`); }
}

// 1. every labelled recording
for (const file of fs.readdirSync(DATA).filter((f) => f.endsWith(".csv")).sort()) {
  const d = loadCsv(path.join(DATA, file));
  const kind = Object.keys(KIND).find((k) => file.replace(/^realleak\d*_?/, "").startsWith(k));
  const expected = d.faulty && kind ? KIND[kind] : null;
  check(file, d.columns, expected ? d.faulty : 0, expected, d.leak);
}

// 2. sensor faults injected into the max-danger leak
const base = loadCsv(path.join(DATA, "realleak_maxdanger.csv"));
const n = base.columns[0].length;
const STRESS = [
  ["frozen from 199", ["Stuck"], (c) => { for (let t = 199; t < n; t++) c[t] = c[199]; }],
  ["reads 15.5 from 121", ["Drift high"], (c) => { for (let t = 121; t < n; t++) c[t] = +(15.5 + 0.02 * gauss()).toFixed(4); }],
  ["+0.012 per reading from 250", ["Drift high"], (c) => { for (let t = 250; t < n; t++) c[t] += 0.012 * (t - 250); }],
  ["+0.8 noise from 250", ["Noisy"], (c) => { for (let t = 250; t < n; t++) c[t] = Math.max(0.01, c[t] + 0.8 * gauss()); }],
  ["blank 220-240, 330-345, 450-end", ["No signal"], (c) => {
    for (const [a, b] of [[220, 240], [330, 345], [450, n - 1]]) for (let t = a; t <= b; t++) c[t] = NaN;
  }],
  ["+4 spikes", ["Spike"], (c) => { for (const t of [230, 300, 370, 440, 510, 580]) if (t < n) c[t] += 4; }],
  ["reads 0 from 300", ["No signal"], (c) => { for (let t = 300; t < n; t++) c[t] = 0; }],
];
for (const [label, expected, inject] of STRESS) {
  for (const sensor of [1, 3, 5]) {
    const columns = base.columns.map((c) => c.slice());
    inject(columns[sensor - 1]);
    check(`maxdanger + S${sensor} ${label}`, columns, sensor, expected, true);
  }
}

console.log(`\n${passed}/${passed + failed} passed`);
process.exit(failed ? 1 : 0);
