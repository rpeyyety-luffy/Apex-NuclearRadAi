/* Tests for the exported report: trust levels and recommendation rules (report_rules.js) and the
   data aggregation (report_data.js), with a rendering sanity check (report_render.js).
   Run: node ml/test_report.js   (exits with code 1 if any check fails) */
"use strict";
const fs = require("fs");
const path = require("path");
const Detector = require("../detector.js");
const Rules = require("../report_rules.js");
const ReportData = require("../report_data.js");
const ReportRender = require("../report_render.js");
const SENSORS = require("../sensors.js");

const DATA = path.join(__dirname, "data");
const NAMES = ["RAD-01", "RAD-02", "RAD-03", "RAD-04", "RAD-05"];
const AT = new Date("2026-10-25T10:00:00Z");
let passed = 0, failed = 0;
function check(label, ok, detail = "") {
  if (ok) { passed++; console.log(`ok   ${label}`); }
  else { failed++; console.log(`FAIL ${label}${detail ? `\n     ${detail}` : ""}`); }
}

function loadCsv(file) {
  const [header, ...lines] = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const cols = header.split(",");
  const rows = lines.map((l) => l.split(","));
  const col = (name) => rows.map((r) => (r[cols.indexOf(name)] === "" ? NaN : Number(r[cols.indexOf(name)])));
  return {
    columns: cols.filter((c) => /^Sensor[_ ]\d+/.test(c)).map(col),
    labels: rows.map((r) => r[0]),
    faulty: cols.includes("Faulty_Sensor") ? Math.max(0, ...col("Faulty_Sensor").filter(Number.isFinite)) : 0,
    leak: cols.includes("Real_Leak") && col("Real_Leak").some((v) => v === 1),
  };
}
function report(columns, labels, names = NAMES, extra = {}) {
  return ReportData.buildReport({ names, columns, labels, indexLabel: "Time_s", detection: Detector.analyse(names, columns), sourceLabel: "test", ...extra },
    { sensorsMeta: SENSORS, generatedAt: AT });
}
const row = (over) => ({ id: "RAD-09", status: "Healthy", fault: null, summary: "", health: 95, offline: false, silentReadings: 0,
  missingShare: 0, maintenance: { action: "None", urgency: "-", text: "" }, ...over });
const withTrust = (r) => ({ ...r, trust: Rules.trustLevel(r).level });

// ------------------------------------------------------------------ 1. trust levels
check("healthy -> Trusted", Rules.trustLevel(row({})).level === "Trusted");
check("faulty -> Untrusted", Rules.trustLevel(row({ status: "Faulty", fault: "Stuck" })).level === "Untrusted");
check("watch -> Needs review", Rules.trustLevel(row({ status: "Watch", fault: "Spike" })).level === "Needs review");
check("at risk -> Needs review", Rules.trustLevel(row({ status: "At risk", fault: "Drift high" })).level === "Needs review");
check("offline wins over faulty", Rules.trustLevel(row({ status: "Faulty", fault: "No signal", offline: true, silentReadings: 40 })).level === "Offline");

// ------------------------------------------------------------------ 2. recommendation rules
const act = (r, plant) => Rules.recommendActions([withTrust(row(r))], plant || {});
const one = (r) => act(r)[0] || {};
check("replace -> High", one({ status: "Faulty", fault: "Stuck", maintenance: { action: "Replace", urgency: "Immediate", text: "frozen" } }).priority === "High");
check("immediate repair -> High", one({ status: "Faulty", fault: "No signal", maintenance: { action: "Repair", urgency: "Immediate", text: "gaps" } }).priority === "High");
check("planned repair -> Medium", one({ status: "Faulty", fault: "Noisy", maintenance: { action: "Repair", urgency: "Planned", text: "noisy" } }).priority === "Medium");
check("recalibrate -> Medium", one({ status: "Faulty", fault: "Drift high", maintenance: { action: "Recalibrate", urgency: "Now", text: "7% off" } }).rule === "recalibrate");
check("at risk -> Medium recalibration", one({ status: "At risk", fault: "Drift high", summary: "Trending up" }).rule === "at-risk");
check("watch -> Low inspection", one({ status: "Watch", fault: "Spike", summary: "spikes", maintenance: { action: "Inspect", urgency: "Routine", text: "" } }).priority === "Low");
check("healthy with low health -> Low monitor", one({ health: 70 }).rule === "monitor");
check("healthy with good health -> no action", act({}).length === 0);
check("offline -> High silent-sensor action", one({ status: "Faulty", fault: "No signal", offline: true, silentReadings: 300, lastSeenLabel: "299" }).rule === "offline");
const exclusive = act({ status: "Faulty", fault: "Drift high", maintenance: { action: "Replace", urgency: "Immediate", text: "40% off" } });
check("maintenance rules are exclusive per sensor", exclusive.filter((a) => a.sensors[0] === "RAD-09").length === 1 && exclusive[0].rule === "replace");
check("data gaps add a separate Low action", act({ status: "Watch", fault: "No signal", summary: "gaps", missingShare: 0.2 }).map((a) => a.rule).join() === "inspect,data-gaps");
const plant = Rules.recommendActions([withTrust(row({ id: "RAD-01" })), withTrust(row({ id: "RAD-02", health: 60 }))], { verdict: { level: "event", text: "Leak" } });
check("plant-wide event is the first High action and names every sensor", plant[0].rule === "plant-event" && plant[0].sensors.join() === "RAD-01,RAD-02");
const mixed = Rules.recommendActions(["Low", "High", "Medium"].map((p, i) => withTrust(row({
  id: `RAD-0${i + 1}`, status: p === "Low" ? "Watch" : "Faulty", fault: "Drift high", summary: "s",
  maintenance: p === "High" ? { action: "Replace", urgency: "Immediate", text: "x" } : p === "Medium" ? { action: "Recalibrate", urgency: "Now", text: "y" } : { action: "Inspect", urgency: "Routine", text: "" },
}))));
check("actions sorted High -> Medium -> Low", mixed.map((a) => a.priority).join() === "High,Medium,Low");
check("every action names sensors, evidence and a next step", mixed.every((a) => a.sensors.length && a.reason && a.nextStep));

// ------------------------------------------------------------------ 3. aggregation on the labelled recordings
for (const file of fs.readdirSync(DATA).filter((f) => f.endsWith(".csv")).sort()) {
  const d = loadCsv(path.join(DATA, file));
  const r = report(d.columns, d.labels);
  const sum = Object.values(r.summary.counts).reduce((a, b) => a + b, 0);
  const faultyRow = d.faulty ? r.rows[d.faulty - 1] : null;
  const others = r.rows.filter((x) => !faultyRow || x.id !== faultyRow.id);
  const problems = [];
  if (sum !== 5 || r.summary.total !== 5) problems.push(`counts ${JSON.stringify(r.summary.counts)}`);
  if (faultyRow && faultyRow.trust === "Trusted") problems.push(`${faultyRow.id} should not be Trusted`);
  if (others.some((x) => x.trust !== "Trusted")) problems.push(`others: ${others.map((x) => x.trust).join(",")}`);
  if (r.rows.some((x) => x.trust !== "Trusted" && !r.actions.some((a) => a.sensors.includes(x.id)))) problems.push("a flagged sensor has no action");
  if (faultyRow && !r.charts.trends.series.some((s) => s.id === faultyRow.id)) problems.push("faulty sensor missing from the trend chart");
  if (r.charts.trends.x.length > ReportData.MAX_POINTS) problems.push("trend not thinned");
  if (d.leak !== Boolean(r.actions[0] && r.actions[0].rule === "plant-event")) problems.push(`plant-event action ${d.leak ? "missing" : "unexpected"}`);
  check(`aggregation ${file}`, !problems.length, problems.join("; "));
}
const drift = report(...Object.values(loadCsv(path.join(DATA, "drift1.csv"))).slice(0, 2));
check("drift1: RAD-01 Untrusted with a recalibration action", drift.rows[0].trust === "Untrusted" && drift.actions.some((a) => a.rule === "recalibrate" && a.sensors[0] === "RAD-01"));
const stuck = report(...Object.values(loadCsv(path.join(DATA, "stuck1.csv"))).slice(0, 2));
check("stuck1: RAD-04 replace is High", stuck.actions[0].rule === "replace" && stuck.actions[0].sensors[0] === "RAD-04" && stuck.actions[0].priority === "High");
check("header carries time range and sensor metadata", drift.header.timeRange.from === "1" && drift.header.timeRange.samples === 600 && drift.rows[0].name === "Reactor cavity area monitor");

// ------------------------------------------------------------------ 4. edge cases
const max = loadCsv(path.join(DATA, "realleak_maxdanger.csv"));
const dead = max.columns.map((c) => c.slice());
for (let t = 300; t < dead[0].length; t++) dead[0][t] = 0;          // RAD-01 reads 0 (no signal) from 300
const off = report(dead, max.labels);
check("missing readings: silent sensor is Offline with a High action", off.rows[0].trust === "Offline" && off.actions.some((a) => a.rule === "offline" && a.sensors[0] === "RAD-01" && a.priority === "High"));
check("missing readings: last reading falls back to the last value seen", Number.isFinite(off.rows[0].lastReading) && off.rows[0].lastSeenLabel === "300");
const empty = ReportData.buildReport({ names: [], columns: [], labels: [], detection: { sensors: [], verdict: null } }, { generatedAt: AT });
check("no sensors: empty summary, no actions", empty.summary.total === 0 && empty.actions.length === 0 && /No sensors/.test(empty.summary.overall));
check("no sensors: renders without error", /No sensors were found/.test(ReportRender.renderReportHtml(empty)));

let seed = 3;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
const many = Array.from({ length: 200 }, (_, i) => `Sensor <${i}> ${"x".repeat(i % 7 ? 0 : 160)} & "co"`);
const manyCols = many.map(() => Array.from({ length: 120 }, () => 100 + rand()));
const big = report(manyCols, manyCols[0].map((_, i) => String(i)), many);
check("200 sensors: one row each, counts add up", big.rows.length === 200 && Object.values(big.summary.counts).reduce((a, b) => a + b, 0) === 200);
const bigHtml = ReportRender.renderReportHtml(big);
check("long and unsafe names are escaped", !bigHtml.includes("Sensor <0>") && bigHtml.includes("Sensor &lt;0&gt;") && bigHtml.includes("&amp; &quot;co&quot;"));
check("200 sensors: table has 200 body rows and a repeating header", (bigHtml.match(/<tr>\s*<td>/g) || []).length === 200 && /thead\{display:table-header-group\}/.test(bigHtml));

// ------------------------------------------------------------------ 5. rendering sanity
const html = ReportRender.renderReportHtml(drift);
check("report has title, summary, table, three charts, actions and footer", [/Sensor Verification Report/, /Executive summary/, /<thead>/, /Recommended actions/, /counter\(page\)/, /Generated by/].every((re) => re.test(html)) && (html.match(/<svg class="chart"/g) || []).length === 3);
check("charts have titles, axis labels and a legend", /Sensors by trust level/.test(html) && /Number of sensors/.test(html) && /Reading \(% of baseline\)/.test(html) && /Network median/.test(html));
check("deterministic: same input, same report", JSON.stringify(report(...Object.values(loadCsv(path.join(DATA, "drift1.csv"))).slice(0, 2))) === JSON.stringify(drift));

console.log(`\n${passed}/${passed + failed} passed`);
process.exit(failed ? 1 : 0);
