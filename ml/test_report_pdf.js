/* Smoke test for the exported report: renders real reports to PDF with headless Chrome (the same
   print path a user takes with "Save as PDF") and checks the files are non-empty, valid PDFs.
   Run: node ml/test_report_pdf.js [--out sample.pdf]
   Needs Chrome or Chromium; set CHROME_PATH if it is not in a standard location. Without a browser
   the test reports SKIPPED and exits with code 2, so it is never mistaken for a pass. */
"use strict";
const fs = require("fs");
const os = require("os");
const path = require("path");
const { execFileSync } = require("child_process");
const Detector = require("../detector.js");
const ReportData = require("../report_data.js");
const ReportRender = require("../report_render.js");
const SENSORS = require("../sensors.js");

const CANDIDATES = [
  process.env.CHROME_PATH,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Google Chrome .app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/bin/chromium", "/usr/bin/chromium-browser",
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
].filter(Boolean);
const chrome = CANDIDATES.find((p) => fs.existsSync(p));
if (!chrome) { console.log("SKIPPED: no Chrome/Chromium found (set CHROME_PATH)"); process.exit(2); }

const outArg = process.argv.indexOf("--out");
const samplePath = outArg > 0 ? path.resolve(process.argv[outArg + 1]) : null;
const dir = fs.mkdtempSync(path.join(os.tmpdir(), "nuclearrad-report-"));
const NAMES = ["RAD-01", "RAD-02", "RAD-03", "RAD-04", "RAD-05"];
const AT = new Date("2026-10-25T10:00:00Z");

function csv(file) {
  const [header, ...lines] = fs.readFileSync(path.join(__dirname, "data", file), "utf8").trim().split(/\r?\n/);
  const cols = header.split(",");
  const idx = cols.map((c, i) => (/^Sensor[_ ]\d+/.test(c) ? i : -1)).filter((i) => i >= 0);
  return { columns: idx.map((i) => lines.map((l) => { const v = l.split(",")[i]; return v === "" ? NaN : Number(v); })), labels: lines.map((l) => l.split(",")[0]) };
}
function build(names, columns, labels, source) {
  return ReportData.buildReport({ names, columns, labels, indexLabel: "Time_s", detection: Detector.analyse(names, columns), sourceLabel: source },
    { sensorsMeta: SENSORS, generatedAt: AT });
}
function toPdf(name, data) {
  const html = path.join(dir, `${name}.html`), pdf = path.join(dir, `${name}.pdf`);
  fs.writeFileSync(html, ReportRender.renderReportHtml(data, { embedded: true }));
  execFileSync(chrome, ["--headless=new", "--disable-gpu", "--no-pdf-header-footer", `--user-data-dir=${path.join(dir, "profile")}`,
    `--print-to-pdf=${pdf}`, `file://${html}`], { stdio: "ignore", timeout: 120000 });
  return pdf;
}
function inspect(pdf) {
  const buf = fs.readFileSync(pdf);
  const text = buf.toString("latin1");
  return { bytes: buf.length, header: text.startsWith("%PDF-"), eof: /%%EOF\s*$/.test(text), pages: (text.match(/\/Type\s*\/Page(?!s)/g) || []).length };
}

let seed = 5;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
const many = Array.from({ length: 150 }, (_, i) => `RAD-${String(i + 1).padStart(3, "0")}`);
const manyCols = many.map(() => Array.from({ length: 120 }, () => 100 + rand()));

const cases = [
  ["drift1", build(NAMES, ...Object.values(csv("drift1.csv")), "drift1.csv"), 1],
  ["realleak_maxdanger", build(NAMES, ...Object.values(csv("realleak_maxdanger.csv")), "realleak_maxdanger.csv"), 1],
  ["empty", ReportData.buildReport({ names: [], columns: [], labels: [], detection: { sensors: [], verdict: null } }, { generatedAt: AT }), 1],
  ["150_sensors", build(many, manyCols, manyCols[0].map((_, i) => String(i)), "synthetic 150-sensor stress test"), 3],
];
let failed = 0;
for (const [name, data, minPages] of cases) {
  const r = inspect(toPdf(name, data));
  const ok = r.header && r.eof && r.bytes > 5000 && r.pages >= minPages;
  if (!ok) failed++;
  console.log(`${ok ? "ok  " : "FAIL"} ${name}.pdf: ${r.bytes} bytes, ${r.pages} page(s), %PDF header ${r.header}, %%EOF ${r.eof}`);
}
if (samplePath) {
  fs.copyFileSync(path.join(dir, "drift1.pdf"), samplePath);
  console.log(`sample report written to ${samplePath}`);
}
console.log(failed ? `${failed} PDF check(s) failed` : "all PDF checks passed");
process.exit(failed ? 1 : 0);
