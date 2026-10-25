/* Rendering for the exported report: report data (report_data.js) -> one self-contained, print-
   optimised HTML document with static SVG charts. The browser's print dialog turns it into a PDF
   ("Save as PDF"); headless Chrome does the same in the tests. No data logic lives here.
   Runs in the browser (window.ReportRender) and in Node (module.exports). */
(function (root) {
  "use strict";

  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const finite = (v) => typeof v === "number" && Number.isFinite(v);
  function num(v, digits) {
    if (!finite(v)) return "—";
    const d = digits ?? (Math.abs(v) >= 100 ? 1 : Math.abs(v) >= 1 ? 2 : 3);
    return v.toFixed(d);
  }

  // trust levels: colour plus a symbol and a border style, so they survive black-and-white printing
  const LEVEL_STYLE = {
    Trusted: { color: "#13795b", symbol: "✓", border: "solid", fill: "url(#p-trusted)" },
    "Needs review": { color: "#a35f00", symbol: "!", border: "dashed", fill: "url(#p-review)" },
    Untrusted: { color: "#b3261e", symbol: "✕", border: "double", fill: "url(#p-untrusted)" },
    Offline: { color: "#4b5563", symbol: "○", border: "dotted", fill: "url(#p-offline)" },
  };
  const SERIES_STYLE = [
    { color: "#b3261e", dash: "" }, { color: "#a35f00", dash: "7 4" },
    { color: "#1d4ed8", dash: "2 3" }, { color: "#6d28d9", dash: "9 3 2 3" },
  ];
  const PRIORITY_STYLE = { High: "#b3261e", Medium: "#a35f00", Low: "#1d4ed8" };

  const PATTERNS = `<defs>
    <pattern id="p-trusted" width="6" height="6" patternUnits="userSpaceOnUse"><rect width="6" height="6" fill="#13795b"/></pattern>
    <pattern id="p-review" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="#f6d9a8"/><rect width="2.5" height="6" fill="#a35f00"/></pattern>
    <pattern id="p-untrusted" width="6" height="6" patternUnits="userSpaceOnUse"><rect width="6" height="6" fill="#f3c4c0"/><path d="M0 0L6 6M6 0L0 6" stroke="#b3261e" stroke-width="1.4"/></pattern>
    <pattern id="p-offline" width="5" height="5" patternUnits="userSpaceOnUse"><rect width="5" height="5" fill="#e5e7eb"/><circle cx="2.5" cy="2.5" r="1.1" fill="#4b5563"/></pattern>
  </defs>`;

  function badge(level, score) {
    const st = LEVEL_STYLE[level] || LEVEL_STYLE.Offline;
    return `<span class="badge" style="color:${st.color};border-color:${st.color};border-style:${st.border}"><b aria-hidden="true">${st.symbol}</b>${esc(level)}${finite(score) ? ` · ${score}` : ""}</span>`;
  }

  // ------------------------------------------------------------------ charts (static SVG)
  function barChart({ title, items, xLabel, yLabel, fillFor, width = 330, emptyText = "No data" }) {
    const height = 60 + items.length * 34, pad = { l: 104, r: 34, t: 34, b: 38 };
    const max = Math.max(1, ...items.map((d) => d.value));
    const plotW = width - pad.l - pad.r;
    const ticks = Array.from(new Set([0, Math.ceil(max / 2), max]));
    const x = (v) => pad.l + (v / max) * plotW;
    const total = items.reduce((t, d) => t + d.value, 0);
    return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(title)}">${PATTERNS}
      <text x="0" y="16" class="ct">${esc(title)}</text>
      ${ticks.map((t) => `<line x1="${x(t)}" x2="${x(t)}" y1="${pad.t - 4}" y2="${height - pad.b}" class="grid"/><text x="${x(t)}" y="${height - pad.b + 14}" text-anchor="middle" class="tick">${t}</text>`).join("")}
      ${items.map((d, i) => {
        const y = pad.t + i * 34;
        return `<text x="${pad.l - 8}" y="${y + 15}" text-anchor="end" class="lbl">${esc(d.label)}</text>`
          + `<rect x="${pad.l}" y="${y + 3}" width="${Math.max(d.value ? 2 : 0, x(d.value) - pad.l)}" height="18" fill="${fillFor(d)}" stroke="#374151" stroke-width=".6"/>`
          + `<text x="${x(d.value) + 6}" y="${y + 16}" class="val">${d.value}</text>`;
      }).join("")}
      <text x="${pad.l + plotW / 2}" y="${height - 6}" text-anchor="middle" class="axis">${esc(xLabel)}</text>
      <text transform="translate(10 ${pad.t + (items.length * 34) / 2}) rotate(-90)" text-anchor="middle" class="axis">${esc(yLabel)}</text>
      ${total ? "" : `<text x="${pad.l + plotW / 2}" y="${pad.t + 20}" text-anchor="middle" class="empty">${esc(emptyText)}</text>`}
    </svg>`;
  }

  function lineChart(trends, width = 690, height = 270) {
    const pad = { l: 56, r: 16, t: 34, b: 66 };
    const all = [...trends.series, trends.reference].flatMap((s) => s.values).filter(finite);
    const title = `Readings over time, as % of each sensor's baseline — ${trends.selection}`;
    if (!all.length || !trends.x.length) {
      return `<svg class="chart" viewBox="0 0 ${width} 90" role="img" aria-label="${esc(title)}"><text x="0" y="16" class="ct">${esc(title)}</text><text x="${width / 2}" y="60" text-anchor="middle" class="empty">No readings to chart</text></svg>`;
    }
    let lo = Math.min(...all), hi = Math.max(...all);
    if (hi - lo < 4) { lo -= 2; hi += 2; }
    const span = hi - lo; lo -= span * 0.06; hi += span * 0.06;
    const plotW = width - pad.l - pad.r, plotH = height - pad.t - pad.b;
    const n = trends.x.length;
    const X = (i) => pad.l + (n > 1 ? (i / (n - 1)) * plotW : plotW / 2);
    const Y = (v) => pad.t + ((hi - v) / (hi - lo)) * plotH;
    const path = (values) => {
      let d = "", pen = false;
      values.forEach((v, i) => { if (finite(v)) { d += `${pen ? "L" : "M"}${X(i).toFixed(1)} ${Y(v).toFixed(1)}`; pen = true; } else pen = false; });
      return d;
    };
    const yTicks = Array.from({ length: 5 }, (_, k) => lo + ((hi - lo) * k) / 4);
    const xTicks = Array.from(new Set([0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * (n - 1)))));
    const shown = trends.series.map((s, k) => ({ ...s, style: SERIES_STYLE[k % SERIES_STYLE.length] }));
    const hundred = 100 >= lo && 100 <= hi ? `<line x1="${pad.l}" x2="${pad.l + plotW}" y1="${Y(100)}" y2="${Y(100)}" stroke="#6b7280" stroke-width=".8" stroke-dasharray="3 3"/>` : "";
    const legendItems = [...shown.map((s) => ({ label: `${s.id} (${s.trust})`, color: s.style.color, dash: s.style.dash, w: 2 })),
      { label: "Network median", color: "#6b7280", dash: "1 2", w: 1.4 }];
    let lx = pad.l;
    const legend = legendItems.map((it) => {
      const out = `<line x1="${lx}" x2="${lx + 22}" y1="${height - 12}" y2="${height - 12}" stroke="${it.color}" stroke-width="${it.w}" stroke-dasharray="${it.dash}"/><text x="${lx + 27}" y="${height - 8}" class="lbl">${esc(it.label)}</text>`;
      lx += 36 + it.label.length * 5.6;
      return out;
    }).join("");
    return `<svg class="chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(title)}">
      <text x="0" y="16" class="ct">${esc(title)}</text>
      ${yTicks.map((v) => `<line x1="${pad.l}" x2="${pad.l + plotW}" y1="${Y(v)}" y2="${Y(v)}" class="grid"/><text x="${pad.l - 6}" y="${Y(v) + 3.5}" text-anchor="end" class="tick">${v.toFixed(span < 20 ? 1 : 0)}</text>`).join("")}
      ${hundred}
      <path d="${path(trends.reference.values)}" fill="none" stroke="#6b7280" stroke-width="1.2" stroke-dasharray="1 2"/>
      ${shown.map((s) => `<path d="${path(s.values)}" fill="none" stroke="${s.style.color}" stroke-width="1.8" stroke-dasharray="${s.style.dash}" stroke-linejoin="round"/>`).join("")}
      <line x1="${pad.l}" x2="${pad.l + plotW}" y1="${pad.t + plotH}" y2="${pad.t + plotH}" stroke="#374151"/>
      ${xTicks.map((i) => `<text x="${X(i)}" y="${pad.t + plotH + 14}" text-anchor="${i === 0 ? "start" : i === n - 1 ? "end" : "middle"}" class="tick">${esc(String(trends.x[i]).slice(-12))}</text>`).join("")}
      <text x="${pad.l + plotW / 2}" y="${pad.t + plotH + 32}" text-anchor="middle" class="axis">${esc(trends.axisLabel)}</text>
      <text transform="translate(12 ${pad.t + plotH / 2}) rotate(-90)" text-anchor="middle" class="axis">Reading (% of baseline)</text>
      ${legend}
    </svg>`;
  }

  // ------------------------------------------------------------------ document
  const CSS = `
    @page { size: A4; margin: 16mm 13mm 18mm; }
    *{box-sizing:border-box}
    html{-webkit-print-color-adjust:exact;print-color-adjust:exact}
    body{margin:0;background:#eef1f3;color:#111827;font:10.5pt/1.45 "Inter","Helvetica Neue",Arial,sans-serif}
    .sheet{max-width:210mm;margin:16px auto;padding:14mm 13mm;background:#fff;box-shadow:0 2px 12px rgba(0,0,0,.12)}
    .toolbar{position:sticky;top:0;z-index:2;display:flex;gap:10px;align-items:center;justify-content:center;padding:10px;background:#111827;color:#e5e7eb;font-size:13px}
    .toolbar button{font:600 13px Inter,Arial,sans-serif;padding:8px 14px;border-radius:6px;border:0;background:#14e0a0;color:#04140e;cursor:pointer}
    h1{font-size:19pt;margin:0 0 2px;letter-spacing:-.3px} h2{font-size:12.5pt;margin:18px 0 8px;padding-bottom:4px;border-bottom:1.5px solid #111827;break-after:avoid}
    .kicker{font:700 8pt Arial,sans-serif;letter-spacing:1.2px;text-transform:uppercase;color:#4b5563}
    .meta{display:grid;grid-template-columns:repeat(3,1fr);gap:6px 16px;margin-top:10px;padding:10px 12px;border:1px solid #d1d5db;border-radius:6px;background:#f9fafb;font-size:9pt}
    .meta div span{display:block;color:#4b5563;font-size:7.5pt;text-transform:uppercase;letter-spacing:.6px}
    .meta div b{font-weight:600;overflow-wrap:anywhere}
    .counts{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:8px 0}
    .count{border:1px solid #d1d5db;border-radius:6px;padding:7px 9px}
    .count span{display:block;font-size:7.5pt;color:#4b5563;text-transform:uppercase;letter-spacing:.5px}
    .count b{font:700 16pt/1.1 "JetBrains Mono",Menlo,monospace}
    .overall{margin:6px 0 0;font-size:10pt}
    .verdict{margin-top:6px;padding:7px 10px;border-left:3px solid #111827;background:#f3f4f6;font-size:9pt}
    .badge{display:inline-flex;align-items:center;gap:4px;padding:1px 7px;border:1.5px solid;border-radius:999px;font-size:8pt;font-weight:700;white-space:nowrap;background:#fff}
    table{width:100%;border-collapse:collapse;font-size:8.3pt;table-layout:fixed}
    thead{display:table-header-group} tfoot{display:table-footer-group}
    tr{break-inside:avoid;page-break-inside:avoid}
    th{text-align:left;font-size:7pt;text-transform:uppercase;letter-spacing:.3px;overflow-wrap:break-word;color:#374151;background:#f3f4f6;border-bottom:1.5px solid #111827;padding:5px 5px}
    td{padding:5px;border-bottom:1px solid #e5e7eb;vertical-align:top;overflow-wrap:anywhere}
    td.num{font-family:"JetBrains Mono",Menlo,monospace;white-space:nowrap}
    td .sub{display:block;color:#4b5563;font-size:7.5pt}
    td ul{margin:0;padding-left:12px} td li{margin:0}
    td .badge{margin-bottom:2px}
    .charts{display:grid;grid-template-columns:1fr 1fr;gap:12px;break-inside:avoid}
    .chart{width:100%;height:auto;display:block;break-inside:avoid}
    .chart .ct{font:700 9.5px Arial,sans-serif;fill:#111827} .chart .tick{font:8px Arial,sans-serif;fill:#4b5563}
    .chart .lbl{font:8.5px Arial,sans-serif;fill:#111827} .chart .val{font:700 8.5px Arial,sans-serif;fill:#111827}
    .chart .axis{font:italic 8.5px Arial,sans-serif;fill:#374151} .chart .grid{stroke:#e5e7eb;stroke-width:.8}
    .chart .empty{font:9px Arial,sans-serif;fill:#6b7280}
    .wide{margin-top:12px;break-inside:avoid}
    .actions{list-style:none;margin:0;padding:0}
    .action{break-inside:avoid;margin:0 0 7px;padding:7px 10px 7px 12px;border:1px solid #d1d5db;border-left:4px solid;border-radius:5px}
    .action h3{margin:0 0 2px;font-size:9.8pt} .action p{margin:1px 0;font-size:8.8pt}
    .action .pri{font:700 7.5pt Arial,sans-serif;letter-spacing:.6px;text-transform:uppercase;margin-right:6px}
    .action .lbl{font-weight:700;color:#374151}
    .none{color:#4b5563;font-style:italic}
    .note{margin-top:16px;padding-top:8px;border-top:1px solid #d1d5db;color:#4b5563;font-size:8pt}
    @media print{
      body{background:#fff} .toolbar{display:none} .sheet{margin:0;padding:0;max-width:none;box-shadow:none}
    }
    @media screen and (max-width:760px){ .sheet{margin:0;padding:16px} .meta{grid-template-columns:1fr 1fr} .counts{grid-template-columns:repeat(2,1fr)} .charts{grid-template-columns:1fr} table{font-size:7.5pt} }
  `;

  function pageFooterCss(data) {
    const stamp = `${data.header.system} · ${data.header.title} · Generated ${data.header.generatedAtText}`.replace(/["\\]/g, "");
    return `@page { @bottom-left { content: "${stamp}"; font: 7.5pt Arial, sans-serif; color: #4b5563; }
      @bottom-right { content: "Page " counter(page) " of " counter(pages); font: 7.5pt Arial, sans-serif; color: #4b5563; } }`;
  }

  /** The complete report as an HTML string. opts.autoPrint opens the print dialog once loaded. */
  function renderReportHtml(data, opts = {}) {
    const h = data.header, s = data.summary;
    const filters = Object.entries(h.filters || {}).map(([k, v]) => `${k}: ${v}`).join(" · ") || "None";
    const range = h.timeRange.samples ? `${h.timeRange.axis} ${h.timeRange.from} – ${h.timeRange.to} (${h.timeRange.samples} readings)` : "No readings";
    const rows = data.rows.map((r) => `<tr>
        <td><b>${esc(r.id)}</b>${r.name && r.name !== r.id ? `<span class="sub">${esc(r.name)}</span>` : ""}</td>
        <td>${esc(r.type)}</td><td>${esc(r.location)}</td>
        <td class="num">${num(r.lastReading)}${finite(r.latest) ? "" : '<span class="sub">latest missing</span>'}</td>
        <td class="num">${esc(r.lastSeenLabel ?? "never")}</td>
        <td>${esc(r.status)}${r.fault ? `<span class="sub">${esc(r.fault)}</span>` : ""}</td>
        <td>${badge(r.trust, r.health)}<span class="sub">${esc(r.trustReason)}</span></td>
        <td>${r.flags.length ? `<ul>${r.flags.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>` : "—"}</td>
      </tr>`).join("");
    const table = data.rows.length ? `<table>
        <colgroup><col style="width:13%"><col style="width:13%"><col style="width:14%"><col style="width:9%"><col style="width:7%"><col style="width:11%"><col style="width:18%"><col style="width:15%"></colgroup>
        <thead><tr><th>Sensor</th><th>Type</th><th>Location</th><th>Last reading</th><th>Last seen</th><th>Status</th><th>Trust level · score</th><th>Flags</th></tr></thead>
        <tbody>${rows}</tbody></table>` : `<p class="none">No sensors were found in the data.</p>`;
    const byPriority = ["High", "Medium", "Low"].map((p) => [p, data.actions.filter((a) => a.priority === p)]);
    const actions = data.actions.length ? byPriority.filter(([, list]) => list.length).map(([p, list]) => `<ul class="actions">${list.map((a) => `
        <li class="action" style="border-left-color:${PRIORITY_STYLE[p]}">
          <h3><span class="pri" style="color:${PRIORITY_STYLE[p]}">${p}</span>${esc(a.title)}</h3>
          <p><span class="lbl">Sensor${a.sensors.length > 1 ? "s" : ""}:</span> ${esc(a.sensors.join(", "))}</p>
          <p><span class="lbl">Evidence:</span> ${esc(a.reason)}</p>
          <p><span class="lbl">Next step:</span> ${esc(a.nextStep)}</p>
        </li>`).join("")}</ul>`).join("") : `<p class="none">No actions needed: every sensor agrees with its peers and no data problems were found.</p>`;

    const dist = barChart({ title: "Sensors by trust level", items: s.levels.map((l) => ({ label: l, value: s.counts[l] })),
      xLabel: "Number of sensors", yLabel: "Trust level", fillFor: (d) => LEVEL_STYLE[d.label].fill });
    const faults = barChart({ title: "Abnormal readings by type (all sensors)", items: data.charts.faultCounts.map((f) => ({ label: f.type, value: f.count })),
      xLabel: "Number of readings", yLabel: "Anomaly type", fillFor: () => "#64748b", emptyText: "No abnormal readings" });

    return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
      <title>${esc(h.title)} · ${esc(h.system)} · ${esc(h.generatedAtText)}</title>
      <style>${CSS}${pageFooterCss(data)}</style></head><body>
      ${opts.embedded ? "" : `<div class="toolbar"><span>Printable report · use “Save as PDF” in the print dialog</span><button type="button" onclick="window.print()">Print / Save as PDF</button></div>`}
      <main class="sheet">
        <div class="kicker">${esc(h.system)} · Radiation monitoring &amp; sensor analytics</div>
        <h1>${esc(h.title)}</h1>
        <div class="meta">
          <div><span>Generated</span><b>${esc(h.generatedAtText)}</b></div>
          <div><span>Time range covered</span><b>${esc(range)}</b></div>
          <div><span>Generated by</span><b>${esc(h.generatedBy || "Not recorded (no user accounts)")}</b></div>
          <div><span>Data source</span><b>${esc(h.source)}</b></div>
          <div><span>Filters applied</span><b>${esc(filters)}</b></div>
          <div><span>Sensors</span><b>${s.total}</b></div>
        </div>

        <h2>Executive summary</h2>
        <div class="counts">
          <div class="count"><span>Total sensors</span><b>${s.total}</b></div>
          ${s.levels.map((l) => `<div class="count" style="border-color:${LEVEL_STYLE[l].color};border-style:${LEVEL_STYLE[l].border}"><span>${LEVEL_STYLE[l].symbol} ${esc(l)}</span><b>${s.counts[l]}</b></div>`).join("")}
        </div>
        <p class="overall">${esc(s.overall)}</p>
        ${s.verdict ? `<div class="verdict"><b>Analysis verdict:</b> ${esc(s.verdict.text)}</div>` : ""}

        <h2>Sensors</h2>
        ${table}

        <h2>Charts</h2>
        <div class="charts">${dist}${faults}</div>
        <div class="wide">${lineChart(data.charts.trends)}</div>

        <h2>Recommended actions</h2>
        ${actions}

        <p class="note">Trust levels: Trusted = agrees with its peers; Needs review = intermittent or trending issue; Untrusted = currently faulty; Offline = no recent readings. Scores are model-based health estimates (0–100). Actions come from the deterministic rules in report_rules.js. This report is a training aid produced in the browser from the loaded data; it is not a plant-control or safety system and a flagged sensor indicates a statistical disagreement with its peers, not a confirmed radiation hazard.</p>
      </main>
      ${opts.autoPrint ? `<script>addEventListener("load",()=>setTimeout(()=>print(),250));<\/script>` : ""}
    </body></html>`;
  }

  const api = { renderReportHtml, barChart, lineChart, LEVEL_STYLE };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ReportRender = api;
})(typeof window !== "undefined" ? window : globalThis);
