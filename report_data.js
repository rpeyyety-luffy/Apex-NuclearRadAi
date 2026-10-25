/* Data collection for the exported report: turns one analysed dataset (detector.js output plus the
   readings) into plain report data — header, summary counts, one row per sensor, chart series.
   No DOM, no rendering, no recommendation logic (see report_rules.js), so it can be tested in Node.
   Runs in the browser (window.ReportData) and in Node (module.exports). */
(function (root) {
  "use strict";

  const Rules = typeof ReportRules !== "undefined" ? ReportRules : require("./report_rules.js");
  const MAX_POINTS = 300;        // chart series are thinned to at most this many points
  const MAX_TREND_SENSORS = 4;   // sensors shown in the trend chart
  const FAULT_TYPES = ["Drift high", "Drift low", "Spike", "Noisy", "Stuck", "No signal"];

  const finite = (v) => typeof v === "number" && Number.isFinite(v);
  function median(values) {
    const a = values.filter(finite).sort((x, y) => x - y);
    if (!a.length) return NaN;
    const m = a.length >> 1;
    return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
  }
  function mean(values) {
    const a = values.filter(finite);
    return a.length ? a.reduce((t, v) => t + v, 0) / a.length : NaN;
  }

  // evenly spaced sample indices (always including the first and last), at most `max` of them
  function thin(n, max = MAX_POINTS) {
    if (n <= max) return Array.from({ length: n }, (_, i) => i);
    return Array.from({ length: max }, (_, k) => Math.round((k * (n - 1)) / (max - 1)));
  }

  function sensorRow(name, i, raw, labels, s, meta) {
    // use the detector's cleaned readings when available, so "missing" means the same thing in the
    // report as in the analysis (detector.js treats zero or negative readings as no reading)
    const column = s && Array.isArray(s.x) && s.x.length === raw.length ? s.x : raw;
    const n = column.length;
    const baselineCount = Math.max(5, Math.min(40, Math.floor(n * 0.22)));   // same baseline as the dashboard
    let lastSeenIndex = -1;
    for (let t = n - 1; t >= 0; t--) if (finite(column[t])) { lastSeenIndex = t; break; }
    const silentReadings = n - 1 - lastSeenIndex;
    const tail = Math.max(Rules.SETTINGS.OFFLINE_MIN_READINGS, Math.round(n * Rules.SETTINGS.OFFLINE_TAIL_FRACTION));
    const missing = column.filter((v) => !finite(v)).length;
    const info = meta[name] || {};
    const m = (s && s.maintenance) || null;
    const row = {
      index: i, id: name, name: info.name || name, type: info.type || "—", location: info.where || "—",
      status: s ? s.status : "Unknown", fault: s ? s.fault : null, summary: s ? s.summary : "",
      health: m ? m.health : null,
      maintenance: m ? { action: m.action, urgency: m.urgency, text: m.text, serviceIn: m.serviceIn, replaceIn: m.replaceIn } : null,
      latest: n ? column[n - 1] : NaN,
      lastReading: lastSeenIndex >= 0 ? column[lastSeenIndex] : NaN,
      lastSeenIndex, lastSeenLabel: lastSeenIndex >= 0 ? labels[lastSeenIndex] : null,
      silentReadings, offline: n > 0 && silentReadings >= tail,
      missingShare: n ? missing / n : 0,
      anomalies: s ? s.anomalies : 0, anomalyPct: s ? s.anomalyPct : 0,
      firstIssueLabel: s && s.firstIssue >= 0 ? labels[s.firstIssue] : null,
      devPct: s && finite(s.devPct) ? s.devPct : null,
      baseline: mean(column.slice(0, baselineCount)),
    };
    const t = Rules.trustLevel(row);
    row.trust = t.level;
    row.trustReason = t.reason;
    row.flags = [
      row.offline ? `Silent for ${silentReadings} readings` : null,
      row.missingShare > 0 ? `${(100 * row.missingShare).toFixed(1)}% missing` : null,
      row.anomalies ? `${row.anomalies} abnormal readings` : null,
      row.devPct !== null && Math.abs(row.devPct) >= 1 ? `${row.devPct >= 0 ? "+" : ""}${row.devPct.toFixed(1)}% vs peers` : null,
    ].filter(Boolean);
    return row;
  }

  function overallSentence(counts, total, verdict) {
    if (!total) return "No sensors were found in the data, so nothing could be assessed.";
    const parts = [`${counts.Trusted} of ${total} sensor${total === 1 ? " is" : "s are"} trusted`];
    for (const level of ["Needs review", "Untrusted", "Offline"]) if (counts[level]) parts.push(`${counts[level]} ${level.toLowerCase()}`);
    const first = parts.shift();
    const list = parts.length ? `${first}; ${parts.join(", ")}.` : `${first}.`;
    if (verdict && verdict.level === "event") return `A plant-wide event was detected: the sensors changed together, which points to the plant rather than the instruments. ${list}`;
    if (counts.Trusted === total) return `All sensors agree with their peers and no data-quality anomalies were found. ${list}`;
    return `Some sensors disagree with their peers or have data problems. ${list}`;
  }

  /**
   * dataset: { names, columns, labels, indexLabel, detection, sourceLabel, filters }
   *   names    sensor IDs, columns[i] the readings of names[i] (NaN = missing), labels the x-axis values
   *   detection the result of Detector.analyse(names, columns)
   * opts: { sensorsMeta (array from sensors.js), generatedAt (Date), generatedBy, title, system }
   */
  function buildReportData(dataset, opts = {}) {
    const names = dataset.names || [];
    const columns = dataset.columns || [];
    const n = columns.length ? columns[0].length : 0;
    const labels = dataset.labels && dataset.labels.length === n ? dataset.labels.map(String) : Array.from({ length: n }, (_, i) => String(i));
    const detection = dataset.detection || { sensors: [], verdict: null, leak: null };
    const meta = Object.fromEntries((opts.sensorsMeta || []).map((s) => [s.id, s]));
    const generatedAt = opts.generatedAt instanceof Date ? opts.generatedAt : new Date();

    const rows = names.map((name, i) => sensorRow(name, i, columns[i] || [], labels, detection.sensors[i], meta));
    const cleaned = names.map((_, i) => { const s = detection.sensors[i]; return s && Array.isArray(s.x) && s.x.length === n ? s.x : columns[i] || []; });
    const counts = Object.fromEntries(Rules.TRUST_LEVELS.map((l) => [l, rows.filter((r) => r.trust === l).length]));
    const verdict = detection.verdict ? { level: detection.verdict.level, kind: detection.verdict.kind || null, text: detection.verdict.text } : null;

    // trend chart: the sensors that matter most — worst trust level first, then lowest health score
    const severity = { Offline: 0, Untrusted: 1, "Needs review": 2, Trusted: 3 };
    const ranked = [...rows].sort((a, b) => severity[a.trust] - severity[b.trust] || (a.health ?? 101) - (b.health ?? 101) || a.index - b.index);
    const flagged = ranked.filter((r) => r.trust !== "Trusted");
    const chosen = (flagged.length ? flagged : ranked).slice(0, flagged.length ? MAX_TREND_SENSORS : Math.min(3, ranked.length));
    const idx = thin(n);
    const pct = (v, base) => (finite(v) && finite(base) && base !== 0 ? (100 * v) / base : NaN);
    const networkBase = median(rows.map((r) => r.baseline));
    const trends = {
      axisLabel: dataset.indexLabel || "Sample",
      x: idx.map((t) => labels[t]),
      series: chosen.map((r) => ({ id: r.id, trust: r.trust, values: idx.map((t) => pct(cleaned[r.index][t], r.baseline)) })),
      reference: { id: "Network median", values: idx.map((t) => pct(median(cleaned.map((c) => c[t])), networkBase)) },
      selection: flagged.length ? "flagged sensors (worst first)" : "lowest health scores (no sensor is flagged)",
    };
    const faultCounts = FAULT_TYPES.map((type) => ({
      type, count: (detection.sensors || []).reduce((t, s) => t + ((s && s.counts && s.counts[type]) || 0), 0),
    }));

    const plant = { verdict, leak: detection.leak ? { kind: detection.leak.kind || null } : null };
    return {
      header: {
        title: opts.title || "Sensor Verification Report",
        system: opts.system || "NuclearRad AI",
        generatedAt: generatedAt.toISOString(),
        generatedAtText: generatedAt.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }),
        generatedBy: opts.generatedBy || null,
        source: dataset.sourceLabel || "Uploaded data",
        filters: dataset.filters || {},
        timeRange: { axis: dataset.indexLabel || "Sample", from: n ? labels[0] : null, to: n ? labels[n - 1] : null, samples: n },
      },
      summary: { total: rows.length, counts, levels: Rules.TRUST_LEVELS, verdict, overall: overallSentence(counts, rows.length, verdict) },
      rows,
      charts: { distribution: Rules.TRUST_LEVELS.map((level) => ({ level, count: counts[level] })), trends, faultCounts },
      plant,
    };
  }

  /** Report data plus its recommended actions (report_rules.js). */
  function buildReport(dataset, opts = {}) {
    const data = buildReportData(dataset, opts);
    data.actions = Rules.recommendActions(data.rows, data.plant);
    return data;
  }

  const api = { buildReportData, buildReport, thin, MAX_POINTS, FAULT_TYPES };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ReportData = api;
})(typeof window !== "undefined" ? window : globalThis);
