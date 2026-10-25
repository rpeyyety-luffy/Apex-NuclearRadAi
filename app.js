const $ = (id) => document.getElementById(id);
const expectedSensors = ['RAD-01', 'RAD-02', 'RAD-03', 'RAD-04', 'RAD-05'];
// chart palette: a healthy channel's raw line is neutral, so only flagged channels stand out
const colors = { raw: '#d5ddda', peer: '#4fc1e3', threshold: '#f0b03c', grid: '#1e2926', axis: '#7f8e98', warn: '#f0b03c', bad: '#ef5d5d' };

function median(values) {
  const a = values.filter(Number.isFinite).sort((x, y) => x - y);
  return a.length ? a[Math.floor(a.length / 2)] : NaN;
}
function mean(values) {
  const a = values.filter(Number.isFinite);
  return a.length ? a.reduce((total, value) => total + value, 0) / a.length : NaN;
}
function numeric(value) {
  if (value === null || value === undefined || value === '') return NaN;
  const n = Number(value);
  return Number.isFinite(n) ? n : NaN;
}
function notice(text, kind = 'info') {
  const box = $('message');
  box.textContent = text;
  box.className = `message show ${kind}`;
}
function clearNotice() { $('message').className = 'message'; }

function normal(random) {
  return Math.sqrt(-2 * Math.log(random() || 0.01)) * Math.cos(2 * Math.PI * random());
}
function demoRows() {
  let seed = 7;
  const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  return Array.from({ length: 180 }, (_, sample) => {
    const row = { sample };
    for (let i = 1; i <= 5; i += 1) row[`RAD-0${i}`] = 100 * (1 + 0.012 * normal(random));
    if (sample >= 80) row['RAD-03'] *= 1 + 0.007 * (sample - 80);
    return row;
  });
}

function normalizeRows(rawRows) {
  if (!rawRows.length) throw new Error('The selected sheet has no data rows.');
  const headers = Object.keys(rawRows[0]);
  // a sensor column is found by its number: "RAD-01", "Sensor_1_MPa" and "Sensor 1 (uSv/h)" are all sensor 1;
  // answer/label columns such as Anomaly_Active, Real_Leak or Faulty_Sensor are never readings
  const sensorNumber = (header) => {
    const h = String(header).trim();
    if (/anomal|leak|faulty|label/i.test(h)) return null;
    const m = h.replace(/\(.*?\)/g, '').match(/^(?:rad|sensor)[\s_-]*0*(\d+)/i);
    return m ? parseInt(m[1], 10) : null;
  };
  const headerFor = new Map(expectedSensors.map((sensor, i) => [sensor, headers.find((h) => sensorNumber(h) === i + 1)]));
  const missing = expectedSensors.filter((sensor) => !headerFor.get(sensor));
  if (missing.length) throw new Error(`Missing required column${missing.length > 1 ? 's' : ''}: ${missing.join(', ')}. The sheet must contain five sensor columns (RAD-01 to RAD-05, or Sensor_1 to Sensor_5).`);
  const sensors = expectedSensors;
  const indexHeader = headers.find((header) => /^(sample|time|time_s|timestamp|date)$/i.test(String(header).trim()));
  const rows = rawRows.map((row, index) => ({
    label: indexHeader && row[indexHeader] !== undefined && row[indexHeader] !== '' ? String(row[indexHeader]) : String(index),
    values: sensors.map((sensor) => numeric(row[headerFor.get(sensor)]))
  })).filter((row) => row.values.some(Number.isFinite));
  if (rows.length < 5) throw new Error('At least five rows with numeric sensor readings are required.');
  const secondsPerReading = indexHeader ? Detector.secondsPerReading(rows.map((row) => row.label), String(indexHeader)) : null;
  return { sensors, rows, secondsPerReading, indexLabel: indexHeader ? String(indexHeader).trim() : 'Sample' };
}

// detector.js status and fault type -> the wording used on this page
const ISSUE = { 'No signal': 'Missing readings', Stuck: 'Stuck signal', Spike: 'Spike / outlier', Noisy: 'Noisy signal', 'Drift high': 'Drift high', 'Drift low': 'Drift low' };
function issueFor(s) {
  if (s.status === 'Healthy') return 'Healthy';
  const fault = ISSUE[s.fault] || s.fault;
  if (s.status === 'At risk') return `At risk · ${fault}`;
  if (s.status === 'Watch') return `Watch · ${fault}`;
  return fault;
}

function analyse(data) {
  const { sensors, rows } = data;
  const baselineCount = Math.max(5, Math.min(40, Math.floor(rows.length * 0.22)));
  const columns = sensors.map((_, i) => rows.map((row) => row.values[i]));
  // the same detector and machine-learned models as fault_detector.html (detector.js, faultml.js)
  const detection = Detector.analyse(sensors, columns);
  const results = sensors.map((sensor, sensorIndex) => {
    const series = columns[sensorIndex];
    const baseline = mean(series.slice(0, baselineCount));
    const peer = rows.map((row) => median(row.values.filter((_, i) => i !== sensorIndex)));
    const latest = series.at(-1);
    const s = detection.sensors[sensorIndex];
    const issue = issueFor(s);
    const recommendation = s.advice.length ? s.advice.join(' ') : 'Continue routine monitoring; readings agree with peer channels.';
    return { sensor, series, peer, baseline, latest, issue, status: s.status, summary: s.summary, recommendation, advice: s.advice, maintenance: s.maintenance };
  });
  const active = results.filter((result) => result.issue !== 'Healthy');
  const network = results.filter((result) => Number.isFinite(result.latest)).map((result) => result.latest);
  return { ...data, baselineCount, results, active, latestMedian: median(network), verdict: detection.verdict, leak: detection.leak };
}

// ------------------------------------------------------------------ small helpers
const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmtNum = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : '—');
const toneOf = (status) => (status === 'Healthy' ? 'ok' : status === 'Faulty' ? 'bad' : 'warn');

function setSource(text, tone = '', busy = false) {
  const el = $('sourceStatus');
  if (!el) return;
  el.className = `badge ${tone}${busy ? ' busy' : ''}`;
  el.innerHTML = `<span>${esc(text)}</span>`;
  el.title = `Data source: ${text}`;
}
function setFileStatus(html) { const el = $('fileStatus'); if (el) el.innerHTML = html; }

// ------------------------------------------------------------------ charts (hand-drawn SVG)
// charts are drawn at their real pixel width (and redrawn on resize) so text stays a fixed, readable size
const CHART = { pad: { l: 64, r: 16, t: 14, b: 40 } };
const chartState = new Map();   // sensor -> { result, labels, domain, width, height, index }

function renderPolyline(values, color, domain, width, height, pad, strokeWidth = 1.6, extra = '') {
  const { min, max } = domain;
  const span = max - min || 1;
  const points = values.map((value, index) => {
    const x = pad.l + index / Math.max(1, values.length - 1) * (width - pad.l - pad.r);
    const y = Number.isFinite(value) ? pad.t + (max - value) / span * (height - pad.t - pad.b) : null;
    return y === null ? null : `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  // missing readings break the line instead of being joined across the gap
  const segments = [];
  let current = [];
  points.forEach((point) => { if (point) current.push(point); else if (current.length) { segments.push(current); current = []; } });
  if (current.length) segments.push(current);
  return segments.map((segment) => `<polyline points="${segment.join(' ')}" fill="none" stroke="${color}" stroke-width="${strokeWidth}" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"${extra}/>`).join('');
}

function tickDigits(span) { return span < 0.5 ? 3 : span < 5 ? 2 : span < 50 ? 1 : 0; }
function shortLabel(label) { const s = String(label); return s.length > 11 ? s.slice(-8) : s; }

function renderChart(result, labels, opts = {}) {
  const { pad } = CHART;
  const width = opts.width || 760;
  const height = opts.tall ? 380 : width < 460 ? 220 : 240;
  const threshold = result.baseline * 1.3;
  // same scale as before: readings, peer estimate and the 130% line, with a 13% margin
  const all = [...result.series, ...result.peer, threshold].filter(Number.isFinite);
  const low = Math.min(...all), high = Math.max(...all);
  const margin = Math.max((high - low) * 0.13, 1);
  const domain = { min: low - margin, max: high + margin };
  const plotW = width - pad.l - pad.r, plotH = height - pad.t - pad.b;
  const y = (value) => pad.t + (domain.max - value) / (domain.max - domain.min) * plotH;
  const digits = tickDigits(domain.max - domain.min);
  const grid = [0, .25, .5, .75, 1].map((step) => {
    const value = domain.min + (domain.max - domain.min) * step;
    return `<line x1="${pad.l}" x2="${width - pad.r}" y1="${y(value).toFixed(1)}" y2="${y(value).toFixed(1)}" stroke="${colors.grid}"/>`
      + `<text x="${pad.l - 8}" y="${(y(value) + 4).toFixed(1)}" text-anchor="end" fill="${colors.axis}" font-size="11" font-family="JetBrains Mono, monospace">${value.toFixed(digits)}</text>`;
  }).join('');
  const nTicks = Math.max(3, Math.min(opts.tall ? 9 : 7, Math.floor((width - pad.l - pad.r) / 90) + 1));
  const xTicks = Array.from({ length: nTicks }, (_, k) => k / (nTicks - 1)).map((step) => {
    const index = Math.round((labels.length - 1) * step);
    const x = pad.l + step * plotW;
    const anchor = step === 0 ? 'start' : step === 1 ? 'end' : 'middle';
    return `<line x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${pad.t + plotH}" y2="${pad.t + plotH + 4}" stroke="${colors.axis}" opacity=".6"/>`
      + `<text x="${x.toFixed(1)}" y="${pad.t + plotH + 17}" text-anchor="${anchor}" fill="${colors.axis}" font-size="11" font-family="JetBrains Mono, monospace">${esc(shortLabel(labels[index]))}</text>`;
  }).join('');
  const tone = toneOf(result.status);
  const rawColor = tone === 'bad' ? colors.bad : tone === 'warn' ? colors.warn : colors.raw;
  const thrY = y(threshold);
  chartState.set(result.sensor, { result, labels, domain, width, height, index: chartState.get(result.sensor)?.index ?? null });
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(result.sensor)} readings chart: raw reading and peer estimate over ${labels.length} samples">`
    + `<rect x="${pad.l}" y="${pad.t}" width="${plotW}" height="${plotH}" fill="#0e1513"/>${grid}`
    + `<line x1="${pad.l}" x2="${pad.l + plotW}" y1="${pad.t + plotH}" y2="${pad.t + plotH}" stroke="${colors.axis}" opacity=".5"/>`
    + `<line x1="${pad.l}" x2="${width - pad.r}" y1="${thrY.toFixed(1)}" y2="${thrY.toFixed(1)}" stroke="${colors.threshold}" stroke-opacity=".6" stroke-dasharray="6 5"/>`
    + `<text x="${width - pad.r - 4}" y="${(thrY - 6).toFixed(1)}" text-anchor="end" fill="${colors.threshold}" fill-opacity=".85" font-size="11">130% of baseline</text>`
    + renderPolyline(result.peer, colors.peer, domain, width, height, pad, 1.4, ' stroke-opacity=".9"')
    + renderPolyline(result.series, rawColor, domain, width, height, pad, tone === 'ok' ? 1.4 : 1.8)
    + xTicks
    + `<text x="${pad.l + plotW / 2}" y="${height - 4}" text-anchor="middle" fill="${colors.axis}" font-size="11">${esc(opts.indexLabel || 'Sample')}</text>`
    + `<text transform="translate(11 ${pad.t + plotH / 2}) rotate(-90)" text-anchor="middle" fill="${colors.axis}" font-size="11">Reading</text>`
    + `<g class="cursor" visibility="hidden"><line x1="0" x2="0" y1="${pad.t}" y2="${pad.t + plotH}" stroke="#e4eae7" stroke-opacity=".35"/>`
    + `<circle class="c-peer" r="3.5" fill="${colors.peer}" stroke="#0e1513" stroke-width="1.5"/><circle class="c-raw" r="4" fill="${rawColor}" stroke="#0e1513" stroke-width="1.5"/></g>`
    + `<rect class="hit" x="${pad.l}" y="${pad.t}" width="${plotW}" height="${plotH}" fill="transparent"/></svg>`;
}

function chartCard(result, report, expanded = false) {
  const tone = toneOf(result.status);
  const rawColor = tone === 'bad' ? colors.bad : tone === 'warn' ? colors.warn : colors.raw;
  return `<article class="chart-card ${tone}${expanded ? ' expanded' : ''}" data-sensor="${esc(result.sensor)}">`
    + `<header class="chart-head"><h3>${esc(result.sensor)}</h3><span class="badge ${tone}">${esc(result.issue)}</span>`
    + `<span class="latest">Latest <b>${fmtNum(result.latest)}</b> · baseline <b>${fmtNum(result.baseline)}</b></span>`
    + `<button class="btn btn-ghost btn-sm expand" type="button" aria-pressed="${expanded}" aria-label="${expanded ? 'Collapse' : 'Expand'} ${esc(result.sensor)} chart">${expanded ? 'Collapse' : 'Expand'}</button></header>`
    + `<div class="chart-body" tabindex="0" aria-label="${esc(result.sensor)} chart. Use the left and right arrow keys to read values."><div class="chart-tip" aria-hidden="true"></div></div>`
    + `<footer class="chart-legend"><span><i style="border-color:${rawColor}"></i>Raw reading</span><span><i style="border-color:${colors.peer}"></i><span data-tip="Median of the other four channels at the same sample: what this channel would read if it agreed with its peers." tabindex="0">Peer estimate</span></span><span><i class="dash" style="border-color:${colors.threshold}"></i>130% of baseline</span></footer>`
    + `</article>`;
}

function showCursor(card, index) {
  const st = chartState.get(card.dataset.sensor);
  if (!st) return;
  const { result, labels, domain, width, height } = st;
  const n = labels.length;
  index = Math.max(0, Math.min(n - 1, index));
  st.index = index;
  const { pad } = CHART;
  const plotW = width - pad.l - pad.r, plotH = height - pad.t - pad.b;
  const x = pad.l + index / Math.max(1, n - 1) * plotW;
  const y = (v) => pad.t + (domain.max - v) / (domain.max - domain.min) * plotH;
  const svg = card.querySelector('svg'), g = svg.querySelector('.cursor');
  g.setAttribute('visibility', 'visible');
  g.querySelector('line').setAttribute('x1', x); g.querySelector('line').setAttribute('x2', x);
  for (const [cls, v] of [['.c-raw', result.series[index]], ['.c-peer', result.peer[index]]]) {
    const c = g.querySelector(cls);
    c.setAttribute('visibility', Number.isFinite(v) ? 'visible' : 'hidden');
    if (Number.isFinite(v)) { c.setAttribute('cx', x); c.setAttribute('cy', y(v)); }
  }
  const tone = toneOf(result.status);
  const rawColor = tone === 'bad' ? colors.bad : tone === 'warn' ? colors.warn : colors.raw;
  const tip = card.querySelector('.chart-tip');
  const raw = result.series[index], peer = result.peer[index];
  tip.innerHTML = `<div class="t">${esc(st.indexLabel || chartLabelName)} ${esc(labels[index])}</div>`
    + `<div class="r"><span><i style="border-color:${rawColor}"></i>Raw reading</span><b>${Number.isFinite(raw) ? fmtNum(raw, 3) : 'missing'}</b></div>`
    + `<div class="r"><span><i style="border-color:${colors.peer}"></i>Peer estimate</span><b>${fmtNum(peer, 3)}</b></div>`
    + (Number.isFinite(raw) && Number.isFinite(peer) && peer ? `<div class="r"><span>Difference</span><b>${(100 * (raw / peer - 1)).toFixed(1)}%</b></div>` : '');
  const box = card.querySelector('.chart-body').getBoundingClientRect(), r = svg.getBoundingClientRect();
  const px = r.left - box.left + x * r.width / width;
  const tipW = tip.offsetWidth || 170;
  tip.style.left = `${px + 14 + tipW > box.width ? Math.max(4, px - tipW - 14) : px + 14}px`;
  tip.classList.add('show');
}
let chartLabelName = 'Sample';
function hideCursor(card) {
  card.querySelector('.cursor')?.setAttribute('visibility', 'hidden');
  card.querySelector('.chart-tip')?.classList.remove('show');
}

function drawChart(card, report) {
  const body = card.querySelector('.chart-body');
  const result = report.results.find((r) => r.sensor === card.dataset.sensor);
  const width = Math.max(260, Math.round(body.clientWidth - 24));   // .chart-body has 12px side padding
  body.querySelector('svg')?.remove();
  body.insertAdjacentHTML('afterbegin', renderChart(result, report.rows.map((row) => row.label),
    { tall: card.classList.contains('expanded'), indexLabel: report.indexLabel, width }));
}
function drawCharts(report) { $('charts').querySelectorAll('.chart-card').forEach((card) => drawChart(card, report)); }

function wireCharts(report) {
  const host = $('charts');
  if (!host || host.dataset.wired) return;
  host.dataset.wired = '1';
  // redraw at the new width when the layout changes (window resize, rotation, column change)
  let lastWidth = host.clientWidth, pending = 0;
  new ResizeObserver(() => {
    if (host.clientWidth === lastWidth || !host._report) return;
    lastWidth = host.clientWidth;
    cancelAnimationFrame(pending);
    pending = requestAnimationFrame(() => drawCharts(host._report));
  }).observe(host);
  const indexAt = (card, clientX) => {
    const svg = card.querySelector('svg'), r = svg.getBoundingClientRect(), st = chartState.get(card.dataset.sensor);
    const { pad } = CHART, { width } = st;
    const xv = (clientX - r.left) * width / r.width;
    return Math.round((xv - pad.l) / (width - pad.l - pad.r) * (st.labels.length - 1));
  };
  host.addEventListener('pointermove', (e) => {
    const body = e.target.closest('.chart-body');
    if (body) showCursor(body.closest('.chart-card'), indexAt(body.closest('.chart-card'), e.clientX));
  });
  host.addEventListener('pointerleave', () => host.querySelectorAll('.chart-card').forEach(hideCursor), true);
  host.addEventListener('pointerout', (e) => {
    const body = e.target.closest('.chart-body');
    if (body && !body.contains(e.relatedTarget)) hideCursor(body.closest('.chart-card'));
  });
  host.addEventListener('keydown', (e) => {
    const body = e.target.closest('.chart-body');
    if (!body) return;
    const card = body.closest('.chart-card'), st = chartState.get(card.dataset.sensor), n = st.labels.length;
    const step = e.shiftKey ? Math.max(1, Math.round(n / 20)) : 1;
    const moves = { ArrowRight: (i) => i + step, ArrowLeft: (i) => i - step, Home: () => 0, End: () => n - 1 };
    if (!moves[e.key]) return;
    e.preventDefault();
    showCursor(card, moves[e.key](st.index ?? n - 1));
  });
  host.addEventListener('focusout', (e) => { const body = e.target.closest('.chart-body'); if (body) hideCursor(body.closest('.chart-card')); });
  host.addEventListener('click', (e) => {
    const btn = e.target.closest('.expand');
    if (!btn) return;
    const card = btn.closest('.chart-card'), report = host._report;
    const result = report.results.find((r) => r.sensor === card.dataset.sensor);
    const wasExpanded = card.classList.contains('expanded');
    card.outerHTML = chartCard(result, report, !wasExpanded);
    const fresh = host.querySelector(`.chart-card[data-sensor="${CSS.escape(result.sensor)}"]`);
    drawChart(fresh, report);
    fresh.querySelector('.expand').focus({ preventScroll: true });
    if (!wasExpanded) fresh.scrollIntoView({ block: 'nearest', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  });
}

// ------------------------------------------------------------------ 3D model hand-off
function renderCore(results = expectedSensors.map((sensor) => ({ sensor, issue: 'Awaiting data', latest: NaN }))) {
  // the 3D model (reactor3d.js) draws each monitor's status from these results
  window.coreResults = results;
  window.dispatchEvent(new CustomEvent('core-results', { detail: results }));
}

// ------------------------------------------------------------------ channel assessment cards
// predicted sensor health and when it should be repaired or replaced (detector.js)
function maintenanceHtml(m, secondsPerReading) {
  const tone = m.health >= 80 ? 'var(--ok)' : m.health >= 40 ? 'var(--warn)' : 'var(--crit)';
  const when = (label, readings) => `<span>${label}: <b>${Detector.whenText(readings, secondsPerReading)}</b></span>`;
  const dates = m.action === 'None' || m.action === 'Replace' || m.action === 'Inspect' ? '' : when(m.action, m.serviceIn) + when('Replace', m.replaceIn);
  return `<div class="finding-action">`
    + `<div class="row"><span class="act act-${m.action.toLowerCase()}">${m.action === 'None' ? 'No maintenance due' : m.action}</span>${m.action === 'None' ? '' : `<span class="urgency">${m.urgency}</span>`}</div>`
    + `<div class="healthbar" role="img" aria-label="Health ${m.health} out of 100"><i style="width:${m.health}%;background:${tone}"></i></div>`
    + (dates ? `<div class="when">${dates}</div>` : '')
    + (m.action === 'None' ? '' : `<p>${esc(m.text)}</p>`) + '</div>';
}

function findingHtml(result, report) {
  const type = toneOf(result.status);
  const m = result.maintenance;
  const steps = result.status === 'Healthy' || !result.advice || !result.advice.length ? ''
    : `<details class="steps"><summary>Recommended steps</summary><ul>${result.advice.map((a) => `<li>${esc(a)}</li>`).join('')}</ul></details>`;
  return `<article class="finding ${type}"><h3><span>${esc(result.sensor)}</span><span class="scenario ${type}">${esc(result.issue)}</span></h3>`
    + `<dl class="kv"><div><dt>Latest reading</dt><dd>${fmtNum(result.latest)}</dd></div><div><dt>Baseline</dt><dd>${fmtNum(result.baseline)}</dd></div>`
    + `<div><dt><span data-tip="Model-based score from 100 (agrees with its peers) to 0 (at the replacement limit). An estimate, not a calibration result." tabindex="0">Health score</span></dt><dd>${m.health}<small>/100</small></dd></div></dl>`
    + `<p class="finding-summary">${esc(result.summary)}</p>${maintenanceHtml(m, report.secondsPerReading)}${steps}</article>`;
}

// ------------------------------------------------------------------ page render
function metricHtml(label, value, unit, context, dot, tip) {
  const l = tip ? `<span data-tip="${esc(tip)}" tabindex="0">${label}</span>` : label;
  return `<div class="metric"><span class="metric-label">${l}</span>${dot ? `<span class="dot ${dot}" aria-hidden="true"></span>` : ''}`
    + `<span class="metric-value">${value}${unit ? `<small>${unit}</small>` : ''}</span><span class="metric-context">${context}</span></div>`;
}

function render(report) {
  const n = report.active.length;
  const anyFaulty = report.results.some((r) => r.status === 'Faulty');
  const flaggedIds = report.active.map((r) => r.sensor).join(', ');
  $('summary').innerHTML = [
    metricHtml('Samples analysed', report.rows.length, 'rows', esc(report.sourceLabel || 'Uploaded data set')),
    metricHtml('Monitored channels', report.sensors.length, '', `${report.sensors[0]} to ${report.sensors[report.sensors.length - 1]}`),
    metricHtml('Latest network median', Number.isFinite(report.latestMedian) ? report.latestMedian.toFixed(2) : '—', 'input units', 'Median of the latest readings', '',
      'Median of the most recent reading across all channels, in the units of the source file.'),
    metricHtml('Channels requiring review', n, `of ${report.sensors.length}`, n ? `Flagged: ${esc(flaggedIds)}` : 'All channels agree with their peers', n ? (anyFaulty ? 'bad' : 'warn') : 'ok'),
  ].join('');
  $('summary').classList.add('show');

  const { level, text } = report.verdict;
  const channels = `${n} channel${n === 1 ? '' : 's'}`;
  const plus = n ? ` + ${n} channel${n === 1 ? '' : 's'} flagged` : '';
  const leak = report.leak;
  const eventBanner = !leak ? ['var(--crit)', 'Plant-wide event · not a sensor fault', 'Possible real radiation rise across most channels']
    : leak.kind === 'full' ? ['var(--crit)', 'Plant-wide event · full leak pattern', `Full pressure-loss pattern: every sensor near zero${plus}`]
    : ['var(--crit)', 'Plant-wide event · leak pattern', `Leak pattern: pressure ${leak.stable ? 'dropped and stabilised' : 'dropping'} at about ${leak.endValue.toFixed(1)}${plus}`];
  const banner = {
    event: eventBanner,
    fault: ['var(--warn)', 'Sensor review', `${channels} need${n === 1 ? 's' : ''} review`],
    watch: ['var(--warn)', 'Attention', `${channels} need${n === 1 ? 's' : ''} attention`],
    ok: ['var(--ok)', 'No anomalies detected', 'No data-quality anomalies detected'],
  }[level] || ['var(--warn)', 'Attention', `${channels} need review`];
  const [color, label, headline] = banner;
  $('status').innerHTML = `<span class="tone" style="background:${color}"></span><div class="eye" style="color:${color}">${label} · ${esc(report.sourceLabel || 'uploaded data')}</div>`
    + `<b>${esc(headline)}</b><span class="sub">${esc(text)}</span>`
    + `<span class="disclaimer">Recommendations are a training aid; escalate operational decisions through approved plant procedures.</span>`;
  $('status').style.borderLeftColor = '';
  $('status').classList.add('show');
  renderCore(report.results);

  if ($('analysisEmpty')) $('analysisEmpty').hidden = true;
  if ($('chartsEmpty')) $('chartsEmpty').hidden = true;
  $('findingsTitle').hidden = false;
  $('findings').innerHTML = report.results.map((result) => findingHtml(result, report)).join('');
  $('findings').classList.add('show');

  $('chartsTitle').hidden = false;
  chartState.clear();
  chartLabelName = report.indexLabel || 'Sample';
  $('charts')._report = report;
  $('charts').innerHTML = report.results.map((result) => chartCard(result, report)).join('');
  $('charts').classList.add('show');
  drawCharts(report);
  wireCharts(report);
}

// ------------------------------------------------------------------ loading data
let loaded = false;
let loadId = 0;   // only the most recent load request may update the page
function runAnalysis(getRows, sourceLabel, sourceBadge, id = ++loadId) {
  setSource(`Analysing ${sourceLabel}…`, 'info', true);
  setFileStatus(`<span class="spinner" aria-hidden="true"></span><span>Analysing <b>${esc(sourceLabel)}</b>…</span>`);
  clearNotice();
  document.body.setAttribute('aria-busy', 'true');
  // let the browser paint the progress indicator before the (synchronous) analysis runs
  setTimeout(() => {
    if (id !== loadId) return;
    try {
      const report = analyse(normalizeRows(getRows()));
      if (id !== loadId) return;
      report.sourceLabel = sourceLabel;
      render(report);
      loaded = true;
      setSource(sourceBadge, 'ok');
      setFileStatus(`<span class="badge ok"><span>Analysed</span></span><b>${esc(sourceLabel)}</b>`);
      notice(`${sourceLabel} loaded. ${report.rows.length} usable samples and ${report.sensors.length} RAD channels were analysed locally.`, 'success');
    } catch (error) {
      if (id !== loadId) return;
      setSource(loaded ? 'Load failed · previous results shown' : 'Load failed', 'bad');
      setFileStatus(`<span class="badge bad"><span>Not analysed</span></span><b>${esc(sourceLabel)}</b>`);
      notice(`${error.message || 'The file could not be read.'}${loaded ? ' The previous results are still shown.' : ''}`, 'error');
    } finally {
      document.body.removeAttribute('aria-busy');
    }
  }, 30);
}

const ACCEPTED = /\.(xlsx|xls|csv)$/i;
const MAX_BYTES = 25 * 1024 * 1024;
function handleFile(file) {
  if (!file) return;
  const id = ++loadId;
  if (!ACCEPTED.test(file.name)) {
    setFileStatus(`<span class="badge bad"><span>Not analysed</span></span><b>${esc(file.name)}</b>`);
    notice(`“${file.name}” is not a supported file type. Choose an .xlsx, .xls or .csv file.`, 'error');
    return;
  }
  if (file.size > MAX_BYTES) {
    notice(`“${file.name}” is ${(file.size / 1048576).toFixed(1)} MB; files up to 25 MB can be analysed in the browser.`, 'error');
    return;
  }
  if (!window.XLSX) { notice('The Excel reader could not load. Check your connection and try again.', 'error'); return; }
  setFileStatus(`<span class="spinner" aria-hidden="true"></span><span>Reading <b>${esc(file.name)}</b>…</span>`);
  const reader = new FileReader();
  reader.onload = () => {
    if (id !== loadId) return;
    runAnalysis(() => {
      const book = XLSX.read(reader.result, { type: 'array' });
      const sheet = book.Sheets[book.SheetNames[0]];
      return XLSX.utils.sheet_to_json(sheet, { defval: '' });
    }, file.name, file.name, id);
  };
  reader.onerror = () => {
    if (id !== loadId) return;
    setFileStatus(`<span class="badge bad"><span>Not analysed</span></span><b>${esc(file.name)}</b>`);
    notice('The selected file could not be read.', 'error');
  };
  reader.readAsArrayBuffer(file);
}

function loadDemo() { runAnalysis(demoRows, 'Synthetic demo data', 'Demo data · synthetic'); }
// kept for the Node checks and any caller that already has parsed rows
function loadRows(rows, source) {
  const report = analyse(normalizeRows(rows));
  report.sourceLabel = source;
  render(report);
  notice(`${source} loaded. ${report.rows.length} usable samples and ${report.sensors.length} RAD channels were analysed locally.`, 'success');
}

$('file').addEventListener('change', (event) => {
  handleFile(event.target.files[0]);
  event.target.value = '';   // choosing the same file again re-runs the analysis
});
$('demo').addEventListener('click', loadDemo);
$('demoTop')?.addEventListener('click', loadDemo);

const zone = $('dropzone');
if (zone) {
  ['dragenter', 'dragover'].forEach((t) => zone.addEventListener(t, (e) => { e.preventDefault(); zone.classList.add('drag'); }));
  ['dragleave', 'dragend'].forEach((t) => zone.addEventListener(t, (e) => { if (!zone.contains(e.relatedTarget)) zone.classList.remove('drag'); }));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('drag');
    const files = e.dataTransfer && e.dataTransfer.files;
    if (files && files.length > 1) notice('Only one file can be analysed at a time; the first file was used.', 'info');
    handleFile(files && files[0]);
  });
  // a file dropped outside the drop area must not navigate away from the page
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => { if (!zone.contains(e.target)) e.preventDefault(); });
}

renderCore();
if (location.hash === '#demo') loadDemo();
