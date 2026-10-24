const $ = (id) => document.getElementById(id);
const expectedSensors = ['RAD-01', 'RAD-02', 'RAD-03', 'RAD-04', 'RAD-05'];
const colors = { raw: '#ff746c', peer: '#8ee2bd', threshold: '#f3b969', grid: '#2d3d38' };

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
function notice(text) {
  const box = $('message');
  box.textContent = text;
  box.classList.add('show');
}
function clearNotice() { $('message').classList.remove('show'); }

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
  const canonical = (header) => String(header).trim().toUpperCase().replace(/[-_ ]/g, '');
  const headerFor = new Map(expectedSensors.map((sensor) => [
    sensor,
    headers.find((header) => canonical(header) === canonical(sensor))
  ]));
  const missing = expectedSensors.filter((sensor) => !headerFor.get(sensor));
  if (missing.length) throw new Error(`Missing required column${missing.length > 1 ? 's' : ''}: ${missing.join(', ')}. The sheet must contain RAD-01 through RAD-05.`);
  const sensors = expectedSensors;
  const indexHeader = headers.find((header) => /^(sample|time|timestamp|date)$/i.test(String(header)));
  const rows = rawRows.map((row, index) => ({
    label: indexHeader && row[indexHeader] !== undefined && row[indexHeader] !== '' ? String(row[indexHeader]) : String(index),
    values: sensors.map((sensor) => numeric(row[headerFor.get(sensor)]))
  })).filter((row) => row.values.some(Number.isFinite));
  if (rows.length < 5) throw new Error('At least five rows with numeric RAD readings are required.');
  return { sensors, rows };
}

function analyse(data) {
  const { sensors, rows } = data;
  const baselineCount = Math.max(5, Math.min(40, Math.floor(rows.length * 0.22)));
  const results = sensors.map((sensor, sensorIndex) => {
    const series = rows.map((row) => row.values[sensorIndex]);
    const baseline = mean(series.slice(0, baselineCount));
    const peer = rows.map((row) => median(row.values.filter((_, i) => i !== sensorIndex)));
    const latest = series.at(-1);
    const missing = series.filter((value) => !Number.isFinite(value)).length;
    const high = series.filter((value) => Number.isFinite(value) && value > baseline * 1.3).length;
    const peerDeviations = series.map((value, i) => Number.isFinite(value) && Number.isFinite(peer[i]) ? Math.abs(value / peer[i] - 1) : 0);
    const maxDeviation = Math.max(...peerDeviations);
    const lateDeviation = mean(peerDeviations.slice(-Math.max(5, Math.floor(rows.length * 0.2))));
    const earlyDeviation = mean(peerDeviations.slice(0, baselineCount));
    let repeats = 0;
    for (let i = 1; i < series.length; i += 1) if (Number.isFinite(series[i]) && series[i] === series[i - 1]) repeats += 1; else repeats = 0;
    let issue = 'Healthy', recommendation = 'Continue routine monitoring; readings agree with peer channels.';
    if (missing) { issue = 'Missing readings'; recommendation = 'Inspect the acquisition path, power, and communications; restore or replace the affected channel.'; }
    else if (repeats >= 4) { issue = 'Stuck signal'; recommendation = 'Inspect and replace or recalibrate the sensor; consecutive identical readings indicate a frozen channel.'; }
    else if (lateDeviation > Math.max(0.05, earlyDeviation * 2.2)) { issue = 'Peer drift'; recommendation = 'Schedule calibration and verify the channel against a reference instrument.'; }
    else if (maxDeviation > 0.2) { issue = 'Spike / outlier'; recommendation = 'Inspect the sensor and its signal conditioning; compare against an independent reading before acting.'; }
    else if (high) { issue = 'Elevated reading'; recommendation = 'Review the affected interval with operators and corroborate with the other RAD channels.'; }
    return { sensor, series, peer, baseline, latest, missing, high, maxDeviation, issue, recommendation };
  });
  const active = results.filter((result) => result.issue !== 'Healthy');
  const network = results.filter((result) => Number.isFinite(result.latest)).map((result) => result.latest);
  return { ...data, baselineCount, results, active, latestMedian: median(network) };
}

function renderPolyline(values, color, domain, width, height, pad) {
  const { min, max } = domain;
  const span = max - min || 1;
  const points = values.map((value, index) => {
    const x = pad.l + index / Math.max(1, values.length - 1) * (width - pad.l - pad.r);
    const y = Number.isFinite(value) ? pad.t + (max - value) / span * (height - pad.t - pad.b) : null;
    return y === null ? null : `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const segments = [];
  let current = [];
  points.forEach((point) => { if (point) current.push(point); else if (current.length) { segments.push(current); current = []; } });
  if (current.length) segments.push(current);
  return segments.map((segment) => `<polyline points="${segment.join(' ')}" fill="none" stroke="${color}" stroke-width="2.2"/>`).join('');
}

function renderChart(result, labels) {
  const width = 760, height = 290, pad = { l: 46, r: 12, t: 16, b: 32 };
  const threshold = result.baseline * 1.3;
  const all = [...result.series, ...result.peer, threshold].filter(Number.isFinite);
  const low = Math.min(...all), high = Math.max(...all);
  const margin = Math.max((high - low) * 0.13, 1);
  const domain = { min: low - margin, max: high + margin };
  const y = (value) => pad.t + (domain.max - value) / (domain.max - domain.min) * (height - pad.t - pad.b);
  const grid = [0, .25, .5, .75, 1].map((step) => {
    const value = domain.min + (domain.max - domain.min) * step;
    return `<line x1="${pad.l}" x2="${width - pad.r}" y1="${y(value)}" y2="${y(value)}" stroke="${colors.grid}"/><text x="4" y="${y(value) + 4}" fill="#aab9b3" font-size="10">${value.toFixed(1)}</text>`;
  }).join('');
  const xTicks = [0, .25, .5, .75, 1].map((step) => {
    const index = Math.round((labels.length - 1) * step);
    const x = pad.l + step * (width - pad.l - pad.r);
    return `<text x="${x}" y="${height - 9}" text-anchor="middle" fill="#aab9b3" font-size="10">${labels[index]}</text>`;
  }).join('');
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${result.sensor} readings chart">${grid}<line x1="${pad.l}" x2="${width-pad.r}" y1="${y(threshold)}" y2="${y(threshold)}" stroke="${colors.threshold}" stroke-dasharray="8 6"/><text x="${pad.l + 3}" y="${y(threshold) - 5}" fill="${colors.threshold}" font-size="10">130% of baseline</text>${renderPolyline(result.peer, colors.peer, domain, width, height, pad)}${renderPolyline(result.series, colors.raw, domain, width, height, pad)}${xTicks}</svg>`;
}

function renderCore(results = expectedSensors.map((sensor) => ({ sensor, issue: 'Awaiting data', latest: NaN }))) {
  const severity = (issue) => issue === 'Healthy' || issue === 'Awaiting data' ? '' : issue === 'Missing readings' || issue === 'Spike / outlier' || issue === 'Stuck signal' ? 'bad' : 'warn';
  $('coreSensors').innerHTML = results.map((result, index) => {
    const reading = Number.isFinite(result.latest) ? result.latest.toFixed(1) : 'awaiting data';
    return `<div class="rad-node n${index} ${severity(result.issue)}">${result.sensor}<span>${reading} · ${result.issue}</span></div>`;
  }).join('');
}

function render(report) {
  clearNotice();
  $('summary').innerHTML = [
    ['Samples', report.rows.length], ['RAD channels', report.sensors.length], ['Latest network median', report.latestMedian.toFixed(2)], ['Issues flagged', report.active.length]
  ].map(([label, value]) => `<div class="metric">${label}<b>${value}</b><span class="small">uploaded data set</span></div>`).join('');
  $('summary').classList.add('show');
  const risk = report.active.length ? 'ATTENTION' : 'SYSTEM NOMINAL';
  $('status').innerHTML = `<div class="eye" style="color:${report.active.length ? 'var(--amber)' : 'var(--mint)'}">${risk} · uploaded data</div><b>${report.active.length ? `${report.active.length} channel${report.active.length > 1 ? 's' : ''} need review` : 'No data-quality anomalies detected'}</b><span class="sub">Recommendations below are based on peer comparison and baseline thresholds. Escalate operational decisions through approved plant procedures.</span>`;
  $('status').style.borderLeftColor = report.active.length ? 'var(--amber)' : 'var(--mint)';
  $('status').classList.add('show');
  renderCore(report.results);
  $('findingsTitle').hidden = false;
  $('findings').innerHTML = report.results.map((result) => {
    const type = result.issue === 'Healthy' ? 'ok' : result.issue === 'Spike / outlier' || result.issue === 'Missing readings' ? 'bad' : 'warn';
    const meta = `Latest: ${Number.isFinite(result.latest) ? result.latest.toFixed(2) : 'missing'} · baseline: ${result.baseline.toFixed(2)}`;
    return `<article class="finding ${type}"><h3>${result.sensor} · ${result.issue}</h3><p>${result.recommendation}</p><p class="small" style="margin-top:9px">${meta}</p></article>`;
  }).join('');
  $('findings').classList.add('show');
  $('chartsTitle').hidden = false;
  $('charts').innerHTML = report.results.map((result) => `<article class="chart-card"><h3>${result.sensor}</h3><p>Latest ${Number.isFinite(result.latest) ? result.latest.toFixed(2) : 'missing'} · ${result.issue}</p>${renderChart(result, report.rows.map(row => row.label))}<div class="legend"><span><i class="dot" style="background:${colors.raw}"></i>Raw reading</span><span><i class="dot" style="background:${colors.peer}"></i>Peer estimate</span><span><i class="dot" style="background:${colors.threshold}"></i>130% baseline</span></div></article>`).join('');
  $('charts').classList.add('show');
}

function loadRows(rows, source) {
  const report = analyse(normalizeRows(rows));
  render(report);
  notice(`${source} loaded. ${report.rows.length} usable samples and ${report.sensors.length} RAD channels were analysed locally.`);
}

$('file').addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (!file) return;
  if (!window.XLSX) { notice('The Excel reader could not load. Check your connection and try again.'); return; }
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const book = XLSX.read(reader.result, { type: 'array' });
      const sheet = book.Sheets[book.SheetNames[0]];
      loadRows(XLSX.utils.sheet_to_json(sheet, { defval: '' }), file.name);
    } catch (error) { notice(error.message || 'The file could not be read.'); }
  };
  reader.onerror = () => notice('The selected file could not be read.');
  reader.readAsArrayBuffer(file);
});
$('demo').addEventListener('click', () => loadRows(demoRows(), 'Demo data'));
renderCore();
