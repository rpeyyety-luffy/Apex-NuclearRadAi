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
  return { sensors, rows, secondsPerReading };
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
    return { sensor, series, peer, baseline, latest, issue, status: s.status, summary: s.summary, recommendation, maintenance: s.maintenance };
  });
  const active = results.filter((result) => result.issue !== 'Healthy');
  const network = results.filter((result) => Number.isFinite(result.latest)).map((result) => result.latest);
  return { ...data, baselineCount, results, active, latestMedian: median(network), verdict: detection.verdict, leak: detection.leak };
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
  // the 3D model (reactor3d.js) draws each monitor's status from these results
  window.coreResults = results;
  window.dispatchEvent(new CustomEvent('core-results', { detail: results }));
}

// predicted sensor health and when it should be repaired or replaced (detector.js)
function maintenanceHtml(m, secondsPerReading) {
  const tone = m.health >= 80 ? 'var(--mint)' : m.health >= 40 ? 'var(--amber)' : 'var(--red)';
  const when = (label, readings) => (m.action === 'None' ? '' : `<span>${label}: <b>${Detector.whenText(readings, secondsPerReading)}</b></span>`);
  const action = m.action === 'Replace' ? 'Replace' : m.action;
  const dates = m.action === 'Replace' || m.action === 'Inspect' ? '' : when(action, m.serviceIn) + when('Replace', m.replaceIn);
  return `<div class="health"><div class="health-top"><span>Health</span><b style="color:${tone}">${m.health}/100</b></div>`
    + `<div class="health-bar"><i style="width:${m.health}%;background:${tone}"></i></div>`
    + `<p class="health-action"><b>${m.action === 'None' ? 'No maintenance due' : `${action} · ${m.urgency}`}</b></p>`
    + (dates ? `<div class="health-when">${dates}</div>` : '')
    + (m.action === 'None' ? '' : `<p class="small" style="margin-top:6px">${m.text}</p>`) + '</div>';
}

function render(report) {
  clearNotice();
  $('summary').innerHTML = [
    ['Samples', report.rows.length], ['RAD channels', report.sensors.length], ['Latest network median', report.latestMedian.toFixed(2)], ['Issues flagged', report.active.length]
  ].map(([label, value]) => `<div class="metric">${label}<b>${value}</b><span class="small">uploaded data set</span></div>`).join('');
  $('summary').classList.add('show');
  const { level, text } = report.verdict;
  const n = report.active.length, channels = `${n} channel${n === 1 ? '' : 's'}`;
  const plus = n ? ` + ${n} faulty sensor${n === 1 ? '' : 's'}` : '';
  const leak = report.leak;
  const eventBanner = !leak ? ['var(--red)', 'PLANT EVENT · NOT A SENSOR FAULT', 'Possible real radiation rise']
    : leak.kind === 'full' ? ['var(--red)', 'FULL LEAK · MAXIMUM DANGER', `Full pressure loss: every sensor near zero${plus}`]
    : ['var(--red)', 'REAL LEAK · NOT A SENSOR FAULT', `Real leak: pressure ${leak.stable ? 'dropped and stabilised' : 'dropping'} at about ${leak.endValue.toFixed(1)}${plus}`];
  const banner = {
    event: eventBanner,
    fault: ['var(--amber)', 'SENSOR FAULT', `${channels} need${n === 1 ? 's' : ''} review`],
    watch: ['var(--amber)', 'ATTENTION', `${channels} need${n === 1 ? 's' : ''} attention`],
    ok: ['var(--mint)', 'SYSTEM NOMINAL', 'No data-quality anomalies detected'],
  }[level] || ['var(--amber)', 'ATTENTION', `${channels} need review`];
  const [color, label, headline] = banner;
  $('status').innerHTML = `<div class="eye" style="color:${color}">${label} · uploaded data</div><b>${headline}</b><span class="sub">${text} Recommendations are a training aid; escalate operational decisions through approved plant procedures.</span>`;
  $('status').style.borderLeftColor = color;
  $('status').classList.add('show');
  renderCore(report.results);
  $('findingsTitle').hidden = false;
  $('findings').innerHTML = report.results.map((result) => {
    const type = result.status === 'Healthy' ? 'ok' : result.status === 'Faulty' ? 'bad' : 'warn';
    const meta = `Latest: ${Number.isFinite(result.latest) ? result.latest.toFixed(2) : 'missing'} · baseline: ${result.baseline.toFixed(2)}`;
    const advice = result.status === 'Healthy' ? '' : `<p style="margin-top:7px">${result.recommendation}</p>`;
    return `<article class="finding ${type}"><h3>${result.sensor} · ${result.issue}</h3><p>${result.summary}</p>${maintenanceHtml(result.maintenance, report.secondsPerReading)}${advice}<p class="small" style="margin-top:9px">${meta}</p></article>`;
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
if (location.hash === '#demo') loadRows(demoRows(), 'Demo data');
