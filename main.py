"""NuclearRad AI monitoring console (simulation and training prototype). Run with: streamlit run main.py

Not a safety system: synthetic data only, never connected to or controlling real equipment.
"""
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import batch_testing as bt
import charts
import evaluation as ev
import fault_detection as fd
import fault_response as fr
import model_training
import nuclearrad as nr
import sensor_health as sh

warnings.filterwarnings("ignore")
st.set_page_config(page_title="NuclearRad | Core Monitor", page_icon="☢", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Manrope:wght@400;500;600;700&display=swap');
:root { --ink:#e8eee9; --muted:#91a19c; --line:#263632; --panel:#101a18; --mint:#8ee2bd; --amber:#f3b969; --red:#ff746c; }
.stApp { background: radial-gradient(ellipse at 62% 0%, #1c302b 0%, #111b19 42%, #0b1211 100%); color:var(--ink); }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
[data-testid="stSidebar"] { background:#101a18; border-right:1px solid var(--line); }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1500px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Barlow Condensed',sans-serif !important; letter-spacing:0 !important; }
h1 { font-size:2.6rem !important; font-weight:700 !important; }
.eyebrow { color:var(--mint); font:600 11px 'IBM Plex Mono',monospace; letter-spacing:1.4px; text-transform:uppercase; }
.subtle { color:var(--muted); font-size:13px; }
.status-strip { padding:16px 20px; background:#14211e; border:1px solid #2a4039; border-left:4px solid var(--mint); border-radius:5px; margin:10px 0 20px; }
.sensor-card { background:linear-gradient(145deg,#14201d,#101816); border:1px solid #293a35; border-radius:5px; padding:13px 14px; min-height:112px; }
.sensor-label { color:#aab9b3; font:500 11px 'IBM Plex Mono',monospace; }
.sensor-value { color:#eef5ef; font:600 24px 'IBM Plex Mono',monospace; margin:8px 0 3px; }
.sensor-foot { color:#91a19c; font:500 10px 'IBM Plex Mono',monospace; }
.section-label { color:#dfe9e3; font:600 20px 'Barlow Condensed',sans-serif; margin:4px 0 2px; }
[data-testid="stMetric"] { background:#14201d; border:1px solid #293a35; border-radius:5px; padding:13px 15px; }
[data-testid="stMetricLabel"] { color:#a8b8b1; }
[data-testid="stMetricValue"] { color:#e8eee9; font-family:'IBM Plex Mono',monospace; }
div[data-baseweb="tab-list"] { gap:8px; }
button[data-baseweb="tab"] { color:#aab9b3; }
@media (max-width:700px) { .block-container { padding:1rem 1rem 2rem; } h1 { font-size:2.1rem !important; } }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ model and session state
@st.cache_resource(show_spinner="Loading the fault model (training it once if no saved model exists)...")
def get_bundle():
    return model_training.load_or_train()


def legacy_corrected(res, warmup):
    """The rule-based monitor's peer-corrected values, rescaled to each sensor's own units."""
    raw = np.nanmean(res["x"][:warmup], axis=0)
    adj = np.nanmean(res["y"][:warmup], axis=0)
    scale = np.where(np.isfinite(raw) & np.isfinite(adj) & (adj != 0), raw / np.where(adj == 0, 1, adj), 1.0)
    return res["corrected"] * scale


def set_run(cfg):
    run = nr.simulate(cfg)
    st.session_state.run = run
    st.session_state.legacy = nr.assess(run.readings, names=run.names, warmup=cfg.warmup)
    for key in ("detection", "rules", "response"):     # results of an older run are no longer valid
        st.session_state.pop(key, None)


if "run" not in st.session_state:
    set_run(nr.ScenarioConfig(scenario="Drift (upward)", seed=7))

bundle = get_bundle()
run = st.session_state.run
cfg = run.config
x, truth = run.readings, run.truth
res = st.session_state.legacy
sensor_count, sample_count = x.shape[1], x.shape[0]
W = cfg.warmup

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown('<div class="eyebrow">Plant simulation</div>', unsafe_allow_html=True)
st.sidebar.title("NuclearRad AI")
st.sidebar.markdown(f"**Current run:** {cfg.scenario}  \nseed {cfg.seed} · id `{run.run_id}`")
t_now = st.sidebar.slider("Simulation time · sample", W, sample_count - 1, min(150, sample_count - 1))
default_focus = run.fault_sensors[0] if run.fault_sensors else 0
focus = st.sidebar.selectbox("Inspect sensor", run.names, index=default_focus)
k = run.names.index(focus)
show_truth = st.sidebar.checkbox("Show healthy reference (simulation only)", False)
st.sidebar.markdown("---")
meta = bundle["meta"]
st.sidebar.caption(f"Model: Random Forest · {meta['version']} · trained {meta['trained_at']} · "
                   f"min_flagged = {bundle['min_flagged']}, min_spike = {bundle.get('min_spike')}")
if st.sidebar.button("Retrain model"):
    with st.spinner("Retraining on the training split..."):
        model_training.load_or_train(retrain=True)
    get_bundle.clear()
    for key in ("detection", "rules", "response", "batch"):
        st.session_state.pop(key, None)
    st.rerun()
st.sidebar.caption("Training simulation only. Synthetic normalized readings. Not a safety system and never "
                   "used to control real equipment.")

# ------------------------------------------------------------------ header and status strip
verdict, action, reason = res["verdict"][t_now], res["action"][t_now], res["reason"][t_now]
st.markdown('<div class="eyebrow">Reactor instrumentation · live simulation</div>', unsafe_allow_html=True)
st.title("Core monitoring")
st.markdown('<div class="subtle">Five-channel radiation watch · peer-checked by NuclearRad AI · '
            'prototype for simulation and training only</div>', unsafe_allow_html=True)

if "leak" in verdict.lower() and not verdict.startswith("Sensor fault"):
    status_color, status_label = "#ff746c", "ESCALATION"
elif verdict.startswith("Sensor fault") or verdict == "Uncertain":
    status_color, status_label = "#f3b969", "ATTENTION"
else:
    status_color, status_label = "#8ee2bd", "SYSTEM NOMINAL"
st.markdown(
    f'<div class="status-strip" style="border-left-color:{status_color}">'
    f'<div class="eyebrow" style="color:{status_color}">{status_label} · {cfg.scenario.upper()} · RULE-BASED MONITOR</div>'
    f'<div style="font-size:18px;font-weight:700;margin:5px 0">{verdict}</div>'
    f'<div class="subtle">{reason} &nbsp;|&nbsp; Recommended action: <b>{action}</b></div></div>',
    unsafe_allow_html=True,
)

tabs = st.tabs(["A · OVERVIEW", "B · SCENARIO SIMULATOR", "C · SENSOR MONITORING", "D · MODEL PREDICTION",
                "E · FAULT-TOLERANCE RESPONSE", "F · EVALUATION", "G · AUTOMATED TESTING"])

# ------------------------------------------------------------------ A. overview
with tabs[0]:
    reading = x[t_now]
    active_faults = int(np.count_nonzero(res["fault"][t_now] != "Healthy"))
    mean_trust = float(np.nanmean(res["trust"][t_now]))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Simulation sample", f"{t_now:03d}", f"of {sample_count - 1}")
    m2.metric("Network median", f"{float(np.nanmedian(reading)):.1f}", "normalized units")
    m3.metric("Sensors flagged", f"{active_faults} / {sensor_count}", "rule-based monitor")
    m4.metric("Mean sensor trust", f"{mean_trust:.0f}%", "behavior confidence")

    diagram_col, detail_col = st.columns([1.35, 1], gap="large")
    with diagram_col:
        st.markdown('<div class="section-label">Reactor vessel · sensor positions</div>', unsafe_allow_html=True)
        st.caption("Illustrative schematic · sensor color reflects the rule-based assessment")
        fig_core = go.Figure()
        fig_core.add_shape(type="circle", x0=-3.4, y0=-3.4, x1=3.4, y1=3.4,
                           line=dict(color="#354942", width=2), fillcolor="rgba(23,42,36,0.35)")
        fig_core.add_shape(type="circle", x0=-2.75, y0=-2.75, x1=2.75, y1=2.75,
                           line=dict(color="#344a42", width=1, dash="dot"), fillcolor="rgba(0,0,0,0)")
        fig_core.add_shape(type="rect", x0=-0.88, y0=-1.25, x1=0.88, y1=1.25,
                           line=dict(color="#8b9f93", width=2), fillcolor="#1a2924")
        fig_core.add_shape(type="rect", x0=-0.7, y0=-1.02, x1=0.7, y1=1.02,
                           line=dict(color="#527064", width=1), fillcolor="#20352c")
        for rod_x in np.linspace(-0.52, 0.52, 5):
            fig_core.add_shape(type="rect", x0=rod_x - 0.035, y0=-0.82, x1=rod_x + 0.035, y1=0.82,
                               line=dict(width=0), fillcolor="#d1a766")
        fig_core.add_annotation(x=0, y=0, text="REACTOR<br>CORE", showarrow=False,
                                font=dict(family="IBM Plex Mono", size=10, color="#dce8df"),
                                bgcolor="rgba(16,26,24,0.86)")
        fig_core.add_annotation(x=0, y=-1.55, text="VESSEL", showarrow=False,
                                font=dict(family="IBM Plex Mono", size=9, color="#91a19c"))

        sensor_x, sensor_y, sensor_colors, sensor_sizes, sensor_hover = [], [], [], [], []
        angles = np.deg2rad([90, 18, -54, -126, 162])
        for i, angle in enumerate(angles[:sensor_count]):
            sx, sy = 3.0 * np.cos(angle), 3.0 * np.sin(angle)
            sensor_x.append(sx)
            sensor_y.append(sy)
            fault = res["fault"][t_now, i]
            trust = res["trust"][t_now, i]
            color = "#ff746c" if fault != "Healthy" else "#f3b969" if trust < nr.UNCERTAIN_TRUST else "#8ee2bd"
            sensor_colors.append(color)
            sensor_sizes.append(25 if i == k else 19)
            current_reading = x[t_now, i]
            reading_text = "missing" if np.isnan(current_reading) else f"{current_reading:.2f}"
            sensor_hover.append(f"{run.names[i]}<br>Reading: {reading_text}<br>Status: {fault}<br>Trust: {trust:.0f}%")
            fig_core.add_shape(type="line", x0=1.0 * np.cos(angle), y0=1.0 * np.sin(angle),
                               x1=sx * 0.86, y1=sy * 0.86, line=dict(color="#354942", width=1))
        fig_core.add_trace(go.Scatter(
            x=sensor_x, y=sensor_y, mode="markers+text", text=run.names,
            textposition=["top center", "middle right", "bottom right", "bottom left", "middle left"][:sensor_count],
            textfont=dict(family="IBM Plex Mono", size=10, color="#dce8df"),
            marker=dict(size=sensor_sizes, color=sensor_colors, line=dict(color="#0b1211", width=3)),
            customdata=sensor_hover, hovertemplate="%{customdata}<extra></extra>", showlegend=False,
        ))
        fig_core.update_layout(
            height=470, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False, range=[-4.1, 4.1], fixedrange=True),
            yaxis=dict(visible=False, range=[-3.8, 3.8], scaleanchor="x", scaleratio=1, fixedrange=True),
            hoverlabel=dict(bgcolor="#17231f", bordercolor="#354942", font=dict(color="#e8eee9")),
        )
        st.plotly_chart(fig_core, width="stretch", config={"displayModeBar": False})

    with detail_col:
        st.markdown('<div class="section-label">Selected channel</div>', unsafe_allow_html=True)
        selected_fault = res["fault"][t_now, k]
        selected_trust = res["trust"][t_now, k]
        selected_reading = x[t_now, k]
        baseline = np.nanmean(x[:W, k])
        delta = (selected_reading / baseline - 1) * 100 if np.isfinite(selected_reading) and baseline else np.nan
        st.metric(focus, "Missing" if np.isnan(selected_reading) else f"{selected_reading:.2f} units",
                  "no signal" if np.isnan(delta) else f"{delta:+.1f}% vs baseline")
        st.metric("Rule-based assessment", selected_fault, f"Trust {selected_trust:.0f}%")
        if selected_fault != "Healthy":
            st.warning(f"Recommended response: {nr.ACTIONS.get(selected_fault, 'CHECK')} {focus}")
        else:
            st.info(f"{focus} is reporting. Peer consensus and trust are shown at the selected sample.")
        st.markdown('<div class="section-label">All channels</div>', unsafe_allow_html=True)
        cards = st.columns(sensor_count)
        for i, card in enumerate(cards):
            fault = res["fault"][t_now, i]
            trust = res["trust"][t_now, i]
            value = x[t_now, i]
            color = "#ff746c" if fault != "Healthy" else "#f3b969" if trust < nr.UNCERTAIN_TRUST else "#8ee2bd"
            value_text = "--" if np.isnan(value) else f"{value:.1f}"
            card.markdown(
                f'<div class="sensor-card" style="border-top:2px solid {color}">'
                f'<div class="sensor-label">{run.names[i]}</div>'
                f'<div class="sensor-value">{value_text}</div>'
                f'<div class="sensor-foot" style="color:{color}">{fault.upper()} · {trust:.0f}%</div></div>',
                unsafe_allow_html=True,
            )
        st.caption("Sensor values are normalized synthetic readings. Trust measures sensor behavior, not radiation level.")

    st.markdown('<div class="section-label">How the prototype works</div>', unsafe_allow_html=True)
    st.markdown(
        "1. **Simulate** (tab B): a reproducible run with five sensors. The healthy reference is kept apart "
        "from the readings and is used only for evaluation.\n"
        "2. **Monitor** (tab C): the original rule-based peer check (`nuclearrad.assess`).\n"
        "3. **Predict** (tab D): a Random Forest trained on simulated runs classifies every sample of every "
        "sensor from causal features (past and present readings only), then a run-level rule turns those "
        "into a verdict. The rule-based detector is shown side by side.\n"
        "4. **Respond** (tab E): a recommended response and a simulated peer-based correction that never "
        "uses the healthy reference and never trusts a sensor already detected as faulty.\n"
        "5. **Evaluate / test** (tabs F, G): metrics on the held-out test split and on fresh batch runs.")
    st.warning("This is a research and training prototype on synthetic data. It is not a safety-certified "
               "system, is not validated on real plant data, and never controls equipment.")

# ------------------------------------------------------------------ B. scenario simulator
with tabs[1]:
    st.markdown('<div class="section-label">Configure a scenario</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    scen = c1.selectbox("Scenario", list(nr.SIM_SCENARIOS), index=list(nr.SIM_SCENARIOS).index(cfg.scenario))
    seed = c1.number_input("Random seed", 0, 99_999, int(cfg.seed))
    noise_pct = c1.slider("Healthy sensor noise (%)", 0.5, 3.0, float(round(cfg.noise * 100, 1)), 0.1)
    sensor = c2.selectbox("Faulty sensor", nr.NAMES, index=cfg.fault_sensor)
    second = c2.selectbox("Second faulty sensor (Two faulty sensors only)", nr.NAMES,
                          index=cfg.second_sensor, disabled=scen != "Two faulty sensors")
    n_affected = c2.slider("Sensors affected by a real leak", 3, nr.N_SENSORS, int(cfg.n_affected),
                           disabled="Real leak" not in scen)
    start = c3.slider("Fault start (sample)", 40, nr.N_SAMPLES - 10, int(cfg.fault_start))
    until_end = c3.checkbox("Fault lasts until the end of the run", cfg.duration is None)
    duration = None if until_end else c3.slider("Fault duration (samples)", 5, nr.N_SAMPLES - start,
                                                 min(cfg.duration or 60, nr.N_SAMPLES - start))
    default, lo, hi, meaning = nr.MAGNITUDES[scen]
    if lo == hi:
        magnitude = None
        c3.caption(f"Magnitude: {meaning}")
    else:
        current = cfg.magnitude if cfg.scenario == scen and cfg.magnitude is not None else default
        magnitude = c3.slider(f"Magnitude: {meaning}", float(lo), float(hi), float(current), 0.01,
                              key=f"mag_{scen}")

    if st.button("Generate scenario", type="primary"):
        new = nr.ScenarioConfig(scenario=scen, seed=int(seed), noise=noise_pct / 100,
                                fault_sensor=nr.NAMES.index(sensor), second_sensor=nr.NAMES.index(second),
                                fault_start=int(start), duration=duration, magnitude=magnitude,
                                n_affected=int(n_affected))
        try:
            set_run(new)
            st.rerun()
        except ValueError as err:
            st.error(f"Invalid configuration: {err}")

    st.markdown('<div class="section-label">Current run</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Run id", run.run_id)
    b.metric("Expected category", run.expected_category)
    c.metric("Instrument-faulty sensors", ", ".join(run.names[i] for i in run.fault_sensors) or "none")
    d.metric("Fault window", "none" if cfg.scenario == "Normal" else f"{cfg.fault_start}–{cfg.fault_end - 1}")
    st.plotly_chart(charts.run_figure(run, show_truth=show_truth), width="stretch")
    st.caption("Shaded: configured fault window. Dotted vertical line: end of the healthy warm-up used for "
               "calibration. The healthy reference is generated separately and never fed to the model.")
    frame = run.to_long_frame()
    with st.expander("Labelled data for this run (one row per sample and sensor)"):
        st.dataframe(frame, width="stretch", height=300)
    st.download_button("Download this run as CSV", frame.to_csv(index=False).encode(),
                       file_name=f"run_{run.run_id}.csv", mime="text/csv")

# ------------------------------------------------------------------ C. sensor monitoring
with tabs[2]:
    st.markdown('<div class="section-label">Peer comparison · selected sensor</div>', unsafe_allow_html=True)
    st.caption("Rule-based monitor (the original nuclearrad.assess). Move the time slider in the sidebar.")
    tt = np.arange(sample_count)
    upto = slice(0, t_now + 1)
    corrected_all = legacy_corrected(res, W)
    baseline_raw = np.nanmean(x[:W, k])
    reading_fig = go.Figure()
    for i in range(sensor_count):
        if i != k:
            reading_fig.add_trace(go.Scatter(x=tt[upto], y=x[upto, i], name=run.names[i],
                                             mode="lines", line=dict(color="#52645d", width=1)))
    reading_fig.add_trace(go.Scatter(x=tt[upto], y=x[upto, k], name=f"{focus} raw", mode="lines",
                                     line=dict(color="#ff746c", width=2.5)))
    reading_fig.add_trace(go.Scatter(x=tt[upto], y=corrected_all[upto, k], name="Rule-based corrected estimate",
                                     mode="lines", line=dict(color="#8ee2bd", width=2, dash="dash")))
    if show_truth:
        reading_fig.add_trace(go.Scatter(x=tt[upto], y=truth[upto, k], name="Healthy reference",
                                         mode="lines", line=dict(color="#d5c08e", width=1.5, dash="dot")))
    flags = np.flatnonzero(res["fault"][upto, k] != "Healthy")
    valid_flags = flags[~np.isnan(x[flags, k])]
    if valid_flags.size:
        reading_fig.add_trace(go.Scatter(x=valid_flags, y=x[valid_flags, k], mode="markers",
                                         name="Rule flagged", marker=dict(color="#ff746c", size=8, symbol="x")))
    if np.isfinite(baseline_raw):
        reading_fig.add_hline(y=1.3 * baseline_raw, line_dash="dash", line_color="#f3b969",
                              annotation_text="130% baseline")
    reading_fig.add_vline(x=t_now, line_color="#87958f")
    reading_fig.update_layout(height=390, margin=dict(l=10, r=10, t=25, b=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#c5d0ca"), xaxis=dict(title="Sample", gridcolor="#263632"),
                              yaxis=dict(title="Normalized reading", gridcolor="#263632"),
                              legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(reading_fig, width="stretch")

    st.markdown('<div class="section-label">Sensor confidence over time</div>', unsafe_allow_html=True)
    trust_fig = go.Figure()
    for i in range(sensor_count):
        trust_fig.add_trace(go.Scatter(x=tt[upto], y=res["trust"][upto, i], name=run.names[i], mode="lines",
                                       line=dict(width=2.5 if i == k else 1.2)))
    trust_fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#c5d0ca"), xaxis=dict(title="Sample", gridcolor="#263632"),
                            yaxis=dict(range=[0, 100], title="Trust (%)", gridcolor="#263632"),
                            legend=dict(orientation="h", y=-0.35))
    st.plotly_chart(trust_fig, width="stretch")

    st.markdown('<div class="section-label">Individual sensor timelines</div>', unsafe_allow_html=True)
    chart_columns = st.columns(2, gap="large")
    for i in range(sensor_count):
        baseline_sensor = np.nanmean(x[:W, i])
        sensor_fig = go.Figure()
        sensor_fig.add_trace(go.Scatter(x=tt[upto], y=x[upto, i], name="Raw reading", mode="lines",
                                        line=dict(color="#ff746c", width=2)))
        sensor_fig.add_trace(go.Scatter(x=tt[upto], y=corrected_all[upto, i], name="Rule-based corrected",
                                        mode="lines", line=dict(color="#8ee2bd", width=2, dash="dash")))
        if show_truth:
            sensor_fig.add_trace(go.Scatter(x=tt[upto], y=truth[upto, i], name="Healthy reference", mode="lines",
                                            line=dict(color="#d5c08e", width=1.5, dash="dot")))
        sensor_flags = np.flatnonzero(res["fault"][upto, i] != "Healthy")
        sensor_flags = sensor_flags[~np.isnan(x[sensor_flags, i])]
        if sensor_flags.size:
            sensor_fig.add_trace(go.Scatter(
                x=sensor_flags, y=x[sensor_flags, i], name="Rule flagged", mode="markers",
                marker=dict(color="#ff746c", size=7, symbol="x"), text=res["fault"][sensor_flags, i],
                hovertemplate="Sample %{x}<br>%{text}<br>Reading %{y:.2f}<extra></extra>",
            ))
        if np.isfinite(baseline_sensor):
            sensor_fig.add_hline(y=1.3 * baseline_sensor, line_dash="dash", line_color="#f3b969",
                                 annotation_text="130% baseline")
        sensor_fig.add_vline(x=W, line_dash="dot", line_color="#87958f")
        sensor_fig.add_vline(x=t_now, line_color="#87958f")
        sensor_fig.update_layout(
            title=dict(text=f"{run.names[i]} · {res['fault'][t_now, i]}", font=dict(size=14, color="#e8eee9")),
            height=330, margin=dict(l=10, r=10, t=42, b=10), paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#c5d0ca"),
            xaxis=dict(title="Sample", gridcolor="#263632"),
            yaxis=dict(title="Normalized reading", gridcolor="#263632"), legend=dict(orientation="h", y=-0.28))
        with chart_columns[i % 2]:
            st.plotly_chart(sensor_fig, width="stretch", key=f"individual_sensor_{i}")

    st.markdown('<div class="section-label">Threshold alarm vs rule-based monitor</div>', unsafe_allow_html=True)
    base = np.nanmean(x[:W], axis=0)
    alarm = np.flatnonzero((np.nan_to_num(x) > 1.3 * base).any(axis=1))
    alarm = alarm[alarm >= W]
    ai = np.flatnonzero((res["fault"][:, k] != "Healthy") & (np.arange(sample_count) >= cfg.fault_start))
    a, b, c = st.columns(3)
    a.metric("Threshold alarm (130% of baseline) first fires", "never" if not alarm.size else f"sample {alarm[0]}")
    b.metric(f"Rules first flag {focus} after fault start", "no flag" if not ai.size else f"sample {ai[0]}")
    if k in run.fault_sensors:
        cm = ev.correction_metrics(run, corrected_all, sensors=[k])
        c.metric("MAE in fault window · raw → rule-corrected",
                 f"{ev.fmt(cm['raw_mae'], 2)} → {ev.fmt(cm['corrected_mae'], 2)}")
    else:
        c.metric("MAE in fault window", "N/A", f"{focus} is not faulty in this run")

# ------------------------------------------------------------------ D. model prediction
with tabs[3]:
    st.markdown('<div class="section-label">Machine-learning prediction for the current run</div>',
                unsafe_allow_html=True)
    st.caption("The model sees only the readings. The scenario name, labels and healthy reference are hidden from it.")
    if st.button("Predict", type="primary"):
        st.session_state.detection = fd.predict_run(bundle, x, W, run.names)
        st.session_state.rules = fd.rule_based_run(x, W, bundle["min_flagged"], run.names,
                                                     bundle.get("min_spike"))
        st.session_state.pop("response", None)
    det = st.session_state.get("detection")
    rules = st.session_state.get("rules")
    if det is None:
        st.info("Press **Predict** to run the model on the current scenario.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Fault present", "yes" if det["fault_present"] else "no")
        b.metric("Predicted category", det["category"])
        c.metric("Affected sensor", det["sensors"][0]["name"] if det["sensors"] else "none")
        d.metric("Model vote share (uncalibrated)", ev.fmt(det["primary_confidence"], 2, pct=True)
                 if det["primary_confidence"] is not None else "N/A")
        st.caption("Vote share = fraction of trees voting for the predicted fault type, averaged over the flagged "
                   "samples. It is not a calibrated probability that the prediction is correct.")
        for note in det["notes"]:
            st.warning(note)
        if det["sensors"]:
            st.dataframe(pd.DataFrame([{"sensor": s["name"], "fault type": s["fault"],
                                        "first detection (sample)": s["first_detection"],
                                        "flagged samples": s["flagged_samples"],
                                        "vote share (uncalibrated)": ev.fmt(s["confidence"], 2)}
                                       for s in det["sensors"]]), width="stretch", hide_index=True)

        st.markdown('<div class="section-label">Machine learning vs rule-based detector</div>', unsafe_allow_html=True)
        first_ml = fd.detection_time_for(det, run.fault_sensors)
        first_rule = fd.detection_time_for(rules, run.fault_sensors)
        start = cfg.fault_start if run.fault_sensors else None
        comparison = pd.DataFrame([
            {"": "Simulation ground truth (for comparison only)", "category": run.expected_category,
             "sensors": ", ".join(run.names[i] for i in run.fault_sensors) or "none", "detection delay": "-"},
            {"": det["source"], "category": det["category"],
             "sensors": ", ".join(s["name"] for s in det["sensors"]) or "none",
             "detection delay": ev.fmt(ev.detection_delay(first_ml, start), suffix=" samples")},
            {"": rules["source"], "category": rules["category"],
             "sensors": ", ".join(s["name"] for s in rules["sensors"]) or "none",
             "detection delay": ev.fmt(ev.detection_delay(first_rule, start), suffix=" samples")},
        ])
        st.dataframe(comparison, width="stretch", hide_index=True)
        st.caption(f"Both detectors use the same run-level rule: a sensor is reported once it has "
                   f"{bundle['min_flagged']} flagged samples, or {bundle.get('min_spike')} spike samples "
                   "(both tuned on the validation split). "
                   "N/A = not detected, or no fault to detect.")

        st.markdown(f'<div class="section-label">Per-sample class votes · {focus}</div>', unsafe_allow_html=True)
        st.plotly_chart(charts.probability_figure(det, k, nr.CLASSES), width="stretch")

        st.markdown(f'<div class="section-label">Sensor health at sample {t_now}</div>', unsafe_allow_html=True)
        health = sh.assess_health(det, x, W, t=t_now)
        st.dataframe(pd.DataFrame([{
            "sensor": h["name"], "health score": h["health_score"], "condition": h["condition"],
            "predicted fault": h["fault"], "maintenance": h["action"], "urgency": h["urgency"],
            "severity": ", ".join(f"{key} {val:.2f}" if isinstance(val, float) else f"{key} {val}"
                                  for key, val in h["severity"].items()) or "-",
        } for h in health]), width="stretch", hide_index=True,
            column_config={"health score": st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                                           format="%.0f")})
        st.plotly_chart(charts.health_figure(sh.health_timeline(det), run.names, t_now,
                                             sh.HEALTHY_SCORE, sh.FAULTY_SCORE), width="stretch")
        st.caption(f"Health score = 100 × (1 − mean model probability of an instrument fault over the last "
                   f"{sh.HEALTH_WINDOW} samples). Like the vote share it is uncalibrated. Condition: Healthy "
                   f"≥ {sh.HEALTHY_SCORE}; Watch = below that but not confirmed; a confirmed fault is Degraded, "
                   f"Faulty (< {sh.FAULTY_SCORE}) or Failed (stuck, or signal lost). Move the time slider in the "
                   "sidebar to see how health changes.")

# ------------------------------------------------------------------ E. fault-tolerance response
with tabs[4]:
    st.markdown('<div class="section-label">Recommended response and simulated correction</div>',
                unsafe_allow_html=True)
    st.caption("Everything here happens inside the simulation. No real equipment is controlled.")
    det = st.session_state.get("detection")
    if det is None:
        st.info("Run **Predict** in tab D first: the response is based on the model's detection.")
    else:
        min_peers = st.slider("Minimum trusted peers for an estimate", 1, sensor_count - 1, fr.MIN_TRUSTED_PEERS)
        if st.button("Simulate response", type="primary"):
            st.session_state.response = dict(rec=fr.recommend(det),
                                             corr=fr.apply_correction(x, det, W, min_peers=min_peers))
        resp = st.session_state.get("response")
        if resp is None:
            st.info("Press **Simulate response**.")
        else:
            rec, corr = resp["rec"], resp["corr"]
            st.markdown(f"**Detected:** {rec['category']} · sensors: {rec['sensors']}")
            for item in rec["items"]:
                st.markdown(f"- **{item['category']} → {item['action']}.** {item['text']}")

            st.markdown('<div class="section-label">Suggested maintenance (simulated work orders)</div>',
                        unsafe_allow_html=True)
            orders = [h for h in sh.assess_health(det, x, W, t=t_now) if h["action"] != "NONE"]
            if not orders:
                st.success(f"No maintenance needed at sample {t_now}: every sensor agrees with its peers.")
            for h in orders:
                text = (f"**{h['name']} → {h['action']}** ({h['urgency']}) · condition {h['condition']}, "
                        f"health {h['health_score']:.0f}/100. {h['reason']}")
                box = {"REPLACE": st.error, "REPAIR": st.warning}.get(h["action"], st.info)
                box(text)
            with st.expander("How the maintenance action is chosen"):
                for fault, rule in sh.MAINTENANCE.items():
                    st.markdown(f"- **{fault}:** {rule}")
                st.caption("The model predicts the fault type and health score; the action follows from that "
                           "type and the severity measured in the readings. Limits are illustrative values for "
                           "this simulator, not limits of any real instrument.")

            st.markdown(f"**Correction method:** {corr['method']}")
            for warning in corr["warnings"]:
                st.warning(warning)
            if corr["status"]:
                st.dataframe(pd.DataFrame(corr["status"].values()), width="stretch", hide_index=True)
                shown = st.selectbox("Sensor", [corr["status"][i]["name"] for i in corr["status"]])
                s_idx = run.names.index(shown)
                st.plotly_chart(charts.correction_figure(run, corr, s_idx, show_truth=True), width="stretch")
                if s_idx in run.fault_sensors:
                    cm = ev.correction_metrics(run, corr["corrected"], sensors=[s_idx])
                    window = "the configured fault window"
                else:
                    cm = ev.correction_metrics(run, corr["corrected"], sensors=[s_idx], window=slice(W, None))
                    window = "the whole run after warm-up (this sensor is healthy: a false alarm)"
                a, b, c, d = st.columns(4)
                a.metric("Raw MAE", ev.fmt(cm["raw_mae"], 2))
                b.metric("Corrected MAE", ev.fmt(cm["corrected_mae"], 2))
                c.metric("Improvement", ev.fmt(cm["improvement_pct"], 1, suffix="%"))
                d.metric("Compared samples", cm["n_compared"])
                st.caption(f"MAE against the healthy reference over {window}, on samples where both a raw "
                           f"reading and an estimate exist. Missing samples: {cm['gap_samples']}, estimate "
                           f"coverage {ev.fmt(cm['gap_coverage'], 1, pct=True)}, gap-estimate MAE "
                           f"{ev.fmt(cm['gap_estimate_mae'], 2)}. Improvement is N/A when raw MAE is 0 or undefined. "
                           "The healthy reference is used for this evaluation only, never for the correction.")

# ------------------------------------------------------------------ F. evaluation
with tabs[5]:
    st.markdown('<div class="section-label">Model evaluation</div>', unsafe_allow_html=True)
    st.caption(f"Model {meta['version']} trained {meta['trained_at']}. Runs per split: "
               f"train {meta['n_runs']['train']}, validation {meta['n_runs']['validation']}, "
               f"test {meta['n_runs']['test']} (separate seed ranges; no run is shared between splits). "
               "The model was fitted on train, `min_flagged` and `min_spike` were chosen on validation, and the test split was "
               "evaluated once afterwards.")

    def split_table(split):
        o, r, s = meta[split]["observation"], meta[split]["observation_rules"], meta[split]["summary"]
        rows = []
        for label, m in (("Observation · ML", o), ("Observation · rules", r),
                         ("Run · ML", s["run_detection"]), ("Run · rules", s["rule_run_detection"])):
            rows.append({"level · detector": label, "n": m["n"], "TP": m["TP"], "TN": m["TN"], "FP": m["FP"],
                         "FN": m["FN"], "accuracy": ev.fmt(m["accuracy"]), "precision": ev.fmt(m["precision"]),
                         "recall": ev.fmt(m["recall"]), "specificity": ev.fmt(m["specificity"]),
                         "FPR": ev.fmt(m["false_positive_rate"])})
        return pd.DataFrame(rows), s

    for split, label in (("test", "Held-out TEST split"), ("validation", "VALIDATION split (used for tuning)")):
        st.markdown(f"#### {label}")
        table, s = split_table(split)
        st.dataframe(table, width="stretch", hide_index=True)
        a, b, c, d = st.columns(4)
        a.metric("Run category accuracy · ML", ev.fmt(s["category_accuracy"]),
                 f"rules {ev.fmt(s['rule_category_accuracy'])}", delta_color="off")
        b.metric("Mean detection delay", ev.fmt(s["mean_detection_delay"], 1, suffix=" samples"))
        c.metric("Mean MAE raw → corrected", f"{ev.fmt(s['mean_raw_mae'], 2)} → {ev.fmt(s['mean_corrected_mae'], 2)}")
        d.metric("Correction improvement", ev.fmt(s["mean_improvement_pct"], 1, suffix="%"))
        a, b, c, _ = st.columns(4)
        a.metric("Maintenance action correct", ev.fmt(s.get("maintenance_accuracy")), "share of fault runs",
                 delta_color="off")
        b.metric("Runs with work on a healthy sensor", s.get("runs_with_unneeded_maintenance", "N/A"))
        c.metric("Lowest health · faulty vs healthy sensors",
                 f"{ev.fmt(s.get('mean_min_health_faulty'), 1)} vs {ev.fmt(s.get('mean_min_health_healthy'), 1)}")

    st.markdown("#### Test split · observation-level confusion matrix (rows = actual, columns = predicted)")
    conf = meta["test"]["observation_confusion"]
    conf_df = pd.DataFrame(conf).set_index("actual") if "actual" in conf else pd.DataFrame(conf)
    st.dataframe(conf_df, width="stretch")
    st.markdown("#### Test split · per class")
    per_class = pd.DataFrame(meta["test"]["observation_per_class"])
    for col in ("precision", "recall"):
        per_class[col] = [ev.fmt(v) for v in per_class[col]]
    st.dataframe(per_class, width="stretch", hide_index=True)
    with st.expander("Feature importance and threshold tuning"):
        st.dataframe(pd.DataFrame(meta["feature_importance"]), width="stretch", hide_index=True)
        st.dataframe(pd.DataFrame(meta["threshold_tuning"]), width="stretch", hide_index=True)
        st.caption(f"Training-set observation accuracy {ev.fmt(meta['train_observation_accuracy'])} "
                   "(optimistic: the model has seen these samples).")
    st.download_button("Download test-split results (CSV)", bundle["test_table"].to_csv(index=False).encode(),
                       file_name="test_results.csv", mime="text/csv")
    st.info("All metrics come from synthetic simulations of this prototype. They say nothing about performance "
            "on real plant instruments.")

# ------------------------------------------------------------------ G. automated testing
with tabs[6]:
    st.markdown('<div class="section-label">Automated batch testing</div>', unsafe_allow_html=True)
    st.caption("Generates fresh runs from a seed range that training, validation and test never use, then "
               "detects, responds and corrects each one and compares the outcome with its ground truth.")
    c1, c2 = st.columns(2)
    per_scenario = c1.number_input("Runs per scenario", 1, 50, 3)
    seed_base = c2.number_input("Seed base (≥ 100000)", 100_000, 9_000_000, bt.BATCH_SEED_BASE)
    if st.button("Run batch test", type="primary"):
        with st.spinner(f"Testing {per_scenario * len(nr.SIM_SCENARIOS)} runs..."):
            st.session_state.batch = bt.run_batch(bundle, int(per_scenario), int(seed_base))
    batch = st.session_state.get("batch")
    if batch is None:
        st.info("Press **Run batch test**.")
    else:
        s = batch["summary"]
        r = s["run_detection"]
        a, b, c, d = st.columns(4)
        a.metric("Runs", s["runs"], batch["split"], delta_color="off")
        b.metric("Detection accuracy / recall / FPR",
                 f"{ev.fmt(r['accuracy'], 2)} / {ev.fmt(r['recall'], 2)} / {ev.fmt(r['false_positive_rate'], 2)}")
        c.metric("Category accuracy · ML vs rules",
                 f"{ev.fmt(s['category_accuracy'], 2)} vs {ev.fmt(s['rule_category_accuracy'], 2)}")
        d.metric("Missed faults / false alarms", f"{s['missed_faults']} / {s['false_alarms']}")
        a, b, _, _ = st.columns(4)
        a.metric("Maintenance action correct", ev.fmt(s["maintenance_accuracy"], 2))
        b.metric("Runs with work on a healthy sensor", s["runs_with_unneeded_maintenance"])
        st.markdown("#### By expected category")
        st.dataframe(bt.by_category(batch["table"]), width="stretch", hide_index=True)
        st.markdown("#### Every run")
        st.dataframe(batch["table"], width="stretch", hide_index=True, height=350)
        st.download_button("Download batch results (CSV)", batch["table"].to_csv(index=False).encode(),
                           file_name="batch_results.csv", mime="text/csv")
