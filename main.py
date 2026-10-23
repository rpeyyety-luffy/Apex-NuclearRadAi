"""NuclearRad AI monitoring console. Run with: streamlit run main.py"""
import warnings

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import nuclearrad as nr

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

# Scenario controls drive the same synthetic signal and assessment pipeline as the original demo.
st.sidebar.markdown('<div class="eyebrow">Plant simulation</div>', unsafe_allow_html=True)
st.sidebar.title("NuclearRad AI")
scenario = st.sidebar.selectbox("Operating scenario", nr.SCENARIOS, index=3)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=7)
noise_pct = st.sidebar.slider("Sensor noise (%)", 0.5, 3.0, 1.2, 0.1)
show_truth = st.sidebar.checkbox("Show simulated true value", False)
st.sidebar.markdown("---")
st.sidebar.caption("Training simulation only. Synthetic normalized readings; not a plant control or safety system.")

x, truth = nr.generate(scenario, seed=int(seed), noise=noise_pct / 100)
res = nr.assess(x)
sensor_count, sample_count = x.shape[1], x.shape[0]
t_now = st.slider("Simulation time · sample (1 sample = 1 minute)", nr.WARMUP, sample_count - 1,
                  min(150, sample_count - 1))
focus = st.selectbox("Inspect sensor", nr.NAMES, index=nr.FAULT_SENSOR)
k = nr.NAMES.index(focus)
verdict, action, reason = res["verdict"][t_now], res["action"][t_now], res["reason"][t_now]

st.markdown('<div class="eyebrow">Reactor instrumentation · live simulation</div>', unsafe_allow_html=True)
st.title("Core monitoring")
st.markdown('<div class="subtle">Five-channel radiation watch · peer-checked by NuclearRad AI</div>', unsafe_allow_html=True)

if "leak" in verdict.lower() and not verdict.startswith("Sensor fault"):
    status_color, status_label = "#ff746c", "ESCALATION"
elif verdict.startswith("Sensor fault") or verdict == "Uncertain":
    status_color, status_label = "#f3b969", "ATTENTION"
else:
    status_color, status_label = "#8ee2bd", "SYSTEM NOMINAL"
st.markdown(
    f'<div class="status-strip" style="border-left-color:{status_color}">'
    f'<div class="eyebrow" style="color:{status_color}">{status_label} · {scenario.upper()}</div>'
    f'<div style="font-size:18px;font-weight:700;margin:5px 0">{verdict}</div>'
    f'<div class="subtle">{reason} &nbsp;|&nbsp; Recommended action: <b>{action}</b></div></div>',
    unsafe_allow_html=True,
)

reading = x[t_now]
active_faults = int(np.count_nonzero(res["fault"][t_now] != "Healthy"))
mean_trust = float(np.nanmean(res["trust"][t_now]))
median_reading = float(np.nanmedian(reading))
m1, m2, m3, m4 = st.columns(4)
m1.metric("Simulation sample", f"{t_now:03d}", f"minute {t_now}")
m2.metric("Network median", f"{median_reading:.1f}", "normalized units")
m3.metric("Sensors flagged", f"{active_faults} / {sensor_count}", "AI fault rules")
m4.metric("Mean sensor trust", f"{mean_trust:.0f}%", "behavior confidence")

st.markdown("<br>", unsafe_allow_html=True)
overview_tab, diagnostics_tab = st.tabs(["CORE OVERVIEW", "SENSOR DIAGNOSTICS"])

with overview_tab:
    diagram_col, detail_col = st.columns([1.35, 1], gap="large")
    with diagram_col:
        st.markdown('<div class="section-label">Reactor vessel · sensor positions</div>', unsafe_allow_html=True)
        st.caption("Illustrative schematic · sensor color reflects AI assessment")
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
        for i, angle in enumerate(angles):
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
            sensor_hover.append(f"{nr.NAMES[i]}<br>Reading: {reading_text}<br>Status: {fault}<br>Trust: {trust:.0f}%")
            fig_core.add_shape(type="line", x0=1.0 * np.cos(angle), y0=1.0 * np.sin(angle),
                               x1=sx * 0.86, y1=sy * 0.86, line=dict(color="#354942", width=1))
        fig_core.add_trace(go.Scatter(
            x=sensor_x, y=sensor_y, mode="markers+text", text=nr.NAMES,
            textposition=["top center", "middle right", "bottom right", "bottom left", "middle left"],
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
        baseline = np.nanmean(x[:nr.WARMUP, k])
        delta = (selected_reading / baseline - 1) * 100 if np.isfinite(selected_reading) and baseline else np.nan
        st.metric(focus, "Missing" if np.isnan(selected_reading) else f"{selected_reading:.2f} units",
                 "no signal" if np.isnan(delta) else f"{delta:+.1f}% vs baseline")
        st.metric("AI assessment", selected_fault, f"Trust {selected_trust:.0f}%")
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
                f'<div class="sensor-label">{nr.NAMES[i]}</div>'
                f'<div class="sensor-value">{value_text}</div>'
                f'<div class="sensor-foot" style="color:{color}">{fault.upper()} · {trust:.0f}%</div></div>',
                unsafe_allow_html=True,
            )
        st.caption("Sensor values are normalized synthetic readings. Trust measures sensor behavior, not radiation level.")

with diagnostics_tab:
    st.markdown('<div class="section-label">Peer comparison · selected sensor</div>', unsafe_allow_html=True)
    tt = np.arange(sample_count)
    upto = slice(0, t_now + 1)
    baseline_raw = np.nanmean(x[:nr.WARMUP, k])
    baseline_adjusted = np.nanmean(res["y"][:nr.WARMUP, k])
    corrected = res["corrected"][:, k] * baseline_raw / baseline_adjusted
    reading_fig = go.Figure()
    for i in range(sensor_count):
        if i != k:
            reading_fig.add_trace(go.Scatter(x=tt[upto], y=x[upto, i], name=nr.NAMES[i],
                                             mode="lines", line=dict(color="#52645d", width=1)))
    reading_fig.add_trace(go.Scatter(x=tt[upto], y=x[upto, k], name=f"{focus} raw", mode="lines",
                                     line=dict(color="#ff746c", width=2.5)))
    reading_fig.add_trace(go.Scatter(x=tt[upto], y=corrected[upto], name="AI-corrected estimate",
                                     mode="lines", line=dict(color="#8ee2bd", width=2, dash="dash")))
    if show_truth:
        reading_fig.add_trace(go.Scatter(x=tt[upto], y=truth[upto, k], name="Simulated true value",
                                         mode="lines", line=dict(color="#d5c08e", width=1.5, dash="dot")))
    flags = np.flatnonzero(res["fault"][upto, k] != "Healthy")
    valid_flags = flags[~np.isnan(x[flags, k])]
    if valid_flags.size:
        reading_fig.add_trace(go.Scatter(x=valid_flags, y=x[valid_flags, k], mode="markers",
                                         name="AI flagged", marker=dict(color="#ff746c", size=8, symbol="x")))
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
        trust_fig.add_trace(go.Scatter(x=tt[upto], y=res["trust"][upto, i], name=nr.NAMES[i], mode="lines",
                                       line=dict(width=2.5 if i == k else 1.2)))
    trust_fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#c5d0ca"), xaxis=dict(title="Sample", gridcolor="#263632"),
                            yaxis=dict(range=[0, 100], title="Trust (%)", gridcolor="#263632"),
                            legend=dict(orientation="h", y=-0.35))
    st.plotly_chart(trust_fig, width="stretch")

    st.markdown('<div class="section-label">Threshold alarm vs NuclearRad AI</div>', unsafe_allow_html=True)
    summary = nr.summarise(res, truth, k)
    a, b, c = st.columns(3)
    a.metric("Threshold alarm first fires", "never" if summary["first_alarm"] is None else f"sample {summary['first_alarm']}")
    b.metric("AI first flags this sensor", "no fault" if summary["first_ai"] is None else f"sample {summary['first_ai']}")
    c.metric("Reading error · raw → corrected", f"{summary['err_before']:.1f}% → {summary['err_after']:.1f}%")

with st.expander("Assessment model"):
    st.markdown("""
- Sensors are calibrated against peer readings during the first 40 samples.
- A peer cross-check distinguishes a shared rise from a single sensor drifting away from consensus.
- Rules detect spikes, drift, stuck readings, excess noise, and dropouts.
- Trust is a 0-100 behavior-confidence score; it is not a radiation measurement.
- Shared rises remain visible as possible real leaks and are escalated, even when another sensor is faulty.
- Flagged readings are replaced with a peer-derived estimate for this demonstration only.
""")