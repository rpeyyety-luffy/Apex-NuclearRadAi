"""NuclearRad AI - Streamlit demo.   Run:  streamlit run main.py"""
import warnings
import numpy as np
import plotly.graph_objects as go
import streamlit as st

import nuclearrad as nr

warnings.filterwarnings("ignore")
st.set_page_config(page_title="NuclearRad AI", page_icon="☢️", layout="wide")

# ----------------------------------------------------------------- sidebar
st.sidebar.title("NuclearRad AI")
st.sidebar.caption("Radiation Sensor Intelligence System")
scenario = st.sidebar.selectbox("Scenario", nr.SCENARIOS, index=3)
seed = st.sidebar.number_input("Random seed", 0, 9999, 7)
noise_pct = st.sidebar.slider("Sensor noise (%)", 0.5, 3.0, 1.2, 0.1)
show_truth = st.sidebar.checkbox("Show true value (hidden from the AI)", True)
st.sidebar.markdown("---")
st.sidebar.caption("Synthetic, normalised readings. Supports the safety system; does not replace it.")

# --------------------------------------------------------------------- run
x, truth = nr.generate(scenario, seed=int(seed), noise=noise_pct / 100)
res = nr.assess(x)
S, T = x.shape[1], x.shape[0]

t_now = st.slider("Time (sample, 1 minute each)", nr.WARMUP, T - 1, min(150, T - 1))
focus = st.selectbox("Sensor to inspect", nr.NAMES, index=nr.FAULT_SENSOR)
k = nr.NAMES.index(focus)

# ------------------------------------------------------------------ header
verdict, action, reason = res["verdict"][t_now], res["action"][t_now], res["reason"][t_now]
st.title("☢️ NuclearRad AI")
if "leak" in verdict.lower() and not verdict.startswith("Sensor fault"):
    st.error(f"**{verdict}**   ·   Action: {action}")          # real leak: always escalated
elif verdict.startswith("Sensor fault") or verdict == "Uncertain":
    st.warning(f"**{verdict}**   ·   Action: {action}")
else:
    st.success(f"**{verdict}**")
st.write(f"**Why:** {reason}")

# ------------------------------------------------------------ trust cards
st.subheader("Trust per sensor")
cols = st.columns(S)
for i, c in enumerate(cols):
    f = res["fault"][t_now, i]
    tr = res["trust"][t_now, i]
    if f != "Healthy":
        label, color = f.upper(), "#dc2626"
    elif tr < nr.UNCERTAIN_TRUST:
        label, color = "UNCERTAIN", "#d97706"
    else:
        label, color = "HEALTHY", "#059669"
    c.markdown(
        f"<div style='border:1px solid #e5e7eb;border-left:6px solid {color};"
        f"border-radius:8px;padding:10px 12px'>"
        f"<div style='font-size:13px;opacity:.7'>{nr.NAMES[i]}</div>"
        f"<div style='font-size:28px;font-weight:700;color:{color}'>{tr:.0f}%</div>"
        f"<div style='font-size:12px;font-weight:600;color:{color}'>{label}</div></div>",
        unsafe_allow_html=True)
st.caption("Trust is not radiation level. 31% means this sensor's current behaviour is only 31% trustworthy.")

# --------------------------------------------------------------- main chart
tt = np.arange(T)
upto = slice(0, t_now + 1)
base_k = np.nanmean(x[:nr.WARMUP, k])

fig = go.Figure()
for i in range(S):
    if i != k:
        fig.add_trace(go.Scatter(x=tt[upto], y=x[upto, i], name=nr.NAMES[i], mode="lines",
                                 line=dict(color="#9ca3af", width=1)))
fig.add_trace(go.Scatter(x=tt[upto], y=x[upto, k], name=f"{focus} raw", mode="lines",
                         line=dict(color="#dc2626", width=2.5)))
fig.add_trace(go.Scatter(x=tt[upto], y=res["corrected"][upto, k] * np.nanmean(x[:nr.WARMUP, k]) /
                         np.nanmean(res["y"][:nr.WARMUP, k]), name=f"{focus} corrected by AI",
                         mode="lines", line=dict(color="#059669", width=2.5, dash="dash")))
if show_truth:
    fig.add_trace(go.Scatter(x=tt[upto], y=truth[upto, k], name="True value", mode="lines",
                             line=dict(color="black", width=1.5, dash="dot")))
flag = np.where(res["fault"][upto, k] != "Healthy")[0]
if flag.size:
    fig.add_trace(go.Scatter(x=flag, y=np.nan_to_num(x[flag, k], nan=base_k), mode="markers",
                             name="Flagged by AI", marker=dict(color="#dc2626", size=7, symbol="x")))
fig.add_hline(y=1.3 * base_k, line_dash="dash", line_color="#d97706",
              annotation_text="Threshold alarm (130)", annotation_position="top left")
fig.add_vline(x=t_now, line_color="#6b7280")
fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                  xaxis=dict(title="Sample (1 minute each)", range=[0, T]),
                  yaxis=dict(title="Reading"), legend=dict(orientation="h", y=-0.25))
st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------- trust over time
fig2 = go.Figure()
for i in range(S):
    fig2.add_trace(go.Scatter(x=tt[upto], y=res["trust"][upto, i], name=nr.NAMES[i], mode="lines",
                              line=dict(width=3 if i == k else 1.2)))
fig2.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10),
                   xaxis=dict(range=[0, T], title="Sample"),
                   yaxis=dict(range=[0, 100], title="Trust (%)"), legend=dict(orientation="h", y=-0.4))
st.plotly_chart(fig2, width="stretch")

# ------------------------------------------------- alarm vs AI comparison
st.subheader("Threshold alarm vs NuclearRad AI")
m = nr.summarise(res, truth, k)
a, b, c = st.columns(3)
a.metric("Threshold alarm first fires", "never" if m["first_alarm"] is None else f"sample {m['first_alarm']}",
         help="Any reading above 1.3x its baseline. It cannot say if the cause is a leak or a broken sensor.")
b.metric("AI first flags the sensor", "no fault" if m["first_ai"] is None else f"sample {m['first_ai']}")
c.metric("Reading error: raw → corrected",
         f"{m['err_before']:.1f}% → {m['err_after']:.1f}%",
         help="RMS relative difference from the noise-free true value while the fault is active.")

with st.expander("How it works"):
    st.markdown("""
1. **Calibrate** each sensor against its peers over the first 40 healthy samples.
2. **Peer cross-check:** `r = reading / median(other sensors) - 1`. A real event moves everyone, so `r` stays near 0. A fault moves one sensor.
3. **Fault rules:** spike, drift, stuck, noise, dropout (see `nuclearrad.py`).
4. **Trust** = 1 - smoothed fault penalty, shown as 0-100% per sensor.
5. **Verdict:** if most healthy sensors rise together it is a *possible real leak* and is always escalated, never treated as a fault.
6. **Fix:** a flagged sensor is replaced by a virtual reading from its peers, so leak watch never goes blind.
""")