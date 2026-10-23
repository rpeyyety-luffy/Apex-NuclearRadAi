"""Standalone Streamlit page with an individual reading chart for each sensor.

Run with: streamlit run sensor_charts.py
"""
import warnings

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import nuclearrad as nr


warnings.filterwarnings("ignore")
st.set_page_config(page_title="Sensor Readings", page_icon="☢️", layout="wide")
st.title("Individual Sensor Readings")
st.caption("Each chart shows one sensor's raw readings, AI-corrected estimate, and optional truth.")

scenario = st.sidebar.selectbox("Scenario", nr.SCENARIOS, index=3)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=7)
noise_pct = st.sidebar.slider("Sensor noise (%)", 0.5, 3.0, 1.2, 0.1)
show_truth = st.sidebar.checkbox("Show true value", True)

x, truth = nr.generate(scenario, seed=int(seed), noise=noise_pct / 100)
res = nr.assess(x)
sample_count, sensor_count = x.shape
current_sample = st.slider(
    "Show readings through sample",
    min_value=0,
    max_value=sample_count - 1,
    value=sample_count - 1,
)
samples = np.arange(current_sample + 1)

for sensor_index, name in enumerate(nr.NAMES[:sensor_count]):
    baseline_raw = np.nanmean(x[:nr.WARMUP, sensor_index])
    baseline_adjusted = np.nanmean(res["y"][:nr.WARMUP, sensor_index])
    corrected_raw = res["corrected"][:, sensor_index] * baseline_raw / baseline_adjusted
    fault = res["fault"][:, sensor_index]
    trust = res["trust"][current_sample, sensor_index]

    st.subheader(name)
    st.caption(
        f"At sample {current_sample}: reading "
        f"{x[current_sample, sensor_index]:.2f} · "
        f"{fault[current_sample]} · trust {trust:.0f}%"
        if not np.isnan(x[current_sample, sensor_index])
        else f"At sample {current_sample}: reading missing · {fault[current_sample]} · trust {trust:.0f}%"
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=samples,
            y=x[: current_sample + 1, sensor_index],
            name="Raw reading",
            mode="lines",
            line=dict(color="#dc2626", width=2),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=samples,
            y=corrected_raw[: current_sample + 1],
            name="AI-corrected estimate",
            mode="lines",
            line=dict(color="#059669", width=2, dash="dash"),
        )
    )
    if show_truth:
        figure.add_trace(
            go.Scatter(
                x=samples,
                y=truth[: current_sample + 1, sensor_index],
                name="True value",
                mode="lines",
                line=dict(color="#111827", width=1.5, dash="dot"),
            )
        )

    flagged_samples = np.flatnonzero(fault[: current_sample + 1] != "Healthy")
    valid_flagged_samples = flagged_samples[
        ~np.isnan(x[flagged_samples, sensor_index])
    ]
    if valid_flagged_samples.size:
        figure.add_trace(
            go.Scatter(
                x=valid_flagged_samples,
                y=x[valid_flagged_samples, sensor_index],
                name="Flagged by AI",
                mode="markers",
                marker=dict(color="#7f1d1d", size=8, symbol="x"),
                text=fault[valid_flagged_samples],
                hovertemplate="Sample %{x}<br>%{text}<br>Reading %{y:.2f}<extra></extra>",
            )
        )

    if np.isfinite(baseline_raw):
        figure.add_hline(
            y=1.3 * baseline_raw,
            line_dash="dash",
            line_color="#d97706",
            annotation_text="130% of baseline",
            annotation_position="top left",
        )
    figure.add_vline(x=nr.WARMUP, line_dash="dot", line_color="#6b7280")
    figure.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Sample",
        yaxis_title="Reading",
        legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(figure, width="stretch", key=f"sensor_chart_{sensor_index}")
