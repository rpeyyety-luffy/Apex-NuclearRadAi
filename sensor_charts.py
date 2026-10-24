"""Standalone Streamlit page with an individual reading chart for each sensor.

Run with: streamlit run sensor_charts.py
"""
import warnings

import numpy as np
import streamlit as st

import charts
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

for sensor_index, name in enumerate(nr.NAMES[:sensor_count]):
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
    figure = charts.sensor_figure(res, sensor_index, upto=current_sample,
                                  truth=truth if show_truth else None)
    st.plotly_chart(figure, width="stretch", key=f"sensor_chart_{sensor_index}")
