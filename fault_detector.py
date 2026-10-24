"""Upload your own sensor readings (Excel or CSV) and get per-sensor charts,
a failure-risk ranking and recommended fixes.

Run with: streamlit run fault_detector.py
"""
import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st

import charts
import nuclearrad as nr

MAX_ROWS = 2000     # the pipeline runs in the browser, so keep the most recent rows only
MIN_ROWS = 30
TIME_HINTS = ("time", "date", "timestamp", "sample")
STATUS_COLOR = {"Faulty now": "#dc2626", "High risk": "#d97706", "Watch": "#ca8a04", "Healthy": "#059669"}

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Sensor Fault Detector", page_icon="☢️", layout="wide")
st.title("Sensor Fault Detector")
st.caption("Upload readings from three or more sensors. Each sensor is checked against its peers, "
           "so a real event that moves every sensor is not mistaken for a fault.")


@st.cache_data
def template_xlsx():
    x, _ = nr.generate("Drift")
    frame = pd.DataFrame(np.round(x, 3), columns=nr.NAMES)
    frame.insert(0, "Timestamp", pd.date_range("2026-01-01", periods=len(x), freq="min"))
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False, sheet_name="Readings")
    return buffer.getvalue()


@st.cache_data
def read_table(data, filename, sheet=None):
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(data))
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet or 0)


@st.cache_data
def analyse(readings, names, warmup, horizon):
    res = nr.assess(readings, names=names, warmup=warmup)
    return res, nr.forecast(res, horizon=horizon)


with st.sidebar:
    st.header("1. Your data")
    upload = st.file_uploader("Sensor readings", type=["xlsx", "csv"])
    st.download_button("Download example file", template_xlsx(), "sensor_readings_example.xlsx",
                       help="One column per sensor, one row per reading. An optional time column comes first.")
    st.caption("Format: one row per reading time, one column per sensor. "
               "Leave a cell empty when a reading is missing.")

if upload is None:
    st.info("Upload an Excel (.xlsx) or CSV file in the sidebar, or download the example file to see the format.")
    st.stop()

data = upload.getvalue()
sheet = None
if upload.name.lower().endswith(".xlsx"):
    sheets = pd.ExcelFile(io.BytesIO(data)).sheet_names
    if len(sheets) > 1:
        sheet = st.sidebar.selectbox("Sheet", sheets)
try:
    table = read_table(data, upload.name, sheet)
except Exception as error:
    st.error(f"Could not read {upload.name}: {error}")
    st.stop()

# a column counts as a sensor when most of its cells are numbers
numeric = table.apply(pd.to_numeric, errors="coerce")
sensor_like = [c for c in table.columns
               if numeric[c].notna().mean() >= 0.5 and not str(c).lower().startswith(TIME_HINTS)]
time_options = ["(row number)"] + [c for c in table.columns if c not in sensor_like]
time_options += [c for c in table.columns if c in sensor_like and str(c).lower().startswith(TIME_HINTS)]

with st.sidebar:
    st.header("2. Columns")
    time_column = st.selectbox("Time column", time_options, index=1 if len(time_options) > 1 else 0)
    sensors = st.multiselect("Sensor columns", [c for c in table.columns if c != time_column],
                             default=[c for c in sensor_like if c != time_column])

if len(sensors) < 3:
    st.warning("Pick at least three sensor columns. Each sensor is judged against the others, so two is not enough.")
    st.stop()

readings = numeric[sensors].to_numpy(dtype=float)
x_values = None if time_column == "(row number)" else table[time_column].to_numpy()
if len(readings) > MAX_ROWS:
    st.info(f"Your file has {len(readings):,} rows; analysing the most recent {MAX_ROWS:,}.")
    readings = readings[-MAX_ROWS:]
    x_values = None if x_values is None else x_values[-MAX_ROWS:]
if len(readings) < MIN_ROWS:
    st.warning(f"Need at least {MIN_ROWS} rows of readings; this file has {len(readings)}.")
    st.stop()

with st.sidebar:
    st.header("3. Settings")
    warmup = st.slider("Healthy warm-up rows", 10, len(readings) // 2, min(nr.WARMUP, len(readings) // 4),
                       help="The first rows are used to learn each sensor's normal behaviour. "
                            "Choose a stretch where you know every sensor was working.")
    horizon = st.slider("Prediction horizon (rows ahead)", 10, 200, 30)

names = [str(s) for s in sensors]
res, outlook = analyse(readings, names, warmup, horizon)
by_name = {row["sensor"]: row for row in outlook}

# ---------------------------------------------------------------- summary
verdict, action, reason = res["verdict"][-1], res["action"][-1], res["reason"][-1]
if "leak" in verdict.lower():
    st.error(f"**{verdict}** — {reason} Recommended action: **{action}**")
elif verdict.startswith("Sensor fault") or verdict == "Uncertain":
    st.warning(f"**{verdict}** — {reason} Recommended action: **{action}**")
else:
    st.success(f"**{verdict}** — {reason}")

counts = pd.Series([row["status"] for row in outlook]).value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sensors analysed", len(names))
c2.metric("Faulty now", int(counts.get("Faulty now", 0)))
c3.metric("High risk", int(counts.get("High risk", 0)))
c4.metric("Average trust", f"{np.mean(res['trust'][-1]):.0f}%")

st.subheader("Failure risk ranking")
ranking = pd.DataFrame([{k: v for k, v in row.items() if k != "remedies"} for row in outlook])
st.dataframe(
    ranking,
    hide_index=True,
    width="stretch",
    column_config={
        "sensor": "Sensor",
        "status": "Status",
        "risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%d"),
        "likely_fault": "Likely fault",
        "trust": st.column_config.NumberColumn("Trust", format="%d%%"),
        "deviation_pct": st.column_config.NumberColumn("Off peers", format="%.2f%%"),
        "samples_to_drift_limit": "Rows to drift limit",
        "flagged_recent_pct": st.column_config.NumberColumn("Recently flagged", format="%d%%"),
        "outlook": st.column_config.TextColumn("Outlook", width="large"),
    },
)

flags = pd.DataFrame(res["fault"], columns=names)
flags.insert(0, "time" if x_values is not None else "row", x_values if x_values is not None else np.arange(len(flags)))
d1, d2 = st.columns(2)
d1.download_button("Download risk report (CSV)", ranking.to_csv(index=False), "sensor_risk_report.csv")
d2.download_button("Download per-row fault flags (CSV)", flags.to_csv(index=False), "sensor_fault_flags.csv")

# ---------------------------------------------------------- per-sensor charts
st.subheader("Sensor readings")
for sensor_index, name in enumerate(names):
    row = by_name[name]
    color = STATUS_COLOR[row["status"]]
    last = readings[-1, sensor_index]
    st.markdown(f"### {name} <span style='color:{color};font-size:0.7em'>● {row['status']}</span>",
                unsafe_allow_html=True)
    st.caption(f"Latest: reading {'missing' if np.isnan(last) else f'{last:.2f}'} · "
               f"{res['fault'][-1, sensor_index]} · trust {row['trust']}% · risk {row['risk']}/100")
    st.plotly_chart(charts.sensor_figure(res, sensor_index, x_values=x_values),
                    width="stretch", key=f"upload_chart_{sensor_index}")
    if row["status"] != "Healthy":
        with st.expander(f"Prediction and recommended fixes for {name}", expanded=row["status"] != "Watch"):
            st.write(row["outlook"])
            for remedy in row["remedies"]:
                st.markdown(f"- {remedy}")

st.caption("Decision support only. Confirm every finding with plant procedures before acting.")
