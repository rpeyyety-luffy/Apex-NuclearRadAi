"""Per-sensor reading chart shared by the sensor pages."""
import numpy as np
import plotly.graph_objects as go


def sensor_figure(res, sensor_index, upto=None, truth=None, x_values=None):
    """Raw reading, AI-corrected estimate, optional truth and AI flags for one sensor.

    upto     : last sample to show (defaults to all)
    x_values : labels for the x axis, e.g. timestamps (defaults to sample numbers)
    """
    x, fault, warmup = res["x"], res["fault"][:, sensor_index], res["warmup"]
    upto = len(x) - 1 if upto is None else upto
    xs = np.arange(len(x)) if x_values is None else np.asarray(x_values)
    shown = slice(0, upto + 1)

    baseline_raw = np.nanmean(x[:warmup, sensor_index])
    baseline_adjusted = np.nanmean(res["y"][:warmup, sensor_index])
    corrected_raw = res["corrected"][:, sensor_index] * baseline_raw / baseline_adjusted

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=xs[shown], y=x[shown, sensor_index], name="Raw reading",
                                mode="lines", line=dict(color="#dc2626", width=2)))
    figure.add_trace(go.Scatter(x=xs[shown], y=corrected_raw[shown], name="AI-corrected estimate",
                                mode="lines", line=dict(color="#059669", width=2, dash="dash")))
    if truth is not None:
        figure.add_trace(go.Scatter(x=xs[shown], y=truth[shown, sensor_index], name="True value",
                                    mode="lines", line=dict(color="#111827", width=1.5, dash="dot")))

    flagged = np.flatnonzero(fault[shown] != "Healthy")
    flagged = flagged[~np.isnan(x[flagged, sensor_index])]
    if flagged.size:
        figure.add_trace(go.Scatter(
            x=xs[flagged], y=x[flagged, sensor_index], name="Flagged by AI", mode="markers",
            marker=dict(color="#7f1d1d", size=8, symbol="x"), text=fault[flagged],
            hovertemplate="%{x}<br>%{text}<br>Reading %{y:.2f}<extra></extra>",
        ))

    if np.isfinite(baseline_raw):
        figure.add_hline(y=1.3 * baseline_raw, line_dash="dash", line_color="#d97706",
                         annotation_text="130% of baseline", annotation_position="top left")
    if warmup < len(xs):
        figure.add_vline(x=xs[warmup], line_dash="dot", line_color="#6b7280")
    figure.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Sample" if x_values is None else "Time",
        yaxis_title="Reading",
        legend=dict(orientation="h", y=-0.25),
    )
    return figure


# ---------------------------------------------------------- figures for the simulator dashboard
DARK = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#c5d0ca"))
GRID = "#263632"
SENSOR_COLORS = ["#8ee2bd", "#7fb7ff", "#f3b969", "#d79bff", "#ff9f7a"]


def _dark(figure, height, y_title="Normalized reading", x_title="Sample"):
    figure.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), **DARK,
                         xaxis=dict(title=x_title, gridcolor=GRID), yaxis=dict(title=y_title, gridcolor=GRID),
                         legend=dict(orientation="h", y=-0.25))
    return figure


def run_figure(run, show_truth=False, upto=None):
    """Every sensor's observed reading, the configured fault window and (optionally) the healthy reference."""
    cfg, x = run.config, run.readings
    upto = len(x) - 1 if upto is None else upto
    t = np.arange(upto + 1)
    figure = go.Figure()
    for i, name in enumerate(run.names):
        color = SENSOR_COLORS[i % len(SENSOR_COLORS)]
        width = 2.5 if i in run.fault_sensors else 1.3
        figure.add_trace(go.Scatter(x=t, y=x[: upto + 1, i], name=name, mode="lines",
                                    line=dict(color=color, width=width)))
        if show_truth:
            figure.add_trace(go.Scatter(x=t, y=run.truth[: upto + 1, i], name=f"{name} healthy reference",
                                        mode="lines", line=dict(color=color, width=1, dash="dot"),
                                        showlegend=i == 0, legendgroup="truth"))
    if cfg.scenario != "Normal":
        figure.add_vrect(x0=cfg.fault_start, x1=cfg.fault_end - 1, fillcolor="#ff746c", opacity=0.08,
                         line_width=0, annotation_text="configured fault window", annotation_position="top left")
    figure.add_vline(x=cfg.warmup, line_dash="dot", line_color="#87958f")
    return _dark(figure, 380)


def probability_figure(detection, sensor, classes):
    """Model vote share per class over time for one sensor (uncalibrated)."""
    proba = detection["proba"][:, sensor, :]
    t = np.arange(len(proba))
    figure = go.Figure()
    for j, cls in enumerate(classes):
        if proba[:, j].max() > 0.05:
            figure.add_trace(go.Scatter(x=t, y=proba[:, j], name=cls, mode="lines", stackgroup="one"))
    return _dark(figure, 300, y_title="Vote share (uncalibrated)")


def correction_figure(run, correction, sensor, show_truth=True):
    """Raw reading, simulated correction and healthy reference for one sensor."""
    x, corrected = run.readings[:, sensor], correction["corrected"][:, sensor]
    t = np.arange(len(x))
    est = correction["estimated"][:, sensor]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=t, y=x, name="Raw reading", mode="lines", line=dict(color="#ff746c", width=2)))
    figure.add_trace(go.Scatter(x=t, y=np.where(est, corrected, np.nan), name="Peer-based estimate",
                                mode="lines+markers", marker=dict(size=3),
                                line=dict(color="#8ee2bd", width=2, dash="dash")))
    if show_truth:
        figure.add_trace(go.Scatter(x=t, y=run.truth[:, sensor], name="Healthy reference (evaluation only)",
                                    mode="lines", line=dict(color="#d5c08e", width=1.5, dash="dot")))
    gaps = np.flatnonzero(correction["unavailable"][:, sensor])
    if gaps.size:
        figure.add_trace(go.Scatter(x=gaps, y=np.full(gaps.size, np.nanmin(x)), name="Estimate unavailable",
                                    mode="markers", marker=dict(color="#f3b969", symbol="line-ns-open", size=10)))
    return _dark(figure, 360)
