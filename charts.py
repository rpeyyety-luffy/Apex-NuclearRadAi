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
