"""Melt-quench visualization helpers (temperature-time diagram)."""

from __future__ import annotations

import json
from typing import Any


def build_temperature_time_plot(
    stages: list[dict[str, Any]], *, timestep_fs: float, cooling_rate: float | None = None
) -> str | None:
    """Build a Plotly temperature-vs-time JSON string from the executed melt-quench stages.

    Each stage is drawn as a linear segment from ``temperature_start`` to
    ``temperature_end`` over ``n_steps * timestep_fs``. The longest cooling
    stage is highlighted and annotated with *cooling_rate*.
    """
    if not stages:
        return None

    fs_to_ns = 1e-6
    times_ns: list[float] = []
    temps: list[float] = []
    labels: list[str] = []
    t_offset = 0.0
    cooling_segment: tuple[float, float, float, float] | None = None
    longest_cooling = 0.0

    def _add_point(t: float, temp: float, label: str) -> None:
        # Merge coinciding stage boundaries so each hover target is a single point.
        if times_ns and times_ns[-1] == t and temps[-1] == temp:
            labels[-1] = f"{labels[-1]} / {label}"
            return
        times_ns.append(t)
        temps.append(temp)
        labels.append(label)

    for i, stage in enumerate(stages, start=1):
        t0 = t_offset
        t1 = t_offset + stage["n_steps"] * timestep_fs * fs_to_ns
        temp_start = float(stage["temperature_start"])
        temp_end = float(stage["temperature_end"])
        _add_point(t0, temp_start, f"Start of stage {i}")
        _add_point(t1, temp_end, f"End of stage {i}")
        if temp_end < temp_start and t1 - t0 > longest_cooling:
            longest_cooling = t1 - t0
            cooling_segment = (t0, temp_start, t1, temp_end)
        t_offset = t1

    layout: dict[str, Any] = {
        "xaxis": {"title": {"text": "Time (ns)", "standoff": 10}},
        "yaxis": {"title": {"text": "Temperature (K)", "standoff": 10}},
        "hovermode": "closest",
        "height": 400,
        "margin": {"l": 80, "r": 20, "t": 20, "b": 60},
    }

    if cooling_segment is not None:
        x0, y0, x1, y1 = cooling_segment
        layout["shapes"] = [
            {
                "type": "line",
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "line": {"color": "#d62728", "width": 3},
                "layer": "above",
            }
        ]
        if cooling_rate:
            if cooling_rate >= 1e12:
                rate_str = f"{cooling_rate / 1e12:.0f} \u00d7 10\u00b9\u00b2 K/s"
            else:
                rate_str = f"{cooling_rate:.1e} K/s"
            layout["annotations"] = [
                {
                    "x": (x0 + x1) / 2,
                    "y": (y0 + y1) / 2,
                    "text": f"Cooling: {rate_str}",
                    "showarrow": True,
                    "arrowhead": 2,
                    "ax": 60,
                    "ay": -40,
                    "font": {"size": 12, "color": "#d62728"},
                }
            ]

    color = "#667eea"
    hovertemplate = "%{customdata}<br>Time: %{x:.4g} ns<br>Temp: %{y:.0f} K<extra></extra>"
    fig = {
        "data": [
            {
                "x": times_ns,
                "y": temps,
                "customdata": labels,
                "mode": "lines+markers",
                "line": {"width": 2.5, "color": color},
                "marker": {"size": 6, "color": color},
                "name": "Temperature",
                "showlegend": False,
                "hovertemplate": hovertemplate,
            },
            {
                "x": [times_ns[0], times_ns[-1]],
                "y": [temps[0], temps[-1]],
                "customdata": ["Start of melt-quench", "End of melt-quench"],
                "mode": "markers",
                "marker": {"symbol": "diamond", "size": 12, "color": color, "line": {"width": 1, "color": "#333"}},
                "name": "Start / end",
                "showlegend": False,
                "hovertemplate": hovertemplate,
            },
        ],
        "layout": layout,
    }
    return json.dumps(fig)
