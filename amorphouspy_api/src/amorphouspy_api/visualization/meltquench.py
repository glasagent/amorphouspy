"""Melt-quench visualization helpers (temperature-time diagram)."""

from __future__ import annotations

import json
from typing import Any


def build_temperature_time_plot(mq_data: dict[str, Any]) -> str | None:
    """Build a Plotly temperature-vs-time JSON dict from melt-quench data.

    Uses the melt-quench protocol parameters to reconstruct the melt-quench
    temperature profile only. Post-quench structural-analysis sampling is a
    separate workflow step and is intentionally not shown here.
    """
    timestep_fs = mq_data.get("timestep", 1.0)
    cooling_rate = mq_data.get("cooling_rate")
    t_high = mq_data.get("temperature_high")
    t_low = mq_data.get("temperature_low", 300.0)

    if not all([cooling_rate, t_high]):
        return None

    assert cooling_rate is not None
    assert t_high is not None

    fs_to_ps = 1e-3
    seconds_to_fs = 1e15
    dt_ps = timestep_fs * fs_to_ps

    # Reconstruct protocol stages (no heating ramp -- the melt starts at T_high):
    # Stage 0: Langevin + nve/limit pre-equilibration at T_high (10k steps)
    pre_equilibration_steps = 10_000
    # Stage 1: Equilibration at T_high (10k steps)
    equil_high_steps = 10_000
    # Stage 2: Cool T_high -> T_low
    delta_t = t_high - t_low
    cooling_steps = int((delta_t / (timestep_fs * cooling_rate)) * seconds_to_fs)
    # Stage 3: Pressure release at T_low (10k steps)
    pressure_release_steps = 10_000

    # Build time and temperature arrays
    ps_to_ns = 1e-3
    times_ns: list[float] = []
    temps: list[float] = []
    t_offset = 0.0

    def _add_segment(n_steps: int, t_start: float, t_end: float) -> float:
        """Add a linear segment, return end time."""
        nonlocal t_offset
        t0 = t_offset
        t1 = t_offset + n_steps * dt_ps
        times_ns.append(t0 * ps_to_ns)
        temps.append(t_start)
        times_ns.append(t1 * ps_to_ns)
        temps.append(t_end)
        t_offset = t1
        return t1

    # Stage 0: Pre-equilibration at T_high
    _add_segment(pre_equilibration_steps, t_high, t_high)
    # Stage 1: Equil at T_high
    _add_segment(equil_high_steps, t_high, t_high)
    # Stage 2: Cooling
    _add_segment(cooling_steps, t_high, t_low)
    # Stage 3: Pressure release
    _add_segment(pressure_release_steps, t_low, t_low)

    # Cooling rate annotation
    cooling_mid_time = (times_ns[4] + times_ns[5]) / 2  # midpoint of cooling stage
    cooling_mid_temp = (t_high + t_low) / 2

    # Format cooling rate
    if cooling_rate >= 1e12:
        rate_str = f"{cooling_rate / 1e12:.0f} \u00d7 10\u00b9\u00b2 K/s"
    else:
        rate_str = f"{cooling_rate:.1e} K/s"

    fig = {
        "data": [
            {
                "x": times_ns,
                "y": temps,
                "mode": "lines",
                "line": {"width": 2.5, "color": "#667eea"},
                "name": "Temperature",
                "hovertemplate": "Time: %{x:.3f} ns<br>Temp: %{y:.0f} K<extra></extra>",
            }
        ],
        "layout": {
            "xaxis": {"title": {"text": "Time (ns)", "standoff": 10}},
            "yaxis": {"title": {"text": "Temperature (K)", "standoff": 10}},
            "hovermode": "closest",
            "height": 400,
            "margin": {"l": 80, "r": 20, "t": 20, "b": 60},
            "annotations": [
                {
                    "x": cooling_mid_time,
                    "y": cooling_mid_temp,
                    "text": f"Cooling: {rate_str}",
                    "showarrow": True,
                    "arrowhead": 2,
                    "ax": 60,
                    "ay": -40,
                    "font": {"size": 12, "color": "#d62728"},
                }
            ],
            "shapes": [
                {
                    "type": "line",
                    "x0": times_ns[4],
                    "y0": t_high,
                    "x1": times_ns[5],
                    "y1": t_low,
                    "line": {"color": "#d62728", "width": 3},
                    "layer": "above",
                }
            ],
        },
    }
    return json.dumps(fig)
