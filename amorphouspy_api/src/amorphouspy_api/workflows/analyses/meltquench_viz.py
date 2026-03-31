"""Melt-quench visualization helpers (temperature-time diagram, timing)."""

from __future__ import annotations

import json
import logging
import pickle
from typing import Any

logger = logging.getLogger(__name__)


def _format_runtime(seconds: float) -> str:
    """Format runtime as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    hours = seconds / 3600
    return f"{hours:.1f} h"


def get_step_timings(request_hash: str) -> dict[str, tuple[float, int]]:
    """Read per-step wall-clock runtimes and core counts from executorlib cache.

    Opens each step's HDF5 output file once and reads both ``runtime``
    and ``resource_dict`` in a single pass.

    Returns:
        Dict mapping step name to ``(runtime_seconds, cores)``.
    """
    import h5py

    from amorphouspy_api.config import MELTQUENCH_PROJECT_DIR

    cache_dir = MELTQUENCH_PROJECT_DIR
    step_names = ["structure_generation", "melt_quench", "structure", "viscosity", "cte", "elastic"]
    timings: dict[str, tuple[float, int]] = {}

    for name in step_names:
        h5_path = cache_dir / f"{request_hash}_{name}_o.h5"
        if not h5_path.exists():
            continue
        try:
            with h5py.File(h5_path, "r") as hdf:
                rt = pickle.loads(hdf["runtime"][()])  # noqa: S301
                if rt <= 0:
                    continue
                cores = 1
                if "resource_dict" in hdf:
                    rd = pickle.loads(hdf["resource_dict"][()])  # noqa: S301
                    cores = max(rd.get("cores", 1), rd.get("threads_per_core", 1))
                timings[name] = (rt, cores)
        except Exception:
            logger.debug("Could not read cache for %s", name)

    return timings


def prepare_timing_context(request_hash: str) -> dict[str, Any]:
    """Build template context for step timings.

    Returns:
        Dict with ``step_timings`` list of {name, runtime_formatted} dicts
        and ``total_runtime_formatted``.
    """
    raw = get_step_timings(request_hash)
    if not raw:
        return {}

    display_names = {
        "structure_generation": "Structure Generation",
        "melt_quench": "Melt-Quench",
        "structure": "Structural Analysis",
        "viscosity": "Viscosity",
        "cte": "CTE",
        "elastic": "Elastic Properties",
    }

    items = []
    for name, (seconds, _cores) in raw.items():
        items.append(
            {
                "name": display_names.get(name, name),
                "runtime": _format_runtime(seconds),
            }
        )

    total = sum(seconds for seconds, _cores in raw.values())

    # Core-hours: multiply each step's wall-clock time by its actual core count
    core_seconds = sum(seconds * cores for seconds, cores in raw.values())
    core_hours = core_seconds / 3600
    return {
        "step_timings": items,
        "total_runtime": _format_runtime(total),
        "total_core_hours": f"{core_hours:.1f} h" if core_hours >= 1 else f"{core_hours * 60:.0f} min",
    }


def build_temperature_time_plot(mq_data: dict[str, Any]) -> str | None:
    """Build a Plotly temperature-vs-time JSON dict from melt-quench data.

    Uses the protocol stage parameters to reconstruct the full T-t profile
    (the stored trajectory is only from the final equilibration stage).
    """
    timestep_fs = mq_data.get("timestep", 1.0)
    cooling_rate = mq_data.get("cooling_rate")
    heating_rate = mq_data.get("heating_rate")
    t_high = mq_data.get("temperature_high")
    t_low = mq_data.get("temperature_low", 300.0)

    if not all([cooling_rate, heating_rate, t_high]):
        return None

    fs_to_ps = 1e-3
    seconds_to_fs = 1e15
    dt_ps = timestep_fs * fs_to_ps

    # Reconstruct protocol stages (PMMCS-like):
    # Stage 1: Heat T_low → T_high
    delta_t = t_high - t_low
    heating_steps = int((delta_t / (timestep_fs * heating_rate)) * seconds_to_fs)
    # Stage 2: Equilibration at T_high (10k steps)
    equil_high_steps = 10_000
    # Stage 3: Cool T_high → T_low
    cooling_steps = int((delta_t / (timestep_fs * cooling_rate)) * seconds_to_fs)
    # Stage 4: Pressure release at T_low (10k steps)
    pressure_release_steps = 10_000
    # Stage 5: Long equilibration at T_low (100k steps)
    long_equil_steps = 100_000

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

    # Stage 1: Heating
    _add_segment(heating_steps, t_low, t_high)
    # Stage 2: Equil at T_high
    _add_segment(equil_high_steps, t_high, t_high)
    # Stage 3: Cooling
    _add_segment(cooling_steps, t_high, t_low)
    # Stage 4: Pressure release
    _add_segment(pressure_release_steps, t_low, t_low)
    # Stage 5: Long equilibration
    _add_segment(long_equil_steps, t_low, t_low)

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
