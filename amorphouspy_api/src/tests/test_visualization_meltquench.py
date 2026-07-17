"""Tests for amorphouspy_api.visualization.meltquench."""

from __future__ import annotations

import json

from amorphouspy_api.visualization.meltquench import build_temperature_time_plot


def test_build_temperature_time_plot_excludes_structural_sampling_stage() -> None:
    """The melt-quench plot reconstructs only melt-quench protocol stages."""
    plot_json = build_temperature_time_plot(
        {
            "timestep": 1.0,
            "cooling_rate": 1e12,
            "temperature_high": 3000.0,
            "temperature_low": 300.0,
            # Structural-analysis sampling is now a separate step and should
            # not extend the melt-quench profile even if present in payloads.
            "n_averaging_frames": 500,
        }
    )

    assert plot_json is not None
    fig = json.loads(plot_json)

    x = fig["data"][0]["x"]
    y = fig["data"][0]["y"]

    # Four protocol stages -> eight endpoints in the polyline.
    assert len(x) == 8
    assert len(y) == 8
    # No heating ramp: the profile starts at the melt temperature.
    assert y[0] == 3000.0
    assert y[-1] == 300.0
