"""Tests for amorphouspy_api.visualization.meltquench."""

from __future__ import annotations

import json

import pytest
from amorphouspy.fabrication.meltquench_protocols import protocol_stage_schedule
from amorphouspy_api.routers.jobs_helpers import (
    _compute_total_md_steps,
    _format_stage_summary,
    _melt_quench_stages,
)
from amorphouspy_api.visualization.meltquench import build_temperature_time_plot


def _stage(n_steps: int, t_start: float, t_end: float | None = None) -> dict:
    return {"n_steps": n_steps, "temperature_start": t_start, "temperature_end": t_start if t_end is None else t_end}


def test_plot_follows_arbitrary_stages() -> None:
    """Every stage becomes one segment with its own duration, including low-T stages before melting."""
    stages = [_stage(20_000, 300.0), _stage(100_000, 4000.0), _stage(3_700, 4000.0, 300.0), _stage(50_000, 300.0)]
    fig = json.loads(build_temperature_time_plot(stages, timestep_fs=2.0, cooling_rate=1e15))

    x, y = fig["data"][0]["x"], fig["data"][0]["y"]
    # Coinciding boundaries are merged; the instantaneous 300 K -> 4000 K jump keeps both points.
    assert y == [300.0, 300.0, 4000.0, 4000.0, 300.0, 300.0]
    # Cumulative ns with a 2 fs timestep.
    assert x == pytest.approx([0, 0.04, 0.04, 0.24, 0.2474, 0.3474])
    # The cooling stage is highlighted, not a hardcoded segment index.
    shape = fig["layout"]["shapes"][0]
    assert (shape["x0"], shape["x1"]) == pytest.approx((0.24, 0.2474))
    assert (shape["y0"], shape["y1"]) == (4000.0, 300.0)
    assert "1000 \u00d7 10\u00b9\u00b2 K/s" in fig["layout"]["annotations"][0]["text"]


def test_stage_boundaries_are_hoverable_with_stage_labels() -> None:
    """Every stage boundary is a marker whose hover text names the adjacent stages."""
    stages = [_stage(10_000, 5000.0), _stage(4_700, 5000.0, 300.0)]
    trace = json.loads(build_temperature_time_plot(stages, timestep_fs=1.0))["data"][0]
    assert "markers" in trace["mode"]
    assert "%{customdata}" in trace["hovertemplate"]
    assert trace["customdata"] == ["Start of stage 1", "End of stage 1 / Start of stage 2", "End of stage 2"]


def test_start_and_end_marked_with_diamonds() -> None:
    """The first and last points of the profile get filled diamond markers."""
    stages = [_stage(10_000, 5000.0), _stage(4_700, 5000.0, 300.0)]
    fig = json.loads(build_temperature_time_plot(stages, timestep_fs=1.0))
    ends = fig["data"][1]
    assert ends["marker"]["symbol"] == "diamond"
    assert ends["x"] == pytest.approx([0.0, 0.0147])
    assert ends["y"] == [5000.0, 300.0]


def test_plot_without_cooling_stage_has_no_highlight() -> None:
    """Profiles without a decreasing segment are drawn without cooling shape/annotation."""
    fig = json.loads(build_temperature_time_plot([_stage(1000, 300.0)], timestep_fs=1.0, cooling_rate=1e12))
    assert "shapes" not in fig["layout"]
    assert "annotations" not in fig["layout"]


def test_plot_returns_none_without_stages() -> None:
    """No stages, no plot."""
    assert build_temperature_time_plot([], timestep_fs=1.0) is None


_MQ = {"timestep": 1.0, "cooling_rate": 1e15, "temperature_high": 5000.0, "temperature_low": 300.0}


def test_recorded_stages_take_precedence() -> None:
    """Stages recorded during the run are used as-is."""
    recorded = [_stage(123, 5000.0), _stage(4700, 5000.0, 300.0)]
    assert _melt_quench_stages({**_MQ, "stages": recorded}, {"potential": "pmmcs"}) == recorded


def test_legacy_job_replays_protocol_with_requested_equilibration_steps() -> None:
    """Jobs without recorded stages are reconstructed from the request, not from fixed defaults."""
    stages = _melt_quench_stages(_MQ, {"potential": "pmmcs", "simulation": {"equilibration_steps": 5000}})
    assert [s["n_steps"] for s in stages] == [10_000, 5_000, 4_700, 5_000]
    assert _compute_total_md_steps(stages) == "24,700"


def test_legacy_job_uses_protocol_defaults_when_not_overridden() -> None:
    """Without an override, PMMCS runs 1M-step equilibration stages, and the plot must show that."""
    stages = _melt_quench_stages(_MQ, {"potential": "pmmcs", "simulation": {"equilibration_steps": None}})
    assert [s["n_steps"] for s in stages] == [10_000, 1_000_000, 4_700, 1_000_000]


@pytest.mark.parametrize("potential", ["du_teter", "du_teter_dbx_generalized", "bmp-harmonic", "yang2026", "shik"])
def test_legacy_reconstruction_supports_api_potential_names(potential: str) -> None:
    """API potential identifiers resolve to their protocol and yield its full stage list."""
    stages = _melt_quench_stages(_MQ, {"potential": potential, "simulation": {"equilibration_steps": 77}})
    expected = protocol_stage_schedule(
        potential, temperature_high=5000.0, temperature_low=300.0, cooling_rate=1e15, equilibration_steps=77
    )
    assert stages == expected


def test_legacy_reconstruction_unknown_potential_returns_none() -> None:
    """Unknown potentials yield no plot instead of a wrong one."""
    assert _melt_quench_stages(_MQ, {"potential": "unknown"}) is None


def test_stage_summary_reports_durations() -> None:
    """The description lists the duration and temperatures of every executed stage."""
    text = _format_stage_summary([_stage(10_000, 5000.0), _stage(4_700, 5000.0, 300.0)], timestep_fs=1.0)
    assert text == "Stages run: 0.01\u2009ns at 5000\u2009K; 0.0047\u2009ns 5000\u2009\u2192\u2009300\u2009K."
