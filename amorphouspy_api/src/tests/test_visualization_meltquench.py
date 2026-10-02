"""Tests for amorphouspy_api.visualization.meltquench."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from amorphouspy.fabrication.meltquench_protocols import protocol_stage_schedule
from amorphouspy_api.routers.jobs_helpers import (
    _compute_total_md_steps,
    _format_stage_summary,
    _melt_quench_stages,
    build_visualization_context,
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


@pytest.mark.parametrize(
    ("rate", "expected"),
    [(1e11, "0.1"), (5e10, "0.05"), (2.5e12, "2.5"), (1e15, "1000")],
)
def test_cooling_rate_always_in_units_of_1e12(rate: float, expected: str) -> None:
    """Rates are always given as a multiple of 10^12 K/s, keeping significant decimals for slow rates."""
    fig = json.loads(build_temperature_time_plot([_stage(10, 1000.0, 300.0)], timestep_fs=1.0, cooling_rate=rate))
    assert fig["layout"]["annotations"][0]["text"] == f"Cooling: {expected} \u00d7 10\u00b9\u00b2 K/s"


def test_cooling_highlight_without_rate_has_no_annotation() -> None:
    """Without a known cooling rate the segment is highlighted but not labelled."""
    fig = json.loads(build_temperature_time_plot([_stage(10, 1000.0, 300.0)], timestep_fs=1.0))
    assert "shapes" in fig["layout"]
    assert "annotations" not in fig["layout"]


def test_longest_cooling_stage_is_highlighted() -> None:
    """With several cooling stages, the longest one (the actual quench) is highlighted."""
    stages = [_stage(10, 5000.0, 4000.0), _stage(1000, 4000.0, 300.0), _stage(5, 300.0, 200.0)]
    shape = json.loads(build_temperature_time_plot(stages, timestep_fs=1.0))["layout"]["shapes"][0]
    assert (shape["y0"], shape["y1"]) == (4000.0, 300.0)


@pytest.mark.parametrize(
    ("mq", "request_data"),
    [
        ({}, {"potential": "pmmcs"}),
        (_MQ, None),
        ({**_MQ, "cooling_rate": None}, {"potential": "pmmcs"}),
    ],
)
def test_stages_unavailable_without_required_data(mq: dict, request_data: dict | None) -> None:
    """Missing potential, cooling rate or melt temperature yields no stages rather than guessed ones."""
    assert _melt_quench_stages(mq, request_data) is None


def test_empty_stages_give_placeholders() -> None:
    """No stages: total steps is N/A and no summary is produced."""
    assert _compute_total_md_steps(None) == "N/A"
    assert _format_stage_summary([], timestep_fs=1.0) == ""


# ---------------------------------------------------------------------------
# build_visualization_context: melt-quench section end to end
# ---------------------------------------------------------------------------


def test_context_uses_recorded_stages_everywhere() -> None:
    """Plot, total MD steps and description are all derived from the same recorded stages."""
    stages = [_stage(10_000, 5000.0), _stage(2_000, 5000.0), _stage(4_700, 5000.0, 300.0), _stage(3_000, 300.0)]
    ctx = build_visualization_context(
        "job", {"melt_quench": {**_MQ, "stages": stages}}, request_data={"potential": "pmmcs"}
    )

    assert ctx["total_md_steps"] == "19,700"
    assert ctx["protocol_description"].startswith("PMMCS (Pedone")
    assert ctx["protocol_description"].endswith(_format_stage_summary(stages, timestep_fs=1.0))
    assert "(1\u2009ns)" not in ctx["protocol_description"]  # no stale hardcoded durations
    plot = json.loads(ctx["temperature_time_plot"])
    assert plot["data"][0]["x"][-1] == pytest.approx(0.0197)


def test_context_for_potential_without_static_description() -> None:
    """Potentials without a qualitative description still get the generated stage summary."""
    ctx = build_visualization_context(
        "job",
        {"melt_quench": dict(_MQ)},
        request_data={"potential": "yang2026", "simulation": {"equilibration_steps": 100}},
    )
    assert ctx["protocol_description"].startswith("Stages run: ")
    assert ctx["protocol_description"].count(";") == 6  # Yang2026: pre-equilibration + 6 stages
    assert ctx["potential"] == "YANG2026"


def test_context_without_melt_quench_has_no_plot() -> None:
    """Jobs without melt-quench results render without plot or step count."""
    ctx = build_visualization_context("job", {}, request_data={"potential": "pmmcs"})
    assert "temperature_time_plot" not in ctx
    assert ctx["total_md_steps"] == "N/A"


def test_context_compares_requested_and_simulated_composition() -> None:
    """Requested (mol%) and simulated (from formula units) compositions are listed in the requested order."""
    result_data = {
        "melt_quench": {**_MQ, "composition": {"Na2O": 25, "SiO2": 75}},
        "structure_generation": {"atoms_dict": {"total_atoms": 3000, "formula_units": {"SiO2": 740, "Na2O": 260}}},
    }
    ctx = build_visualization_context("job", result_data, request_data={"potential": "pmmcs"})
    assert ctx["composition"] == "SiO2 75.0 - Na2O 25.0"
    assert ctx["n_atoms"] == "3,000"
    assert ctx["actual_composition"] == "SiO2 74.0 - Na2O 26.0"
    assert ctx["actual_composition_items"] == [{"oxide": "SiO2", "mol": "74.0"}, {"oxide": "Na2O", "mol": "26.0"}]


def test_context_falls_back_to_defaults_when_structure_rendering_fails() -> None:
    """A broken structure-analysis payload must not take down the whole results page."""
    with patch("amorphouspy_api.visualization.structure.prepare_structure_context", side_effect=RuntimeError("boom")):
        ctx = build_visualization_context(
            "job", {"structure_characterization": {"x": 1}}, request_data={"composition": {"SiO2": 100}}
        )
    assert ctx["density"] == "N/A"
    assert ctx["structure_xyz"] == ""
    assert ctx["composition"] == "SiO2 100.0"


def test_context_includes_timings_for_known_request_hash() -> None:
    """Step timings are looked up by request hash and merged into the context."""
    with patch(
        "amorphouspy_api.visualization.timing.prepare_timing_context", return_value={"wall_time": "1 min"}
    ) as mock_timing:
        ctx = build_visualization_context("job", {}, request_hash="abc")
    mock_timing.assert_called_once_with("abc")
    assert ctx["wall_time"] == "1 min"
