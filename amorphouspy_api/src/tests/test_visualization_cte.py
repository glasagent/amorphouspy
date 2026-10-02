"""Tests for amorphouspy_api.visualization.cte."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import ClassVar

import pytest
from amorphouspy_api.routers.jobs_helpers import _add_optional_analyses
from amorphouspy_api.visualization.cte import (
    _build_cte_convergence_plot,
    _build_cte_summary_plot,
    _build_cte_vt_plot,
    _cumulative_mean_and_uncertainty,
    prepare_cte_plots,
)
from jinja2 import Environment, FileSystemLoader

# ---------------------------------------------------------------------------
# _cumulative_mean_and_uncertainty
# ---------------------------------------------------------------------------


class TestCumulativeMeanAndUncertainty:
    """Tests for _cumulative_mean_and_uncertainty."""

    def test_single_value(self) -> None:
        """Single value gives that value as mean with zero uncertainty."""
        means, uncs = _cumulative_mean_and_uncertainty([5.0])
        assert means == [5.0]
        assert uncs == [0.0]

    def test_known_sequence(self) -> None:
        """Running mean of [2, 4] should be [2, 3]."""
        means, _ = _cumulative_mean_and_uncertainty([2.0, 4.0])
        assert means[0] == pytest.approx(2.0)
        assert means[1] == pytest.approx(3.0)

    def test_uncertainty_decreases_with_samples(self) -> None:
        """Uncertainty should generally decrease as more samples are added."""
        values = [1.0, 2.0, 1.5, 1.8, 1.9, 2.1, 1.7]
        _, uncs = _cumulative_mean_and_uncertainty(values)
        # After a few points, uncertainty should be smaller than at 2 points
        assert uncs[-1] < uncs[1]

    def test_constant_values_zero_uncertainty(self) -> None:
        """Identical values produce zero uncertainty."""
        means, uncs = _cumulative_mean_and_uncertainty([3.0, 3.0, 3.0, 3.0])
        assert all(m == pytest.approx(3.0) for m in means)
        assert all(u == pytest.approx(0.0) for u in uncs)


# ---------------------------------------------------------------------------
# _build_cte_convergence_plot
# ---------------------------------------------------------------------------


class TestBuildCTEConvergencePlot:
    """Tests for _build_cte_convergence_plot."""

    @staticmethod
    def _make_data(n_runs: int = 5) -> dict:
        return {
            "run_index": list(range(1, n_runs + 1)),
            "CTE_x": [7e-6] * n_runs,
            "CTE_y": [7e-6] * n_runs,
            "CTE_z": [7e-6] * n_runs,
        }

    def test_returns_figure_for_valid_data(self) -> None:
        """Valid data produces a Plotly figure dict."""
        fig = _build_cte_convergence_plot(self._make_data())
        assert fig is not None
        assert "data" in fig
        assert "layout" in fig

    def test_returns_none_for_empty_run_index(self) -> None:
        """Empty run_index returns None."""
        assert _build_cte_convergence_plot({"run_index": []}) is None

    def test_returns_none_for_missing_cte_components(self) -> None:
        """Missing CTE_x/y/z returns None."""
        data = {"run_index": [1, 2], "CTE_x": [1e-6, 1e-6]}
        assert _build_cte_convergence_plot(data) is None

    def test_metadata_affects_title(self) -> None:
        """Temperature from metadata appears in the plot title."""
        fig = _build_cte_convergence_plot(
            self._make_data(),
            metadata={"temperature": 300, "production_steps": 100000, "timestep": 1.0},
        )
        assert "300" in fig["layout"]["title"]["text"]

    def test_x_axis_uses_time_when_metadata_available(self) -> None:
        """With production_steps in metadata, x-axis shows simulation time."""
        fig = _build_cte_convergence_plot(
            self._make_data(),
            metadata={"production_steps": 1_000_000, "timestep": 1.0},
        )
        assert "Time" in fig["layout"]["xaxis"]["title"]["text"]


# ---------------------------------------------------------------------------
# _build_cte_summary_plot
# ---------------------------------------------------------------------------


class TestBuildCTESummaryPlot:
    """Tests for _build_cte_summary_plot."""

    def test_returns_figure_for_valid_summary(self) -> None:
        """Valid summary data produces a bar chart figure."""
        summary = {
            "CTE_x_mean": 7e-6,
            "CTE_y_mean": 7e-6,
            "CTE_z_mean": 7e-6,
            "CTE_x_uncertainty": 1e-7,
            "CTE_y_uncertainty": 1e-7,
            "CTE_z_uncertainty": 1e-7,
            "temperature": 300,
        }
        fig = _build_cte_summary_plot(summary)
        assert fig is not None
        assert fig["data"][0]["type"] == "bar"
        assert fig["data"][0]["y"] == pytest.approx([7.0])
        assert fig["layout"]["yaxis"]["title"]["text"] == "CTE (ppm/K)"

    def test_returns_none_for_missing_keys(self) -> None:
        """Missing CTE mean keys return None."""
        assert _build_cte_summary_plot({"CTE_x_mean": 7e-6}) is None

    def test_temperature_in_title(self) -> None:
        """Temperature appears in the title when provided."""
        summary = {
            "CTE_x_mean": 7e-6,
            "CTE_y_mean": 7e-6,
            "CTE_z_mean": 7e-6,
            "temperature": 500,
        }
        fig = _build_cte_summary_plot(summary)
        assert "500" in fig["layout"]["title"]["text"]


# ---------------------------------------------------------------------------
# _build_cte_vt_plot
# ---------------------------------------------------------------------------


class TestBuildCTEVTPlot:
    """Tests for _build_cte_vt_plot."""

    @staticmethod
    def _make_vt_data() -> dict:
        return {
            "data": {
                "run_index": [1, 2, 3],
                "T": [300.0, 500.0, 700.0],
                "V": [1005.0, 1055.0, 1100.0],
                "Lx": [10.0, 10.2, 10.3],
            },
            "metadata": {"temperatures": [300, 500, 700]},
        }

    def test_returns_figure_for_valid_data(self) -> None:
        """Valid V-T data produces a scatter plot."""
        fig = _build_cte_vt_plot(self._make_vt_data())
        assert fig is not None
        assert fig["data"][0]["x"] == [300.0, 500.0, 700.0]
        assert fig["data"][0]["y"] == [1005.0, 1055.0, 1100.0]

    def test_returns_none_for_insufficient_temps(self) -> None:
        """Fewer than 2 temperature points returns None."""
        data = {"data": {"T": [300.0], "V": [1000.0]}}
        assert _build_cte_vt_plot(data) is None

    def test_returns_none_without_data(self) -> None:
        """Missing ``data`` or ``T``/``V`` arrays returns None."""
        assert _build_cte_vt_plot({}) is None
        assert _build_cte_vt_plot({"data": {"T": [300.0, 500.0]}}) is None

    def test_returns_none_for_mismatched_lengths(self) -> None:
        """T and V arrays of different length are rejected."""
        data = {"data": {"T": [300.0, 500.0], "V": [1000.0]}}
        assert _build_cte_vt_plot(data) is None

    def test_sorts_by_temperature_and_skips_nan(self) -> None:
        """Points are sorted by T and non-finite values are dropped."""
        data = {"data": {"T": [700.0, 300.0, 500.0], "V": [1100.0, 1000.0, float("nan")]}}
        fig = _build_cte_vt_plot(data)
        assert fig is not None
        assert fig["data"][0]["x"] == [300.0, 700.0]
        assert fig["data"][0]["y"] == [1000.0, 1100.0]

    def test_axis_titles(self) -> None:
        """Axis titles use the dict form that current Plotly.js actually renders."""
        layout = _build_cte_vt_plot(self._make_vt_data())["layout"]
        assert layout["xaxis"]["title"]["text"] == "Temperature (K)"
        assert layout["yaxis"]["title"]["text"] == "Volume (\u00c5\u00b3)"

    def test_linear_fit_recovers_known_cte(self) -> None:
        """For exactly linear V(T), the dashed fit reproduces the data and the legend reports the input CTE."""
        alpha_v = 3.0e-5
        temps = [300.0, 400.0, 500.0, 600.0]
        vols = [1000.0 * (1 + alpha_v * (t - 300.0)) for t in temps]
        fig = _build_cte_vt_plot({"data": {"T": temps, "V": vols}})

        assert len(fig["data"]) == 2
        fit = fig["data"][1]
        assert fit["line"]["dash"] == "dash"
        assert fit["x"] == [300.0, 600.0]
        assert fit["y"] == pytest.approx([vols[0], vols[-1]])
        assert "30.00 ppm/K" in fit["name"]  # alpha_V
        assert "10.00 ppm/K" in fit["name"]  # alpha_L = alpha_V / 3
        assert "R\u00b2 = 1.0000" in fit["name"]
        assert fig["layout"]["showlegend"] is True

    def test_fit_line_is_least_squares_for_noisy_data(self) -> None:
        """With scatter, the fit line passes through the data centroid with the least-squares slope."""
        temps = [300.0, 400.0, 500.0, 600.0]
        vols = [1000.0, 1004.0, 1005.0, 1009.0]
        fit = _build_cte_vt_plot({"data": {"T": temps, "V": vols}})["data"][1]
        slope = (fit["y"][1] - fit["y"][0]) / (fit["x"][1] - fit["x"][0])
        assert slope == pytest.approx(0.028)
        assert fit["y"][0] + slope * (450.0 - 300.0) == pytest.approx(sum(vols) / 4)


# ---------------------------------------------------------------------------
# prepare_cte_plots
# ---------------------------------------------------------------------------


class TestPrepareCTEPlots:
    """Tests for the prepare_cte_plots entry point."""

    def test_fluctuations_path(self) -> None:
        """Fluctuations data produces convergence and summary plots."""
        cte_data = {
            "summary": {
                "CTE_x_mean": 7e-6,
                "CTE_y_mean": 7e-6,
                "CTE_z_mean": 7e-6,
                "CTE_x_uncertainty": 1e-7,
                "CTE_y_uncertainty": 1e-7,
                "CTE_z_uncertainty": 1e-7,
                "temperature": 300,
            },
            "data": {
                "run_index": [1, 2, 3],
                "CTE_x": [7e-6, 7.1e-6, 6.9e-6],
                "CTE_y": [7e-6, 7.2e-6, 6.8e-6],
                "CTE_z": [7e-6, 6.9e-6, 7.1e-6],
            },
            "metadata": {"temperature": 300, "production_steps": 100000, "timestep": 1.0},
        }
        plots = prepare_cte_plots(cte_data)
        assert "convergence" in plots
        assert "summary" in plots
        # Values should be valid JSON
        json.loads(plots["convergence"])
        json.loads(plots["summary"])

    def test_temperature_scan_path(self) -> None:
        """Temperature-scan data produces a volume_temperature plot."""
        cte_data = {
            "data": {"T": [300.0, 500.0, 700.0], "V": [1000.0, 1050.0, 1100.0]},
            "metadata": {"temperatures": [300, 500, 700]},
        }
        plots = prepare_cte_plots(cte_data)
        assert "volume_temperature" in plots
        json.loads(plots["volume_temperature"])

    def test_empty_data_returns_empty(self) -> None:
        """Data with no recognisable keys returns empty plots dict."""
        plots = prepare_cte_plots({})
        assert plots == {}


# ---------------------------------------------------------------------------
# Caption below the temperature-scan figure
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "amorphouspy_api" / "templates"


def _render_cte_caption(cte_request: dict) -> str:
    """Render results.html for a T-scan result and return the CTE tab caption as plain text."""
    context: dict = {"job_id": "test", "progress": {}, "tags": []}
    result_data = {"cte": {"data": {"T": [300.0, 400.0, 500.0], "V": [1000.0, 1003.0, 1006.0]}}}
    _add_optional_analyses(context, result_data, request_data={"analyses": [cte_request]})
    html = (
        Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=True)
        .get_template("results.html")
        .render(context)
    )
    tab = html.split('id="tab-cte"', 1)[1].split("</p>", 1)[0]
    return " ".join(re.sub(r"<[^>]+>", "", tab).split())


class TestCTETemperatureScanCaption:
    """The caption documents pre-equilibration and per-temperature equilibration."""

    _BASE: ClassVar[dict] = {
        "type": "cte",
        "method": "temperature_scan",
        "temperatures": [300.0, 400.0, 500.0],
        "timestep": 1.0,
    }

    def test_explicit_pre_equilibration(self) -> None:
        """Explicit settings appear with durations converted from steps to ns."""
        text = _render_cte_caption(
            {
                **self._BASE,
                "pre_equilibration_steps": 10_000,
                "pre_equilibration_temperature": 800.0,
                "equilibration_steps": 20_000,
                "production_steps": 50_000,
            }
        )
        assert "at 300, 400, 500 K" in text
        assert "pre-equilibration of 0.01 ns at 800 K was performed" in text
        assert "equilibrated for 0.02 ns, followed by a 0.05 ns production run" in text

    def test_disabled_pre_equilibration_is_stated(self) -> None:
        """pre_equilibration_steps=0 is reported explicitly rather than omitted."""
        text = _render_cte_caption({**self._BASE, "pre_equilibration_steps": 0})
        assert "No one-time pre-equilibration was performed" in text
        assert "pre-equilibration of" not in text

    def test_legacy_job_reports_core_default(self) -> None:
        """Jobs stored before the field existed ran with the 500k-step default at the highest scan T."""
        text = _render_cte_caption(dict(self._BASE))
        assert "pre-equilibration of 0.5 ns at 500 K was performed" in text
        assert "equilibrated for 0.1 ns, followed by a 0.2 ns production run" in text
