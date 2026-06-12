"""Tests for amorphouspy_api.visualization.cte."""

from __future__ import annotations

import json

import pytest
from amorphouspy_api.visualization.cte import (
    _build_cte_convergence_plot,
    _build_cte_summary_plot,
    _build_cte_vt_plot,
    _cumulative_mean_and_uncertainty,
    prepare_cte_plots,
)

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
            "01_300K": {"run1": {"V": 1000.0}, "run2": {"V": 1010.0}},
            "02_500K": {"run1": {"V": 1050.0}, "run2": {"V": 1060.0}},
            "03_700K": {"run1": {"V": 1100.0}},
        }

    def test_returns_figure_for_valid_data(self) -> None:
        """Valid V-T data produces a scatter plot."""
        fig = _build_cte_vt_plot(self._make_vt_data())
        assert fig is not None
        assert len(fig["data"][0]["x"]) == 3

    def test_returns_none_for_insufficient_temps(self) -> None:
        """Fewer than 2 temperature points returns None."""
        data = {"01_300K": {"run1": {"V": 1000.0}}}
        assert _build_cte_vt_plot(data) is None

    def test_ignores_non_temperature_keys(self) -> None:
        """Keys without the expected format are skipped."""
        data = {
            "metadata": {"something": True},
            "01_300K": {"run1": {"V": 1000.0}},
            "02_500K": {"run1": {"V": 1050.0}},
        }
        fig = _build_cte_vt_plot(data)
        assert fig is not None
        assert len(fig["data"][0]["x"]) == 2


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
            "01_300K": {"run1": {"V": 1000.0}},
            "02_500K": {"run1": {"V": 1050.0}},
            "03_700K": {"run1": {"V": 1100.0}},
        }
        plots = prepare_cte_plots(cte_data)
        assert "volume_temperature" in plots
        json.loads(plots["volume_temperature"])

    def test_empty_data_returns_empty(self) -> None:
        """Data with no recognisable keys returns empty plots dict."""
        plots = prepare_cte_plots({})
        assert plots == {}
