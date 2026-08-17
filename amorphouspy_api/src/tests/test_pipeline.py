"""Tests for amorphouspy_api.pipeline helper functions and step registry."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
from amorphouspy_api.pipeline import (
    _SUBMITTERS,
    ANALYSES,
    ANALYSIS_NAMES,
    BASE_STEPS,
    STEPS,
    _accumulate_step,
    _analysis_uses_lammps,
    _merge_results,
    _run_analysis,
    _run_structural_analysis,
    submit_pipeline,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# _accumulate_step
# ---------------------------------------------------------------------------


class TestAccumulateStep:
    """Tests for _accumulate_step."""

    def test_merges_step_result_into_accumulated(self) -> None:
        """Step result is merged under its name into accumulated dict."""

        def fake_step(_sub, _cfg, _acc):  # noqa: ANN202
            return {"atoms": 100}

        result = _accumulate_step(
            step_name="structure_generation",
            step_fn=fake_step,
            submission=None,
            config=None,
            accumulated={"existing": True},
        )
        assert result == {"existing": True, "structure_generation": {"atoms": 100}}

    def test_empty_accumulated(self) -> None:
        """Works with an empty accumulated dict."""

        def fake_step(_sub, _cfg, _acc):  # noqa: ANN202
            return {"key": "value"}

        result = _accumulate_step(
            step_name="step1",
            step_fn=fake_step,
            submission=None,
            config=None,
            accumulated={},
        )
        assert result == {"step1": {"key": "value"}}


# ---------------------------------------------------------------------------
# _run_analysis
# ---------------------------------------------------------------------------


class TestRunAnalysis:
    """Tests for _run_analysis."""

    def test_wraps_result_under_step_name(self) -> None:
        """Result is wrapped as {step_name: result}."""

        def fake_analysis(_sub, _cfg, _base):  # noqa: ANN202
            return {"viscosity": 1.5}

        result = _run_analysis(
            step_name="viscosity",
            step_fn=fake_analysis,
            submission=None,
            config=None,
            base_result={"melt_quench": {}},
        )
        assert result == {"viscosity": {"viscosity": 1.5}}


class TestRunStructuralAnalysis:
    """Tests for _run_structural_analysis."""

    @patch("amorphouspy.properties.structural.all.run_structural_analysis")
    def test_applies_trajectory_storage_mode_to_sampling_history(self, mock_run_structural_analysis: MagicMock) -> None:
        """When NVT averaging is used, sampling_history follows trajectory_storage_mode."""
        mean_data = MagicMock()
        mean_data.model_dump.return_value = {"density": 2.5}
        mock_run_structural_analysis.return_value = (
            mean_data,
            None,
            3,
            [
                {
                    "positions": [[1.0], [2.0], [3.0]],
                    "cells": [[[1.0]], [[2.0]], [[3.0]]],
                    "velocities": [[0.1], [0.2], [0.3]],
                    "forces": [[0.4], [0.5], [0.6]],
                    "temperature": [300.0, 301.0, 302.0],
                }
            ],
        )

        submission = SimpleNamespace(
            potential="pmmcs",
            simulation=SimpleNamespace(
                timestep=1.0,
                n_atoms=100,
                cores=1,
                structural_analysis_trajectory_storage_mode="last_frame_drop_velocities_and_forces",
            ),
        )
        config = SimpleNamespace(n_averaging_frames=3, rdf_cutoff=10.0, bin_width=0.02, n_jobs=3)
        result = {
            "melt_quench": {"final_structure": object()},
            "structure_generation": {"potential": object()},
        }

        out = _run_structural_analysis(submission, config, result)

        assert out["density"] == 2.5
        assert out["n_averaging_frames"] == 3
        assert out["sampling_history"] == [
            {
                "positions": [[3.0]],
                "cells": [[[3.0]]],
                "temperature": [300.0, 301.0, 302.0],
            }
        ]
        assert mock_run_structural_analysis.call_args.kwargs["n_jobs"] == 3

    @patch("amorphouspy.properties.structural.all.run_structural_analysis")
    def test_omits_sampling_history_when_not_present(self, mock_run_structural_analysis: MagicMock) -> None:
        """Single-frame path should not add sampling_history to output."""
        mean_data = MagicMock()
        mean_data.model_dump.return_value = {"density": 2.5}
        mock_run_structural_analysis.return_value = (mean_data, None, 1, None)

        submission = SimpleNamespace(
            potential="pmmcs",
            simulation=SimpleNamespace(
                timestep=1.0,
                n_atoms=100,
                cores=1,
                structural_analysis_trajectory_storage_mode="last_frame_all_data",
            ),
        )
        config = SimpleNamespace(n_averaging_frames=1, rdf_cutoff=10.0, bin_width=0.02, n_jobs=1)
        result = {
            "melt_quench": {"final_structure": object()},
            "structure_generation": {"potential": object()},
        }

        out = _run_structural_analysis(submission, config, result)

        assert out["density"] == 2.5
        assert out["n_averaging_frames"] == 1
        assert "sampling_history" not in out
        assert mock_run_structural_analysis.call_args.kwargs["n_jobs"] == 1

    @patch("amorphouspy.properties.structural.all.run_structural_analysis")
    def test_no_dump_data_omits_sampling_history(self, mock_run_structural_analysis: MagicMock) -> None:
        """The structural-analysis-only no_dump_data mode should store no dump payload."""
        mean_data = MagicMock()
        mean_data.model_dump.return_value = {"density": 2.5}
        mock_run_structural_analysis.return_value = (
            mean_data,
            None,
            3,
            [
                {
                    "positions": [[1.0], [2.0], [3.0]],
                    "cells": [[[1.0]], [[2.0]], [[3.0]]],
                    "temperature": [300.0, 301.0, 302.0],
                }
            ],
        )

        submission = SimpleNamespace(
            potential="pmmcs",
            simulation=SimpleNamespace(
                timestep=1.0,
                n_atoms=100,
                cores=1,
                structural_analysis_trajectory_storage_mode="no_dump_data",
            ),
        )
        config = SimpleNamespace(n_averaging_frames=3, rdf_cutoff=10.0, bin_width=0.02, n_jobs=3)
        result = {
            "melt_quench": {"final_structure": object()},
            "structure_generation": {"potential": object()},
        }

        out = _run_structural_analysis(submission, config, result)

        assert out["density"] == 2.5
        assert out["n_averaging_frames"] == 3
        assert "sampling_history" not in out
        assert mock_run_structural_analysis.call_args.kwargs["n_jobs"] == 3

    @patch("amorphouspy.properties.structural.all.run_structural_analysis")
    def test_last_frame_mode_reduces_ndarray_sampling_history(self, mock_run_structural_analysis: MagicMock) -> None:
        """last_frame modes should also reduce ndarray-backed frame series."""
        mean_data = MagicMock()
        mean_data.model_dump.return_value = {"density": 2.5}
        mock_run_structural_analysis.return_value = (
            mean_data,
            None,
            3,
            [
                {
                    "positions": np.array([[[1.0]], [[2.0]], [[3.0]]]),
                    "cells": np.array([[[[1.0]]], [[[2.0]]], [[[3.0]]]]),
                    "temperature": [300.0, 301.0, 302.0],
                }
            ],
        )

        submission = SimpleNamespace(
            potential="pmmcs",
            simulation=SimpleNamespace(
                timestep=1.0,
                n_atoms=100,
                cores=1,
                structural_analysis_trajectory_storage_mode="last_frame_all_data",
            ),
        )
        config = SimpleNamespace(n_averaging_frames=3, rdf_cutoff=10.0, bin_width=0.02, n_jobs=3)
        result = {
            "melt_quench": {"final_structure": object()},
            "structure_generation": {"potential": object()},
        }

        out = _run_structural_analysis(submission, config, result)

        assert out["sampling_history"][0]["positions"] == [[[3.0]]]
        assert out["sampling_history"][0]["cells"] == [[[[3.0]]]]
        assert mock_run_structural_analysis.call_args.kwargs["n_jobs"] == 3


# ---------------------------------------------------------------------------
# _merge_results
# ---------------------------------------------------------------------------


class TestMergeResults:
    """Tests for _merge_results."""

    def test_merges_base_with_analyses(self) -> None:
        """Base result is merged with analysis dicts."""
        base = {"structure_generation": {}, "melt_quench": {}}
        a1 = {"viscosity": {"eta": 1.0}}
        a2 = {"cte": {"alpha": 7e-6}}

        result = _merge_results(base, analysis_1=a1, analysis_2=a2)
        assert result["structure_generation"] == {}
        assert result["melt_quench"] == {}
        assert result["viscosity"] == {"eta": 1.0}
        assert result["cte"] == {"alpha": 7e-6}

    def test_no_analyses(self) -> None:
        """Merge with no analysis results returns a copy of base."""
        base = {"a": 1, "b": 2}
        result = _merge_results(base)
        assert result == base
        assert result is not base


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------


class TestStepRegistry:
    """Tests for the STEPS/ANALYSES/BASE_STEPS registries."""

    def test_base_steps_are_subset_of_steps(self) -> None:
        """BASE_STEPS names exist in STEPS."""
        for name in BASE_STEPS:
            assert name in STEPS

    def test_analyses_exclude_base_steps(self) -> None:
        """ANALYSES does not contain base steps."""
        for name in BASE_STEPS:
            assert name not in ANALYSES

    def test_expected_analysis_keys(self) -> None:
        """Known analysis types are registered."""
        for name in ("structure_characterization", "cte", "elastic"):
            assert name in ANALYSES

    def test_submitters_contain_viscosity(self) -> None:
        """Viscosity manages its own sub-DAG via _SUBMITTERS."""
        assert "viscosity" in _SUBMITTERS
        assert callable(_SUBMITTERS["viscosity"])

    def test_analysis_names_includes_all(self) -> None:
        """ANALYSIS_NAMES is the union of ANALYSES and _SUBMITTERS."""
        assert frozenset(ANALYSES) | frozenset(_SUBMITTERS) == ANALYSIS_NAMES
        assert "viscosity" in ANALYSIS_NAMES

    def test_all_steps_are_callable(self) -> None:
        """Every registered step function is callable."""
        for fn in STEPS.values():
            assert callable(fn)


class TestAnalysisUsesLammps:
    """Tests for the ``_analysis_uses_lammps`` helper."""

    def test_structure_characterization_single_frame_is_false(self) -> None:
        """n_averaging_frames=1 does not need LAMMPS-sized resources."""
        config = SimpleNamespace(n_averaging_frames=1)
        assert _analysis_uses_lammps("structure_characterization", config) is False

    def test_structure_characterization_multi_frame_is_true(self) -> None:
        """n_averaging_frames>1 needs LAMMPS-sized resources."""
        config = SimpleNamespace(n_averaging_frames=2)
        assert _analysis_uses_lammps("structure_characterization", config) is True

    def test_other_step_names_are_false(self) -> None:
        """Non-structural-analysis steps never need LAMMPS resources from this helper."""
        config = SimpleNamespace(n_averaging_frames=2)
        assert _analysis_uses_lammps("cte", config) is False


class _DummyExecutor:
    """Minimal executor mock collecting submit kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def submit(self, fn: Callable[..., object], **kwargs: object) -> dict[str, int]:
        self.calls.append({"fn": fn, "kwargs": kwargs})
        return {"future": len(self.calls)}


class TestSubmitPipelineResources:
    """Tests for resource selection in submit_pipeline."""

    @patch("amorphouspy_api.executor.get_lammps_resource_dict")
    @patch("amorphouspy_api.executor.get_base_resource_dict")
    @patch("amorphouspy_api.executor._is_slurm")
    def test_structure_single_frame_uses_base_resources(
        self,
        mock_is_slurm: MagicMock,
        mock_base_resources: MagicMock,
        mock_lammps_resources: MagicMock,
    ) -> None:
        """structure_characterization with n_averaging_frames=1 should not request LAMMPS-sized resources."""
        mock_is_slurm.return_value = True
        mock_base_resources.return_value = {"base_token": 1}
        mock_lammps_resources.return_value = {"threads_per_core": 24}

        submission = SimpleNamespace(
            potential="pmmcs",
            simulation=SimpleNamespace(n_atoms=15_000, cores=24),
            analyses=[SimpleNamespace(type="structure_characterization", n_averaging_frames=1)],
        )
        executor = _DummyExecutor()

        submit_pipeline(executor=executor, submission=submission, cache_key="abc123")

        structure_call = next(c for c in executor.calls if c["kwargs"].get("step_name") == "structure_characterization")
        assert structure_call["kwargs"]["resource_dict"]["base_token"] == 1
        assert "threads_per_core" not in structure_call["kwargs"]["resource_dict"]

    @patch("amorphouspy_api.executor.get_lammps_resource_dict")
    @patch("amorphouspy_api.executor.get_base_resource_dict")
    @patch("amorphouspy_api.executor._is_slurm")
    def test_structure_averaging_uses_lammps_resources(
        self,
        mock_is_slurm: MagicMock,
        mock_base_resources: MagicMock,
        mock_lammps_resources: MagicMock,
    ) -> None:
        """structure_characterization with n_averaging_frames>1 should request LAMMPS-sized resources."""
        mock_is_slurm.return_value = True
        mock_base_resources.return_value = {"base_token": 1}
        mock_lammps_resources.return_value = {"threads_per_core": 24}

        submission = SimpleNamespace(
            potential="pmmcs",
            simulation=SimpleNamespace(n_atoms=15_000, cores=24),
            analyses=[SimpleNamespace(type="structure_characterization", n_averaging_frames=2)],
        )
        executor = _DummyExecutor()

        submit_pipeline(executor=executor, submission=submission, cache_key="abc123")

        structure_call = next(c for c in executor.calls if c["kwargs"].get("step_name") == "structure_characterization")
        assert structure_call["kwargs"]["resource_dict"]["threads_per_core"] == 24
