"""Tests for amorphouspy_api.pipeline helper functions and step registry."""

from __future__ import annotations

from amorphouspy_api.pipeline import (
    ANALYSES,
    BASE_STEPS,
    STEPS,
    _accumulate_step,
    _merge_results,
    _run_analysis,
)

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
        for name in ("structure_characterization", "viscosity", "cte", "elastic"):
            assert name in ANALYSES

    def test_all_steps_are_callable(self) -> None:
        """Every registered step function is callable."""
        for fn in STEPS.values():
            assert callable(fn)
