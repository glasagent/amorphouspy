"""Tests for the generic DAG-based step-timing logic in visualization.timing."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pytest
from amorphouspy_api.visualization.timing import (
    _critical_path_seconds,
    _fold_steps,
    _format_runtime,
    _label_from_filename,
    _read_h5_node,
    _TimingNode,
    get_step_timings,
    prepare_timing_context,
)

if TYPE_CHECKING:
    from pathlib import Path


class _FakeFuture:
    """Stand-in for executorlib's ``FutureItem`` (duck-typed via ``_file_name``)."""

    def __init__(self, file_name: str) -> None:
        self._file_name = file_name


def _pickled(value: object) -> np.ndarray:
    """Pickle a value the way executorlib stores it (as a uint8 byte array)."""
    return np.frombuffer(pickle.dumps(value), dtype="uint8")


def _write_cache(
    path: Path,
    runtime: float | None,
    cores: int | None = None,
    deps: list[str] | None = None,
) -> None:
    """Write a minimal executorlib-style output cache file.

    ``deps`` are dependency *filenames* stored as ``FutureItem`` inputs.
    """
    with h5py.File(path, "w") as hdf:
        if runtime is not None:
            hdf["runtime"] = _pickled(runtime)
        if cores is not None:
            hdf["resource_dict"] = _pickled({"cores": cores})
        if deps:
            kwargs = {f"result_{i}": _FakeFuture(name) for i, name in enumerate(deps)}
            hdf["input_kwargs"] = _pickled(kwargs)


def _build_job(cache_dir: Path, request_hash: str = "abc") -> None:
    """Write a small representative pipeline: a viscosity branch parallel to melt-quench.

    structure_generation -> melt_quench           (base chain)
    structure_generation -> viscosity_2000K -> viscosity (collect)
    root merge depends on melt_quench + viscosity.
    """
    h = request_hash
    _write_cache(cache_dir / f"{h}_structure_generation_o.h5", runtime=10.0, cores=1)
    _write_cache(
        cache_dir / f"{h}_melt_quench_o.h5",
        runtime=100.0,
        cores=2,
        deps=[f"{h}_structure_generation_o.h5"],
    )
    _write_cache(
        cache_dir / f"{h}_viscosity_2000K_o.h5",
        runtime=500.0,
        cores=2,
        deps=[f"{h}_structure_generation_o.h5"],
    )
    _write_cache(
        cache_dir / f"{h}_viscosity_o.h5",
        runtime=5.0,
        cores=1,
        deps=[f"{h}_viscosity_2000K_o.h5"],
    )
    _write_cache(
        cache_dir / f"{h}_o.h5",
        runtime=2.0,
        cores=1,
        deps=[f"{h}_melt_quench_o.h5", f"{h}_viscosity_o.h5"],
    )


# ---------------------------------------------------------------------------
# _format_runtime
# ---------------------------------------------------------------------------


class TestFormatRuntime:
    """Tests for _format_runtime."""

    def test_seconds(self) -> None:
        """Sub-minute values are formatted in seconds."""
        assert _format_runtime(45) == "45s"

    def test_minutes(self) -> None:
        """Sub-hour values are formatted in minutes."""
        assert _format_runtime(90) == "1.5 min"

    def test_hours(self) -> None:
        """Multi-hour values are formatted in hours."""
        assert _format_runtime(7200) == "2.0 h"


# ---------------------------------------------------------------------------
# _label_from_filename
# ---------------------------------------------------------------------------


class TestLabelFromFilename:
    """Tests for _label_from_filename."""

    def test_root_merge_node(self) -> None:
        """The bare ``{hash}_o.h5`` file maps to the empty label."""
        assert _label_from_filename("abc_o.h5", "abc") == ""

    def test_named_step(self) -> None:
        """A named step file maps to its step label."""
        assert _label_from_filename("abc_melt_quench_o.h5", "abc") == "melt_quench"

    def test_sub_job(self) -> None:
        """A fan-out sub-job keeps its full label."""
        assert _label_from_filename("abc_viscosity_2000K_o.h5", "abc") == "viscosity_2000K"

    def test_other_hash_returns_none(self) -> None:
        """Files belonging to a different request hash are rejected."""
        assert _label_from_filename("other_melt_quench_o.h5", "abc") is None

    def test_non_output_file_returns_none(self) -> None:
        """Non ``_o.h5`` files are rejected."""
        assert _label_from_filename("abc_melt_quench_i.h5", "abc") is None


# ---------------------------------------------------------------------------
# _read_h5_node
# ---------------------------------------------------------------------------


class TestReadH5Node:
    """Tests for _read_h5_node."""

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        """A missing cache file yields None."""
        assert _read_h5_node(tmp_path / "abc_melt_quench_o.h5", "abc") is None

    def test_reads_runtime_cores_and_deps(self, tmp_path: Path) -> None:
        """Runtime, cores, and dependency labels are recovered from the cache."""
        _write_cache(tmp_path / "abc_structure_generation_o.h5", runtime=10.0, cores=1)
        _write_cache(
            tmp_path / "abc_melt_quench_o.h5",
            runtime=100.0,
            cores=4,
            deps=["abc_structure_generation_o.h5"],
        )
        node = _read_h5_node(tmp_path / "abc_melt_quench_o.h5", "abc")
        assert node is not None
        assert node.label == "melt_quench"
        assert node.runtime == 100.0
        assert node.cores == 4
        assert node.deps == ["structure_generation"]

    def test_defaults_cores_to_one(self, tmp_path: Path) -> None:
        """Cores default to one when no resource_dict is present."""
        _write_cache(tmp_path / "abc_viscosity_o.h5", runtime=10.0)
        node = _read_h5_node(tmp_path / "abc_viscosity_o.h5", "abc")
        assert node is not None
        assert node.cores == 1

    def test_non_positive_runtime_returns_none(self, tmp_path: Path) -> None:
        """A non-positive runtime is treated as missing."""
        _write_cache(tmp_path / "abc_cte_o.h5", runtime=0.0, cores=2)
        assert _read_h5_node(tmp_path / "abc_cte_o.h5", "abc") is None


# ---------------------------------------------------------------------------
# _critical_path_seconds
# ---------------------------------------------------------------------------


class TestCriticalPathSeconds:
    """Tests for _critical_path_seconds."""

    def test_parallel_branches_take_the_max(self) -> None:
        """Parallel branches overlap, so only the slowest chain counts."""
        nodes = {
            "gen": _TimingNode("gen", runtime=10.0, cores=1, deps=[]),
            "fast": _TimingNode("fast", runtime=20.0, cores=1, deps=["gen"]),
            "slow": _TimingNode("slow", runtime=200.0, cores=1, deps=["gen"]),
            "": _TimingNode("", runtime=1.0, cores=1, deps=["fast", "slow"]),
        }
        # 1 (root) + 200 (slow) + 10 (gen) = 211, not the 231 total sum.
        assert _critical_path_seconds(nodes) == pytest.approx(211.0)

    def test_edges_outside_the_map_are_ignored(self) -> None:
        """Dependencies not present in the map contribute nothing (subtree timing)."""
        nodes = {"leaf": _TimingNode("leaf", runtime=50.0, cores=1, deps=["absent"])}
        assert _critical_path_seconds(nodes) == pytest.approx(50.0)

    def test_empty_map(self) -> None:
        """No nodes gives zero."""
        assert _critical_path_seconds({}) == 0.0


# ---------------------------------------------------------------------------
# _fold_steps
# ---------------------------------------------------------------------------


class TestFoldSteps:
    """Tests for _fold_steps."""

    def test_folds_sub_jobs_into_parent(self) -> None:
        """Fan-out sub-jobs are grouped under the shorter-labelled parent step."""
        nodes = {
            "viscosity_2000K": _TimingNode("viscosity_2000K", runtime=500.0, cores=2, deps=["structure_generation"]),
            "viscosity_1500K": _TimingNode("viscosity_1500K", runtime=300.0, cores=2, deps=["structure_generation"]),
            "viscosity": _TimingNode("viscosity", runtime=5.0, cores=1, deps=["viscosity_2000K", "viscosity_1500K"]),
        }
        steps = _fold_steps(nodes)
        assert set(steps) == {"viscosity"}
        wall, core_seconds = steps["viscosity"]
        # Slowest sub-job (500) + collect (5); deps to base are outside the subtree.
        assert wall == pytest.approx(505.0)
        # (500 + 300) * 2 cores + 5 * 1.
        assert core_seconds == pytest.approx(1605.0)

    def test_simple_steps_are_not_folded(self) -> None:
        """Independent steps keep their own runtime and core-seconds."""
        nodes = {
            "melt_quench": _TimingNode("melt_quench", runtime=100.0, cores=2, deps=["structure_generation"]),
            "structure_generation": _TimingNode("structure_generation", runtime=10.0, cores=1, deps=[]),
        }
        steps = _fold_steps(nodes)
        assert steps["melt_quench"] == pytest.approx((100.0, 200.0))
        assert steps["structure_generation"] == pytest.approx((10.0, 10.0))

    def test_root_merge_node_excluded(self) -> None:
        """The empty-label merge node is not shown as a step."""
        nodes = {
            "": _TimingNode("", runtime=2.0, cores=1, deps=["melt_quench"]),
            "melt_quench": _TimingNode("melt_quench", runtime=100.0, cores=1, deps=[]),
        }
        assert set(_fold_steps(nodes)) == {"melt_quench"}


# ---------------------------------------------------------------------------
# get_step_timings / prepare_timing_context (integration over real cache files)
# ---------------------------------------------------------------------------


class TestTimingIntegration:
    """End-to-end tests reading representative cache files."""

    @staticmethod
    def _patch_cache_dir(monkeypatch: pytest.MonkeyPatch, cache_dir: Path) -> None:
        monkeypatch.setattr("amorphouspy_api.config.MELTQUENCH_PROJECT_DIR", cache_dir)

    def test_get_step_timings_folds_viscosity(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Viscosity sub-jobs are folded generically, without special casing."""
        _build_job(tmp_path)
        self._patch_cache_dir(monkeypatch, tmp_path)

        steps = get_step_timings("abc")
        assert set(steps) == {"structure_generation", "melt_quench", "viscosity"}
        # Viscosity wall = slowest sub-job (500) + collect (5).
        assert steps["viscosity"][0] == pytest.approx(505.0)

    def test_total_wall_is_critical_path_not_sum(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Total wall time follows the critical path, not the sum of all steps."""
        _build_job(tmp_path)
        self._patch_cache_dir(monkeypatch, tmp_path)

        ctx = prepare_timing_context("abc")
        # root(2) + viscosity collect(5) + viscosity_2000K(500) + structure_generation(10) = 517.
        assert ctx["total_runtime"] == _format_runtime(517.0)
        # Sum of every task's core-seconds: 10 + 200 + 1000 + 5 + 2 = 1217 core-s.
        assert ctx["total_core_hours"] == f"{1217 / 3600 * 60:.0f} min"

    def test_step_rows_ordered_and_named(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Step rows use display names in the canonical order."""
        _build_job(tmp_path)
        self._patch_cache_dir(monkeypatch, tmp_path)

        ctx = prepare_timing_context("abc")
        names = [item["name"] for item in ctx["step_timings"]]
        assert names == ["Structure Generation", "Melt-Quench", "Viscosity"]

    def test_empty_when_no_cache_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty cache dir yields an empty timing context."""
        self._patch_cache_dir(monkeypatch, tmp_path)
        assert prepare_timing_context("missing") == {}
