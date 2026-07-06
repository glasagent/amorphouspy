"""Tests for dynamic LAMMPS core selection in amorphouspy_api.executor."""

from __future__ import annotations

import pytest
from amorphouspy.lammps.potentials.potential import DEFAULT_MIN_ATOMS_PER_CORE, get_min_atoms_per_core
from amorphouspy_api.executor import (
    compute_lammps_cores,
    get_lammps_server_kwargs,
    get_max_cores,
)


class TestGetMinAtomsPerCore:
    """Tests for get_min_atoms_per_core."""

    def test_known_potential(self) -> None:
        """A potential declaring MIN_ATOMS_PER_CORE returns its value."""
        assert get_min_atoms_per_core("pmmcs") == 500

    def test_case_insensitive(self) -> None:
        """Lookup is case-insensitive."""
        assert get_min_atoms_per_core("PMMCS") == get_min_atoms_per_core("pmmcs")

    def test_unknown_potential_warns_and_uses_default(self) -> None:
        """An unknown potential warns and falls back to the default minimum."""
        with pytest.warns(UserWarning, match="Unknown potential"):
            assert get_min_atoms_per_core("not_a_real_potential") == DEFAULT_MIN_ATOMS_PER_CORE

    def test_missing_attribute_warns_and_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A potential without MIN_ATOMS_PER_CORE warns and falls back to the default."""
        from amorphouspy.lammps.potentials import potential as potential_mod

        mod = potential_mod._POTENTIAL_MODULES["pmmcs"]
        monkeypatch.delattr(mod, "MIN_ATOMS_PER_CORE", raising=False)
        with pytest.warns(UserWarning, match="does not define MIN_ATOMS_PER_CORE"):
            assert get_min_atoms_per_core("pmmcs") == DEFAULT_MIN_ATOMS_PER_CORE


class TestGetMaxCores:
    """Tests for get_max_cores."""

    def test_reads_lammps_max_cores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LAMMPS_MAX_CORES is read when set."""
        monkeypatch.setenv("LAMMPS_MAX_CORES", "16")
        assert get_max_cores() == 16

    def test_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Defaults to 4 when LAMMPS_MAX_CORES is not set."""
        monkeypatch.delenv("LAMMPS_MAX_CORES", raising=False)
        assert get_max_cores() == 4


class TestComputeLammpsCores:
    """Tests for compute_lammps_cores."""

    def test_uses_max_when_workload_is_large(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Large workloads use the full system maximum."""
        monkeypatch.setenv("LAMMPS_MAX_CORES", "8")
        # 8 cores * 500 atoms/core = 4000; 100000 atoms easily saturates them.
        assert compute_lammps_cores("pmmcs", 100_000) == 8

    def test_scales_down_for_small_workload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Small workloads scale down below the system maximum."""
        monkeypatch.setenv("LAMMPS_MAX_CORES", "16")
        # 6000 atoms / 500 min-atoms-per-core = 12 efficient cores, below the max.
        assert compute_lammps_cores("pmmcs", 6000) == 12

    def test_floor_keeps_efficiency_above_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Flooring keeps the realised atoms/core at or above the minimum."""
        monkeypatch.setenv("LAMMPS_MAX_CORES", "16")
        # 5750 / 500 floors to 11 cores -> ~523 atoms/core (>= minimum).
        assert compute_lammps_cores("pmmcs", 5750) == 11

    def test_minimum_one_core_for_tiny_workload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """At least one core is used even for tiny systems."""
        monkeypatch.setenv("LAMMPS_MAX_CORES", "8")
        assert compute_lammps_cores("pmmcs", 100) == 1

    def test_never_exceeds_max(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The core count never exceeds the system maximum."""
        monkeypatch.setenv("LAMMPS_MAX_CORES", "2")
        assert compute_lammps_cores("pmmcs", 1_000_000) == 2

    def test_explicit_override_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit cores override bypasses auto-selection."""
        monkeypatch.setenv("LAMMPS_MAX_CORES", "16")
        assert compute_lammps_cores("pmmcs", 100, cores=6) == 6

    def test_override_clamped_to_at_least_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-positive override is clamped to one core."""
        monkeypatch.setenv("LAMMPS_MAX_CORES", "16")
        assert compute_lammps_cores("pmmcs", 6000, cores=0) == 1

    def test_override_above_max_warns_but_is_honoured(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An override above LAMMPS_MAX_CORES is honoured with a warning."""
        monkeypatch.setenv("LAMMPS_MAX_CORES", "4")
        with caplog.at_level("WARNING"):
            assert compute_lammps_cores("pmmcs", 6000, cores=32) == 32
        assert "exceeds LAMMPS_MAX_CORES" in caplog.text


class TestGetLammpsServerKwargs:
    """Tests for get_lammps_server_kwargs."""

    def test_returns_cores_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """server_kwargs carries the computed core count."""
        monkeypatch.setenv("LAMMPS_MAX_CORES", "16")
        assert get_lammps_server_kwargs("pmmcs", 6000) == {"cores": 12}

    def test_explicit_override_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit cores override is reflected in server_kwargs."""
        monkeypatch.setenv("LAMMPS_MAX_CORES", "16")
        assert get_lammps_server_kwargs("pmmcs", 6000, cores=8) == {"cores": 8}
