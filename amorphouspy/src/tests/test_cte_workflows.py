"""Tests for CTE workflow orchestration in amorphouspy.workflows.cte.

These tests mock _run_lammps_md to verify the orchestration logic of
cte_from_fluctuations_simulation and temperature_scan_simulation without
requiring a LAMMPS installation.
"""

import logging

import numpy as np
import pandas as pd
import pytest
from amorphouspy.properties import cte as cte_module
from amorphouspy.properties.cte import (
    cte_from_fluctuations_simulation,
    temperature_scan_simulation,
)
from ase import Atoms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LOGGER_NAME = "amorphouspy.properties.cte"


def _clean_logger_handlers() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def _make_structure() -> Atoms:
    """Create a simple SiO2-like structure for testing."""
    return Atoms(
        "Si2O4", positions=[[0, 0, 0], [2, 2, 2], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 2, 3]], cell=[5, 5, 5], pbc=True
    )


def _make_potential() -> pd.DataFrame:
    """Create a mock potential dataframe."""
    return pd.DataFrame({"Name": ["mock_potential"], "Config": [[]]})


def _make_parsed_output(n: int = 100, seed: int = 42) -> dict:
    """Build a mock parsed_output dict mimicking what LAMMPS returns."""
    rng = np.random.default_rng(seed)
    pressures = np.zeros((n, 3, 3))
    pressures[:, 0, 0] = rng.normal(0, 0.01, n)
    pressures[:, 1, 1] = rng.normal(0, 0.01, n)
    pressures[:, 2, 2] = rng.normal(0, 0.01, n)
    return {
        "generic": {
            "steps": np.arange(n, dtype=float).tolist(),
            "temperature": (np.full(n, 300.0) + rng.normal(0, 1, n)).tolist(),
            "energy_tot": rng.normal(-100.0, 1.0, n).tolist(),
            "volume": (np.full(n, 125.0) + rng.normal(0, 0.5, n)).tolist(),
            "pressures": pressures,
        },
        "lammps": {
            "Lx": (np.full(n, 5.0) + rng.normal(0, 0.01, n)).tolist(),
            "Ly": (np.full(n, 5.0) + rng.normal(0, 0.01, n)).tolist(),
            "Lz": (np.full(n, 5.0) + rng.normal(0, 0.01, n)).tolist(),
        },
    }


_call_counter = 0


def _mock_run_lammps_md(*_args, **_kwargs):
    """Mock that returns a structure and parsed output with varying random data."""
    global _call_counter  # noqa: PLW0603
    _call_counter += 1
    structure = _make_structure()
    parsed_output = _make_parsed_output(n=200, seed=_call_counter * 7)
    return structure, parsed_output


@pytest.fixture(autouse=True)
def _patch_run_lammps_md(monkeypatch):
    """Reset mock call counter and patch _run_lammps_md for each test."""
    global _call_counter  # noqa: PLW0603
    _call_counter = 0
    monkeypatch.setattr(cte_module, "_run_lammps_md", _mock_run_lammps_md)


# ---------------------------------------------------------------------------
# cte_from_fluctuations_simulation
# ---------------------------------------------------------------------------


def test_fluctuations_simulation_returns_summary_and_data(tmp_path, monkeypatch) -> None:
    """Workflow returns dict with 'summary' and 'data' keys."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        result = cte_from_fluctuations_simulation(
            structure=_make_structure(),
            potential=_make_potential(),
            temperature=300.0,
            pressure=1e-4,
            timestep=1.0,
            equilibration_steps=100_000,
            production_steps=100_000,
            min_production_runs=2,
            max_production_runs=3,
            CTE_uncertainty_criterion=1.0,
            n_dump=100_000,
            n_log=100,
        )
        assert "summary" in result
        assert "data" in result
        assert "CTE_V_mean" in result["summary"]
        assert "is_converged" in result["summary"]
    finally:
        _clean_logger_handlers()


def test_fluctuations_simulation_converges_with_loose_criterion(tmp_path, monkeypatch) -> None:
    """With a very loose criterion, the workflow marks convergence as True."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        result = cte_from_fluctuations_simulation(
            structure=_make_structure(),
            potential=_make_potential(),
            temperature=300.0,
            production_steps=100_000,
            min_production_runs=2,
            max_production_runs=5,
            CTE_uncertainty_criterion=1.0,
            n_log=100,
            equilibration_steps=100_000,
        )
        assert result["summary"]["is_converged"] == "True"
    finally:
        _clean_logger_handlers()


def test_fluctuations_simulation_not_converged_with_tight_criterion(tmp_path, monkeypatch) -> None:
    """With an impossibly tight criterion, convergence is not reached."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        result = cte_from_fluctuations_simulation(
            structure=_make_structure(),
            potential=_make_potential(),
            temperature=300.0,
            production_steps=100_000,
            min_production_runs=2,
            max_production_runs=2,
            CTE_uncertainty_criterion=1e-30,
            n_log=100,
            equilibration_steps=100_000,
        )
        assert result["summary"]["is_converged"] == "False"
    finally:
        _clean_logger_handlers()


def test_fluctuations_simulation_data_has_cte_keys(tmp_path, monkeypatch) -> None:
    """Result 'data' dict contains CTE arrays."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        result = cte_from_fluctuations_simulation(
            structure=_make_structure(),
            potential=_make_potential(),
            temperature=300.0,
            production_steps=100_000,
            min_production_runs=2,
            max_production_runs=2,
            CTE_uncertainty_criterion=1.0,
            n_log=100,
            equilibration_steps=100_000,
        )
        data = result["data"]
        for key in ("CTE_V", "CTE_x", "CTE_y", "CTE_z"):
            assert key in data
            assert len(data[key]) >= 2
    finally:
        _clean_logger_handlers()


def test_fluctuations_simulation_aniso_flag(tmp_path, monkeypatch) -> None:
    """The aniso=True flag does not break the workflow."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        result = cte_from_fluctuations_simulation(
            structure=_make_structure(),
            potential=_make_potential(),
            temperature=300.0,
            production_steps=100_000,
            min_production_runs=2,
            max_production_runs=2,
            CTE_uncertainty_criterion=1.0,
            n_log=100,
            equilibration_steps=100_000,
            aniso=True,
        )
        assert "summary" in result
    finally:
        _clean_logger_handlers()


# ---------------------------------------------------------------------------
# temperature_scan_simulation
# ---------------------------------------------------------------------------


def test_temperature_scan_returns_data(tmp_path, monkeypatch) -> None:
    """temperature_scan_simulation returns dict with 'data' key."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        result = temperature_scan_simulation(
            structure=_make_structure(),
            potential=_make_potential(),
            temperature=[300, 400, 500],
            production_steps=100_000,
            n_dump=100_000,
            n_log=100,
        )
        assert "data" in result
    finally:
        _clean_logger_handlers()


def test_temperature_scan_correct_number_of_runs(tmp_path, monkeypatch) -> None:
    """Data arrays have one entry per temperature."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        temps = [300, 400, 500.0]
        result = temperature_scan_simulation(
            structure=_make_structure(),
            potential=_make_potential(),
            temperature=temps,
            production_steps=100_000,
            n_dump=100_000,
            n_log=100,
        )
        data = result["data"]
        assert len(data["T"]) == len(temps)
        assert len(data["V"]) == len(temps)
    finally:
        _clean_logger_handlers()


def test_temperature_scan_default_temperatures(tmp_path, monkeypatch) -> None:
    """Default temperature list is used when None is passed."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        result = temperature_scan_simulation(
            structure=_make_structure(),
            potential=_make_potential(),
            temperature=None,
            production_steps=100_000,
            n_dump=100_000,
            n_log=100,
        )
        assert len(result["data"]["T"]) == 4
    finally:
        _clean_logger_handlers()


def test_temperature_scan_aniso_flag(tmp_path, monkeypatch) -> None:
    """The aniso=True flag does not break temperature_scan_simulation."""
    monkeypatch.chdir(tmp_path)
    _clean_logger_handlers()
    try:
        result = temperature_scan_simulation(
            structure=_make_structure(),
            potential=_make_potential(),
            temperature=[300, 400],
            production_steps=100_000,
            n_dump=100_000,
            n_log=100,
            aniso=True,
        )
        assert "data" in result
    finally:
        _clean_logger_handlers()
