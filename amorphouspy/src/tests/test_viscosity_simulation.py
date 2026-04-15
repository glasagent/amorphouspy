"""Tests for viscosity_simulation and viscosity_ensemble using mocked LAMMPS calls.

These tests exercise the convergence-loop logic and ensemble averaging without
requiring a real LAMMPS installation.  Both `_viscosity_simulation` (the single-
temperature run) and `viscosity_simulation` (the iterative wrapper) are patched
so that synthetic pressure-tensor data is returned instead of real MD trajectories.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from amorphouspy.workflows.viscosity import (
    helfand_viscosity,
    viscosity_ensemble,
    viscosity_simulation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pressure_block(n: int = 1000, seed: int = 0) -> dict:
    """Return a generic-style MD result dict with synthetic pressure tensors."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 5, n)
    pressures = np.zeros((n, 3, 3))
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        pressures[:, i, j] = np.exp(-t) + 0.005 * rng.standard_normal(n)
    for k in range(3):
        pressures[:, k, k] = 0.005 * rng.standard_normal(n)
    return {
        "pressures": pressures,
        "volume": np.ones(n) * 1e5,
        "temperature": np.ones(n) * 1000.0,
    }


def _make_mock_sim_result(n: int = 1000) -> dict:
    """Return the dict that `_viscosity_simulation` is expected to produce."""
    return {
        "result": _make_pressure_block(n),
        "structure": MagicMock(),
    }


def _make_viscosity_sim_output(n: int = 1000) -> dict:
    """Return the dict that `viscosity_simulation` is expected to produce."""
    md_result = {"result": _make_pressure_block(n)}
    helfand_data = helfand_viscosity(md_result, timestep=1.0)
    return {
        "viscosity_data": helfand_data,
        "result": _make_pressure_block(n),
        "structure": MagicMock(),
        "total_production_steps": n,
        "iterations": 1,
        "converged": True,
        "seed": 12345,
    }


def _minimal_potential() -> pd.DataFrame:
    return pd.DataFrame({"Name": ["mock_potential"], "Config": [[]]})


# ---------------------------------------------------------------------------
# viscosity_simulation — time-budget exhaustion path
# ---------------------------------------------------------------------------


@patch("amorphouspy.workflows.viscosity._viscosity_simulation")
def test_viscosity_simulation_budget_exhausted(mock_sim: MagicMock) -> None:
    """viscosity_simulation returns after hitting the time budget without converging."""
    n = 1000
    mock_sim.return_value = _make_mock_sim_result(n)

    result = viscosity_simulation(
        structure=MagicMock(),
        potential=_minimal_potential(),
        temperature_sim=1000.0,
        timestep=1.0,
        initial_production_steps=n,
        # max_total_time_ns so small that max_steps == initial_production_steps
        max_total_time_ns=n * 1e-6,
        max_iterations=10,
    )

    assert "viscosity_data" in result
    assert "converged" in result
    assert "total_production_steps" in result
    assert "iterations" in result


@patch("amorphouspy.workflows.viscosity._viscosity_simulation")
def test_viscosity_simulation_returns_structure(mock_sim: MagicMock) -> None:
    """viscosity_simulation passes back a structure object in the result."""
    n = 1000
    mock_sim.return_value = _make_mock_sim_result(n)

    result = viscosity_simulation(
        structure=MagicMock(),
        potential=_minimal_potential(),
        temperature_sim=1000.0,
        timestep=1.0,
        initial_production_steps=n,
        max_total_time_ns=n * 1e-6,
        max_iterations=5,
    )

    assert "structure" in result


@patch("amorphouspy.workflows.viscosity._run_lammps_md")
@patch("amorphouspy.workflows.viscosity._viscosity_simulation")
def test_viscosity_simulation_extension_step(mock_sim: MagicMock, mock_lammps: MagicMock) -> None:
    """viscosity_simulation can extend the trajectory by one 100-ps block."""
    n = 500
    block = _make_pressure_block(n)
    mock_sim.return_value = {"result": block, "structure": MagicMock()}

    # _run_lammps_md is called for extensions; return a parsed_output with a "generic" key.
    ext_block = _make_pressure_block(100)
    mock_lammps.return_value = (MagicMock(), {"generic": ext_block})

    result = viscosity_simulation(
        structure=MagicMock(),
        potential=_minimal_potential(),
        temperature_sim=1000.0,
        timestep=1.0,
        initial_production_steps=n,
        # Allow one extra extension before budget is hit.
        max_total_time_ns=(n + 100) * 1e-6,
        max_iterations=3,
        eta_stable_iters=10,  # never converge via stability
        eta_rel_tol=0.0,
    )

    assert result["total_production_steps"] >= n


# ---------------------------------------------------------------------------
# viscosity_simulation — empty potential guard
# ---------------------------------------------------------------------------


def test_viscosity_simulation_empty_potential_raises() -> None:
    """viscosity_simulation raises ValueError for an empty potential DataFrame."""
    with pytest.raises(ValueError, match="No matching potential"):
        viscosity_simulation(
            structure=MagicMock(),
            potential=pd.DataFrame(columns=["Name", "Config"]),
            temperature_sim=1000.0,
        )


# ---------------------------------------------------------------------------
# viscosity_ensemble — sequential execution
# ---------------------------------------------------------------------------


@patch("amorphouspy.workflows.viscosity.viscosity_simulation")
def test_viscosity_ensemble_sequential_runs_n_replicas(mock_sim: MagicMock) -> None:
    """viscosity_ensemble calls viscosity_simulation exactly n_replicas times."""
    mock_sim.return_value = _make_viscosity_sim_output()

    result = viscosity_ensemble(
        structure=MagicMock(),
        potential=_minimal_potential(),
        n_replicas=3,
        temperature_sim=1000.0,
    )

    assert mock_sim.call_count == 3
    assert result["n_replicas"] == 3
    assert len(result["viscosities"]) == 3


@patch("amorphouspy.workflows.viscosity.viscosity_simulation")
def test_viscosity_ensemble_output_keys(mock_sim: MagicMock) -> None:
    """viscosity_ensemble returns all expected top-level keys."""
    mock_sim.return_value = _make_viscosity_sim_output()

    result = viscosity_ensemble(
        structure=MagicMock(),
        potential=_minimal_potential(),
        n_replicas=2,
    )

    expected_keys = {
        "viscosity",
        "viscosity_fit_residual",
        "viscosity_sem",
        "shear_modulus_inf",
        "bulk_viscosity",
        "maxwell_relaxation_time_ps",
        "mean_pressure_gpa",
        "temperature",
        "n_replicas",
        "seeds",
        "viscosities",
        "converged",
        "results",
    }
    assert expected_keys.issubset(result.keys())


@patch("amorphouspy.workflows.viscosity.viscosity_simulation")
def test_viscosity_ensemble_single_replica_zero_std(mock_sim: MagicMock) -> None:
    """With n_replicas=1, std and sem are both 0."""
    mock_sim.return_value = _make_viscosity_sim_output()

    result = viscosity_ensemble(
        structure=MagicMock(),
        potential=_minimal_potential(),
        n_replicas=1,
    )

    assert result["viscosity_fit_residual"] == 0.0
    assert result["viscosity_sem"] == 0.0


# ---------------------------------------------------------------------------
# viscosity_ensemble — seed handling
# ---------------------------------------------------------------------------


@patch("amorphouspy.workflows.viscosity.viscosity_simulation")
def test_viscosity_ensemble_explicit_seeds(mock_sim: MagicMock) -> None:
    """Explicit seeds list is passed through to the result."""
    mock_sim.return_value = _make_viscosity_sim_output()
    seeds = [111, 222, 333]

    result = viscosity_ensemble(
        structure=MagicMock(),
        potential=_minimal_potential(),
        n_replicas=3,
        seeds=seeds,
    )

    assert result["seeds"] == seeds


def test_viscosity_ensemble_seed_count_mismatch_raises() -> None:
    """Raises ValueError when len(seeds) != n_replicas."""
    with pytest.raises(ValueError, match="does not match n_replicas"):
        viscosity_ensemble(
            structure=MagicMock(),
            potential=_minimal_potential(),
            n_replicas=3,
            seeds=[1, 2],
        )


# ---------------------------------------------------------------------------
# viscosity_ensemble — parallel execution
# ---------------------------------------------------------------------------


@patch("amorphouspy.workflows.viscosity.viscosity_simulation")
def test_viscosity_ensemble_parallel(mock_sim: MagicMock) -> None:
    """parallel=True produces the same structure of output as sequential."""
    mock_sim.return_value = _make_viscosity_sim_output()

    result = viscosity_ensemble(
        structure=MagicMock(),
        potential=_minimal_potential(),
        n_replicas=2,
        parallel=True,
    )

    assert result["n_replicas"] == 2
    assert len(result["viscosities"]) == 2


# ---------------------------------------------------------------------------
# viscosity_ensemble — seed file persistence
# ---------------------------------------------------------------------------


@patch("amorphouspy.workflows.viscosity.viscosity_simulation")
def test_viscosity_ensemble_saves_seed_file(mock_sim: MagicMock, tmp_path) -> None:
    """Seeds are written to disk when tmp_working_directory is provided."""
    mock_sim.return_value = _make_viscosity_sim_output()

    viscosity_ensemble(
        structure=MagicMock(),
        potential=_minimal_potential(),
        n_replicas=2,
        tmp_working_directory=str(tmp_path),
    )

    seed_file = tmp_path / "viscosity_ensemble_seeds.json"
    assert seed_file.exists()
    data = json.loads(seed_file.read_text())
    assert "seeds" in data
    assert len(data["seeds"]) == 2
