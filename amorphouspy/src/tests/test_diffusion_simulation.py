"""Tests for ``diffusion_simulation`` using a mocked LAMMPS runner.

The LAMMPS MD call is patched to return a synthetic frame block, so the orchestration
(equilibration + production, frame extraction, MSD/diffusion/cross-check) is exercised
without a real LAMMPS installation.
"""

import gzip
from unittest.mock import patch

import ase.io
import numpy as np
import pandas as pd
import pytest
from amorphouspy.properties.diffusion import compute_msd, diffusion_simulation, log_linear_dump_steps
from ase import Atoms


def _make_structure(n_na=10, n_si=20, n_o=30, box=14.0, seed=0):
    """Build a small random oxide-like structure."""
    rng = np.random.default_rng(seed)
    n_atoms = n_na + n_si + n_o
    positions = rng.uniform(0, box, size=(n_atoms, 3))
    return Atoms("Na" * n_na + "Si" * n_si + "O" * n_o, positions=positions, cell=[box, box, box], pbc=True)


def _make_generic_block(structure, n_frames=60, box=14.0, seed=1):
    """Return a synthetic parsed 'generic' MD block with a random-walk trajectory."""
    rng = np.random.default_rng(seed)
    n_atoms = len(structure)
    walk = np.cumsum(rng.normal(scale=0.15, size=(n_frames, n_atoms, 3)), axis=0)
    unwrapped = structure.get_positions()[None, :, :] + walk
    return {
        "positions": unwrapped % box,
        "unwrapped_positions": unwrapped,
        "cells": np.tile(np.eye(3) * box, (n_frames, 1, 1)),
        "temperature": np.full(n_frames, 3000.0),
        "energy_pot": np.zeros(n_frames),
        "energy_tot": np.zeros(n_frames),
        "volume": np.full(n_frames, box**3),
        "pressures": np.zeros((n_frames, 3, 3)),
    }


def _minimal_potential():
    """Return a minimal non-empty potential DataFrame."""
    return pd.DataFrame({"Name": ["pmmcs"], "Config": [["pair_style buck/coul/long 8.0"]]})


def _expected_schedule(production_steps, timestep, crossover_ps, points_per_decade, linear_interval_ps):
    """Replicate the schedule diffusion_simulation computes internally."""
    return log_linear_dump_steps(
        production_steps,
        crossover_steps=round(crossover_ps * 1000 / timestep),
        points_per_decade=points_per_decade,
        linear_interval_steps=max(1, round(linear_interval_ps * 1000 / timestep)),
    )


def test_diffusion_simulation_returns_log_linear_results():
    """diffusion_simulation runs three MD stages and returns single-origin MSD on a log-then-linear grid."""
    structure = _make_structure()
    schedule = _expected_schedule(60_000, 1.0, 10.0, 9, 2.0)
    generic = _make_generic_block(structure, n_frames=len(schedule))

    with patch("amorphouspy.properties.diffusion._run_lammps_md", autospec=True) as mock_md:
        mock_md.return_value = (structure, {"generic": generic})
        out = diffusion_simulation(
            structure,
            _minimal_potential(),
            production_steps=60_000,
            crossover_ps=10.0,
            points_per_decade=9,
            linear_interval_ps=2.0,
        )

    assert mock_md.call_count == 3
    assert set(out) == {"result", "structure", "frames", "msd", "diffusion", "trajectory_path"}
    assert out["trajectory_path"] is None
    assert len(out["frames"]) == len(schedule)
    assert [frame.info["step"] for frame in out["frames"]] == schedule
    assert out["msd"]["msd_total"][0] == pytest.approx(0.0, abs=1e-9)
    lag_diffs = np.diff(out["msd"]["lag_time_ps"])
    assert not np.allclose(lag_diffs, lag_diffs[0])  # non-uniform (log then linear)
    assert set(out["diffusion"]["per_species"]) == {"Na", "Si", "O"}


def test_diffusion_simulation_injects_log_variable():
    """The production stage injects the decade-log dump variable, reset_timestep and sorted dump_modify."""
    structure = _make_structure()
    schedule = _expected_schedule(60_000, 1.0, 10.0, 9, 2.0)
    generic = _make_generic_block(structure, n_frames=len(schedule))

    with patch("amorphouspy.properties.diffusion._run_lammps_md", autospec=True) as mock_md:
        mock_md.return_value = (structure, {"generic": generic})
        diffusion_simulation(
            structure, _minimal_potential(), production_steps=60_000, crossover_ps=10.0, linear_interval_ps=2.0
        )

    production_call = mock_md.call_args_list[-1]
    config = production_call.kwargs["potential"]["Config"].iloc[0]
    assert any("variable amx_dump equal" in line for line in config)
    assert config[-1] == "reset_timestep 0"
    assert production_call.kwargs["input_control_file"] == {"dump_modify": "1 every v_amx_dump first yes sort id"}


def test_diffusion_simulation_saves_trajectory(tmp_path):
    """save_trajectory writes a gzipped extXYZ that reloads into compute_msd."""
    structure = _make_structure()
    schedule = _expected_schedule(60_000, 1.0, 10.0, 9, 2.0)
    generic = _make_generic_block(structure, n_frames=len(schedule))
    path = tmp_path / "traj.xyz.gz"

    with patch("amorphouspy.properties.diffusion._run_lammps_md", autospec=True) as mock_md:
        mock_md.return_value = (structure, {"generic": generic})
        out = diffusion_simulation(
            structure,
            _minimal_potential(),
            production_steps=60_000,
            crossover_ps=10.0,
            linear_interval_ps=2.0,
            save_trajectory=path,
        )

    assert out["trajectory_path"] == str(path)
    assert path.exists()
    assert path.stat().st_size > 0
    with gzip.open(path, "rt") as handle:
        reloaded = ase.io.read(handle, index=":", format="extxyz")
    assert len(reloaded) == len(schedule)
    assert "unwrapped_positions" in reloaded[0].arrays
    msd = compute_msd(reloaded, method="single_origin")
    assert msd["msd_total"][0] == pytest.approx(0.0, abs=1e-9)


def test_diffusion_simulation_empty_potential_raises():
    """An empty potential is rejected before any MD is launched."""
    structure = _make_structure()
    with pytest.raises(ValueError, match="No matching potential"):
        diffusion_simulation(structure, pd.DataFrame())
