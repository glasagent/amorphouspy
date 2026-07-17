"""Tests for amorphouspy.lammps.md."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from amorphouspy.lammps.md import md_simulation
from ase import Atoms

if TYPE_CHECKING:
    from pathlib import Path


def _potential(name: str = "pmmcs", config: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame({"Name": [name], "Config": [config or []]})


def test_md_simulation_raises_on_empty_potential() -> None:
    """Empty potential table raises a clear ValueError."""
    with pytest.raises(ValueError, match="No matching potential"):
        md_simulation(
            structure=Atoms("Si"),
            potential=pd.DataFrame(columns=["Name", "Config"]),
        )


@patch("amorphouspy.lammps.md._run_lammps_md")
def test_md_simulation_passes_potential_config_unchanged(mock_run_md: MagicMock) -> None:
    """The potential Config reaches the runner exactly as the caller supplied it."""
    mock_run_md.return_value = (Atoms("Si"), {"generic": {"steps": [0, 1]}})

    config = ["pair_style table spline 500", "pair_modify shift yes"]
    potential = _potential(name="shik", config=list(config))

    md_simulation(structure=Atoms("Si"), potential=potential)

    _, kwargs = mock_run_md.call_args
    assert kwargs["potential"].loc[0, "Config"] == config


@patch("amorphouspy.lammps.md._run_lammps_md")
def test_md_simulation_forwards_all_runtime_arguments(mock_run_md: MagicMock, tmp_path: Path) -> None:
    """Runtime knobs are forwarded unchanged to _run_lammps_md."""
    structure = Atoms("Si")
    parsed = {"generic": {"temperature": [300.0]}}
    mock_run_md.return_value = (structure, parsed)

    md_simulation(
        structure=structure,
        potential=_potential(),
        temperature_sim=2200.0,
        timestep=2.0,
        production_steps=1234,
        n_dump=50,
        n_print_thermo=10,
        server_kwargs={"cores": 2},
        temperature_end=1800.0,
        pressure=0.1,
        pressure_end=0.2,
        langevin=True,
        seed=987,
        tmp_working_directory=tmp_path,
    )

    _, kwargs = mock_run_md.call_args
    assert kwargs["temperature"] == 2200.0
    assert kwargs["temperature_end"] == 1800.0
    assert kwargs["n_ionic_steps"] == 1234
    assert kwargs["timestep"] == 2.0
    assert kwargs["initial_temperature"] == 2200.0
    assert kwargs["pressure"] == 0.1
    assert kwargs["pressure_end"] == 0.2
    assert kwargs["n_dump"] == 50
    assert kwargs["n_print_thermo"] == 10
    assert kwargs["langevin"] is True
    assert kwargs["seed"] == 987
    assert kwargs["server_kwargs"] == {"cores": 2}
    assert kwargs["tmp_working_directory"] == tmp_path


@patch("amorphouspy.lammps.md._run_lammps_md")
def test_md_simulation_returns_structure_and_generic_result(mock_run_md: MagicMock) -> None:
    """Return payload exposes final structure and parsed generic result."""
    final_structure = Atoms("Si2")
    parsed = {"generic": {"steps": [0, 1, 2]}, "lammps": {}}
    mock_run_md.return_value = (final_structure, parsed)

    out = md_simulation(structure=Atoms("Si"), potential=_potential())

    assert out["structure"] is final_structure
    assert out["result"] == {"steps": [0, 1, 2]}
