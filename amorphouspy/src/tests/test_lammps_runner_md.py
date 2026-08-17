"""Tests for MD runner logic in amorphouspy.lammps.runner."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from amorphouspy.lammps.runner import _run_lammps_md, get_lammps_command
from ase import Atoms

if TYPE_CHECKING:
    from pathlib import Path


def _structure() -> Atoms:
    return Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)


def test_run_lammps_md_requires_pressure_with_pressure_end() -> None:
    """pressure_end requires pressure to be specified."""
    with pytest.raises(ValueError, match="pressure must be set"):
        _run_lammps_md(
            structure=_structure(),
            potential="dummy",
            temperature=300.0,
            n_ionic_steps=10,
            timestep=1.0,
            initial_temperature=300.0,
            pressure=None,
            pressure_end=0.1,
        )


def test_run_lammps_md_requires_scalar_pressure_for_ramp() -> None:
    """pressure_end does not accept a 6-component pressure list."""
    with pytest.raises(ValueError, match="pressure must be a scalar"):
        _run_lammps_md(
            structure=_structure(),
            potential="dummy",
            temperature=300.0,
            n_ionic_steps=10,
            timestep=1.0,
            initial_temperature=300.0,
            pressure=[0.1, 0.1, 0.1, None, None, None],
            pressure_end=0.2,
        )


@patch("amorphouspy.lammps.runner.structure_from_parsed_output")
@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_run_lammps_md_injects_pressure_ramp_and_clamps_output_frequency(
    mock_run_capture: MagicMock,
    mock_structure_from_output: MagicMock,
    tmp_path: Path,
) -> None:
    """Pressure ramp injects fix npt and n_dump/n_print are clamped to n_ionic_steps."""
    structure = _structure()
    parsed_output = {"generic": {}, "lammps": {}}
    mock_run_capture.return_value = parsed_output
    mock_structure_from_output.return_value = structure

    new_structure, out = _run_lammps_md(
        structure=structure,
        potential="dummy",
        temperature=300.0,
        temperature_end=500.0,
        n_ionic_steps=50,
        timestep=1.0,
        initial_temperature=300.0,
        pressure=0.5,
        pressure_end=1.0,
        n_dump=100,
        n_print_thermo=200,
        tmp_working_directory=tmp_path,
    )

    assert new_structure is structure
    assert out is parsed_output

    kwargs = mock_run_capture.call_args.kwargs
    calc_kwargs = kwargs["calc_kwargs"]
    input_control = kwargs["input_control_file"]

    assert calc_kwargs["n_print"] == 50
    assert calc_kwargs["pressure"] == 0.5
    assert input_control["dump_modify"] == "1 every 50 first yes"
    assert input_control["thermo"] == "50"
    assert "iso 5000.0 10000.0 1.0" in input_control["fix"]


@patch("amorphouspy.lammps.runner.structure_from_parsed_output")
@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_run_lammps_md_without_pressure_ramp_uses_passed_pressure(
    mock_run_capture: MagicMock,
    mock_structure_from_output: MagicMock,
    tmp_path: Path,
) -> None:
    """Without pressure_end, pressure is forwarded unchanged and no custom fix is injected."""
    structure = _structure()
    parsed_output = {"generic": {}, "lammps": {}}
    mock_run_capture.return_value = parsed_output
    mock_structure_from_output.return_value = structure
    pressure = [0.1, 0.1, 0.1, None, None, None]

    _run_lammps_md(
        structure=structure,
        potential="dummy",
        temperature=300.0,
        n_ionic_steps=20,
        timestep=1.0,
        initial_temperature=300.0,
        pressure=pressure,
        n_dump=None,
        n_print_thermo=None,
        tmp_working_directory=tmp_path,
    )

    kwargs = mock_run_capture.call_args.kwargs
    calc_kwargs = kwargs["calc_kwargs"]
    input_control = kwargs["input_control_file"]

    assert calc_kwargs["n_print"] == 20
    assert calc_kwargs["pressure"] == pressure
    assert input_control["thermo"] == "20"
    assert "fix" not in input_control


@patch("amorphouspy.lammps.runner.structure_from_parsed_output")
@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_run_lammps_md_merges_input_control_overrides(
    mock_run_capture: MagicMock,
    mock_structure_from_output: MagicMock,
    tmp_path: Path,
) -> None:
    """Caller-provided input_control_file overrides defaults in the generated controls."""
    structure = _structure()
    parsed_output = {"generic": {}, "lammps": {}}
    mock_run_capture.return_value = parsed_output
    mock_structure_from_output.return_value = structure

    _run_lammps_md(
        structure=structure,
        potential="dummy",
        temperature=300.0,
        n_ionic_steps=20,
        timestep=1.0,
        initial_temperature=300.0,
        input_control_file={"thermo": "7", "thermo_style": "custom step temp"},
        tmp_working_directory=tmp_path,
    )

    input_control = mock_run_capture.call_args.kwargs["input_control_file"]
    assert input_control["thermo"] == "7"
    assert input_control["thermo_style"] == "custom step temp"


def test_get_lammps_command_defaults_to_single_core() -> None:
    """Default command uses one MPI rank."""
    assert get_lammps_command() == "mpiexec -n 1 lmp_mpi -in lmp.in"


def test_get_lammps_command_uses_server_cores() -> None:
    """Server kwargs with cores overrides rank count."""
    assert get_lammps_command({"cores": 4}) == "mpiexec -n 4 lmp_mpi -in lmp.in"
