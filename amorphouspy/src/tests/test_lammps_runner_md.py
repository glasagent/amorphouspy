"""Tests for MD runner logic in amorphouspy.lammps.runner."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from amorphouspy.lammps.runner import _run_lammps_md, get_lammps_command, thermostat_ensembles
from ase import Atoms

if TYPE_CHECKING:
    from pathlib import Path


def _structure() -> Atoms:
    return Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)


def _structure_with_velocities() -> Atoms:
    structure = _structure()
    structure.set_velocities([[0.01, 0.0, 0.0]])
    return structure


@pytest.mark.parametrize(
    ("ensemble", "extra", "match"),
    [
        ("nvp", {"temperature": 300.0}, "ensemble must be one of"),
        ("nvt", {"temperature": 300.0, "npt_pressure": 0.1}, "can only be used with an npt ensemble"),
        ("nvt_langevin", {"temperature": 300.0, "npt_pressure_end": 0.1}, "can only be used with an npt ensemble"),
        ("npt", {"temperature": 300.0}, "npt_pressure must be set"),
        ("npt", {"temperature": 300.0, "npt_pressure_end": 0.1}, "npt_pressure must be set"),
        (
            "npt",
            {"temperature": 300.0, "npt_pressure": [0.1, 0.1, 0.1, None, None, None], "npt_pressure_end": 0.2},
            "npt_pressure must be a scalar",
        ),
        ("npt_langevin", {"temperature": 300.0, "npt_pressure": 0.1, "npt_pressure_end": 0.0}, "pressure ramp"),
        ("nvt", {}, "temperature must be set"),
        ("npt", {"npt_pressure": 0.1}, "temperature must be set"),
        ("nve", {"temperature": 300.0}, "cannot be used with ensemble 'nve'"),
        ("nve", {"temperature_end": 500.0}, "cannot be used with ensemble 'nve'"),
        ("nve", {"npt_pressure": 0.1}, "can only be used with an npt ensemble"),
    ],
)
@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_run_lammps_md_rejects_arguments_that_do_not_fit_the_ensemble(
    mock_run_capture: MagicMock, ensemble: str, extra: dict, match: str
) -> None:
    """Every ensemble-specific argument is validated against the chosen ensemble before LAMMPS runs."""
    with pytest.raises(ValueError, match=match):
        _run_lammps_md(
            structure=_structure_with_velocities(),
            potential="dummy",
            n_ionic_steps=10,
            timestep=1.0,
            ensemble=ensemble,  # ty: ignore[invalid-argument-type]
            **extra,
        )
    mock_run_capture.assert_not_called()


@pytest.mark.parametrize(
    ("ensemble", "extra", "temperature", "pressure", "langevin"),
    [
        ("nvt", {"temperature": 300.0}, 300.0, None, False),
        ("nvt_langevin", {"temperature": 300.0}, 300.0, None, True),
        ("npt", {"temperature": 300.0, "npt_pressure": 0.1}, 300.0, 0.1, False),
        ("npt_langevin", {"temperature": 300.0, "npt_pressure": 0.1}, 300.0, 0.1, True),
        ("nve", {}, None, None, False),
    ],
)
@patch("amorphouspy.lammps.runner.structure_from_parsed_output")
@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_run_lammps_md_maps_ensemble_to_parser_kwargs(
    mock_run_capture: MagicMock,
    mock_structure_from_output: MagicMock,
    tmp_path: Path,
    ensemble: str,
    extra: dict,
    temperature: float | None,
    pressure: float | None,
    langevin: bool,  # noqa: FBT001
) -> None:
    """Each ensemble selects the parser's temperature, pressure and langevin settings."""
    structure = _structure_with_velocities()
    mock_run_capture.return_value = {"generic": {}, "lammps": {}}
    mock_structure_from_output.return_value = structure

    _run_lammps_md(
        structure=structure,
        potential="dummy",
        n_ionic_steps=20,
        timestep=1.0,
        ensemble=ensemble,  # ty: ignore[invalid-argument-type]
        tmp_working_directory=tmp_path,
        **extra,
    )

    kwargs = mock_run_capture.call_args.kwargs
    assert kwargs["calc_kwargs"]["temperature"] == temperature
    assert kwargs["calc_kwargs"]["pressure"] == pressure
    assert kwargs["calc_kwargs"]["langevin"] is langevin
    assert "fix" not in kwargs["input_control_file"]


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
        ensemble="npt",
        npt_pressure=0.5,
        npt_pressure_end=1.0,
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
    """Without npt_pressure_end, npt_pressure is forwarded unchanged and no custom fix is injected."""
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
        ensemble="npt",
        npt_pressure=pressure,
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
        ensemble="nvt",
        input_control_file={"thermo": "7", "thermo_style": "custom step temp"},
        tmp_working_directory=tmp_path,
    )

    input_control = mock_run_capture.call_args.kwargs["input_control_file"]
    assert input_control["thermo"] == "7"
    assert input_control["thermo_style"] == "custom step temp"


@patch("amorphouspy.lammps.runner.structure_from_parsed_output")
@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_run_lammps_md_default_calc_kwargs_unchanged(
    mock_run_capture: MagicMock,
    mock_structure_from_output: MagicMock,
    tmp_path: Path,
) -> None:
    """The nvt path sends the same calc_kwargs and controls as the former pressure=None, langevin=False default."""
    structure = _structure()
    mock_run_capture.return_value = {"generic": {}, "lammps": {}}
    mock_structure_from_output.return_value = structure

    _run_lammps_md(structure, "dummy", 20, 1.0, 300.0, ensemble="nvt", tmp_working_directory=tmp_path)

    kwargs = mock_run_capture.call_args.kwargs
    assert kwargs["calc_kwargs"] == {
        "temperature": 300.0,
        "n_ionic_steps": 20,
        "time_step": 1.0,
        "n_print": 20,
        "initial_temperature": 600.0,
        "seed": 12345,
        "pressure": None,
        "langevin": False,
    }
    assert kwargs["input_control_file"] == {
        "dump_modify": "1 every 20 first yes",
        "thermo": "20",
        "thermo_style": "custom step temp density pe etotal pxx pxy pxz pyy pyz pzz vol",
        "thermo_modify": "flush no",
    }


@patch("amorphouspy.lammps.runner.structure_from_parsed_output")
@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_run_lammps_md_nve_keeps_structure_velocities(
    mock_run_capture: MagicMock,
    mock_structure_from_output: MagicMock,
    tmp_path: Path,
) -> None:
    """ensemble="nve" without initial_temperature sends initial_temperature=0 (keep the structure's velocities)."""
    structure = _structure_with_velocities()
    mock_run_capture.return_value = {"generic": {}, "lammps": {}}
    mock_structure_from_output.return_value = structure

    _run_lammps_md(structure, "dummy", 20, 1.0, ensemble="nve", tmp_working_directory=tmp_path)

    calc_kwargs = mock_run_capture.call_args.kwargs["calc_kwargs"]
    assert calc_kwargs["temperature"] is None
    assert calc_kwargs["initial_temperature"] == 0


@patch("amorphouspy.lammps.runner.structure_from_parsed_output")
@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_run_lammps_md_nve_passes_explicit_initial_temperature(
    mock_run_capture: MagicMock,
    mock_structure_from_output: MagicMock,
    tmp_path: Path,
) -> None:
    """An explicit positive initial_temperature creates velocities, so no velocities are required."""
    structure = _structure()
    mock_run_capture.return_value = {"generic": {}, "lammps": {}}
    mock_structure_from_output.return_value = structure

    _run_lammps_md(
        structure, "dummy", 20, 1.0, initial_temperature=500.0, ensemble="nve", tmp_working_directory=tmp_path
    )

    calc_kwargs = mock_run_capture.call_args.kwargs["calc_kwargs"]
    assert calc_kwargs["temperature"] is None
    assert calc_kwargs["initial_temperature"] == 500.0


@pytest.mark.parametrize("velocities", ["absent", "zero"])
@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_run_lammps_md_nve_requires_velocities(mock_run_capture: MagicMock, velocities: str) -> None:
    """ensemble="nve" without initial_temperature fails when the structure has no (or only zero) velocities."""
    structure = _structure()
    if velocities == "zero":
        structure.set_velocities([[0.0, 0.0, 0.0]])
    assert structure.has("momenta") is (velocities == "zero")
    with pytest.raises(ValueError, match="non-zero velocities"):
        _run_lammps_md(structure, "dummy", 20, 1.0, ensemble="nve")
    mock_run_capture.assert_not_called()


@patch("amorphouspy.lammps.runner.structure_from_parsed_output")
@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_run_lammps_md_does_not_modify_input_velocities(
    mock_run_capture: MagicMock,
    mock_structure_from_output: MagicMock,
    tmp_path: Path,
) -> None:
    """In-place velocity rescaling by the parser must not reach the caller's structure."""
    structure = _structure_with_velocities()

    def rescale_velocities_in_place(**kwargs: object) -> dict:
        passed = kwargs["structure"]
        passed.set_velocities(passed.get_velocities() * 1000)
        return {"generic": {}, "lammps": {}}

    mock_run_capture.side_effect = rescale_velocities_in_place
    mock_structure_from_output.return_value = structure

    _run_lammps_md(structure, "dummy", 20, 1.0, ensemble="nve", tmp_working_directory=tmp_path)

    np.testing.assert_allclose(structure.get_velocities(), [[0.01, 0.0, 0.0]])


def test_thermostat_ensembles() -> None:
    """Each thermostat maps to its NVT and NPT ensemble; unknown names are rejected."""
    assert thermostat_ensembles("nose_hoover") == ("nvt", "npt")
    assert thermostat_ensembles("langevin") == ("nvt_langevin", "npt_langevin")
    with pytest.raises(ValueError, match="thermostat must be one of"):
        thermostat_ensembles("berendsen")  # ty: ignore[invalid-argument-type]


def test_get_lammps_command_defaults_to_single_core() -> None:
    """Default command uses one MPI rank."""
    assert get_lammps_command() == "mpiexec -n 1 lmp_mpi -in lmp.in"


def test_get_lammps_command_uses_server_cores() -> None:
    """Server kwargs with cores overrides rank count."""
    assert get_lammps_command({"cores": 4}) == "mpiexec -n 4 lmp_mpi -in lmp.in"
