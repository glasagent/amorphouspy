"""Tests for the minimisation runners in amorphouspy.lammps.runner."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from amorphouspy.lammps.runner import _run_lammps_minimize, _run_lammps_relax
from ase import Atoms

if TYPE_CHECKING:
    from collections.abc import Callable

SHIFT = np.array([0.1, 0.0, 0.0])
FINAL_FORCES = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
FINAL_PRESSURE = np.array([[0.3, 0.1, 0.0], [0.1, 0.6, 0.0], [0.0, 0.0, 0.9]])


def _structure() -> Atoms:
    structure = Atoms("SiO", positions=[[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)
    structure.set_velocities([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]])
    return structure


def _fake_lammps(
    criterion: str = "force tolerance", last_dump_step: int = 100, *, stats: bool = True
) -> Callable[..., dict]:
    """Stand-in for run_lammps_with_error_capture: writes log.lammps and dump.out, returns a parsed output."""

    def run(working_directory: str, **kwargs: Atoms) -> dict:
        log = "Step Temp PotEng\n0 0 -10.0\n100 0 -10.5\nLoop time of 0.1 on 1 procs for 100 steps\n"
        if stats:
            log += f"  Stopping criterion = {criterion}\n  Iterations, force evaluations = 100 200\n"
        Path(working_directory, "log.lammps").write_text(log)
        Path(working_directory, "dump.out").write_text(
            f"ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\nITEM: TIMESTEP\n{last_dump_step}\nITEM: NUMBER OF ATOMS\n2\n"
        )
        positions = kwargs["structure"].get_positions()
        generic = {
            "steps": np.array([0, 100]),
            "positions": np.array([positions, positions + SHIFT]),
            "forces": np.array([np.ones((2, 3)), FINAL_FORCES]),
            "cells": np.array([np.eye(3) * 5.0, np.eye(3) * 5.5]),
            "energy_pot": np.array([-10.0, -10.5]),
            "pressures": np.array([np.zeros((3, 3)), FINAL_PRESSURE]),
            "volume": np.array([125.0, 166.375]),
            "velocities": np.array([np.zeros((2, 3)), np.full((2, 3), 0.5)]),
            "indices": np.array([[0, 1], [0, 1]]),
        }
        return {"generic": generic, "lammps": {}}

    return run


@pytest.mark.parametrize(("pressure", "control"), [(None, None), (0.5, {"neighbor": "2.0 bin"})])
@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_run_lammps_minimize_passes_minimize_arguments(
    mock_run_capture: MagicMock, pressure: float | None, control: dict | None
) -> None:
    """The parser gets minimize mode, the calc kwargs, no final-structure dump and a zero-velocity copy."""
    mock_run_capture.side_effect = _fake_lammps()
    structure = _structure()

    _run_lammps_minimize(structure, "dummy", pressure=pressure, input_control_file=control)

    kwargs = mock_run_capture.call_args.kwargs
    assert kwargs["calc_mode"] == "minimize"
    assert kwargs["calc_kwargs"] == {
        "ionic_force_tolerance": 1e-4,
        "ionic_energy_tolerance": 0.0,
        "max_iter": 100_000,
        "pressure": pressure,
        "n_print": 100,
        "style": "cg",
    }
    assert kwargs["units"] == "metal"
    assert kwargs["dump_final_structure"] is False
    # Forwarded unchanged: the runner adds no dump_modify or thermo_modify that would drop the 15-digit formats.
    assert kwargs["input_control_file"] is control
    assert kwargs["structure"] is not structure
    np.testing.assert_array_equal(kwargs["structure"].get_velocities(), 0.0)


@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_run_lammps_minimize_summary(mock_run_capture: MagicMock) -> None:
    """Every summary value follows from the final frame and the log; the input structure is left unchanged."""
    mock_run_capture.side_effect = _fake_lammps()
    structure = _structure()
    positions, velocities = structure.get_positions(), structure.get_velocities()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        new_structure, _, summary = _run_lammps_minimize(structure, "dummy")

    np.testing.assert_array_equal(summary.pop("pressure_gpa"), FINAL_PRESSURE)
    assert summary == {
        "stopping_criterion": "force tolerance",
        "converged": True,
        "n_iterations": 100,
        "n_force_evaluations": 200,
        "energy_initial_ev": -10.0,
        "energy_final_ev": -10.5,
        "energy_change_mev_per_atom": pytest.approx(-250.0),
        "force_norm_ev_per_a": pytest.approx(np.sqrt(26.0)),
        "max_force_ev_per_a": pytest.approx(5.0),
        "mean_pressure_gpa": pytest.approx(0.6),
        "volume_initial_a3": 125.0,
        "volume_final_a3": 166.375,
    }
    np.testing.assert_allclose(new_structure.get_positions(), positions + SHIFT)
    np.testing.assert_allclose(new_structure.cell[:], np.eye(3) * 5.5)
    np.testing.assert_array_equal(new_structure.get_velocities(), 0.0)
    np.testing.assert_array_equal(structure.get_positions(), positions)
    np.testing.assert_array_equal(structure.get_velocities(), velocities)


@pytest.mark.parametrize("criterion", ["max iterations", "linesearch alpha is zero"])
@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_run_lammps_minimize_warns_when_not_converged(mock_run_capture: MagicMock, criterion: str) -> None:
    """A stop on a criterion other than a tolerance warns and still returns the final structure."""
    mock_run_capture.side_effect = _fake_lammps(criterion)
    with pytest.warns(UserWarning, match=rf"stopped on '{criterion}'.*per-atom force 5 eV/Å"):
        new_structure, _, summary = _run_lammps_minimize(_structure(), "dummy")
    assert summary["converged"] is False
    assert summary["stopping_criterion"] == criterion
    np.testing.assert_allclose(new_structure.get_positions(), _structure().get_positions() + SHIFT)


@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_run_lammps_minimize_rejects_non_final_dump(mock_run_capture: MagicMock) -> None:
    """A last dumped step that differs from the last thermo step and the iteration count raises."""
    mock_run_capture.side_effect = _fake_lammps(last_dump_step=90)
    with pytest.raises(RuntimeError, match=r"dumped step \(90\).*thermo step \(100\).*iteration count \(100\)"):
        _run_lammps_minimize(_structure(), "dummy")


@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_run_lammps_minimize_requires_log_statistics(mock_run_capture: MagicMock) -> None:
    """A log.lammps without the minimisation statistics raises."""
    mock_run_capture.side_effect = _fake_lammps(stats=False)
    with pytest.raises(RuntimeError, match="minimisation statistics"):
        _run_lammps_minimize(_structure(), "dummy")


@pytest.mark.parametrize(
    ("runner", "kwargs", "match"),
    [
        (_run_lammps_minimize, {"force_tolerance": -1e-4}, "force_tolerance must be finite"),
        (_run_lammps_minimize, {"energy_tolerance": np.nan}, "energy_tolerance must be finite"),
        (_run_lammps_minimize, {"force_tolerance": 0.0}, "cannot both be 0"),
        (_run_lammps_minimize, {"max_iter": True}, "max_iter must be a positive integer"),
        (_run_lammps_minimize, {"n_print": 0}, "n_print must be a positive integer"),
        (_run_lammps_minimize, {"min_style": "quickmin"}, "min_style must be one of"),
        (_run_lammps_minimize, {"pressure": [0.0, 0.0, 0.0]}, "finite scalar"),
        (_run_lammps_minimize, {"pressure": np.inf}, "finite scalar"),
        (_run_lammps_minimize, {"min_style": "fire", "pressure": 0.0}, "'fire' cannot be combined"),
        (_run_lammps_relax, {"n_cycles": 0}, "n_cycles must be a positive integer"),
        (_run_lammps_relax, {"n_cycles": True}, "n_cycles must be a positive integer"),
        (_run_lammps_relax, {"pressure": np.nan}, "finite scalar"),
    ],
)
@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_minimize_runners_reject_invalid_arguments(
    mock_run_capture: MagicMock, runner: Callable[..., object], kwargs: dict, match: str
) -> None:
    """Invalid arguments raise ValueError before LAMMPS runs."""
    with pytest.raises(ValueError, match=match):
        runner(_structure(), "dummy", **kwargs)
    mock_run_capture.assert_not_called()


@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_run_lammps_minimize_warns_when_n_print_does_not_divide_max_iter(mock_run_capture: MagicMock) -> None:
    """n_print not dividing max_iter warns, because a max-iterations stop would not dump the last step."""
    mock_run_capture.side_effect = _fake_lammps()
    with pytest.warns(UserWarning, match="n_print=100 does not divide max_iter=250"):
        _run_lammps_minimize(_structure(), "dummy", max_iter=250)


@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_run_lammps_relax_chains_cg_box_relax_and_fire(mock_run_capture: MagicMock) -> None:
    """Two cycles run cg with box/relax then FIRE at fixed cell, each stage starting from the previous result."""
    mock_run_capture.side_effect = _fake_lammps()
    structure = _structure()

    new_structure, parsed_output, summaries = _run_lammps_relax(structure, "dummy")

    stages = [("cg", 0.0), ("fire", None), ("cg", 0.0), ("fire", None)]
    calls = [call.kwargs for call in mock_run_capture.call_args_list]
    assert [(c["calc_kwargs"]["style"], c["calc_kwargs"]["pressure"]) for c in calls] == stages
    for stage, kwargs in enumerate(calls):
        np.testing.assert_allclose(kwargs["structure"].get_positions(), structure.get_positions() + stage * SHIFT)
    assert [(s["min_style"], s["pressure_target_gpa"]) for s in summaries] == stages
    np.testing.assert_allclose(parsed_output["generic"]["positions"][-1], structure.get_positions() + 4 * SHIFT)
    np.testing.assert_allclose(new_structure.get_positions(), structure.get_positions() + 4 * SHIFT)
