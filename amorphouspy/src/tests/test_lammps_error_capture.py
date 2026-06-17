"""Tests for run_lammps_with_error_capture error handling."""

import subprocess
from unittest.mock import patch

import pytest
from amorphouspy.lammps.runner import get_lammps_command, run_lammps_with_error_capture
from ase.io import read

from amorphouspy import generate_potential, get_structure_dict

from . import DATA_DIR


def test_lammps_error_contains_diagnostics(tmp_path):
    """Run LAMMPS with overlapping atoms that will crash and verify the error includes log content."""
    # Use a real structure but place all atoms at the origin so LAMMPS blows up
    structure = read(DATA_DIR / "SiO2_glass_300_atoms.xyz")
    structure.set_positions([[0.0, 0.0, 0.0]] * len(structure))

    atoms_dict = get_structure_dict(composition={"SiO2": 100}, target_atoms=9)
    potential = generate_potential(atoms_dict=atoms_dict, potential_type="shik")

    with pytest.raises(RuntimeError, match=r"Pair distance < table inner cutoff") as exc_info:
        run_lammps_with_error_capture(
            working_directory=str(tmp_path),
            structure=structure,
            potential=potential,
            calc_mode="md",
            calc_kwargs={
                "temperature": 300.0,
                "n_ionic_steps": 10,
                "time_step": 1.0,
                "n_print": 10,
                "initial_temperature": 300.0,
                "seed": 42,
                "pressure": None,
                "langevin": False,
            },
            units="metal",
            write_restart_file=False,
            read_restart_file=False,
            restart_file="restart.out",
            input_control_file={},
            lmp_command=get_lammps_command(server_kwargs={"cores": 1}),
        )
    # Verify the log.lammps content is also included
    assert "log.lammps" in str(exc_info.value)


@patch("amorphouspy.lammps.runner.lammps_file_interface_function")
def test_error_capture_includes_stdout_and_stderr(mock_lammps_call, tmp_path):
    """CalledProcessError diagnostics include both stdout and stderr snippets."""
    mock_lammps_call.side_effect = subprocess.CalledProcessError(
        returncode=1,
        cmd="lmp_mpi -in lmp.in",
        output="stdout from lammps",
        stderr="stderr from lammps",
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_lammps_with_error_capture(working_directory=str(tmp_path))

    msg = str(exc_info.value)
    assert "LAMMPS stdout" in msg
    assert "stdout from lammps" in msg
    assert "LAMMPS stderr" in msg
    assert "stderr from lammps" in msg


@patch("amorphouspy.lammps.runner.lammps_file_interface_function")
def test_soft_failure_when_job_crashed_flag_true(mock_lammps_call, tmp_path):
    """A truthy job_crashed flag raises RuntimeError even without subprocess failure."""
    mock_lammps_call.return_value = ("", {"generic": {}, "lammps": {}}, True)

    with pytest.raises(RuntimeError, match="LAMMPS crashed"):
        run_lammps_with_error_capture(working_directory=str(tmp_path))


@patch("amorphouspy.lammps.runner.lammps_file_interface_function")
def test_soft_failure_when_required_output_keys_missing(mock_lammps_call, tmp_path):
    """Missing parsed output keys are treated as a failed run."""
    mock_lammps_call.return_value = ("", {"generic": None, "lammps": {}}, False)

    with pytest.raises(RuntimeError, match="LAMMPS crashed"):
        run_lammps_with_error_capture(working_directory=str(tmp_path))
