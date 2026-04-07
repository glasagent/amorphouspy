"""Tests for run_lammps_with_error_capture error handling."""

import pytest
from amorphouspy.workflows.shared import get_lammps_command, run_lammps_with_error_capture
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
