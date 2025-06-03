import os

from ase.atoms import Atoms
import numpy as np
from pyiron_base import job
from pyiron_atomistics.lammps.lammps import lammps_function
from structuretoolkit.common import center_coordinates_in_unit_cell


def _get_structure(
    structure,
    cell,
    indices,
    positions=None,
    unwrapped_positions=None,
    total_displacements=None,
    wrap_atoms=True,
):
    if indices is not None and len(indices) != len(structure):
        snapshot = Atoms(
            positions=np.zeros(indices.shape + (3,)),
            cell=cell,
            pbc=structure.pbc,
        )
        snapshot.set_array("indices", indices)
    else:
        snapshot = structure.copy()
        if cell is not None:
            snapshot.cell = cell
        if indices is not None:
            snapshot.set_array("indices", indices)
    if wrap_atoms:
        snapshot.positions = positions
        snapshot = center_coordinates_in_unit_cell(snapshot)
        # snapshot.center_coordinates_in_unit_cell()
    elif unwrapped_positions is not None:
        snapshot.positions = unwrapped_positions
    else:
        snapshot.positions += total_displacements
    return snapshot


@job
def melt_quench_simulation(
    structure,
    potential,
    working_directory,
    temperature_high=5000.0,
    temperature_low=300.0,
    n_print=1000,
):
    # Here is an attempt to create a LAMMPS simulation with pyiron_atomistics
    # I am not very sure this is the best way to do it, however, I think it is working

    # Create working directory
    os.makedirs(working_directory, exist_ok=True)

    # 2) Stage 1: Heat up / initial NVT
    _shell_output, parsed_output, _job_crashed = lammps_function(
        working_directory=working_directory,
        structure=structure,
        potential=potential,
        calc_mode="md",
        calc_kwargs={
            "temperature": [
                temperature_low,
                temperature_high,
            ],  # heat from T = 300 K to T = 5000 K
            "n_ionic_steps": 47_000,  # number of MD steps used for the heating
            # can be changes to calculate  specific rate
            "time_step": 1.0,  # 1 fs time step
            "n_print": n_print,  # output every 1000 steps
            "seed": 12345,  # random seed for velocities
            "initial_temperature": temperature_low,  # initialize at 300 K
            #
        },
        cutoff_radius=None,
        units="metal",
        bonds_kwargs={},
        server_kwargs={},
        enable_h5md=False,
        write_restart_file=False,
        read_restart_file=False,
        restart_file="restart.out",
        executable_version=None,
        executable_path=None,
        input_control_file=None,
    )
    structure_1 = _get_structure(
        structure=structure,
        cell=parsed_output["generic"]["cells"][-1],
        indices=parsed_output["generic"]["indices"][-1],
        positions=parsed_output["generic"]["positions"][-1],
        unwrapped_positions=None,
        total_displacements=None,
        wrap_atoms=True,
    )
    # print("velo", parsed_output["generic"]["velocities"][-1])
    structure_1.set_velocities(parsed_output["generic"]["velocities"][-1])
    for filename in os.listdir(working_directory):
        file_path = os.path.join(working_directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # 3) Stage 2: High-T NVT equilibration at e.g. 1000 K
    _shell_output, parsed_output, _job_crashed = lammps_function(
        working_directory=working_directory,
        structure=structure_1,
        potential=potential,
        calc_mode="md",
        calc_kwargs={
            "temperature": temperature_high,
            "n_ionic_steps": 1_000,
            "time_step": 1.0,
            "n_print": n_print,
            "initial_temperature": 0,
            "pressure": None,
        },
        cutoff_radius=None,
        units="metal",
        bonds_kwargs={},
        server_kwargs={},
        enable_h5md=False,
        write_restart_file=False,
        read_restart_file=False,
        restart_file="restart.out",
        executable_version=None,
        executable_path=None,
        input_control_file=None,
    )
    structure_2 = _get_structure(
        structure=structure_1,
        cell=parsed_output["generic"]["cells"][-1],
        indices=parsed_output["generic"]["indices"][-1],
        positions=parsed_output["generic"]["positions"][-1],
        unwrapped_positions=None,
        total_displacements=None,
        wrap_atoms=True,
    )
    structure_2.set_velocities(parsed_output["generic"]["velocities"][-1])
    for filename in os.listdir(working_directory):
        file_path = os.path.join(working_directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Run Lammps just as a function - step3
    _shell_output, parsed_output, _job_crashed = lammps_function(
        working_directory=working_directory,
        structure=structure_2,
        potential=potential,
        calc_mode="md",
        calc_kwargs={
            "temperature": [
                temperature_high,
                temperature_low,
            ],  # cooling  from 5000 down to 300 K
            # number of MD steps used for the cooling can be changes to calculate  specific rate.
            "n_ionic_steps": 47_000,
            "time_step": 1.0,
            "n_print": n_print,
            "initial_temperature": 0,
            "pressure": None,
        },
        cutoff_radius=None,
        units="metal",
        bonds_kwargs={},
        server_kwargs={},
        enable_h5md=False,
        write_restart_file=False,
        read_restart_file=False,
        restart_file="restart.out",
        executable_version=None,
        executable_path=None,
        input_control_file=None,
    )
    structure_3 = _get_structure(
        structure=structure_2,
        cell=parsed_output["generic"]["cells"][-1],
        indices=parsed_output["generic"]["indices"][-1],
        positions=parsed_output["generic"]["positions"][-1],
        unwrapped_positions=None,
        total_displacements=None,
        wrap_atoms=True,
    )
    structure_3.set_velocities(parsed_output["generic"]["velocities"][-1])
    for filename in os.listdir(working_directory):
        file_path = os.path.join(working_directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Run Lammps just as a function - step4
    _shell_output, parsed_output, _job_crashed = lammps_function(
        working_directory=working_directory,
        structure=structure_3,
        potential=potential,
        calc_mode="md",
        calc_kwargs={
            "temperature": temperature_low,  # cooling  from 5000 down to 300 K
            "pressure": 0.0,  # 0 MPa, release the pressure
            # number of MD steps used for the cooling can be changes to calculate  specific rate.
            "n_ionic_steps": 10_000,
            "time_step": 1.0,
            "n_print": n_print,
            "initial_temperature": 0,
        },
        cutoff_radius=None,
        units="metal",
        bonds_kwargs={},
        server_kwargs={},
        enable_h5md=False,
        write_restart_file=False,
        read_restart_file=False,
        restart_file="restart.out",
        executable_version=None,
        executable_path=None,
        input_control_file=None,
    )
    structure_4 = _get_structure(
        structure=structure_3,
        cell=parsed_output["generic"]["cells"][-1],
        indices=parsed_output["generic"]["indices"][-1],
        positions=parsed_output["generic"]["positions"][-1],
        unwrapped_positions=None,
        total_displacements=None,
        wrap_atoms=True,
    )
    structure_4.set_velocities(parsed_output["generic"]["velocities"][-1])
    for filename in os.listdir(working_directory):
        file_path = os.path.join(working_directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Run Lammps just as a function - step5
    _shell_output, parsed_output, _job_crashed = lammps_function(
        working_directory=working_directory,
        structure=structure_4,
        potential=potential,
        calc_mode="md",
        calc_kwargs={
            "temperature": temperature_low,  # cooling  from 5000 down to 300 K
            # number of MD steps used for the cooling can be changes to calculate  specific rate.
            "n_ionic_steps": 100_000,
            "time_step": 1.0,
            "n_print": n_print,
            "initial_temperature": 0,
        },
        cutoff_radius=None,
        units="metal",
        bonds_kwargs={},
        server_kwargs={},
        enable_h5md=False,
        write_restart_file=False,
        read_restart_file=False,
        restart_file="restart.out",
        executable_version=None,
        executable_path=None,
        input_control_file=None,
    )
    for filename in os.listdir(working_directory):
        file_path = os.path.join(working_directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    os.rmdir(working_directory)
    return {
        "steps": parsed_output["generic"]["steps"],
        "temperature": parsed_output["generic"]["temperature"],
    }
