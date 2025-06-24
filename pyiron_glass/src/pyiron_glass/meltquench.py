import os
import shutil
import numpy as np
from ase.atoms import Atoms
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
    """
    Return an updated `Atoms` object based on the provided information.

    Parameters
    ----------
    structure : Atoms
        The reference atomic structure.
    cell : ndarray
        The simulation cell to assign to the new structure.
    indices : ndarray
        Indices of the atoms to include in the new snapshot.
    positions : ndarray, optional
        Wrapped atomic positions.
    unwrapped_positions : ndarray, optional
        Unwrapped atomic positions.
    total_displacements : ndarray, optional
        Total atomic displacements to be added to the initial positions.
    wrap_atoms : bool, optional
        Whether to wrap atoms inside the unit cell (default is True).

    Returns
    -------
    Atoms
        The newly constructed atomic structure with updated positions and cell.
    """

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
    elif unwrapped_positions is not None:
        snapshot.positions = unwrapped_positions
    else:
        snapshot.positions += total_displacements

    return snapshot


def _run_lammps_md(
    structure,
    potential,
    working_directory,
    temperature,
    n_ionic_steps,
    timestep,
    n_print,
    initial_temperature,
    temperature_end=None,
    pressure=None,
    langevin=False,
    seed=12345,
):  # pylint: disable=too-many-positional-arguments
    """
    Run a LAMMPS MD calculation with given parameters and return the final structure and parsed output.

    Parameters
    ----------
    structure : Atoms
        The atomic structure to simulate.
    potential : str
        The potential file to be used for the simulation.
    working_directory : str
        The directory where the simulation files will be stored.
    temperature : float or list
        The target temperature for the MD run. Can be a single value or a list [start, end].
    n_ionic_steps : int
        Number of MD steps to run.
    timestep : float
        Time step for integration in femtoseconds.
    n_print : int
        Frequency of output writing in simulation steps.
    initial_temperature : None or float
        Initial temperature according to which the initial velocity field is created. If None, the initial
        temperature will be twice the target temperature (which would go immediately down to the target temperature
        as described in equipartition theorem). If 0, the velocity field is not initialized (in which case the
        initial velocity given in structure will be used and seed to initialize velocities will be ignored).
    temperature_end : float, optional
        Final temperature for ramping. If None, no temperature ramp is applied.
    pressure : float, optional
        Target pressure for NPT simulations. If None, NVT is used.
    langevin : bool, optional
        Whether to use Langevin dynamics
    seed : int, optional
        Random seed for velocity initialization (default is 12345). Ignored if `initial_temperature` is 0.

    Returns
    -------
    tuple
        A tuple (structure, parsed_output) with the final structure and the simulation output dictionary.
    """

    temp_setting = [temperature, temperature_end] if temperature_end is not None else temperature

    _shell_output, parsed_output, _job_crashed = lammps_function(
        working_directory=working_directory,
        structure=structure,
        potential=potential,
        calc_mode="md",
        calc_kwargs={
            "temperature": temp_setting,
            "n_ionic_steps": n_ionic_steps,
            "time_step": timestep,
            "n_print": n_print,
            "initial_temperature": initial_temperature,
            "seed": seed,
            "pressure": pressure,
            "langevin": langevin,
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

    new_structure = _get_structure(
        structure=structure,
        cell=parsed_output["generic"]["cells"][-1],
        indices=parsed_output["generic"]["indices"][-1],
        positions=parsed_output["generic"]["positions"][-1],
        wrap_atoms=True,
    )
    new_structure.set_velocities(parsed_output["generic"]["velocities"][-1])

    # see issue #32: Consider implementing a more robust cleanup procedure for temporary files.
    _clean_directory(working_directory)

    return new_structure, parsed_output


def _clean_directory(directory):
    """
    Remove all files in the specified directory.

    Parameters
    ----------
    directory : str
        Path to the directory to be cleaned.
    """

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


@job
def melt_quench_simulation(
    structure,
    potential,
    working_directory,
    temperature_high=5000.0,
    temperature_low=300.0,
    timestep=1.0,
    heating_rate=1e12,
    cooling_rate=1e12,
    n_print=1000,
    langevin=False,
    seed=12345,
):  # pylint: disable=too-many-positional-arguments
    """
    Perform a melt-quench simulation using LAMMPS via pyiron_atomistics.
    This function heats a structure to a high temperature, equilibrates it,
    and then cools it down to a low temperature, simulating a melt-quench process.
    The heating and cooling rates are given in K/s, and the conversion into simulation steps is done automatically.

    Parameters
    ----------
    structure : Atoms
        The initial atomic structure to be melted and quenched.
    potential : str
        The potential file to be used for the simulation.
    working_directory : str
        The directory where the simulation files will be stored.
    temperature_high : float, optional
        The high temperature to which the structure will be heated (default is 5000.0 K).
    temperature_low : float, optional
        The low temperature to which the structure will be cooled (default is 300.0 K).
    n_print : int, optional
        The frequency of output during the simulation (default is 1000).
    heating_rate : float, optional
        The rate at which the temperature is increased during the heating phase, in K/s (default is 1e12 K/s).
    cooling_rate : float, optional
        The rate at which the temperature is decreased during the cooling phase, in K/s (default is 1e12 K/s).
    langevin : bool, optional
        Whether to use Langevin dynamics.
    seed : int, optional
        Random seed for velocity initialization (default is 12345). Ignored if `initial_temperature` is 0.

    Returns
    -------
    dict
        A dictionary containing the simulation steps and temperature data.
    """
    os.makedirs(working_directory, exist_ok=True)

    seconds_to_femtos = 1e15
    heating_steps = int(((temperature_high - temperature_low) / (timestep * heating_rate)) * seconds_to_femtos)
    cooling_steps = int(((temperature_high - temperature_low) / (timestep * cooling_rate)) * seconds_to_femtos)

    # Stage 1: Heating from low to high T
    structure, _ = _run_lammps_md(
        structure=structure,
        potential=potential,
        working_directory=working_directory,
        temperature=temperature_low,
        temperature_end=temperature_high,
        n_ionic_steps=heating_steps,
        timestep=timestep,
        n_print=n_print,
        initial_temperature=temperature_low,
        langevin=langevin,
        seed=seed,
    )

    # Stage 2: Equilibration at high T
    structure, _ = _run_lammps_md(
        structure=structure,
        potential=potential,
        working_directory=working_directory,
        temperature=temperature_high,
        n_ionic_steps=1_000,
        timestep=timestep,
        n_print=n_print,
        initial_temperature=0,
        langevin=langevin,
    )

    # Stage 3: Cooling from high to low T
    structure, _ = _run_lammps_md(
        structure=structure,
        potential=potential,
        working_directory=working_directory,
        temperature=temperature_high,
        temperature_end=temperature_low,
        n_ionic_steps=cooling_steps,
        timestep=timestep,
        n_print=n_print,
        initial_temperature=0,
        langevin=langevin,
    )

    # Stage 4: Pressure release at low T
    structure, _ = _run_lammps_md(
        structure=structure,
        potential=potential,
        working_directory=working_directory,
        temperature=temperature_low,
        n_ionic_steps=10_000,
        timestep=timestep,
        n_print=n_print,
        initial_temperature=0,
        pressure=0.0,
        langevin=langevin,
    )

    # Stage 5: Long equilibration at low T
    structure_final, parsed_output = _run_lammps_md(
        structure=structure,
        potential=potential,
        working_directory=working_directory,
        temperature=temperature_low,
        n_ionic_steps=100_000,
        timestep=timestep,
        n_print=n_print,
        initial_temperature=0,
        langevin=langevin,
    )

    # see issue #32: Consider implementing a more robust cleanup procedure for temporary files.
    shutil.rmtree(working_directory)

    return {
        "structure": structure_final,
        "steps": parsed_output["generic"]["steps"],
        "temperature": parsed_output["generic"]["temperature"],
        "generic": parsed_output["generic"],
    }
