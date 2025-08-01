"""Viscosity simulation workflows for glass systems using LAMMPS and Green-Kubo method."""

from pathlib import Path

import numpy as np
from ase.atoms import Atoms
from pyiron_atomistics.lammps.lammps import lammps_function
from pyiron_base import job
from structuretoolkit.common import center_coordinates_in_unit_cell


def _get_structure(
    structure: Atoms,
    cell: np.ndarray,
    indices: np.ndarray,
    positions: np.ndarray | None = None,
    unwrapped_positions: np.ndarray | None = None,
    total_displacements: np.ndarray | None = None,
    *,
    wrap_atoms: bool = True,
) -> Atoms:
    """Return an updated `Atoms` object based on the provided information.

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
            positions=np.zeros((*indices.shape, 3)),
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
    structure: Atoms,
    potential: str,
    working_directory: str,
    temperature: float | list[float],
    n_ionic_steps: int,
    timestep: float,
    n_print: int,
    initial_temperature: float,
    pressure: float | None = None,
    *,
    delete_folder: bool = False,
    langevin: bool = False,
    seed: int = 12345,
) -> tuple[Atoms, dict]:  # pylint: disable=too-many-positional-arguments
    """Run a LAMMPS MD calculation with given parameters and return the final structure and parsed output.

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
    pressure : float, optional
        Target pressure for NPT simulations. If None, NVT is used.
    delete_folder : bool, optional
        decide to keep the simulation output on the disk or delete it.
    langevin : bool, optional
        Whether to use Langevin dynamics
    seed : int, optional
        Random seed for velocity initialization (default is 12345). Ignored if `initial_temperature` is 0.

    Returns
    -------
    tuple
        A tuple (structure, parsed_output) with the final structure and the simulation output dictionary.

    """
    temp_setting = temperature

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
        input_control_file={
            "dump_modify": f"1 every {n_ionic_steps} first yes",
            "thermo_style": "custom step temp density pe etotal pxx pxy pxz pyy pyz pzz vol",
        },
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
    if delete_folder:
        _clean_directory(working_directory)

    return new_structure, parsed_output


def _clean_directory(directory: str) -> None:
    """Remove all files in the specified directory.

    Parameters
    ----------
    directory : str
        Path to the directory to be cleaned.

    """
    directory_path = Path(directory)
    for file_path in directory_path.iterdir():
        if file_path.is_file():
            file_path.unlink()


@job
def viscosity_simulation(
    structure: Atoms,
    potential: str,
    working_directory: str,
    temperature_sim: float = 5000.0,
    timestep: float = 1.0,
    production_steps: int = 10_000_000,
    n_print: int = 1,
    *,
    langevin: bool = False,
    seed: int = 12345,
) -> dict:  # pylint: disable=too-many-positional-arguments
    """Perform a viscosity simulation using LAMMPS via pyiron_atomistics.

    This function equilibrate a structure at predefined temperature, and preform
    a production run to get instantaneous off diagonal stresses or pressures for viscosity calculation.
    The number of steps used here is only for testing purposes.

    Parameters
    ----------
    structure : Atoms
        The initial atomic structure to be melted and quenched.
    potential : str
        The potential file to be used for the simulation.
    working_directory : str
        The directory where the simulation files will be stored.
    temperature_sim : float, optional
        The temperature at which the structure will be equilibrated (default is 5000.0 K).
    timestep : float, optional
        Time step for integration in femtoseconds (default is 1.0 fs).
    production_steps : float, optional
        The number of steps for the production.
    n_print : int, optional
        The frequency of output during the simulation (default is 1000).
    langevin : bool, optional
        Whether to use Langevin dynamics.
    seed : int, optional
        Random seed for velocity initialization (default is 12345). Ignored if `initial_temperature` is 0.

    Returns
    -------
    dict
        A dictionary containing the simulation steps and temperature data.

    """
    Path(working_directory).mkdir(parents=True, exist_ok=True)

    # Stage 1: NVT at T
    structure1, _ = _run_lammps_md(
        structure=structure,
        potential=potential,
        working_directory=working_directory,
        temperature=temperature_sim,
        n_ionic_steps=100_000,
        timestep=timestep,
        n_print=1000,
        initial_temperature=temperature_sim,
        langevin=langevin,
        seed=seed,
        delete_folder=True,
    )

    # Stage 2: Equilibration in NPT at T
    structure2, _ = _run_lammps_md(
        structure=structure1,
        potential=potential,
        working_directory=working_directory,
        temperature=temperature_sim,
        n_ionic_steps=1_000_000,
        timestep=timestep,
        n_print=1000,
        initial_temperature=0,
        pressure=0.0,
        langevin=langevin,
        delete_folder=True,
    )

    # Stage 3: Equilibration NVT at T
    structure3, _ = _run_lammps_md(
        structure=structure2,
        potential=potential,
        working_directory=working_directory,
        temperature=temperature_sim,
        n_ionic_steps=10_000,
        timestep=timestep,
        n_print=1000,
        initial_temperature=0,
        langevin=langevin,
        delete_folder=True,
    )

    # Stage 4: Production simulation for viscosity at T
    structure_final, parsed_output = _run_lammps_md(
        structure=structure3,
        potential=potential,
        working_directory=working_directory,
        temperature=temperature_sim,
        n_ionic_steps=production_steps,
        timestep=timestep,
        n_print=n_print,
        initial_temperature=0,
        langevin=langevin,
        delete_folder=True,
    )

    return {
        "structure": structure_final,
        "generic": parsed_output["generic"],
    }
