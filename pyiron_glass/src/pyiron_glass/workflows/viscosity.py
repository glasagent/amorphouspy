"""Viscosity simulation workflows for glass systems using LAMMPS and Green-Kubo method."""

import tempfile
from pathlib import Path

from ase.atoms import Atoms
from pyiron_atomistics.lammps.lammps import lammps_function
from pyiron_base import job

from pyiron_glass.io_utils import structure_from_parsed_output


def _run_lammps_md(
    structure: Atoms,
    potential: str,
    temperature: float | list[float],
    n_ionic_steps: int,
    timestep: float,
    n_print: int,
    initial_temperature: float,
    pressure: float | None = None,
    server_kwargs: dict | None = None,
    *,
    langevin: bool = False,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
) -> tuple[Atoms, dict]:  # pylint: disable=too-many-positional-arguments
    """Run a LAMMPS MD calculation with given parameters and return the final structure and parsed output.

    Parameters
    ----------
    structure : Atoms
        The atomic structure to simulate.
    potential : str
        The potential file to be used for the simulation.
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
    server_kwargs : dict | None, optional
        Additional keyword arguments for the server.
    langevin : bool, optional
        Whether to use Langevin dynamics
    seed : int, optional
        Random seed for velocity initialization (default is 12345). Ignored if `initial_temperature` is 0.
    tmp_working_directory : str | Path | None
        Specifies the location of the temporary directory to run the simulations. Per default (None), the
        directory is located in the operating systems location for temperary files. With the specification
        of tmp_working_directory, the temporary directory is created in the specified location. Therefore,
        tmp_working_directory needs to exist beforehand.


    Returns
    -------
    tuple
        A tuple (structure, parsed_output) with the final structure and the simulation output dictionary.

    """
    # Creates a temporary directory for the simulation in the specified working directory.
    with tempfile.TemporaryDirectory(dir=tmp_working_directory) as tmpdir:
        tmp_path = str(Path(tmpdir))

        # defines the temperature protocol
        temp_setting = temperature

        # Sets up the LAMMPS simulations
        _shell_output, parsed_output, _job_crashed = lammps_function(
            working_directory=tmp_path,
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
            server_kwargs=server_kwargs,
            enable_h5md=False,
            write_restart_file=False,
            read_restart_file=False,
            restart_file="restart.out",
            executable_path=None,
            input_control_file={
                "dump_modify": f"1 every {n_ionic_steps} first yes",
                "thermo_style": "custom step temp density pe etotal pxx pxy pxz pyy pyz pzz vol",
                "thermo_modify": "flush yes",
            },
        )

        # Retrives the final structure from the parsed output
        new_structure = structure_from_parsed_output(initial_structure=structure, parsed_output=parsed_output)

    return new_structure, parsed_output


@job
def viscosity_simulation(
    structure: Atoms,
    potential: str,
    temperature_sim: float = 5000.0,
    timestep: float = 1.0,
    production_steps: int = 10_000_000,
    n_print: int = 1,
    server_kwargs: dict | None = None,
    *,
    langevin: bool = False,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
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
    tmp_working_directory : str
        The directory where the simulation files will be stored.
    temperature_sim : float, optional
        The temperature at which the structure will be equilibrated (default is 5000.0 K).
    timestep : float, optional
        Time step for integration in femtoseconds (default is 1.0 fs).
    production_steps : float, optional
        The number of steps for the production.
    n_print : int, optional
        The frequency of output during the simulation (default is 1000).
    server_kwargs : dict | None, optional
        Additional arguments for the server.
    langevin : bool, optional
        Whether to use Langevin dynamics.
    seed : int, optional
        Random seed for velocity initialization (default is 12345). Ignored if `initial_temperature` is 0.

    Returns
    -------
    dict
        A dictionary containing the simulation steps and temperature data.

    """
    # Stage 0: Langevin dynamics at T
    structure0, _ = _run_lammps_md(
        structure=structure,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=5000,
        n_ionic_steps=10_000,
        timestep=timestep,
        n_print=1000,
        initial_temperature=temperature_sim,
        langevin=True,
        seed=seed,
        server_kwargs=server_kwargs,
    )

    # Stage 1: cooling to T of interest in NVT
    structure1, _ = _run_lammps_md(
        structure=structure0,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=[5000, temperature_sim],
        n_ionic_steps=10_000,
        timestep=timestep,
        n_print=1000,
        initial_temperature=temperature_sim,
        langevin=langevin,
        seed=seed,
        server_kwargs=server_kwargs,
    )

    # Stage 2: Equilibration in NVT at T
    structure2, _ = _run_lammps_md(
        structure=structure1,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=temperature_sim,
        n_ionic_steps=10_000,
        timestep=timestep,
        n_print=1000,
        initial_temperature=temperature_sim,
        langevin=langevin,
        seed=seed,
        server_kwargs=server_kwargs,
    )

    # Stage 3: Equilibration in NPT at T
    structure3, _ = _run_lammps_md(
        structure=structure2,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=temperature_sim,
        n_ionic_steps=10_000,
        timestep=timestep,
        n_print=1000,
        initial_temperature=0,
        pressure=0.0,
        langevin=langevin,
        server_kwargs=server_kwargs,
    )

    # Stage 4: Equilibration NVT at T
    structure4, _ = _run_lammps_md(
        structure=structure3,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=temperature_sim,
        n_ionic_steps=10_000,
        timestep=timestep,
        n_print=1000,
        initial_temperature=0,
        langevin=langevin,
        server_kwargs=server_kwargs,
    )

    # Stage 5: Production simulation for viscosity at T
    structure_final, parsed_output = _run_lammps_md(
        structure=structure4,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=temperature_sim,
        n_ionic_steps=production_steps,
        timestep=timestep,
        n_print=n_print,
        initial_temperature=0,
        langevin=langevin,
        server_kwargs=server_kwargs,
    )

    result = parsed_output.get("generic", None)

    if result is None:
        msg = "The 'generic' key is missing from parsed_output."
        raise KeyError(msg)
        result = {}

    return {"result": result}
