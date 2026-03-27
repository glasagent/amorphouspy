"""Shared module for amorphouspy simulation workflows.

This module contains shared functionality which is reused in the individual workflows.
"""

import tempfile
from pathlib import Path
from typing import Any

from ase.atoms import Atoms
from lammpsparser.compatibility.file import lammps_file_interface_function

from amorphouspy.io_utils import structure_from_parsed_output


def _run_lammps_md(
    structure: Atoms,
    potential: str,
    temperature: float | list[float],
    n_ionic_steps: int,
    timestep: float,
    n_print: int,
    initial_temperature: float,
    pressure: float | None = None,
    server_kwargs: dict[str, Any] | None = None,
    *,
    langevin: bool = False,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
) -> tuple[Atoms, dict[str, Any]]:  # pylint: disable=too-many-positional-arguments
    """Run a LAMMPS MD calculation with given parameters and return the final structure and parsed output.

    Args:
        structure: The atomic structure to simulate.
        potential: The potential file to be used for the simulation.
        temperature: The target temperature for the MD run. Can be a single value or a list [start, end].
        n_ionic_steps: Number of MD steps to run.
        timestep: Time step for integration in femtoseconds.
        n_print: Frequency of output writing in simulation steps.
        initial_temperature: Initial temperature for velocity initialization. If None, the initial
            temperature will be twice the target temperature (which would go immediately down to the target temperature
            as described in equipartition theorem). If 0, the velocity field is not initialized (in which case the
            initial velocity given in structure will be used and seed to initialize velocities will be ignored).
        pressure: Target pressure for NPT simulations. If None, NVT is used.
        server_kwargs: Additional keyword arguments for the server.
        langevin: Whether to use Langevin dynamics.
        seed: Random seed for velocity initialization (default is 12345). Ignored if `initial_temperature` is 0.
        tmp_working_directory: Specifies the location of the temporary directory to run the simulations.
            Per default (None), the directory is located in the operating systems location for temporary files.
            With the specification of tmp_working_directory, the temporary directory is created in the specified
            location. Therefore, tmp_working_directory needs to exist beforehand.

    Returns:
        A tuple containing:
            - structure_final: The final atomic structure.
            - parsed_output: The parsed output dictionary.

    """
    # Creates a temporary directory for the simulation in the specified working directory.
    with tempfile.TemporaryDirectory(dir=tmp_working_directory) as tmpdir:
        tmp_path = str(Path(tmpdir))

        # defines the temperature protocol
        temp_setting = temperature

        # Sets up the LAMMPS simulations
        _shell_output, parsed_output, _job_crashed = lammps_file_interface_function(
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
            units="metal",
            write_restart_file=False,
            read_restart_file=False,
            restart_file="restart.out",
            input_control_file={
                "dump_modify": f"1 every {n_ionic_steps} first yes",
                "thermo_style": "custom step temp density pe etotal pxx pxy pxz pyy pyz pzz vol",
                "thermo_modify": "flush yes",
            },
            lmp_command=get_lammps_command(server_kwargs=server_kwargs),
        )

        # Retrieves the final structure from the parsed output
        new_structure = structure_from_parsed_output(initial_structure=structure, parsed_output=parsed_output)

    return new_structure, parsed_output


def get_lammps_command(server_kwargs: dict | None = None) -> str:
    """Generate a LAMMPS command, by default this function returns: "mpiexec -n 1 --oversubscribe lmp_mpi -in lmp.in".

    Args:
        server_kwargs: Server dictionary for example: {"cores": 2}.

    Returns:
        LAMMPS command as a string.

    """
    lmp_command = "mpiexec -n 1 --oversubscribe lmp_mpi -in lmp.in"
    if server_kwargs is not None:
        if isinstance(server_kwargs, dict) and len(server_kwargs) == 1:
            if "cores" in server_kwargs:
                lmp_command = "mpiexec -n {} --oversubscribe lmp_mpi -in lmp.in".format(str(server_kwargs["cores"]))
            else:
                raise ValueError("Server dictionary error: " + str(server_kwargs))
        elif isinstance(server_kwargs, (dict, list)) and len(server_kwargs) == 0:
            pass
        else:
            raise ValueError("Server dictionary error: " + str(server_kwargs))
    return lmp_command
