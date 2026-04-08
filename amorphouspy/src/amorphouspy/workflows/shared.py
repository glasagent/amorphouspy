"""Shared module for amorphouspy simulation workflows.

This module contains shared functionality which is reused in the individual workflows.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

import pandas as pd
from ase.atoms import Atoms
from lammpsparser.compatibility.file import lammps_file_interface_function

from amorphouspy.io_utils import structure_from_parsed_output

LammpsPotential = str | pd.DataFrame | dict[str, Any]


def run_lammps_with_error_capture(working_directory: str, **kwargs: Any) -> dict:  # noqa: ANN401
    """Wrap ``lammps_file_interface_function``, capturing LAMMPS output on failure.

    On ``subprocess.CalledProcessError`` the wrapper reads any available stdout,
    stderr and the tail of ``log.lammps`` from *working_directory* and re-raises
    as a ``RuntimeError`` so the caller (and eventually the API) gets actionable
    diagnostics instead of just an exit-code message.

    Also checks the ``job_crashed`` flag and validates that the parsed output
    contains ``generic`` and ``lammps`` keys, raising on soft failures.

    All keyword arguments are forwarded to ``lammps_file_interface_function``.

    Returns:
        The parsed LAMMPS output dictionary.
    """
    try:
        _shell_output, parsed_output, job_crashed = lammps_file_interface_function(
            working_directory=working_directory, **kwargs
        )
    except subprocess.CalledProcessError as exc:
        details = [str(exc)]
        if exc.output:
            details.append(f"LAMMPS stdout:\n{exc.output[-2000:]}")
        if exc.stderr:
            details.append(f"LAMMPS stderr:\n{exc.stderr[-2000:]}")
        log_file = Path(working_directory) / "log.lammps"
        if log_file.exists():
            with log_file.open("rb") as _lf:
                _lf.seek(max(0, log_file.stat().st_size - 2000))
                log_tail = _lf.read().decode(errors="replace")
            details.append(f"log.lammps (last 2000 chars):\n{log_tail}")
        raise RuntimeError("\n".join(details)) from exc

    if job_crashed or parsed_output.get("generic") is None or parsed_output.get("lammps") is None:
        details = [f"LAMMPS crashed in {working_directory}."]
        log_file = Path(working_directory) / "log.lammps"
        if log_file.exists():
            with log_file.open("rb") as _lf:
                _lf.seek(max(0, log_file.stat().st_size - 2000))
                log_tail = _lf.read().decode(errors="replace")
            details.append(f"log.lammps (last 2000 chars):\n{log_tail}")
        raise RuntimeError("\n".join(details))

    return parsed_output


def _run_lammps_md(
    structure: Atoms,
    potential: LammpsPotential,
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
        parsed_output = run_lammps_with_error_capture(
            working_directory=tmp_path,
            structure=structure,
            potential=cast("Any", potential),
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
    if server_kwargs is not None and isinstance(server_kwargs, dict) and "cores" in server_kwargs:
        lmp_command = f"mpiexec -n {server_kwargs['cores']} --oversubscribe lmp_mpi -in lmp.in"
    return lmp_command
