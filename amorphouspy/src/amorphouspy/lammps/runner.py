"""Shared module for amorphouspy simulation workflows.

This module contains shared functionality which is reused in the individual workflows.
"""

import subprocess
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pandas as pd
from ase.atoms import Atoms
from lammpsparser.compatibility.file import lammps_file_interface_function

from amorphouspy.lammps.io import structure_from_parsed_output

LammpsPotential = str | pd.DataFrame | dict[str, Any]
LammpsPressure = int | float | list[int | float | None] | None


@contextmanager
def simulation_working_directory(tmp_working_directory: str | Path | None) -> Iterator[str]:
    """Yield a working directory for a single LAMMPS run.

    Ownership semantics depend on whether the caller supplies a location:

    * ``tmp_working_directory is None`` -- a directory is created in the
      operating system's temporary location and **removed automatically** when
      the context exits. This is the default, self-cleaning behaviour.
    * ``tmp_working_directory`` given -- a uniquely-named sub-directory is
      created inside it and **left in place** on exit. The caller owns it and is
      responsible for removing it. Run artefacts (``log.lammps``, ``lammps.data``,
      dumps) therefore remain available for inspection afterwards.

    Args:
        tmp_working_directory: Parent location for the run directory, or None to
            use an auto-cleaned system temporary directory. When given, it must
            already exist.

    Yields:
        The path to the working directory to run the simulation in.
    """
    if tmp_working_directory is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    else:
        # Caller-owned: unique sub-directory (avoids collisions across the many
        # runs of a multi-stage workflow) that is deliberately not deleted.
        yield tempfile.mkdtemp(dir=tmp_working_directory)


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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*Couldn't determine the LAMMPS to pyiron unit conversion type of quantity.*",
                category=UserWarning,
                module=r"lammpsparser\.units",
            )
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
    temperature: float,
    n_ionic_steps: int,
    timestep: float,
    initial_temperature: float | None = None,
    temperature_end: float | None = None,
    pressure: LammpsPressure = None,
    pressure_end: float | None = None,
    server_kwargs: dict[str, Any] | None = None,
    *,
    n_dump: int | None = None,
    n_print_thermo: int | None = None,
    input_control_file: dict[str, Any] | None = None,
    langevin: bool = False,
    seed: int | None = 12345,
    tmp_working_directory: str | Path | None = None,
    dump_final_structure: bool = True,
) -> tuple[Atoms, dict[str, Any]]:  # pylint: disable=too-many-positional-arguments
    """Run a LAMMPS MD calculation with given parameters and return the final structure and parsed output.

    Args:
        structure: The atomic structure to simulate.
        potential: The potential file to be used for the simulation.
        temperature: Start temperature (or constant temperature when ``temperature_end`` is None).
        n_ionic_steps: Number of MD steps to run.
        timestep: Time step for integration in femtoseconds.
        initial_temperature: Initial temperature for velocity initialization. If None, the initial
            temperature will be twice the target temperature (which would go immediately down to the target temperature
            as described in equipartition theorem). If 0, the velocity field is not initialized (in which case the
            initial velocity given in structure will be used and seed to initialize velocities will be ignored).
        temperature_end: End temperature for a linear temperature ramp. If None, temperature is held constant.
        pressure: Start pressure in GPa for NPT simulations. If None, NVT is used.
            A scalar selects isotropic NPT. A six-element list selects anisotropic
            or triclinic NPT.
        pressure_end: End pressure in GPa for a linear pressure ramp. Requires ``pressure`` to be set.
            The pressure ramp is injected as a custom LAMMPS ``fix npt`` command because the parser does not
            support pressure ramps natively. Does not work in combination with ``langevin``.
        server_kwargs: Additional keyword arguments for the server.
        n_dump: Dump frequency of structural output in simulation steps. If None,
            falls back to ``n_ionic_steps``.
        n_print_thermo: Thermodynamic print frequency in simulation steps. If None,
            falls back to ``n_dump``.
        input_control_file: Optional LAMMPS input overrides merged on top of the
            default generated controls.
        langevin: Whether to use Langevin dynamics for thermostats. Cannot be used in combination with ``pressure_end``.
        seed: Random seed for velocity initialization (default is 12345). May be None
            when the backend should choose a random seed. Ignored if `initial_temperature` is 0.
        tmp_working_directory: Specifies the location of the temporary directory to run the simulations.
            Per default (None), the directory is located in the operating systems location for temporary files
            and is removed automatically once the run finishes.
            With the specification of tmp_working_directory, a uniquely-named sub-directory is created inside
            it and left in place afterwards (the caller owns it and is responsible for removing it), so the run
            artefacts such as ``log.lammps`` remain available. tmp_working_directory needs to exist beforehand.
            Data will be cleaned after the simulation is finished.
        dump_final_structure: Whether to dump the final structure to a file. If False, dumping happens as specified.
            If True, adds an additional dump command to ensure that the final structure is always dumped. Internal
            check avoids that the same structure is dumped twice if the final step is already a dump step. Defaults
            to True.

    Returns:
        A tuple containing:
            - structure_final: The final atomic structure.
            - parsed_output: The parsed output dictionary.

    """
    if pressure_end is not None and pressure is None:
        msg = "pressure must be set when pressure_end is specified."
        raise ValueError(msg)
    if pressure_end is not None and isinstance(pressure, list):
        msg = "pressure must be a scalar when pressure_end is specified."
        raise ValueError(msg)

    # Creates a working directory for the simulation (auto-cleaned when
    # tmp_working_directory is None; caller-owned otherwise).
    with simulation_working_directory(tmp_working_directory) as tmpdir:
        tmp_path = str(Path(tmpdir))

        temp_setting: float | list[float] = (
            [temperature, temperature_end] if temperature_end is not None else temperature
        )
        t_start = temperature
        t_end = temperature_end if temperature_end is not None else temperature

        if n_dump is None:
            n_dump = n_ionic_steps
        if n_print_thermo is None:
            n_print_thermo = n_dump

        effective_n_dump = min(n_dump, n_ionic_steps)
        effective_n_print_thermo = min(n_print_thermo, n_ionic_steps)

        input_control: dict[str, Any] = {
            "dump_modify": f"1 every {effective_n_dump} first yes",
            "thermo": f"{effective_n_print_thermo}",
            "thermo_style": "custom step temp density pe etotal pxx pxy pxz pyy pyz pzz vol",
            "thermo_modify": "flush no",
        }

        # Pressure ramp: the parser cannot express [P_start → P_end] natively, so inject a
        # custom fix npt command that overrides whatever the parser would generate.
        if pressure_end is not None:
            if langevin:
                msg = "langevin cannot be used in combination with pressure ramps via ``pressure_end``."
                raise ValueError(msg)
            assert isinstance(pressure, int | float), "pressure must be a scalar when pressure_end is given"
            p_start_bar = pressure * 10_000  # GPa → bar (LAMMPS metal units)
            p_end_bar = pressure_end * 10_000
            input_control["fix"] = f"ensemble all npt temp {t_start} {t_end} 0.1 iso {p_start_bar} {p_end_bar} 1.0"
            passed_pressure: LammpsPressure = pressure  # scalar to put parser in NPT mode
        else:
            passed_pressure = pressure

        if input_control_file is not None:
            input_control.update(input_control_file)

        if initial_temperature is None:
            initial_temperature = 2 * temperature

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
                "n_print": effective_n_dump,
                "initial_temperature": initial_temperature,
                "seed": seed,
                "pressure": passed_pressure,
                "langevin": langevin,
            },
            units="metal",
            write_restart_file=False,
            read_restart_file=False,
            restart_file="restart.out",
            input_control_file=input_control,
            lmp_command=get_lammps_command(server_kwargs=server_kwargs),
            dump_final_structure=dump_final_structure,
        )

        # Retrieves the final structure from the parsed output
        new_structure = structure_from_parsed_output(initial_structure=structure, parsed_output=parsed_output)

    return new_structure, parsed_output


def get_lammps_command(server_kwargs: dict | None = None) -> str:
    """Generate a portable LAMMPS MPI command.

    Args:
        server_kwargs: Server dictionary for example: {"cores": 2}.

    Returns:
        LAMMPS command as a string.

    """
    lmp_command = "mpiexec -n 1 lmp_mpi -in lmp.in"
    if server_kwargs is not None and isinstance(server_kwargs, dict) and "cores" in server_kwargs:
        lmp_command = f"mpiexec -n {server_kwargs['cores']} lmp_mpi -in lmp.in"
    return lmp_command
