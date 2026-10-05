"""Shared module for amorphouspy simulation workflows.

This module contains shared functionality which is reused in the individual workflows.
"""

import subprocess
import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast, get_args

import numpy as np
import pandas as pd
from ase.atoms import Atoms
from lammpsparser.compatibility.file import lammps_file_interface_function

from amorphouspy.lammps.io import structure_from_parsed_output

LammpsPotential = str | pd.DataFrame | dict[str, Any]
LammpsPressure = int | float | list[int | float | None] | None
Ensemble = Literal["nve", "nvt", "npt", "nvt_langevin", "npt_langevin"]
Thermostat = Literal["nose_hoover", "langevin"]


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


def thermostat_ensembles(thermostat: Thermostat) -> tuple[Ensemble, Ensemble]:
    """Return the NVT and NPT ensemble names for a thermostat.

    Args:
        thermostat: ``"nose_hoover"`` or ``"langevin"``.

    Returns:
        ``("nvt", "npt")`` for ``"nose_hoover"``, ``("nvt_langevin", "npt_langevin")`` for ``"langevin"``.

    Raises:
        ValueError: If ``thermostat`` is unknown.

    """
    if thermostat == "nose_hoover":
        return "nvt", "npt"
    if thermostat == "langevin":
        return "nvt_langevin", "npt_langevin"
    msg = f"thermostat must be one of {get_args(Thermostat)}, got {thermostat!r}."
    raise ValueError(msg)


def _validate_npt_pressure(ensemble: str, npt_pressure: LammpsPressure, npt_pressure_end: float | None) -> None:
    """Check that the pressure arguments of an MD run are consistent with its ensemble.

    Args:
        ensemble: Name of the thermodynamic ensemble.
        npt_pressure: Target (start) pressure in GPa.
        npt_pressure_end: End pressure in GPa for a linear pressure ramp.

    Raises:
        ValueError: If a pressure is given for a non-npt ensemble, missing for an npt ensemble, or the
            pressure ramp is combined with an anisotropic pressure or a Langevin thermostat.
    """
    is_npt = ensemble.startswith("npt")
    if not is_npt and (npt_pressure is not None or npt_pressure_end is not None):
        msg = f"npt_pressure and npt_pressure_end can only be used with an npt ensemble, got {ensemble!r}."
        raise ValueError(msg)
    if is_npt and npt_pressure is None:
        msg = f"npt_pressure must be set for ensemble {ensemble!r}."
        raise ValueError(msg)
    if npt_pressure_end is not None and isinstance(npt_pressure, list):
        msg = "npt_pressure must be a scalar when npt_pressure_end is specified."
        raise ValueError(msg)
    if npt_pressure_end is not None and ensemble == "npt_langevin":
        msg = "npt_pressure_end (pressure ramp) cannot be used with ensemble 'npt_langevin'."
        raise ValueError(msg)


def _validate_ensemble(
    structure: Atoms,
    ensemble: str,
    temperature: float | None,
    temperature_end: float | None,
    npt_pressure: LammpsPressure,
    npt_pressure_end: float | None,
    initial_temperature: float | None,
) -> float | None:
    """Check that the arguments of an MD run are consistent with its ensemble.

    Args:
        structure: The atomic structure to simulate.
        ensemble: One of ``"nve"``, ``"nvt"``, ``"npt"``, ``"nvt_langevin"`` or ``"npt_langevin"``.
        temperature: Target (start) temperature in K.
        temperature_end: End temperature in K for a linear temperature ramp.
        npt_pressure: Target (start) pressure in GPa.
        npt_pressure_end: End pressure in GPa for a linear pressure ramp.
        initial_temperature: Requested temperature for velocity initialization.

    Returns:
        The initial temperature to pass on. For ``"nve"``, None becomes 0, i.e. keep the velocities of ``structure``.

    Raises:
        ValueError: If ``ensemble`` is unknown or any argument does not fit it.
    """
    if ensemble not in get_args(Ensemble):
        msg = f"ensemble must be one of {get_args(Ensemble)}, got {ensemble!r}."
        raise ValueError(msg)
    _validate_npt_pressure(ensemble, npt_pressure, npt_pressure_end)
    if ensemble != "nve":
        if temperature is None:
            msg = f"temperature must be set for ensemble {ensemble!r}."
            raise ValueError(msg)
        return initial_temperature
    if temperature is not None or temperature_end is not None:
        msg = "temperature and temperature_end cannot be used with ensemble 'nve'."
        raise ValueError(msg)
    if initial_temperature is None:
        initial_temperature = 0.0
    if initial_temperature <= 0 and np.allclose(structure.get_velocities(), 0.0):
        msg = "ensemble 'nve' with initial_temperature 0 needs a structure that carries non-zero velocities."
        raise ValueError(msg)
    return initial_temperature


def _run_lammps_md(
    structure: Atoms,
    potential: LammpsPotential,
    n_ionic_steps: int,
    timestep: float,
    temperature: float | None = None,
    initial_temperature: float | None = None,
    temperature_end: float | None = None,
    npt_pressure: LammpsPressure = None,
    npt_pressure_end: float | None = None,
    server_kwargs: dict[str, Any] | None = None,
    *,
    ensemble: Ensemble,
    n_dump: int | None = None,
    n_print_thermo: int | None = None,
    input_control_file: dict[str, Any] | None = None,
    seed: int | None = 12345,
    tmp_working_directory: str | Path | None = None,
    dump_final_structure: bool = True,
) -> tuple[Atoms, dict[str, Any]]:  # pylint: disable=too-many-positional-arguments
    """Run a LAMMPS MD calculation with given parameters and return the final structure and parsed output.

    Args:
        structure: The atomic structure to simulate.
        potential: The potential file to be used for the simulation.
        n_ionic_steps: Number of MD steps to run.
        timestep: Time step for integration in femtoseconds.
        temperature: Start temperature (or constant temperature when ``temperature_end`` is None).
            Required for every ensemble except ``"nve"``, where it must be None.
        initial_temperature: Initial temperature for velocity initialization. If None, the initial
            temperature will be twice the target temperature (which would go immediately down to the target temperature
            as described in equipartition theorem). If 0, the velocity field is not initialized (in which case the
            initial velocity given in structure will be used and seed to initialize velocities will be ignored).
            For ``ensemble="nve"``, None means 0, so the velocities carried by ``structure`` are kept.
        temperature_end: End temperature for a linear temperature ramp. If None, temperature is held constant.
            Not allowed for ``ensemble="nve"``.
        npt_pressure: Start pressure in GPa. Required for the ``"npt"`` and ``"npt_langevin"`` ensembles and not
            allowed otherwise. A scalar selects isotropic NPT. A six-element list selects anisotropic
            or triclinic NPT.
        npt_pressure_end: End pressure in GPa for a linear pressure ramp. Requires a scalar ``npt_pressure``.
            The pressure ramp is injected as a custom LAMMPS ``fix npt`` command because the parser does not
            support pressure ramps natively. Only allowed for ``ensemble="npt"``.
        server_kwargs: Additional keyword arguments for the server.
        n_dump: Dump frequency of structural output in simulation steps. If None,
            falls back to ``n_ionic_steps``.
        n_print_thermo: Thermodynamic print frequency in simulation steps. If None,
            falls back to ``n_dump``.
        input_control_file: Optional LAMMPS input overrides merged on top of the
            default generated controls.
        ensemble: Thermodynamic ensemble. ``"nve"`` runs unthermostatted microcanonical dynamics (``fix nve``)
            starting from the velocities in ``structure`` unless a positive ``initial_temperature`` is given.
            ``"nvt"`` and ``"npt"`` use Nosé-Hoover thermostat/barostat. ``"nvt_langevin"`` uses ``fix nve``
            with a Langevin thermostat, ``"npt_langevin"`` uses ``fix nph`` with a Langevin thermostat.
        seed: Random seed for velocity initialization (default is 12345). May be None
            when the backend should choose a random seed. Ignored if `initial_temperature` is 0.
        tmp_working_directory: Specifies the location of the temporary directory to run the simulations.
            Per default (None), the directory is located in the operating systems location for temporary files
            and is removed automatically once the run finishes.
            With the specification of tmp_working_directory, a uniquely-named sub-directory is created inside
            it and left in place afterwards (the caller owns it and is responsible for removing it), so the run
            artefacts such as ``log.lammps`` remain available. tmp_working_directory needs to exist beforehand.
        dump_final_structure: Whether to dump the final structure to a file. If False, dumping happens as specified.
            If True, adds an additional dump command to ensure that the final structure is always dumped. Internal
            check avoids that the same structure is dumped twice if the final step is already a dump step. Defaults
            to True.

    Returns:
        A tuple containing:
            - structure_final: The final atomic structure.
            - parsed_output: The parsed output dictionary.

    Raises:
        ValueError: If ``ensemble`` is unknown or the temperature/pressure arguments do not fit it (see
            ``_validate_ensemble``).

    """
    initial_temperature = _validate_ensemble(
        structure, ensemble, temperature, temperature_end, npt_pressure, npt_pressure_end, initial_temperature
    )

    # Creates a working directory for the simulation (auto-cleaned when
    # tmp_working_directory is None; caller-owned otherwise).
    with simulation_working_directory(tmp_working_directory) as tmpdir:
        tmp_path = str(Path(tmpdir))

        temp_setting: float | list[float] | None = (
            [temperature, temperature_end] if temperature is not None and temperature_end is not None else temperature
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
        if npt_pressure_end is not None:
            assert isinstance(npt_pressure, int | float), "npt_pressure must be a scalar when npt_pressure_end is given"
            p_start_bar = npt_pressure * 10_000  # GPa → bar (LAMMPS metal units)
            p_end_bar = npt_pressure_end * 10_000
            input_control["fix"] = f"ensemble all npt temp {t_start} {t_end} 0.1 iso {p_start_bar} {p_end_bar} 1.0"

        if input_control_file is not None:
            input_control.update(input_control_file)

        if initial_temperature is None:
            assert temperature is not None, "temperature is validated for every ensemble except nve"
            initial_temperature = 2 * temperature

        # Sets up the LAMMPS simulations
        parsed_output = run_lammps_with_error_capture(
            working_directory=tmp_path,
            # lammpsparser rescales the velocities of the structure it receives in place (Å/fs -> Å/ps),
            # so it gets a copy to keep the caller's velocities valid for a later run.
            structure=structure.copy(),
            potential=cast("Any", potential),
            calc_mode="md",
            calc_kwargs={
                "temperature": temp_setting,
                "n_ionic_steps": n_ionic_steps,
                "time_step": timestep,
                "n_print": effective_n_dump,
                "initial_temperature": initial_temperature,
                "seed": seed,
                "pressure": npt_pressure,
                "langevin": ensemble.endswith("_langevin"),
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
