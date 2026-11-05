"""Shared module for amorphouspy simulation workflows.

This module contains shared functionality which is reused in the individual workflows.
"""

import re
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
MinStyle = Literal["cg", "sd", "hftn", "fire"]

_CONVERGED_CRITERIA = ("energy tolerance", "force tolerance")


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


def _is_positive_int(value: object) -> bool:
    """Return True for a positive integer (Python or numpy), rejecting bool."""
    return isinstance(value, int | np.integer) and not isinstance(value, bool) and bool(value > 0)


def _validate_minimize_arguments(
    force_tolerance: float,
    energy_tolerance: float,
    max_iter: int,
    min_style: str,
    pressure: float | None,
    n_print: int,
) -> None:
    """Check the arguments of a LAMMPS minimisation before anything runs.

    Args:
        force_tolerance: LAMMPS ``ftol`` in eV/Å.
        energy_tolerance: LAMMPS ``etol`` (relative energy change, dimensionless).
        max_iter: Maximum number of minimiser iterations.
        min_style: LAMMPS ``min_style``.
        pressure: Target pressure in GPa for ``fix box/relax``, or None for a fixed cell.
        n_print: Thermo and dump interval in iterations.

    Raises:
        ValueError: If a tolerance is negative or non-finite, both tolerances are 0, ``max_iter`` or ``n_print`` is
            not a positive integer, ``min_style`` is unknown, ``pressure`` is not a finite scalar, or ``"fire"`` is
            combined with a pressure.
    """
    for name, tolerance in (("force_tolerance", force_tolerance), ("energy_tolerance", energy_tolerance)):
        if not np.isfinite(tolerance) or tolerance < 0:
            msg = f"{name} must be finite and non-negative, got {tolerance!r}."
            raise ValueError(msg)
    if force_tolerance == 0 and energy_tolerance == 0:
        msg = "force_tolerance and energy_tolerance cannot both be 0: the minimisation would never converge."
        raise ValueError(msg)
    for name, value in (("max_iter", max_iter), ("n_print", n_print)):
        if not _is_positive_int(value):
            msg = f"{name} must be a positive integer, got {value!r}."
            raise ValueError(msg)
    if min_style not in get_args(MinStyle):
        msg = f"min_style must be one of {get_args(MinStyle)}, got {min_style!r}."
        raise ValueError(msg)
    if pressure is not None:
        is_real_scalar = isinstance(pressure, int | float | np.integer | np.floating) and not isinstance(pressure, bool)
        if not is_real_scalar or not np.isfinite(pressure):
            msg = f"pressure must be None or a finite scalar in GPa, got {pressure!r}."
            raise ValueError(msg)
        if min_style == "fire":
            msg = "min_style 'fire' cannot be combined with a pressure: LAMMPS rejects damped dynamics with box/relax."
            raise ValueError(msg)


def _parse_minimize_log(log_text: str) -> tuple[str, int, int]:
    """Read the minimisation statistics from the text of ``log.lammps``.

    Args:
        log_text: Full content of ``log.lammps``.

    Returns:
        The stopping criterion, the number of iterations and the number of force evaluations of the last
        ``minimize`` command.

    Raises:
        RuntimeError: If the ``Stopping criterion`` or ``Iterations, force evaluations`` line is missing.
    """
    criteria = re.findall(r"Stopping criterion = (.+)", log_text)
    counts = re.findall(r"Iterations, force evaluations = (\d+) (\d+)", log_text)
    if not criteria or not counts:
        msg = "log.lammps lacks the minimisation statistics ('Stopping criterion', 'Iterations, force evaluations')."
        raise RuntimeError(msg)
    n_iterations, n_force_evaluations = counts[-1]
    return criteria[-1].strip(), int(n_iterations), int(n_force_evaluations)


def _last_dump_step(dump_file: Path) -> int | None:
    """Return the timestep of the last frame in a LAMMPS text dump, or None if it has no frame."""
    last_step = None
    with dump_file.open() as f:
        for line in f:
            if line.startswith("ITEM: TIMESTEP"):
                last_step = int(next(f))
    return last_step


def _minimize_summary(
    parsed_output: dict[str, Any], stopping_criterion: str, n_iterations: int, n_force_evaluations: int
) -> dict[str, Any]:
    """Summarise a parsed LAMMPS minimisation (see ``_run_lammps_minimize`` for the keys)."""
    generic = parsed_output["generic"]
    forces = np.asarray(generic["forces"][-1], dtype=float)
    energy_pot = np.asarray(generic["energy_pot"], dtype=float)
    volume = np.asarray(generic["volume"], dtype=float)
    pressure = np.asarray(generic["pressures"][-1], dtype=float)
    return {
        "stopping_criterion": stopping_criterion,
        "converged": stopping_criterion in _CONVERGED_CRITERIA,
        "n_iterations": n_iterations,
        "n_force_evaluations": n_force_evaluations,
        "energy_initial_ev": float(energy_pot[0]),
        "energy_final_ev": float(energy_pot[-1]),
        "energy_change_mev_per_atom": float(1e3 * (energy_pot[-1] - energy_pot[0]) / len(forces)),
        "force_norm_ev_per_a": float(np.linalg.norm(forces)),
        "max_force_ev_per_a": float(np.linalg.norm(forces, axis=1).max()),
        "pressure_gpa": pressure,
        "mean_pressure_gpa": float(np.trace(pressure) / 3),
        "volume_initial_a3": float(volume[0]),
        "volume_final_a3": float(volume[-1]),
    }


def _run_lammps_minimize(
    structure: Atoms,
    potential: LammpsPotential,
    *,
    force_tolerance: float = 1e-4,
    energy_tolerance: float = 0.0,
    max_iter: int = 100_000,
    min_style: MinStyle = "cg",
    pressure: float | None = None,
    n_print: int = 100,
    server_kwargs: dict[str, Any] | None = None,
    tmp_working_directory: str | Path | None = None,
    input_control_file: dict[str, Any] | None = None,
) -> tuple[Atoms, dict[str, Any], dict[str, Any]]:
    """Minimise the potential energy of a structure with LAMMPS and return the final structure and a summary.

    With ``pressure=None`` (the default) the cell is fixed and only the atoms move. With a scalar ``pressure`` the
    cell also relaxes isotropically under ``fix box/relax iso`` towards that pressure (GPa, converted to bar by
    lammpsparser); ``box/relax`` acts on the virial pressure only.

    The force tolerance is LAMMPS ``ftol`` with the default ``min_modify norm two``: the 2-norm of the global force
    vector over all 3N components (eV/Å), not a per-atom maximum. With ``box/relax`` that vector also contains the box
    degrees of freedom. A 2-norm below ``ftol`` still allows single atoms with a large force in a big system, so the
    summary also reports the largest per-atom force. The energy tolerance is LAMMPS ``etol``, the energy change
    between iterations relative to the energy magnitude (dimensionless). lammpsparser caps the force evaluations at
    ``100 * max_iter``.

    ``"fire"`` is damped dynamics with the LAMMPS metal default timestep (0.001 ps); LAMMPS forbids it together with
    ``box/relax``. Its final thermo pressure contains the kinetic term of FIRE's residual velocities.

    The structure is sent to LAMMPS with zero velocities, so the reported pressure is the static (virial) pressure,
    and the returned structure also carries zero velocities. lammpsparser writes the dump and thermo output with
    ``%20.15g``, so the returned positions and energies keep 15 significant digits.

    Args:
        structure: The atomic structure to minimise. It is not modified.
        potential: The LAMMPS potential, as for ``_run_lammps_md``.
        force_tolerance: LAMMPS ``ftol`` in eV/Å (2-norm of the global force vector).
        energy_tolerance: LAMMPS ``etol`` (relative energy change, dimensionless). Both tolerances 0 is rejected.
        max_iter: Maximum number of minimiser iterations.
        min_style: LAMMPS ``min_style``: ``"cg"``, ``"sd"``, ``"hftn"`` or ``"fire"`` (fixed cell only).
        pressure: Target pressure in GPa for an isotropic ``fix box/relax``, or None for a fixed cell.
        n_print: Thermo and dump interval in iterations. It should divide ``max_iter``: on the ``max iterations``
            stop LAMMPS dumps the last step only if it is a multiple of ``n_print``.
        server_kwargs: Additional keyword arguments for the server, for example ``{"cores": 2}``.
        tmp_working_directory: Parent of the run directory, as for ``_run_lammps_md``. None uses a temporary
            directory that is removed after the run.
        input_control_file: LAMMPS input overrides forwarded unchanged to lammpsparser. A ``"dump_modify"`` or
            ``"thermo_modify"`` key replaces the 15-digit output formats and loses precision.

    Returns:
        A tuple containing:
            - structure_final: The minimised structure with zero velocities.
            - parsed_output: The parsed output dictionary.
            - summary: A dictionary with ``"stopping_criterion"`` (LAMMPS text), ``"converged"`` (True for
              ``"energy tolerance"`` or ``"force tolerance"``), ``"n_iterations"``, ``"n_force_evaluations"``,
              ``"energy_initial_ev"``, ``"energy_final_ev"``, ``"energy_change_mev_per_atom"``,
              ``"force_norm_ev_per_a"`` (global 2-norm of the final atomic forces), ``"max_force_ev_per_a"`` (largest
              per-atom force), ``"pressure_gpa"`` (final 3x3 tensor), ``"mean_pressure_gpa"`` (trace / 3),
              ``"volume_initial_a3"`` and ``"volume_final_a3"``.

    Raises:
        ValueError: If an argument is invalid (see ``_validate_minimize_arguments``).
        RuntimeError: If LAMMPS fails, ``log.lammps`` lacks the minimisation statistics, or the last dumped step
            differs from the last thermo step or the iteration count.

    Warns:
        UserWarning: If ``n_print`` does not divide ``max_iter``, or if the minimisation stopped on a criterion other
            than the energy or force tolerance.

    Example:
        ```pycon
        >>> relaxed, parsed, summary = _run_lammps_minimize(glass, potential, force_tolerance=1e-6)  # doctest: +SKIP
        >>> summary["converged"], summary["max_force_ev_per_a"]  # doctest: +SKIP
        ```

    """
    _validate_minimize_arguments(force_tolerance, energy_tolerance, max_iter, min_style, pressure, n_print)
    if max_iter % n_print != 0:
        warnings.warn(
            f"n_print={n_print} does not divide max_iter={max_iter}: a minimisation that stops on 'max iterations' "
            "does not dump its last step and raises RuntimeError.",
            UserWarning,
            stacklevel=2,
        )

    lammps_structure = structure.copy()
    lammps_structure.set_velocities(np.zeros((len(structure), 3)))

    with simulation_working_directory(tmp_working_directory) as tmpdir:
        parsed_output = run_lammps_with_error_capture(
            working_directory=tmpdir,
            structure=lammps_structure,
            potential=cast("Any", potential),
            calc_mode="minimize",
            calc_kwargs={
                "ionic_force_tolerance": force_tolerance,
                "ionic_energy_tolerance": energy_tolerance,
                "max_iter": max_iter,
                "pressure": pressure,
                "n_print": n_print,
                "style": min_style,
            },
            units="metal",
            write_restart_file=False,
            read_restart_file=False,
            restart_file="restart.out",
            input_control_file=input_control_file,
            lmp_command=get_lammps_command(server_kwargs=server_kwargs),
            dump_final_structure=False,
        )
        # Read inside the context: a temporary working directory is removed on exit.
        log_text = (Path(tmpdir) / "log.lammps").read_text()
        last_dump_step = _last_dump_step(Path(tmpdir) / "dump.out")

    stopping_criterion, n_iterations, n_force_evaluations = _parse_minimize_log(log_text)
    last_thermo_step = int(parsed_output["generic"]["steps"][-1])
    if not last_dump_step == last_thermo_step == n_iterations:
        msg = (
            f"The last dumped step ({last_dump_step}) does not match the last thermo step ({last_thermo_step}) and "
            f"the iteration count ({n_iterations}); stopping criterion: {stopping_criterion!r}. The final structure "
            "was not dumped; choose n_print dividing max_iter."
        )
        raise RuntimeError(msg)

    summary = _minimize_summary(parsed_output, stopping_criterion, n_iterations, n_force_evaluations)
    if not summary["converged"]:
        warnings.warn(
            f"Minimisation stopped on {stopping_criterion!r}, not on a tolerance; "
            f"largest per-atom force {summary['max_force_ev_per_a']:.3g} eV/Å.",
            UserWarning,
            stacklevel=2,
        )

    new_structure = structure_from_parsed_output(initial_structure=structure, parsed_output=parsed_output)
    new_structure.set_velocities(np.zeros((len(new_structure), 3)))
    return new_structure, parsed_output, summary


def _run_lammps_relax(
    structure: Atoms,
    potential: LammpsPotential,
    *,
    n_cycles: int = 2,
    pressure: float | None = 0.0,
    force_tolerance: float = 1e-4,
    energy_tolerance: float = 0.0,
    max_iter: int = 100_000,
    n_print: int = 100,
    server_kwargs: dict[str, Any] | None = None,
    tmp_working_directory: str | Path | None = None,
    input_control_file: dict[str, Any] | None = None,
) -> tuple[Atoms, dict[str, Any], list[dict[str, Any]]]:
    """Relax a structure by cycles of conjugate gradient with ``fix box/relax`` followed by FIRE at fixed cell.

    Each cycle runs two separate LAMMPS minimisations through ``_run_lammps_minimize``: ``"cg"`` with ``pressure``
    (cell and atoms relax), then ``"fire"`` with the cell fixed. Each stage starts from the previous stage's final
    structure. All stages share the tolerances, ``max_iter`` and ``n_print``. A stage that does not converge warns
    and the protocol continues.

    Args:
        structure: The atomic structure to relax. It is not modified.
        potential: The LAMMPS potential, as for ``_run_lammps_md``.
        n_cycles: Number of cg + FIRE cycles.
        pressure: Target pressure in GPa of the ``box/relax`` stages, or None to keep the cell fixed throughout.
        force_tolerance: LAMMPS ``ftol`` in eV/Å for every stage.
        energy_tolerance: LAMMPS ``etol`` (dimensionless) for every stage.
        max_iter: Maximum number of iterations per stage.
        n_print: Thermo and dump interval in iterations per stage.
        server_kwargs: Additional keyword arguments for the server, for example ``{"cores": 2}``.
        tmp_working_directory: Parent of the run directories, as for ``_run_lammps_md``.
        input_control_file: LAMMPS input overrides forwarded to every stage (see ``_run_lammps_minimize``).

    Returns:
        A tuple containing:
            - structure_final: The relaxed structure with zero velocities.
            - parsed_output: The parsed output of the last stage.
            - summaries: One summary per stage in run order, as returned by ``_run_lammps_minimize``, with
              ``"min_style"`` and ``"pressure_target_gpa"`` (None for a fixed cell) added.

    Raises:
        ValueError: If ``n_cycles`` is not a positive integer or another argument is invalid.
        RuntimeError: If a stage fails (see ``_run_lammps_minimize``).

    Example:
        ```pycon
        >>> relaxed, parsed, summaries = _run_lammps_relax(glass, potential, pressure=0.0)  # doctest: +SKIP
        >>> [s["stopping_criterion"] for s in summaries]  # doctest: +SKIP
        ```

    """
    if not _is_positive_int(n_cycles):
        msg = f"n_cycles must be a positive integer, got {n_cycles!r}."
        raise ValueError(msg)
    _validate_minimize_arguments(force_tolerance, energy_tolerance, max_iter, "cg", pressure, n_print)

    current = structure
    parsed_output: dict[str, Any] = {}
    summaries: list[dict[str, Any]] = []
    for _ in range(n_cycles):
        for min_style, stage_pressure in (("cg", pressure), ("fire", None)):
            current, parsed_output, summary = _run_lammps_minimize(
                current,
                potential,
                force_tolerance=force_tolerance,
                energy_tolerance=energy_tolerance,
                max_iter=max_iter,
                min_style=min_style,
                pressure=stage_pressure,
                n_print=n_print,
                server_kwargs=server_kwargs,
                tmp_working_directory=tmp_working_directory,
                input_control_file=input_control_file,
            )
            summaries.append({"min_style": min_style, "pressure_target_gpa": stage_pressure, **summary})
    return current, parsed_output, summaries


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
