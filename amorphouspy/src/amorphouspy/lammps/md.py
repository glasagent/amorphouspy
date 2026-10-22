"""Single MD simulation at constant temperature and pressure workflow for glass systems using LAMMPS.

Author: Achraf Atila (achraf.atila@bam.de).
"""

from pathlib import Path

import pandas as pd
from ase.atoms import Atoms

from amorphouspy.lammps.runner import Ensemble, _run_lammps_md


def md_simulation(
    structure: Atoms,
    potential: pd.DataFrame,
    temperature_sim: float | None = None,
    timestep: float = 1.0,
    production_steps: int = 10_000_000,
    n_dump: int | None = 1000,
    n_print_thermo: int | None = None,
    server_kwargs: dict | None = None,
    *,
    ensemble: Ensemble,
    temperature_end: float | None = None,
    npt_pressure: float | None = None,
    npt_pressure_end: float | None = None,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
    input_control_file: dict | None = None,
) -> dict:  # pylint: disable=too-many-positional-arguments
    """Perform a molecular dynamics simulation using LAMMPS.

    This function equilibrates a structure at a predefined temperature and pressure, with optional
    linear ramps for temperature and/or pressure over the course of the simulation.

    Args:
        structure: The initial atomic structure to be melted and quenched.
        potential: The potential file to be used for the simulation.
        temperature_sim: Start temperature in K (or constant temperature when ``temperature_end`` is None).
            Required for every ensemble except ``"nve"``, where it must be None.
        timestep: Time step for integration in femtoseconds (default is 1.0 fs).
        production_steps: The number of steps for the production.
        n_dump: Interval in MD steps for dumping. If None, only the last frame is dumped.
        n_print_thermo: Interval in MD steps for printing thermodynamic information.
            If None, uses ``n_dump``.
        server_kwargs: Additional arguments for the server.
        ensemble: Thermodynamic ensemble, one of ``"nve"``, ``"nvt"``, ``"npt"``, ``"nvt_langevin"`` or
            ``"npt_langevin"``. ``"nve"`` runs unthermostatted dynamics starting from the velocities carried by
            ``structure``, which must be non-zero (e.g. the output structure of a previous NVT run).
        temperature_end: End temperature in K for a linear ramp from ``temperature_sim``.
            If None, temperature is held constant at ``temperature_sim``. Not allowed for ``"nve"``.
        npt_pressure: Start pressure in GPa. Required for ``"npt"`` and ``"npt_langevin"``, not allowed otherwise.
        npt_pressure_end: End pressure in GPa for a linear pressure ramp (``"npt"`` only).
            If None, pressure is held constant at ``npt_pressure``.
        seed: Random seed for velocity initialization (default is 12345). Ignored if ``initial_temperature`` is 0.
        tmp_working_directory: Specifies the location of the temporary directory to run the simulations.
            Per default (None), the directory is located in the operating systems location for temporary files
            and is removed automatically once the run finishes.
            With the specification of tmp_working_directory, a uniquely-named sub-directory is created inside
            it and left in place afterwards (the caller owns it and is responsible for removing it), so the run
            artefacts such as ``log.lammps`` and the dump files remain available. tmp_working_directory needs to
            exist beforehand.
        input_control_file: Optional LAMMPS input overrides merged on top of the
            default generated controls.

    Returns:
        A dictionary containing the simulation steps and temperature data.

    """
    if potential.empty:
        msg = "No matching potential found for the given configuration."
        raise ValueError(msg)
    structure_final, parsed_output = _run_lammps_md(
        structure=structure,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=temperature_sim,
        temperature_end=temperature_end,
        n_ionic_steps=production_steps,
        timestep=timestep,
        initial_temperature=None if ensemble == "nve" else temperature_sim,
        npt_pressure=npt_pressure,
        npt_pressure_end=npt_pressure_end,
        n_dump=n_dump,
        n_print_thermo=n_print_thermo,
        ensemble=ensemble,
        seed=seed,
        server_kwargs=server_kwargs,
        input_control_file=input_control_file,
    )

    result = parsed_output.get("generic", None)

    return {"structure": structure_final, "result": result}
