"""Single MD simulation at constant temperature and pressure workflow for glass systems using LAMMPS.

Author: Achraf Atila (achraf.atila@bam.de).
"""

from pathlib import Path

import pandas as pd
from ase.atoms import Atoms

from amorphouspy.lammps.runner import _run_lammps_md


def md_simulation(
    structure: Atoms,
    potential: pd.DataFrame,
    temperature_sim: float = 5000.0,
    timestep: float = 1.0,
    production_steps: int = 10_000_000,
    n_dump: int | None = 1000,
    n_print_thermo: int | None = None,
    server_kwargs: dict | None = None,
    *,
    temperature_end: float | None = None,
    pressure: float | None = None,
    pressure_end: float | None = None,
    langevin: bool = False,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
) -> dict:  # pylint: disable=too-many-positional-arguments
    """Perform a molecular dynamics simulation using LAMMPS.

    This function equilibrates a structure at a predefined temperature and pressure, with optional
    linear ramps for temperature and/or pressure over the course of the simulation.

    Args:
        structure: The initial atomic structure to be melted and quenched.
        potential: The potential file to be used for the simulation.
        temperature_sim: Start temperature in K (or constant temperature when ``temperature_end`` is None).
        timestep: Time step for integration in femtoseconds (default is 1.0 fs).
        production_steps: The number of steps for the production.
        n_dump: Interval in MD steps for dumping. If None, only the last frame is dumped.
        n_print_thermo: Interval in MD steps for printing thermodynamic information.
            If None, uses ``n_dump``.
        server_kwargs: Additional arguments for the server.
        temperature_end: End temperature in K for a linear ramp from ``temperature_sim``.
            If None, temperature is held constant at ``temperature_sim``.
        pressure: Start pressure in GPa. If None, NVT ensemble is used.
            Provide a value (e.g. ``0.0``) to enable NPT.
        pressure_end: End pressure in GPa for a linear pressure ramp. Requires ``pressure`` to be set.
            If None, pressure is held constant at ``pressure``.
        langevin: Whether to use Langevin dynamics.
        seed: Random seed for velocity initialization (default is 12345). Ignored if ``initial_temperature`` is 0.
        tmp_working_directory: Specifies the location of the temporary directory to run the simulations.
            Per default (None), the directory is located in the operating systems location for temporary files.
            With the specification of tmp_working_directory, the temporary directory is created in the specified
            location. Therefore, tmp_working_directory needs to exist beforehand. Data will be cleaned after the
            simulation is finished.

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
        initial_temperature=temperature_sim,
        pressure=pressure,
        pressure_end=pressure_end,
        n_dump=n_dump,
        n_print_thermo=n_print_thermo,
        langevin=langevin,
        seed=seed,
        server_kwargs=server_kwargs,
    )

    result = parsed_output.get("generic", None)

    return {"structure": structure_final, "result": result}
