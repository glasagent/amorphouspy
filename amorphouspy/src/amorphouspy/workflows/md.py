"""Single MD simulation at constant temperature and pressure workflow for glass systems using LAMMPS.

Author: Achraf Atila (achraf.atila@bam.de).
"""

from pathlib import Path

import pandas as pd
from ase.atoms import Atoms

from amorphouspy.workflows.shared import _run_lammps_md


def md_simulation(
    structure: Atoms,
    potential: pd.DataFrame,
    temperature_sim: float = 5000.0,
    timestep: float = 1.0,
    production_steps: int = 10_000_000,
    n_print: int = 1000,
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
        n_print: The frequency of output during the simulation (default is 1000).
        server_kwargs: Additional arguments for the server.
        temperature_end: End temperature in K for a linear ramp from ``temperature_sim``.
            If None, temperature is held constant at ``temperature_sim``.
        pressure: Start pressure in GPa. If None, NVT ensemble is used.
            Provide a value (e.g. ``0.0``) to enable NPT.
        pressure_end: End pressure in GPa for a linear pressure ramp. Requires ``pressure`` to be set.
            If None, pressure is held constant at ``pressure``.
        langevin: Whether to use Langevin dynamics.
        seed: Random seed for velocity initialization (default is 12345). Ignored if ``initial_temperature`` is 0.
        tmp_working_directory: The directory where the simulation files will be stored.

    Returns:
        A dictionary containing the simulation steps and temperature data.

    """
    if potential.empty:
        msg = "No matching potential found for the given configuration."
        raise ValueError(msg)
    potential_name = potential.loc[0, "Name"]

    if potential_name.lower() == "shik":
        exclude_patterns = [
            "fix langevin all langevin 5000 5000 0.01 48279",
            "fix ensemble all nve/limit 0.5",
            "run 10000",
            "unfix langevin",
            "unfix ensemble",
        ]

        potential["Config"] = potential["Config"].apply(
            lambda lines: [line for line in lines if not any(p in line for p in exclude_patterns)]
        )

    structure_final, parsed_output = _run_lammps_md(
        structure=structure,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=temperature_sim,
        temperature_end=temperature_end,
        n_ionic_steps=production_steps,
        timestep=timestep,
        n_print=n_print,
        initial_temperature=temperature_sim,
        pressure=pressure,
        pressure_end=pressure_end,
        langevin=langevin,
        seed=seed,
        server_kwargs=server_kwargs,
    )

    result = parsed_output.get("generic", None)

    return {"structure": structure_final, "result": result}
