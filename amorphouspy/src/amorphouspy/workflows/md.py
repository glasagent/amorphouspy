"""Single MD simulation at constant temperature and pressure workflow for glass systems using LAMMPS.

Author: Achraf Atila (achraf.atila@bam.de).
"""

from pathlib import Path

from ase.atoms import Atoms

from amorphouspy.workflows.shared import _run_lammps_md


def md_simulation(
    structure: Atoms,
    potential: str,
    temperature_sim: float = 5000.0,
    timestep: float = 1.0,
    production_steps: int = 10_000_000,
    n_print: int = 1000,
    server_kwargs: dict | None = None,
    *,
    pressure: float | None = None,
    langevin: bool = False,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
) -> dict:  # pylint: disable=too-many-positional-arguments
    """Perform a molecular dynamics simulation using LAMMPS.

    This function equilibrate a structure at predefined temperature and pressure.
    The number of steps used here is only for testing purposes.

    Args:
        structure: The initial atomic structure to be melted and quenched.
        potential: The potential file to be used for the simulation.
        temperature_sim: The temperature at which the structure will be equilibrated (default is 5000.0 K).
        timestep: Time step for integration in femtoseconds (default is 1.0 fs).
        production_steps: The number of steps for the production.
        n_print: The frequency of output during the simulation (default is 1000).
        server_kwargs: Additional arguments for the server.
        pressure: The pressure at which the structure will be equilibrated (default is None).
        langevin: Whether to use Langevin dynamics.
        seed: Random seed for velocity initialization (default is 12345). Ignored if `initial_temperature` is 0.
        tmp_working_directory: The directory where the simulation files will be stored.

    Returns:
        A dictionary containing the simulation steps and temperature data.

    """
    potential_name = potential.at[0, "Name"]

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

    # Stage 1: constant temperature or pressure simulation
    structure_final, parsed_output = _run_lammps_md(
        structure=structure,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=temperature_sim,
        n_ionic_steps=production_steps,
        timestep=timestep,
        n_print=n_print,
        initial_temperature=temperature_sim,
        pressure=pressure,
        langevin=langevin,
        seed=seed,
        server_kwargs=server_kwargs,
    )

    result = parsed_output.get("generic", None)

    return {"structure": structure_final, "result": result}
