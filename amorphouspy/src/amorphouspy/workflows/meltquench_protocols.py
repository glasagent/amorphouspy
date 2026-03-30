"""Simulation protocols for melt-quench workflows.

Implementations of various melt-quench protocols for different potentials.

Author
------
Achraf Atila (achraf.atila@bam.de)
"""

from dataclasses import dataclass
from functools import partial

import pandas as pd
from ase.atoms import Atoms


@dataclass
class MeltQuenchParams:
    """Parameters for melt-quench simulation protocols.

    Attributes:
        runner: The function to run LAMMPS MD simulations.
        structure: Initial atomic structure.
        potential: Potential parameters.
        temperature_high: High temperature for melting.
        temperature_low: Low temperature for quenching.
        heating_steps: Number of steps for heating phase.
        cooling_steps: Number of steps for cooling phase.
        timestep: MD timestep.
        n_print: Print frequency.
        langevin: Whether to use Langevin dynamics.
        seed: Random seed.
        server_kwargs: Server configuration.
        tmp_working_directory: Temporary directory path.

    """

    runner: callable
    structure: Atoms
    potential: pd.DataFrame | dict
    temperature_high: float
    temperature_low: float
    heating_steps: int
    cooling_steps: int
    timestep: float
    n_print: int
    langevin: bool
    seed: int
    server_kwargs: dict | None = None
    tmp_working_directory: str | None = None


def pmmcs_protocol(params: MeltQuenchParams) -> tuple[Atoms, dict]:
    """Execute the simulation PMMCS protocol.

    Args:
        params: MeltQuenchParams dataclass containing all simulation parameters.

    Returns:
        Final structure and parsed output.

    """
    # Bind common parameters to runner
    run = partial(
        params.runner,
        potential=params.potential,
        tmp_working_directory=params.tmp_working_directory,
        timestep=params.timestep,
        n_print=params.n_print,
        langevin=params.langevin,
        server_kwargs=params.server_kwargs,
    )

    # Stage 1: Heating from low to high T
    structure, _ = run(
        structure=params.structure,
        temperature=params.temperature_low,
        temperature_end=params.temperature_high,
        n_ionic_steps=params.heating_steps,
        initial_temperature=params.temperature_low,
        seed=params.seed,
    )

    # Stage 2: Equilibration at high T
    structure, _ = run(
        structure=structure,
        temperature=params.temperature_high,
        n_ionic_steps=10_000,
        initial_temperature=0,
    )

    # Stage 3: Cooling from high to low T
    structure, _ = run(
        structure=structure,
        temperature=params.temperature_high,
        temperature_end=params.temperature_low,
        n_ionic_steps=params.cooling_steps,
        initial_temperature=0,
    )

    # Stage 4: Pressure release at low T
    structure, _ = run(
        structure=structure,
        temperature=params.temperature_low,
        n_ionic_steps=10_000,
        initial_temperature=0,
        pressure=0.0,
    )

    # Stage 5: Long equilibration at low T
    structure_final, parsed_output = run(
        structure=structure,
        temperature=params.temperature_low,
        n_ionic_steps=100_000,
        initial_temperature=0,
    )

    return structure_final, parsed_output


def bjp_protocol(params: MeltQuenchParams) -> tuple[Atoms, dict]:
    """Execute the simulation BJP protocol.

    Args:
        params: MeltQuenchParams dataclass containing all simulation parameters.

    Returns:
        Final structure and parsed output.

    """
    # Bind common parameters to runner
    run = partial(
        params.runner,
        potential=params.potential,
        tmp_working_directory=params.tmp_working_directory,
        timestep=params.timestep,
        n_print=params.n_print,
        langevin=params.langevin,
        server_kwargs=params.server_kwargs,
    )

    # Stage 1: Heating from low to high T
    structure, _ = run(
        structure=params.structure,
        temperature=params.temperature_low,
        temperature_end=params.temperature_high,
        n_ionic_steps=params.heating_steps,
        initial_temperature=params.temperature_low,
        pressure=0.0,
        seed=params.seed,
    )

    # Stage 2: Equilibration at high T
    structure, _ = run(
        structure=structure,
        temperature=params.temperature_high,
        n_ionic_steps=100_000,
        initial_temperature=0,
        pressure=0.0,
    )

    # Stage 3: Cooling from high to low T
    structure, _ = run(
        structure=structure,
        temperature=params.temperature_high,
        temperature_end=params.temperature_low,
        n_ionic_steps=params.cooling_steps,
        initial_temperature=0,
        pressure=0.0,
    )

    # Stage 4: Pressure release at low T
    structure, _ = run(
        structure=structure,
        temperature=params.temperature_low,
        n_ionic_steps=100_000,
        initial_temperature=0,
        pressure=0.0,
    )

    # Stage 5: Long equilibration at low T
    structure_final, parsed_output = run(
        structure=structure,
        temperature=params.temperature_low,
        n_ionic_steps=100_000,
        initial_temperature=0,
    )

    return structure_final, parsed_output


def shik_protocol(params: MeltQuenchParams) -> tuple[Atoms, dict]:
    """Execute the simulation SHIK protocol.

    Args:
        params: MeltQuenchParams dataclass containing all simulation parameters.

    Returns:
        Final structure and parsed output.

    """
    # Bind common parameters to runner
    run1 = partial(
        params.runner,
        potential=params.potential,
        tmp_working_directory=params.tmp_working_directory,
        timestep=params.timestep,
        n_print=params.n_print,
        langevin=params.langevin,
        server_kwargs=params.server_kwargs,
    )

    exclude_patterns = [
        "fix langevinnve all langevin 5000 5000 0.01 48279",
        "fix ensemblenve all nve/limit 0.5",
        "run 10000",
        "unfix langevinnve",
        "unfix ensemblenve",
    ]

    # Copy the potential before stripping the init block, so run1 keeps the
    # original Config (with langevin + nve/limit) while run2 uses the stripped version.
    potential2 = params.potential.copy()
    potential2["Config"] = potential2["Config"].apply(
        lambda lines: [line for line in lines if not any(p in line for p in exclude_patterns)]
    )

    run2 = partial(
        params.runner,
        potential=potential2,
        tmp_working_directory=params.tmp_working_directory,
        timestep=params.timestep,
        n_print=params.n_print,
        langevin=params.langevin,
        server_kwargs=params.server_kwargs,
    )

    # Stage 1: heating from 300 to 5000 K for 100 ps
    structure, _ = run1(
        structure=params.structure,
        temperature=params.temperature_high,  # 5000 K
        n_ionic_steps=params.heating_steps,
        initial_temperature=params.temperature_high,
        pressure=None,  # NVT ensemble
        seed=params.seed,
    )

    # Stage 2: NVT equilibration at 5000 K for 100 ps
    structure, _ = run2(
        structure=structure,
        temperature=params.temperature_high,  # 5000 K
        n_ionic_steps=int(100_000 / params.timestep),  # 100 ps / (1 fs timestep) = 1e5 steps
        initial_temperature=params.temperature_high,
        pressure=None,  # NVT ensemble
        seed=params.seed,
    )

    # Stage 3: NPT equilibration at 5000 K and 0.1 GPa for 700 ps
    structure, _ = run2(
        structure=structure,
        temperature=params.temperature_high,
        n_ionic_steps=int(700_000 / params.timestep),  # 700 ps
        initial_temperature=0,
        pressure=0.1,  # GPa
    )

    # Stage 4: Quenching 5000 K -> 300 K in NPT
    structure, _ = run2(
        structure=structure,
        temperature=params.temperature_high,
        temperature_end=params.temperature_low,
        n_ionic_steps=params.cooling_steps,
        initial_temperature=0,
        pressure=[0.1, 0.0],  # ramp pressure from 0.1 -> 0 GPa
    )

    # Stage 5: Annealing at 300 K and 0 GPa for 100 ps in NPT
    structure_final, parsed_output = run2(
        structure=structure,
        temperature=params.temperature_low,
        n_ionic_steps=int(100_000 / params.timestep),  # 100 ps
        initial_temperature=0,
        pressure=0.0,
    )

    return structure_final, parsed_output
