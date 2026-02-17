"""Simulation protocols for melt-quench workflows.

Implementations of various melt-quench protocols for different potentials.

Author
------
Achraf Atila (achraf.atila@bam.de)
"""

from functools import partial

import pandas as pd
from ase.atoms import Atoms


def pmmcs_protocol(
    runner: callable,
    structure: Atoms,
    potential: pd.DataFrame | dict,
    temperature_high: float,
    temperature_low: float,
    heating_steps: int,
    cooling_steps: int,
    timestep: float,
    n_print: int,
    *,
    langevin: bool,
    seed: int,
    server_kwargs: dict | None,
    tmp_working_directory: str | None,
) -> tuple[Atoms, dict]:
    """Execute the simulation PMMCS protocol.

    Args:
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

    Returns:
        Final structure and parsed output.

    """
    # Bind common parameters to runner
    run = partial(
        runner,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        timestep=timestep,
        n_print=n_print,
        langevin=langevin,
        server_kwargs=server_kwargs,
    )

    # Stage 1: Heating from low to high T
    structure, _ = run(
        structure=structure,
        temperature=temperature_low,
        temperature_end=temperature_high,
        n_ionic_steps=heating_steps,
        initial_temperature=temperature_low,
        seed=seed,
    )

    # Stage 2: Equilibration at high T
    structure, _ = run(
        structure=structure,
        temperature=temperature_high,
        n_ionic_steps=10_000,
        initial_temperature=0,
    )

    # Stage 3: Cooling from high to low T
    structure, _ = run(
        structure=structure,
        temperature=temperature_high,
        temperature_end=temperature_low,
        n_ionic_steps=cooling_steps,
        initial_temperature=0,
    )

    # Stage 4: Pressure release at low T
    structure, _ = run(
        structure=structure,
        temperature=temperature_low,
        n_ionic_steps=10_000,
        initial_temperature=0,
        pressure=0.0,
    )

    # Stage 5: Long equilibration at low T
    structure_final, parsed_output = run(
        structure=structure,
        temperature=temperature_low,
        n_ionic_steps=100_000,
        initial_temperature=0,
    )

    return structure_final, parsed_output


def bjp_protocol(
    runner: callable,
    structure: Atoms,
    potential: pd.DataFrame | dict,
    temperature_high: float,
    temperature_low: float,
    heating_steps: int,
    cooling_steps: int,
    timestep: float,
    n_print: int,
    *,
    langevin: bool,
    seed: int,
    server_kwargs: dict | None,
    tmp_working_directory: str | None,
) -> tuple[Atoms, dict]:
    """Execute the simulation BJP protocol.

    Args:
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

    Returns:
        Final structure and parsed output.

    """
    # Bind common parameters to runner
    run = partial(
        runner,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        timestep=timestep,
        n_print=n_print,
        langevin=langevin,
        server_kwargs=server_kwargs,
    )

    # Stage 1: Heating from low to high T
    structure, _ = run(
        structure=structure,
        temperature=temperature_low,
        temperature_end=temperature_high,
        n_ionic_steps=heating_steps,
        initial_temperature=temperature_low,
        pressure=0.0,
        seed=seed,
    )

    # Stage 2: Equilibration at high T
    structure, _ = run(
        structure=structure,
        temperature=temperature_high,
        n_ionic_steps=100_000,
        initial_temperature=0,
        pressure=0.0,
    )

    # Stage 3: Cooling from high to low T
    structure, _ = run(
        structure=structure,
        temperature=temperature_high,
        temperature_end=temperature_low,
        n_ionic_steps=cooling_steps,
        initial_temperature=0,
        pressure=0.0,
    )

    # Stage 4: Pressure release at low T
    structure, _ = run(
        structure=structure,
        temperature=temperature_low,
        n_ionic_steps=100_000,
        initial_temperature=0,
        pressure=0.0,
    )

    # Stage 5: Long equilibration at low T
    structure_final, parsed_output = run(
        structure=structure,
        temperature=temperature_low,
        n_ionic_steps=100_000,
        initial_temperature=0,
    )

    return structure_final, parsed_output


def shik_protocol(
    runner: callable,
    structure: Atoms,
    potential: pd.DataFrame | dict,
    temperature_high: float,
    temperature_low: float,
    heating_steps: int,
    cooling_steps: int,
    timestep: float,
    n_print: int,
    *,
    langevin: bool,
    seed: int,
    server_kwargs: dict | None,
    tmp_working_directory: str | None,
) -> tuple[Atoms, dict]:
    """Execute the simulation SHIK protocol.

    Args:
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

    Returns:
        Final structure and parsed output.

    """
    # Bind common parameters to runner
    run = partial(
        runner,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        timestep=timestep,
        n_print=n_print,
        langevin=langevin,
        server_kwargs=server_kwargs,
    )

    # Stage 1: heating from 300 to 5000 K for 100 ps
    structure, _ = run(
        structure=structure,
        temperature=temperature_high,  # 5000 K
        n_ionic_steps=heating_steps,
        initial_temperature=temperature_high,
        pressure=None,  # NVT ensemble
        seed=seed,
    )

    exclude_patterns = [
        "fix langevin all langevin 5000 5000 0.01 48279",
        "fix ensemble all nve/limit 0.5",
        "run 10000",
        "unfix langevin",
        "unfix ensemble",
    ]

    # Modify potential in-place; the partial function holds a reference to this object,
    # so subsequent calls to run() will automatically use the modified potential
    potential["Config"] = potential["Config"].apply(
        lambda lines: [line for line in lines if not any(p in line for p in exclude_patterns)]
    )

    # Stage 2: NVT equilibration at 5000 K for 100 ps
    structure, _ = run(
        structure=structure,
        temperature=temperature_high,  # 5000 K
        n_ionic_steps=int(100_000 / timestep),  # 100 ps / (1 fs timestep) = 1e5 steps
        initial_temperature=temperature_high,
        pressure=None,  # NVT ensemble
        seed=seed,
    )

    # Stage 3: NPT equilibration at 5000 K and 0.1 GPa for 700 ps
    structure, _ = run(
        structure=structure,
        temperature=temperature_high,
        n_ionic_steps=int(700_000 / timestep),  # 700 ps
        initial_temperature=0,
        pressure=0.1,  # GPa
    )

    # Stage 4: Quenching 5000 K -> 300 K in NPT
    structure, _ = run(
        structure=structure,
        temperature=temperature_high,
        temperature_end=temperature_low,
        n_ionic_steps=cooling_steps,
        initial_temperature=0,
        pressure=[0.1, 0.0],  # ramp pressure from 0.1 -> 0 GPa
    )

    # Stage 5: Annealing at 300 K and 0 GPa for 100 ps in NPT
    structure_final, parsed_output = run(
        structure=structure,
        temperature=temperature_low,
        n_ionic_steps=int(100_000 / timestep),  # 100 ps
        initial_temperature=0,
        pressure=0.0,
    )

    return structure_final, parsed_output


# Protocol registry for easy lookup
PROTOCOLS = {
    "pmmcs": pmmcs_protocol,
    "bjp": bjp_protocol,
    "shik": shik_protocol,
}


def run_protocol(
    protocol_name: str,
    runner: callable,
    structure: Atoms,
    potential: pd.DataFrame | dict,
    temperature_high: float,
    temperature_low: float,
    heating_steps: int,
    cooling_steps: int,
    timestep: float,
    n_print: int,
    *,
    langevin: bool,
    seed: int,
    server_kwargs: dict | None = None,
    tmp_working_directory: str | None = None,
) -> tuple[Atoms, dict]:
    """Run a melt-quench protocol by name.

    Args:
        protocol_name: Name of the protocol ('pmmcs', 'bjp', or 'shik').
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

    Returns:
        Final structure and parsed output.

    Raises:
        ValueError: If protocol_name is not recognized.

    """
    protocol_name = protocol_name.lower()
    if protocol_name not in PROTOCOLS:
        msg = f"Unknown protocol: {protocol_name}. Available protocols: {', '.join(PROTOCOLS.keys())}"
        raise ValueError(msg)

    protocol = PROTOCOLS[protocol_name]
    return protocol(
        runner=runner,
        structure=structure,
        potential=potential,
        temperature_high=temperature_high,
        temperature_low=temperature_low,
        heating_steps=heating_steps,
        cooling_steps=cooling_steps,
        timestep=timestep,
        n_print=n_print,
        langevin=langevin,
        seed=seed,
        server_kwargs=server_kwargs,
        tmp_working_directory=tmp_working_directory,
    )
