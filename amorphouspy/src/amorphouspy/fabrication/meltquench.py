"""Melt-quench simulation workflows for glass systems using LAMMPS.

Implementations of melt-quench simulation workflows for glass systems using LAMMPS.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from pathlib import Path
from typing import Any

import pandas as pd
from ase.atoms import Atoms

from amorphouspy.fabrication.meltquench_protocols import (
    DEFAULT_MELT_TEMPERATURES,
    PROTOCOL_MAP,
    MeltQuenchParams,
)
from amorphouspy.lammps.runner import _run_lammps_md


def melt_quench_simulation(
    structure: Atoms,
    potential: pd.DataFrame,
    temperature_high: float | None = None,
    temperature_low: float = 300.0,
    timestep: float = 1.0,
    heating_rate: float = 1e12,
    cooling_rate: float = 1e12,
    n_dump: int | None = None,
    n_print_thermo: int | None = 100,
    equilibration_steps: int | None = None,
    *,
    server_kwargs: dict | None = None,
    langevin: bool = False,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
    pre_equilibrate: bool = True,
) -> dict:  # pylint: disable=too-many-positional-arguments
    """Perform a melt-quench simulation using LAMMPS.

    This function heats a structure to a high temperature, equilibrates it,
    and then cools it down to a low temperature, simulating a melt-quench process.
    The heating and cooling rates are given in K/s, and the conversion into simulation steps is done automatically.

    Args:
        structure: The initial atomic structure to be melted and quenched.
        potential: The potential file to be used for the simulation.
        temperature_high: The high temperature to which the structure will be heated.
            If None, the protocol's default melt temperature is used (e.g. 4000 K for SHIK, 5000 K for others).
        temperature_low: The low temperature to which the structure will be cooled.
        timestep: Time step for integration in femtoseconds.
        heating_rate: The rate at which the temperature is increased during the heating phase,
            in K/s. Note: the SHIK protocol has no heating stage and ignores `heating_rate` --
            it equilibrates the liquid directly at `temperature_high`.
        cooling_rate: The rate at which the temperature is decreased during the cooling phase,
            in K/s.
        n_dump: Dump frequency in simulation steps. If None, falls back to the
            final step so only the last structure is dumped.
        n_print_thermo: Thermodynamic print frequency in simulation steps.
            If None, uses n_dump.
        equilibration_steps: Override for all fixed equilibration stages inside the protocol.
            If None, each protocol uses its own hardcoded defaults.
        server_kwargs: Additional keyword arguments for the server.
        langevin: Whether to use Langevin dynamics.
        seed: Random seed for velocity initialization. Ignored if `initial_temperature` is 0.
        tmp_working_directory: Specifies the location of the temporary directory to run the simulations.
        pre_equilibrate: Run a Langevin + nve/limit pre-equilibration block at
            `temperature_high` before the first stage. Needed for randomly placed
            structures; set to False when the starting structure is already equilibrated.

    Returns:
        A dictionary containing the simulation steps and temperature data.

    Raises:
        ValueError: If the resolved `temperature_high` equals `temperature_low`
            (zero heating/cooling steps would otherwise be sent to LAMMPS).

    Example:
        >>> result = melt_quench_simulation(
        ...     structure=my_atoms,
        ...     potential=my_potential,
        ...     temperature_high=5000.0,
        ...     temperature_low=300.0,
        ...     cooling_rate=1e12
        ... )

    """
    seconds_to_femtos = 1e15
    potential_name = str(potential.loc[0, "Name"]).lower()

    if temperature_high is None:
        temperature_high = DEFAULT_MELT_TEMPERATURES.get(potential_name, 5000.0)

    if temperature_high == temperature_low:
        msg = (
            f"temperature_high must differ from temperature_low (both are {temperature_high} K): "
            "heating/cooling requires a nonzero temperature range."
        )
        raise ValueError(msg)

    heating_steps = int(((temperature_high - temperature_low) / (timestep * heating_rate)) * seconds_to_femtos)
    cooling_steps = int(((temperature_high - temperature_low) / (timestep * cooling_rate)) * seconds_to_femtos)

    if potential_name in {"bmp-screened-harmonic", "bmp-harmonic"}:
        potential_name = "bmp"  # both variants share the same MD protocol

    # Check if protocol exists
    elif potential_name not in PROTOCOL_MAP:
        available = ", ".join(PROTOCOL_MAP.keys())
        msg = f"Unknown potential: {potential_name}. Available protocols: {available}"
        raise ValueError(msg)

    # Create parameters dataclass
    params = MeltQuenchParams(
        structure=structure,
        potential=potential,
        temperature_high=temperature_high,
        temperature_low=temperature_low,
        heating_steps=heating_steps,
        cooling_steps=cooling_steps,
        timestep=timestep,
        n_dump=n_dump,
        n_print_thermo=n_print_thermo,
        langevin=langevin,
        seed=seed,
        server_kwargs=server_kwargs,
        tmp_working_directory=tmp_working_directory,
        equilibration_steps=equilibration_steps,
        pre_equilibrate=pre_equilibrate,
    )

    # Run the protocol using the function-based approach
    protocol_func = PROTOCOL_MAP[potential_name]
    structure_final, history = protocol_func(_run_lammps_md, params)

    return {
        "structure": structure_final,
        "result": history,
    }


def extract_equilibration_frames(
    final_structure: Atoms,
    simulation_history: list[dict[str, Any]] | None = None,
) -> list[Atoms]:
    """Reconstruct Atoms snapshots from the final equilibration stage.

    Falls back to a single-element list with *final_structure* when no
    simulation history is available or the history contains no position data.

    Args:
        final_structure: The quenched structure from the melt-quench pipeline.
        simulation_history: Full stage-by-stage MD history (optional).

    Returns:
        List of ASE Atoms frames suitable for averaging.
    """
    if not simulation_history:
        return [final_structure]

    last_stage = next((s for s in reversed(simulation_history) if s is not None), None)
    if last_stage is None or "positions" not in last_stage:
        return [final_structure]

    positions = last_stage["positions"]
    cells = last_stage["cells"]
    n_frames = len(positions)

    if n_frames <= 1:
        return [final_structure]

    frames: list[Atoms] = []
    for i in range(n_frames):
        frame = final_structure.copy()
        frame.set_positions(positions[i])
        frame.set_cell(cells[i])
        frame.set_pbc(True)
        frame.wrap()
        frames.append(frame)

    return frames
