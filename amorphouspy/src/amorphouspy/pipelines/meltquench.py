"""Melt-quench pipeline: structure generation and glass quenching.

Combines structure generation (composition → random atoms → potential)
with the LAMMPS melt-quench simulation into a two-step pipeline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from amorphouspy.fabrication.geometry import get_ase_structure, get_structure_dict

if TYPE_CHECKING:
    import pandas as pd
    from ase import Atoms

    from amorphouspy.lammps.potentials._config import InteractionConfig
from amorphouspy.fabrication.meltquench import melt_quench_simulation
from amorphouspy.fabrication.meltquench_protocols import DEFAULT_MELT_TEMPERATURES
from amorphouspy.lammps.potentials.potential import generate_potential

logger = logging.getLogger(__name__)


def generate_structure(
    composition: dict[str, float],
    n_atoms: int = 6000,
    potential_type: str = "pmmcs",
    density: float | None = None,
    structure_seed: int = 42,
    electrostatics_config: InteractionConfig | None = None,
) -> dict[str, Any]:
    """Generate an initial random structure and matching LAMMPS potential.

    Args:
        composition: Oxide composition as ``{oxide: mol%}``.
        n_atoms: Target number of atoms.
        potential_type: Name of the interatomic potential to generate.
        density: Target density in g/cm³; ``None`` uses the Fluegel model.
        structure_seed: Random seed for atom placement.
        electrostatics_config: Electrostatics configuration (e.g. ``DsfConfig``).

    Returns:
        Dict with ``atoms_dict``, ``structure`` (ASE Atoms), and ``potential``.
    """
    atoms_dict = get_structure_dict(
        composition=composition,
        target_atoms=n_atoms,
        density=density,
        random_seed=structure_seed,
    )
    structure = get_ase_structure(atoms_dict=atoms_dict)
    potential = generate_potential(
        atoms_dict=atoms_dict,
        potential_type=potential_type,
        melt=True,
        electrostatics=electrostatics_config,
    )

    return {
        "atoms_dict": atoms_dict,
        "structure": structure,
        "potential": potential,
    }


def run_melt_quench(
    structure: Atoms,
    potential: pd.DataFrame,
    *,
    potential_type: str = "pmmcs",
    heating_rate: float = 1e14,
    cooling_rate: float = 1e12,
    timestep: float = 1.0,
    temperature_high: float | None = None,
    temperature_low: float = 300.0,
    equilibration_steps: int | None = None,
    n_dump: int | None = 100_000,
    n_print_thermo: int | None = None,
    server_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a LAMMPS melt-quench simulation.

    Args:
        structure: Initial ASE Atoms object.
        potential: LAMMPS potential object from ``generate_potential``.
        potential_type: Potential name, used for default ``temperature_high`` lookup.
        heating_rate: Heating rate in K/ps.
        cooling_rate: Cooling rate in K/ps.
        timestep: MD timestep in fs.
        temperature_high: Melt temperature in K; ``None`` uses the protocol default.
        temperature_low: Final quench temperature in K.
        equilibration_steps: Override for equilibration step count; ``None`` = protocol default.
        n_dump: Trajectory dump frequency (steps). ``None`` uses the final step only.
        n_print_thermo: Thermodynamic output frequency (steps). ``None`` uses ``n_dump``.

        server_kwargs: LAMMPS server/resource configuration.

    Returns:
        Dict with ``final_structure``, ``mean_temperature``, ``simulation_steps``,
        ``composition``, trajectory data, and simulation parameters.
    """
    if server_kwargs is None:
        server_kwargs = {}

    if temperature_high is None:
        temperature_high = DEFAULT_MELT_TEMPERATURES.get(potential_type, 5000.0)

    mq = melt_quench_simulation(
        structure=structure,
        potential=potential,
        n_dump=n_dump,
        n_print_thermo=n_print_thermo,
        heating_rate=int(heating_rate),
        cooling_rate=int(cooling_rate),
        timestep=timestep,
        temperature_high=temperature_high,
        temperature_low=temperature_low,
        equilibration_steps=equilibration_steps,
        langevin=False,
        server_kwargs=server_kwargs,
    )

    # Extract only the data of the last stage of the melt-quench simulation to
    # compute selected summaries for the final output. "simulation_history" will
    # still contain the full trajectory of all stages at this point.
    last_stage = next((s for s in reversed(mq["result"]) if s is not None), {})

    return {
        "final_structure": mq["structure"],
        "mean_temperature": float(np.mean(last_stage["temperature"])),
        "simulation_steps": len(last_stage["steps"]),
        "temperature_trajectory": [float(t) for t in last_stage["temperature"]],
        "steps_trajectory": [int(s) for s in last_stage["steps"]],
        "simulation_history": mq["result"],
        "timestep": timestep,
        "cooling_rate": int(cooling_rate),
        "heating_rate": int(heating_rate),
        "temperature_high": temperature_high,
        "temperature_low": temperature_low,
        "n_dump": n_dump,
        "n_print_thermo": (n_dump if n_print_thermo is None else n_print_thermo),
    }
