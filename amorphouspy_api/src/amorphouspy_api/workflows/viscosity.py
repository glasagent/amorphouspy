"""Viscosity workflow for glass simulation.

Runs Green-Kubo viscosity calculations at multiple temperatures using the
quenched structure from a melt-quench simulation.  The structure is
sequentially cooled from the highest to the lowest requested temperature,
and at each step a production MD run is performed followed by post-processing.

Unlike the melt-quench workflow (which is a single executorlib DAG), viscosity
must iterate over temperatures sequentially (each step depends on the previous
cooled structure), so this module is written as a plain synchronous function
intended to be run via :func:`run_viscosity_workflow`.
"""

import logging
from typing import Any

from amorphouspy import (
    generate_potential,
    get_ase_structure,
    get_structure_dict,
    melt_quench_simulation,
)
from amorphouspy.workflows.viscosity import get_viscosity, viscosity_simulation

logger = logging.getLogger(__name__)


def run_viscosity_workflow(
    composition: dict[str, float],
    n_atoms: int,
    potential_type: str,
    heating_rate: float,
    cooling_rate: float,
    temperatures: list[float],
    timestep: float = 1.0,
    n_timesteps: int = 10_000_000,
    n_print: int = 1,
    max_lag: int | None = 1_000_000,
    lammps_resource_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the complete viscosity workflow: structure generation → melt-quench → viscosity at each T.

    Args:
        composition: Oxide glass composition dict (e.g. {"SiO2": 70, "Na2O": 30}).
        n_atoms: Target number of atoms.
        potential_type: Interatomic potential type (pmmcs/bjp/shik).
        heating_rate: Heating rate in K/ps.
        cooling_rate: Cooling rate in K/ps.
        temperatures: List of temperatures (K) for viscosity runs.
        timestep: MD timestep in fs.
        n_timesteps: Number of MD steps per viscosity production run.
        n_print: Thermodynamic output frequency.
        max_lag: Maximum correlation lag (steps) for Green-Kubo.
        lammps_resource_dict: Resource dict for LAMMPS (e.g. {"cores": 4}).

    Returns:
        Result dict suitable for storing in ``result_data["viscosity"]``.
    """
    if lammps_resource_dict is None:
        lammps_resource_dict = {}

    # Step 1: Generate structure and potential
    logger.info("Generating structure for viscosity: %s", composition)
    atoms_dict = get_structure_dict(composition=composition, target_atoms=n_atoms)
    structure = get_ase_structure(atoms_dict=atoms_dict)
    potential = generate_potential(atoms_dict=atoms_dict, potential_type=potential_type)

    # Sequential cooling: highest T → lowest T
    sorted_temps = sorted(temperatures, reverse=True)
    logger.info("Viscosity temperatures (high→low): %s", sorted_temps)

    viscosities: list[float] = []
    all_max_lags: list[float] = []
    sim_steps: list[int] = []
    lag_times_ps: list[list[float]] = []
    sacf_data: list[list[float]] = []
    viscosity_running: list[list[float]] = []

    structure_current = structure

    for idx, temp in enumerate(sorted_temps):
        temp_high = 5000.0 if idx == 0 else sorted_temps[idx - 1]

        # Cool to this temperature via melt-quench
        logger.info("Cooling from %.1f K to %.1f K", temp_high, temp)
        mq_result = melt_quench_simulation(
            structure=structure_current,
            potential=potential,
            temperature_high=float(temp_high),
            temperature_low=float(temp),
            timestep=1.0,
            heating_rate=float(heating_rate),
            cooling_rate=float(cooling_rate),
            n_print=1000,
            langevin=False,
            server_kwargs=lammps_resource_dict,
        )
        structure_current = mq_result["structure"]
        logger.info("Cooled to %.1f K, %d atoms", temp, len(structure_current))

        # Run viscosity production simulation
        logger.info("Running viscosity simulation at %.1f K", temp)
        visc_result = viscosity_simulation(
            structure=structure_current,
            potential=potential,
            temperature_sim=float(temp),
            timestep=float(timestep),
            production_steps=int(n_timesteps),
            n_print=int(n_print),
            langevin=False,
            seed=12345,
            server_kwargs=lammps_resource_dict,
        )

        # Post-process: Green-Kubo analysis
        visc_data = get_viscosity(visc_result, timestep=float(timestep), max_lag=max_lag)
        logger.info("Viscosity at %.1f K: %.3e Pa·s", temp, visc_data["viscosity"])

        viscosities.append(float(visc_data["viscosity"]))
        all_max_lags.append(float(visc_data["max_lag"]))
        sim_steps.append(int(n_timesteps))
        lag_times_ps.append(visc_data.get("lag_time_ps", []))
        sacf_data.append(visc_data.get("sacf", []))
        viscosity_running.append(visc_data.get("viscosity_running", []))

    return {
        "temperatures": sorted_temps,
        "viscosities": viscosities,
        "max_lag": all_max_lags,
        "simulation_steps": sim_steps,
        "lag_times_ps": lag_times_ps,
        "sacf_data": sacf_data,
        "viscosity_running": viscosity_running,
    }
