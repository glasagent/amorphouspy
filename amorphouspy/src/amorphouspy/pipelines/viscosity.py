"""Multi-temperature viscosity pipeline.

Sequentially cools a quenched glass to each target temperature via
melt-quench and runs a Green-Kubo viscosity production simulation
at each one.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from amorphouspy.atoms.shared import downsample_log
from amorphouspy.fabrication.meltquench import melt_quench_simulation

if TYPE_CHECKING:
    import pandas as pd
    from ase import Atoms
from amorphouspy.properties.viscosity import get_viscosity, viscosity_simulation

logger = logging.getLogger(__name__)


def run_viscosity_workflow(
    structure: Atoms,
    potential: pd.DataFrame,
    temperatures: list[float],
    heating_rate: float,
    cooling_rate: float,
    timestep: float = 1.0,
    n_timesteps: int = 10_000_000,
    n_print: int = 1,
    max_lag: int | None = 1_000_000,
    server_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run viscosity analysis at multiple temperatures.

    Starting from the quenched structure, the workflow sequentially cools
    from the highest to the lowest requested temperature.  At each step a
    melt-quench brings the structure to the target temperature, followed by
    a Green-Kubo viscosity production run.

    Args:
        structure: Quenched ASE Atoms from the melt-quench pipeline.
        potential: LAMMPS potential object.
        temperatures: Target temperatures (K) for viscosity runs.
        heating_rate: Heating rate in K/ps.
        cooling_rate: Cooling rate in K/ps.
        timestep: MD timestep in fs.
        n_timesteps: Number of MD steps per viscosity production run.
        n_print: Thermodynamic output frequency.
        max_lag: Maximum correlation lag (steps) for Green-Kubo.
        server_kwargs: LAMMPS server/resource configuration.

    Returns:
        Result dict with ``temperatures``, ``viscosities``, ``max_lag``,
        ``simulation_steps``, ``lag_times_ps``, and ``viscosity_integral``.
    """
    if server_kwargs is None:
        server_kwargs = {}

    sorted_temps = sorted(temperatures, reverse=True)
    logger.info("Viscosity temperatures (high→low): %s", sorted_temps)

    viscosities: list[float] = []
    all_max_lags: list[float] = []
    sim_steps: list[int] = []
    lag_times_ps: list[list[float]] = []
    viscosity_integral: list[list[float]] = []

    structure_current = structure

    for idx, temp in enumerate(sorted_temps):
        temp_high = 5000.0 if idx == 0 else sorted_temps[idx - 1]

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
            server_kwargs=server_kwargs,
        )
        structure_current = mq_result["structure"]
        logger.info("Cooled to %.1f K, %d atoms", temp, len(structure_current))

        logger.info("Running viscosity simulation at %.1f K", temp)
        visc_result = viscosity_simulation(
            structure=structure_current,
            potential=potential,
            temperature_sim=float(temp),
            timestep=float(timestep),
            initial_production_steps=int(n_timesteps),
            n_print=int(n_print),
            langevin=False,
            seed=12345,
            server_kwargs=server_kwargs,
        )

        visc_data = get_viscosity(visc_result, timestep=float(timestep), max_lag=max_lag)
        logger.info("Viscosity at %.1f K: %.3e Pa·s", temp, visc_data["viscosity"])

        viscosities.append(float(visc_data["viscosity"]))
        all_max_lags.append(float(visc_data["max_lag"]))
        sim_steps.append(int(n_timesteps))
        raw_lag_data = cast("list[float]", visc_data.get("lag_time_ps", []))
        raw_visc_data = cast("list[float]", visc_data.get("viscosity_integral", []))
        lag_times_ps.append(downsample_log(raw_lag_data))
        viscosity_integral.append(downsample_log(raw_visc_data))

    return {
        "temperatures": sorted_temps,
        "viscosities": viscosities,
        "max_lag": all_max_lags,
        "simulation_steps": sim_steps,
        "lag_times_ps": lag_times_ps,
        "viscosity_integral": viscosity_integral,
    }
