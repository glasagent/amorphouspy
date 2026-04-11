"""Melt-quench simulation workflows for glass systems using LAMMPS.

Implementations of melt-quench simulation workflows for glass systems using LAMMPS.

Author: Achraf Atila (achraf.atila@bam.de)
"""

import tempfile
from pathlib import Path
from typing import Any, cast

import pandas as pd
from ase.atoms import Atoms

from amorphouspy.io_utils import structure_from_parsed_output
from amorphouspy.workflows.meltquench_protocols import (
    DEFAULT_MELT_TEMPERATURES,
    PROTOCOL_MAP,
    MeltQuenchParams,
)
from amorphouspy.workflows.shared import LammpsPotential, get_lammps_command, run_lammps_with_error_capture


def _run_lammps_md(  # pragma: no cover
    structure: Atoms,
    potential: LammpsPotential,
    temperature: float,
    n_ionic_steps: int,
    timestep: float,
    n_print: int,
    initial_temperature: float,
    temperature_end: float | None = None,
    pressure: float | None = None,
    pressure_end: float | None = None,
    server_kwargs: dict | None = None,
    *,
    langevin: bool = False,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
) -> tuple[Atoms, dict]:  # pylint: disable=too-many-positional-arguments
    """Run a LAMMPS MD calculation with given parameters and return the final structure and parsed output.

    Args:
        structure: The atomic structure to simulate.
        potential: The potential file to be used for the simulation.
        temperature: Start temperature for the MD run.
        n_ionic_steps: Number of MD steps to run.
        timestep: Time step for integration in femtoseconds.
        n_print: Frequency of output writing in simulation steps.
        initial_temperature: Initial temperature for velocity initialization. If None, the initial
            temperature will be twice the target temperature (which would go immediately down to the target temperature
            as described in equipartition theorem). If 0, the velocity field is not initialized (in which case the
            initial velocity given in structure will be used and seed to initialize velocities will be ignored).
        temperature_end: End temperature for a linear ramp. If None, temperature is held constant.
        pressure: Start pressure in GPa for NPT simulations. If None, NVT is used.
        pressure_end: End pressure in GPa for a linear pressure ramp. Requires ``pressure`` to be set.
        server_kwargs: Additional keyword arguments for the server.
        langevin: Whether to use Langevin dynamics.
        seed: Random seed for velocity initialization (default is 12345). Ignored if `initial_temperature` is 0.
        tmp_working_directory: Specifies the location of the temporary directory to run the simulations.

    Returns:
        A tuple containing the final structure and the simulation output dictionary.

    """
    if pressure_end is not None and pressure is None:
        msg = "pressure must be set when pressure_end is specified."
        raise ValueError(msg)

    # Creates a temporary directory for the simulation in the specified working directory.
    with tempfile.TemporaryDirectory(dir=tmp_working_directory) as tmpdir:
        tmp_path = str(Path(tmpdir))

        temp_setting: float | list[float] = (
            [temperature, temperature_end] if temperature_end is not None else temperature
        )
        t_start = temperature
        t_end = temperature_end if temperature_end is not None else temperature

        input_control: dict[str, Any] = {
            "dump_modify": f"1 every {n_ionic_steps} first yes",
            "thermo_style": "custom step temp density pe etotal pxx pxy pxz pyy pyz pzz vol",
            "thermo_modify": "flush yes",
        }

        if pressure_end is not None:
            p_start_bar = pressure * 10_000  # GPa → bar
            p_end_bar = pressure_end * 10_000
            input_control["fix"] = f"ensemble all npt temp {t_start} {t_end} 0.1 iso {p_start_bar} {p_end_bar} 1.0"
            passed_pressure: float | None = pressure
        else:
            passed_pressure = pressure

        parsed_output = run_lammps_with_error_capture(
            working_directory=tmp_path,
            structure=structure,
            potential=cast("Any", potential),
            calc_mode="md",
            calc_kwargs={
                "temperature": temp_setting,
                "n_ionic_steps": n_ionic_steps,
                "time_step": timestep,
                "n_print": n_print,
                "initial_temperature": initial_temperature,
                "seed": seed,
                "pressure": passed_pressure,
                "langevin": langevin,
            },
            units="metal",
            write_restart_file=False,
            read_restart_file=False,
            restart_file="restart.out",
            input_control_file=input_control,
            lmp_command=get_lammps_command(server_kwargs=server_kwargs),
        )

        new_structure = structure_from_parsed_output(initial_structure=structure, parsed_output=parsed_output)

    return new_structure, parsed_output


def melt_quench_simulation(
    structure: Atoms,
    potential: pd.DataFrame,
    temperature_high: float | None = None,
    temperature_low: float = 300.0,
    timestep: float = 1.0,
    heating_rate: float = 1e12,
    cooling_rate: float = 1e12,
    n_print: int = 1000,
    equilibration_steps: int | None = None,
    *,
    server_kwargs: dict | None = None,
    langevin: bool = False,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
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
        temperature_low: The low temperature to which the structure will be cooled (default is 300.0 K).
        timestep: Time step for integration in femtoseconds (default is 1.0 fs).
        heating_rate: The rate at which the temperature is increased during the heating phase,
            in K/s (default is 1e12 K/s).
        cooling_rate: The rate at which the temperature is decreased during the cooling phase,
            in K/s (default is 1e12 K/s).
        n_print: The frequency of output during the simulation (default is 1000).
        equilibration_steps: Override for all fixed equilibration stages inside the protocol.
            If None, each protocol uses its own hardcoded defaults.
        server_kwargs: Additional keyword arguments for the server.
        langevin: Whether to use Langevin dynamics.
        seed: Random seed for velocity initialization (default is 12345). Ignored if `initial_temperature` is 0.
        tmp_working_directory: Specifies the location of the temporary directory to run the simulations.

    Returns:
        A dictionary containing the simulation steps and temperature data.

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
    potential_name = potential.loc[0, "Name"].lower()

    if temperature_high is None:
        temperature_high = DEFAULT_MELT_TEMPERATURES.get(potential_name, 5000.0)

    heating_steps = int(((temperature_high - temperature_low) / (timestep * heating_rate)) * seconds_to_femtos)
    cooling_steps = int(((temperature_high - temperature_low) / (timestep * cooling_rate)) * seconds_to_femtos)

    # Check if protocol exists
    if potential_name not in PROTOCOL_MAP:
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
        n_print=n_print,
        langevin=langevin,
        seed=seed,
        server_kwargs=server_kwargs,
        tmp_working_directory=tmp_working_directory,
        equilibration_steps=equilibration_steps,
    )

    # Run the protocol using the function-based approach
    protocol_func = PROTOCOL_MAP[potential_name]
    structure_final, history = protocol_func(_run_lammps_md, params)

    return {
        "structure": structure_final,
        "result": history,
    }
