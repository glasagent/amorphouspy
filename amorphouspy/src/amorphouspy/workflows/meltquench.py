"""Melt-quench simulation workflows for glass systems using LAMMPS.

Implementations of melt-quench simulation workflows for glass systems using LAMMPS.

Author: Achraf Atila (achraf.atila@bam.de)
"""

import tempfile
from pathlib import Path

from ase.atoms import Atoms
from lammpsparser.compatibility.file import lammps_file_interface_function

from amorphouspy.io_utils import structure_from_parsed_output
from amorphouspy.workflows.meltquench_protocols import (
    MeltQuenchParams,
    bjp_protocol,
    pmmcs_protocol,
    shik_protocol,
)
from amorphouspy.workflows.shared import get_lammps_command


def _run_lammps_md(  # pragma: no cover
    structure: Atoms,
    potential: str,
    temperature: float | list[float],
    n_ionic_steps: int,
    timestep: float,
    n_print: int,
    initial_temperature: float,
    temperature_end: float | None = None,
    pressure: float | list[float] | None = None,
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
        temperature: The target temperature for the MD run. Can be a single value or a list [start, end].
        n_ionic_steps: Number of MD steps to run.
        timestep: Time step for integration in femtoseconds.
        n_print: Frequency of output writing in simulation steps.
        initial_temperature: Initial temperature for velocity initialization. If None, the initial
            temperature will be twice the target temperature (which would go immediately down to the target temperature
            as described in equipartition theorem). If 0, the velocity field is not initialized (in which case the
            initial velocity given in structure will be used and seed to initialize velocities will be ignored).
        temperature_end: Final temperature for ramping. If None, no temperature ramp is applied.
        pressure: Target pressure for NPT simulations. If None, NVT is used.
        server_kwargs: Additional keyword arguments for the server.
        langevin: Whether to use Langevin dynamics.
        seed: Random seed for velocity initialization (default is 12345). Ignored if `initial_temperature` is 0.
        tmp_working_directory: Specifies the location of the temporary directory to run the simulations.

    Returns:
        A tuple containing the final structure and the simulation output dictionary.

    """
    # Creates a temporary directory for the simulation in the specified working directory.
    with tempfile.TemporaryDirectory(dir=tmp_working_directory) as tmpdir:
        tmp_path = str(Path(tmpdir))

        # defines the temperature protocol
        temp_setting = [temperature, temperature_end] if temperature_end is not None else temperature

        # If pressure is a list [P_start, P_end], create a custom fix command
        # This bypasses the parser's inability to handle pressure ramps
        if isinstance(pressure, list) and len(pressure) == 2:  # noqa: PLR2004
            p_start, p_end = pressure[0] * 10000, pressure[1] * 10000
            # Convert to metal units (bar) if necessary, or use as is for GPa
            t_start = temperature if not isinstance(temperature, list) else temperature[0]
            t_end = temperature_end if temperature_end is not None else t_start

            custom_fix = f"ensemble all npt temp {t_start} {t_end} 0.1 iso {p_start} {p_end} 1.0"
            # Set scalar pressure for the parser to avoid crashes
            passed_pressure = p_start
            # Sets up the LAMMPS simulations
            _shell_output, parsed_output, _job_crashed = lammps_file_interface_function(
                working_directory=tmp_path,
                structure=structure,
                potential=potential,
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
                input_control_file={
                    "thermo_modify": "flush yes",
                    "fix": custom_fix,
                },
                lmp_command=get_lammps_command(server_kwargs=server_kwargs),
            )
        else:
            passed_pressure = pressure
            # Sets up the LAMMPS simulations
            _shell_output, parsed_output, _job_crashed = lammps_file_interface_function(
                working_directory=tmp_path,
                structure=structure,
                potential=potential,
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
                input_control_file={
                    "thermo_modify": "flush yes",
                },
                lmp_command=get_lammps_command(server_kwargs=server_kwargs),
            )

        if _job_crashed or "generic" not in parsed_output:
            msg = f"LAMMPS crashed. Check logs in {tmp_path}"
            raise RuntimeError(msg)

        # Retrives the final structure from the parsed output
        new_structure = structure_from_parsed_output(initial_structure=structure, parsed_output=parsed_output)

    return new_structure, parsed_output


def melt_quench_simulation(
    structure: Atoms,
    potential: str,
    temperature_high: float = 5000.0,
    temperature_low: float = 300.0,
    timestep: float = 1.0,
    heating_rate: float = 1e12,
    cooling_rate: float = 1e12,
    n_print: int = 1000,
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
        temperature_high: The high temperature to which the structure will be heated (default is 5000.0 K).
        temperature_low: The low temperature to which the structure will be cooled (default is 300.0 K).
        timestep: Time step for integration in femtoseconds (default is 1.0 fs).
        heating_rate: The rate at which the temperature is increased during the heating phase,
            in K/s (default is 1e12 K/s).
        cooling_rate: The rate at which the temperature is decreased during the cooling phase,
            in K/s (default is 1e12 K/s).
        n_print: The frequency of output during the simulation (default is 1000).
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
    heating_steps = int(((temperature_high - temperature_low) / (timestep * heating_rate)) * seconds_to_femtos)
    cooling_steps = int(((temperature_high - temperature_low) / (timestep * cooling_rate)) * seconds_to_femtos)

    potential_name = potential.at[0, "Name"].lower()

    # Map potential names to protocol functions
    protocol_map = {
        "pmmcs": pmmcs_protocol,
        "bjp": bjp_protocol,
        "shik": shik_protocol,
    }

    # Check if protocol exists
    if potential_name not in protocol_map:
        available = ", ".join(protocol_map.keys())
        msg = f"Unknown potential: {potential_name}. Available protocols: {available}"
        raise ValueError(msg)

    # Create parameters dataclass
    params = MeltQuenchParams(
        runner=_run_lammps_md,
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

    # Run the protocol using the function-based approach
    protocol_func = protocol_map[potential_name]
    structure_final, parsed_output = protocol_func(params)

    result = parsed_output.get("generic", None)

    return {
        "structure": structure_final,
        "result": result,
    }
