"""Melt-quench simulation workflows for glass systems using LAMMPS."""

import tempfile
from pathlib import Path

from ase.atoms import Atoms
from pyiron_base import job
from pyiron_lammps.compatibility.file import lammps_file_interface_function

from pyiron_glass.io_utils import structure_from_parsed_output
from pyiron_glass.workflows.shared import get_lammps_command


def _run_lammps_md(
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

    Parameters
    ----------
    structure : Atoms
        The atomic structure to simulate.
    potential : str
        The potential file to be used for the simulation.
    temperature : float or list
        The target temperature for the MD run. Can be a single value or a list [start, end].
    n_ionic_steps : int
        Number of MD steps to run.
    timestep : float
        Time step for integration in femtoseconds.
    n_print : int
        Frequency of output writing in simulation steps.
    initial_temperature : None or float
        Initial temperature according to which the initial velocity field is created. If None, the initial
        temperature will be twice the target temperature (which would go immediately down to the target temperature
        as described in equipartition theorem). If 0, the velocity field is not initialized (in which case the
        initial velocity given in structure will be used and seed to initialize velocities will be ignored).
    temperature_end : float, optional
        Final temperature for ramping. If None, no temperature ramp is applied.
    pressure : float, optional
        Target pressure for NPT simulations. If None, NVT is used.
    server_kwargs : dict | None, optional
        Additional keyword arguments for the server.
    langevin : bool, optional
        Whether to use Langevin dynamics
    seed : int, optional
        Random seed for velocity initialization (default is 12345). Ignored if `initial_temperature` is 0.
    tmp_working_directory : str | Path | None
        Specifies the location of the temporary directory to run the simulations. Per default (None), the
        directory is located in the operating systems location for temperary files. With the specification
        of tmp_working_directory, the temporary directory is created in the specified location. Therefore,
        tmp_working_directory needs to exist beforehand.


    Returns
    -------
    tuple
        A tuple (structure, parsed_output) with the final structure and the simulation output dictionary.

    """
    # Creates a temporary directory for the simulation in the specified working directory.
    with tempfile.TemporaryDirectory(dir=tmp_working_directory) as tmpdir:
        tmp_path = str(Path(tmpdir))

        # defines the temperature protocol
        temp_setting = [temperature, temperature_end] if temperature_end is not None else temperature

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
                "pressure": pressure,
                "langevin": langevin,
            },
            units="metal",
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


@job
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
    """Perform a melt-quench simulation using LAMMPS via pyiron_atomistics.

    This function heats a structure to a high temperature, equilibrates it,
    and then cools it down to a low temperature, simulating a melt-quench process.
    The heating and cooling rates are given in K/s, and the conversion into simulation steps is done automatically.

    Parameters
    ----------
    structure : Atoms
        The initial atomic structure to be melted and quenched.
    potential : str
        The potential file to be used for the simulation.
    temperature_high : float, optional
        The high temperature to which the structure will be heated (default is 5000.0 K).
    temperature_low : float, optional
        The low temperature to which the structure will be cooled (default is 300.0 K).
    timestep : float, optional
        Time step for integration in femtoseconds (default is 1.0 fs).
    heating_rate : float, optional
        The rate at which the temperature is increased during the heating phase, in K/s (default is 1e12 K/s).
    cooling_rate : float, optional
        The rate at which the temperature is decreased during the cooling phase, in K/s (default is 1e12 K/s).
    n_print : int, optional
        The frequency of output during the simulation (default is 1000).
    server_kwargs : dict | None, optional
        Additional keyword arguments for the server.
    langevin : bool, optional
        Whether to use Langevin dynamics.
    seed : int, optional
        Random seed for velocity initialization (default is 12345). Ignored if `initial_temperature` is 0.
    tmp_working_directory : str | Path | None
        Specifies the location of the temporary directory to run the simulations. Per default (None), the
        directory is located in the operating systems location for temperary files. With the specification
        of tmp_working_directory, the temporary directory is created in the specified location. Therefore,
        tmp_working_directory needs to exist beforehand.

    Returns
    -------
    dict
        A dictionary containing the simulation steps and temperature data.

    """
    seconds_to_femtos = 1e15
    heating_steps = int(((temperature_high - temperature_low) / (timestep * heating_rate)) * seconds_to_femtos)
    cooling_steps = int(((temperature_high - temperature_low) / (timestep * cooling_rate)) * seconds_to_femtos)

    potential_name = potential.at[0, "Name"]

    if potential_name.lower() == "pmmcs":
        # ================================================================
        # Stage 1: Heating from low to high T
        # ================================================================
        structure, _ = _run_lammps_md(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_low,
            temperature_end=temperature_high,
            n_ionic_steps=heating_steps,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=temperature_low,
            langevin=langevin,
            seed=seed,
            server_kwargs=server_kwargs,
        )

        # ================================================================
        # Stage 2: Equilibration at high T
        # ================================================================
        structure, _ = _run_lammps_md(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_high,
            n_ionic_steps=10_000,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        # ================================================================
        # Stage 3: Cooling from high to low T
        # ================================================================
        structure, _ = _run_lammps_md(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_high,
            temperature_end=temperature_low,
            n_ionic_steps=cooling_steps,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        # ================================================================
        # Stage 4: Pressure release at low T
        # ================================================================
        structure, _ = _run_lammps_md(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_low,
            n_ionic_steps=10_000,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            pressure=0.0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        # ================================================================
        # Stage 5: Long equilibration at low T
        # ================================================================
        structure_final, parsed_output = _run_lammps_md(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_low,
            n_ionic_steps=100_000,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        result = parsed_output.get("generic", None)

    elif potential_name.lower() == "bjp":
        # ================================================================
        # Stage 1: Heating from low to high T
        # ================================================================
        structure, _ = _run_lammps_md(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_low,
            temperature_end=temperature_high,
            n_ionic_steps=heating_steps,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=temperature_low,
            pressure=0.0,
            langevin=langevin,
            seed=seed,
            server_kwargs=server_kwargs,
        )

        # ================================================================
        # Stage 2: Equilibration at high T
        # ================================================================
        structure, _ = _run_lammps_md(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_high,
            n_ionic_steps=100_000,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            pressure=0.0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        # ================================================================
        # Stage 3: Cooling from high to low T
        # ================================================================
        structure, _ = _run_lammps_md(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_high,
            temperature_end=temperature_low,
            n_ionic_steps=cooling_steps,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            pressure=0.0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        # ================================================================
        # Stage 4: Pressure release at low T
        # ================================================================
        structure, _ = _run_lammps_md(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_low,
            n_ionic_steps=100_000,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            pressure=0.0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        # ================================================================
        # Stage 5: Long equilibration at low T
        # ================================================================
        structure_final, parsed_output = _run_lammps_md(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_low,
            n_ionic_steps=100_000,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        result = parsed_output.get("generic", None)

    elif potential_name.lower() == "shik":
        # ================================================================
        # Stage 1: heating from 300 to 5000 K for 100 ps
        # ================================================================
        structure, _ = _run_lammps_md(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_high,  # 5000 K
            n_ionic_steps=heating_steps,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=temperature_high,
            pressure=None,  # NVT ensemble
            langevin=langevin,
            seed=seed,
            server_kwargs=server_kwargs,
        )

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

        # ================================================================
        # Stage 2: NVT equilibration at 5000 K for 100 ps
        # ================================================================
        structure, _ = _run_lammps_md(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_high,  # 5000 K
            n_ionic_steps=int(100_000 / timestep),  # 100 ps / (1 fs timestep) = 1e5 steps
            timestep=timestep,
            n_print=n_print,
            initial_temperature=temperature_high,
            pressure=None,  # NVT ensemble
            langevin=langevin,
            seed=seed,
            server_kwargs=server_kwargs,
        )

        # ================================================================
        # Stage 3: NPT equilibration at 5000 K and 0.1 GPa for 700 ps
        # ================================================================
        structure, _ = _run_lammps_md(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_high,
            n_ionic_steps=int(700_000 / timestep),  # 700 ps
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            pressure=0.1,  # GPa
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        # ================================================================
        # Stage 4: Quenching 5000 K → 300 K in NPT
        # Nominal rate ~1 K/ps, so ΔT=4700 K → 4700 ps = 4.7 ns = 4.7e6 fs
        # ================================================================
        structure, _ = _run_lammps_md(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_high,
            temperature_end=temperature_low,
            n_ionic_steps=cooling_steps,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            pressure=[0.1, 0.0],  # ramp pressure from 0.1 → 0 GPa
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        # ================================================================
        # Stage 5: Annealing at 300 K and 0 GPa for 100 ps in NPT
        # ================================================================
        structure_final, parsed_output = _run_lammps_md(
            structure=structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_low,
            n_ionic_steps=int(100_000 / timestep),  # 100 ps
            timestep=timestep,
            n_print=n_print,
            initial_temperature=0,
            pressure=0.0,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        result = parsed_output.get("generic", None)

    if result is None:
        msg = "The 'generic' key is missing from parsed_output."
        raise KeyError(msg)
        result = {}

    return {
        "structure": structure_final,
        "result": result,
    }
