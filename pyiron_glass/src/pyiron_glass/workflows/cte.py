"""CTE simulation workflows for glass systems using LAMMPS.

Implements molecular dynamics workflows and post-processing utilities for
CTE calculations based on H-V fluctuations with NPT simulations.

Author
------
Marcel Sadowski (github.com/Gitdowski)
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from ase.atoms import Atoms
from pyiron_base import job
from lammpsparser.compatibility.file import lammps_file_interface_function

from pyiron_glass.analysis.cte import cte_from_npt_fluctuations
from pyiron_glass.io_utils import structure_from_parsed_output
from pyiron_glass.workflows.shared import get_lammps_command


def _create_logger() -> logging.Logger:
    """Create and configure a logger for CTE workflow warnings."""
    # create logger
    logger = logging.getLogger("pyiron_glass.workflows.cte")
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler("cte.log")
    file_handler.setLevel(logging.INFO)

    # Create a formatter and attach it to the handler
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    return logger


def _run_lammps_md(
    structure: Atoms,
    potential: str,
    temperature: float | list[float | int],
    n_ionic_steps: int,
    timestep: float,
    n_dump: int,
    n_log: int,
    initial_temperature: float,
    pressure: float | list | None = None,
    server_kwargs: dict | None = None,
    *,
    langevin: bool = False,
    seed: int | None = 12345,
    tmp_working_directory: str | Path | None = None,
) -> tuple[Atoms, dict]:  # pylint: disable=too-many-positional-arguments
    """Run a LAMMPS MD calculation with given parameters and return the final structure and parsed output.

    Parameters
    ----------
    structure : Atoms
        The atomic structure to simulate.
    potential : str
        The potential file to be used for the simulation.
    temperature : float, int or list[float | int]
        The target temperature for the MD run. Can be a single value or a list [start, end].
    n_ionic_steps : int
        Number of MD steps to run.
    timestep : float
        Time step for integration in femtoseconds.
    n_dump : int
        Frequency of dump output writing in simulation steps.
    n_log : int
        Frequency of log output writing in simulation steps.
    initial_temperature : None or float
        Initial temperature according to which the initial velocity field is created. If None, the initial
        temperature will be twice the target temperature (which would go immediately down to the target temperature
        as described in equipartition theorem). If 0, the velocity field is not initialized (in which case the
        initial velocity given in structure will be used and seed to initialize velocities will be ignored).
    temperature_end : float, optional
        Final temperature for ramping. If None, no temperature ramp is applied.
    pressure : float | list | None, optional
        Target pressure. If None, NVT is used. If a float or int is provided, isotropic NPT is used. If a list
        of 6 values is provided, anisotropic or tricilinic NPT is used.
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
    structure_final : Atoms
        Final atomic structure from the simulation.
    parsed_output : dict
        Parsed output dictionary returned by `lammps_function`.

    Notes
    -----
    - Automatically manages a temporary working directory and cleans it after execution.
    - Uses `pyiron_atomistics.lammps.lammps_function` as the backend.
    - The `thermo_style` is fixed to report pressure tensor components for post-analysis.

    """
    # Creates a temporary directory for the simulation in the specified working directory.
    with tempfile.TemporaryDirectory(dir=tmp_working_directory) as tmpdir:
        tmp_path = str(Path(tmpdir))

        # Sets up the LAMMPS simulations
        _shell_output, parsed_output, _job_crashed = lammps_file_interface_function(
            working_directory=tmp_path,
            structure=structure,
            potential=potential,
            calc_mode="md",
            calc_kwargs={
                "temperature": temperature,
                "n_ionic_steps": n_ionic_steps,
                "time_step": timestep,
                "n_print": n_log,
                "initial_temperature": initial_temperature,
                "seed": seed,
                "pressure": pressure,
                "langevin": langevin,
            },
            units="metal",
            write_restart_file=False,
            read_restart_file=False,
            restart_file="restart.out",
            lmp_command=get_lammps_command(server_kwargs=server_kwargs),
            input_control_file={
                "dump_modify": f"1 every {n_dump} first yes",
                "thermo_style": "custom step temp pe etotal pxx pxy pxz pyy pyz pzz lx ly lz vol",
                "thermo_modify": "flush no",
            },
        )

        if _job_crashed or parsed_output.get("generic", None) is None or parsed_output.get("lammps", None) is None:
            msg = f"LAMMPS crashed. Check logs in {tmp_path}"
            raise RuntimeError(msg)

        new_structure = structure_from_parsed_output(initial_structure=structure, parsed_output=parsed_output)

    return new_structure, parsed_output


def _initialize_intermediate_datadict() -> dict:
    """Initialize dict to store data over multiple production runs."""
    return {
        "steps": 0,
        "T": np.array([]),
        "E_tot": np.array([]),
        "ptot": np.array([]),
        "pxx": np.array([]),
        "pyy": np.array([]),
        "pzz": np.array([]),
        "V": np.array([]),
        "Lx": np.array([]),
        "Ly": np.array([]),
        "Lz": np.array([]),
    }


def _collect_data(collected_data: dict, subresults: dict, run_key: str) -> dict:
    """Collect data from a specific production run into the cumulative data dictionary."""
    collected_data["steps"] += int(subresults[run_key]["steps"][-1])
    collected_data["T"] = np.append(collected_data["T"], subresults[run_key]["temperature"])
    collected_data["E_tot"] = np.append(collected_data["E_tot"], subresults[run_key]["energy_tot"])

    pxx = subresults[run_key]["pressures"][:, 0, 0] * 1e9  # in Pa
    pyy = subresults[run_key]["pressures"][:, 1, 1] * 1e9  # in Pa
    pzz = subresults[run_key]["pressures"][:, 2, 2] * 1e9  # in Pa
    p_tot = (pxx + pyy + pzz) / 3  # hydrostatic pressure in Pa
    collected_data["ptot"] = np.append(collected_data["ptot"], p_tot)
    collected_data["pxx"] = np.append(collected_data["pxx"], pxx)
    collected_data["pyy"] = np.append(collected_data["pyy"], pyy)
    collected_data["pzz"] = np.append(collected_data["pzz"], pzz)

    collected_data["V"] = np.append(collected_data["V"], subresults[run_key]["volume"])  # in Ang^3
    collected_data["Lx"] = np.append(collected_data["Lx"], subresults[run_key]["Lx"])
    collected_data["Ly"] = np.append(collected_data["Ly"], subresults[run_key]["Ly"])
    collected_data["Lz"] = np.append(collected_data["Lz"], subresults[run_key]["Lz"])

    return collected_data


def _sanity_check_subresults(
    T_target: float,
    T_actual: float,
    p_target: float,
    ptot_actual: float,
    pxx_actual: float,
    pyy_actual: float,
    pzz_actual: float,
    T_key: str,
    run_key: str,
    logger: logging.Logger,
) -> None:
    """Perform sanity checks on actual vs target temperature and pressure values."""
    REL_TOL = 0.05  # 5 %
    ABS_TEMP_TOL = 10  # K
    ABS_PRESS_TOL = 1e7  # Pa

    if abs(T_target - T_actual) > REL_TOL * T_target and abs(T_target - T_actual) > ABS_TEMP_TOL:
        msg = f"\n  Temperature differences (>5% and > 10 K) are observed at {T_key}, {run_key}:\n"
        msg += f"  Specified target temperature = {T_target:.2f} K\n"
        msg += f"  Actual average temperature   = {T_actual:.2f} K.\n"
        msg += "  Using target temperature for CTE calculation."
        logger.warning(msg)

    if abs(p_target - ptot_actual) > REL_TOL * p_target and abs(p_target - ptot_actual) > ABS_PRESS_TOL:
        msg = f"\n  Pressure (ptot) differences (>5% and > 10 MPa) are observed at {T_key}, {run_key}:\n"
        msg += f"  Specified target pressure = {p_target / 1e6:.2e} MPa\n"
        msg += f"  Actual average pressure   = {ptot_actual / 1e6:.2e} MPa.\n"
        msg += "  Using target pressure for CTE calculation."
        logger.warning(msg)

    if abs(p_target - pxx_actual) > REL_TOL * p_target and abs(p_target - pxx_actual) > ABS_PRESS_TOL:
        msg = f"\n  Pressure (pxx) differences (>5% and > 10 MPa) are observed at {T_key}, {run_key}:\n"
        msg += f"  Specified target pressure = {p_target / 1e6:.2e} MPa\n"
        msg += f"  Actual average pressure   = {pxx_actual / 1e6:.2e} MPa.\n"
        msg += "  Using target pressure for CTE calculation."
        logger.warning(msg)

    if abs(p_target - pyy_actual) > REL_TOL * p_target and abs(p_target - pyy_actual) > ABS_PRESS_TOL:
        msg = f"\n  Pressure (pyy) differences (>5% and > 10 MPa) are observed at {T_key}, {run_key}:\n"
        msg += f"  Specified target pressure = {p_target / 1e6:.2e} MPa\n"
        msg += f"  Actual average pressure   = {pyy_actual / 1e6:.2e} MPa.\n"
        msg += "  Using target pressure for CTE calculation."
        logger.warning(msg)

    if abs(p_target - pzz_actual) > REL_TOL * p_target and abs(p_target - pzz_actual) > ABS_PRESS_TOL:
        msg = f"\n  Pressure (pzz) differences (>5% and > 10 MPa) are observed at {T_key}, {run_key}:\n"
        msg += f"  Specified target pressure = {p_target / 1e6:.2e} MPa\n"
        msg += f"  Actual average pressure   = {pzz_actual / 1e6:.2e} MPa.\n"
        msg += "  Using target pressure for CTE calculation."
        logger.warning(msg)


def _cte_fluctuation_workflow_analysis(
    subresults: dict,
    temperature: float,
    p_in_GPa: float,
    T_key: str,
    N_points: int = 1000,
    *,
    use_running_mean: bool = False,
    logger: logging.Logger,
) -> dict:
    """Parse the specific collected output from cte_simulation workflow.

    This function extracts and computes the "instantaneous enthalpy" and volume data from
    the different production runs and computes the isotropic CTE based on the H-V fluctuations.
    This is done in a fashion to be able to check for convergence in the workflow.
    E.g., if multiple production runs are available, the CTE is first calculated for only
    the first run, then for the combined data of the first two runs, etc.

    Parameters
    ----------
    subresults : dict
        Parsed output dictionary from `cte_simulation` workflow for a specific temperature.
        Structure of the dictionary is:
        {   "run01" : {"pressure" : ..., "volume" : ..., "temperature" : ..., etc},
            "run02" : {"pressure" : ..., "volume" : ..., "temperature" : ..., etc},
            ...
        }
    temperature : int | float
        Target temperature in K. If not specified the average temperature from the data is used.
    p_in_GPa : float
        Target pressure in GPa (for consistent usage of pyiron units). Will be converted to Pa internally.
        If not specified, the average pressure from the data is used.
    T_key : str
        Key name for temperature in the subresults dictionary. Needed for warning messages to identify
        the specific temperature step.
    N_points: int
        Window size for running mean calculation if use_running_mean is True. (default 1000) If
        use_running_mean is False, this parameter is ignored.
    use_running_mean: bool
        Conventionally, fluctuations are calculated as difference from the mean of the whole trajectory. If
        use_running_mean is True, running mean values are used to determine fluctuations. This can be useful for
        non-stationary data where drift in volume and energy is still observed. (default False)
    logger : logging.Logger
        Logger for warning messages.

    Returns
    -------
    dict
        Nested dictionary with run_key "run01", "run02, ... as main keys. Under every key, the dictionary
        contains the collected and computed CTE values and other thermodynamic averages calculated up to
        the specific production run. Structure is:
        {   "run01" : { "CTE_V" : ...,     # Isotropic CTE based on volume fluctuations in 1/K
                        "CTE_x" : ...,     # Anisotropic CTE based on cell length x in 1/K
                        "CTE_y" : ...,     # Anisotropic CTE based on cell length y in 1/K
                        "CTE_z" : ...,     # Anisotropic CTE based on cell length z in 1/K
                        "steps" : ...,     # Total number of steps collected up to this run
                        "T" : ...,         # Temperature in K
                        "ptot" : ...,      # Total pressure in GPa
                        "pxx" : ...,       # Pressure xx component in GPa
                        "pyy" : ...,       # Pressure yy component in GPa
                        "pzz" : ...,       # Pressure zz component in GPa
                        "E_tot" : ...,     # Average total energy in eV
                        "V" : ...,         # Average volume
                        "Lx" : ...,        # Average cell length in x
                        "Ly" : ...,        # Average cell length in y
                        "Lz" : ...,        # Average cell length in z
                        },
            "run02" : {"CTE_V" : ..., "CTE_x" : ..., "CTE_y" : ..., "CTE_z" : ..., etc},
            ...
        }

    Notes
    -----
        - Care needs to be taken to ensure the correct "enthalpy" is used. In Lammps, the enthalpy
          is calculated based on the instantaneous properties, H_inst = E_inst + p_inst * V_inst. However,
          the required property here is better reflected if the pressure that defines the ensemble is used:
          H_ens = E_inst + p_target * V_inst.
        - If isotropic NPT simulations are performed (see "iso" keyword in lammps), the apparent hydrostatic
          pressure = (pxx+pyy+pzz)/3 will be close to the user-specified pressure, but individual pxx, pyy
          and pzz components can deviate significantly if the structure is not fully relaxed or shows anisotropic
          features. In this case, it is not entirely clear if the hydrostatic pressure or the actual individual
          pressures should be used to calculate the 1D CTE components. Therefore, we perform anisotropic NPT
          simulations (see "aniso" keyword in lammps) per default and the user-specified pressure is applied to all
          individual components.

    """
    # collect final output here
    cte_data = {}

    # Initialize dict to collect data over multiple production runs
    collected_data = _initialize_intermediate_datadict()

    # convert GPa to Pa
    p = p_in_GPa * 1e9

    for run_key in sorted(subresults.keys()):
        cte_data[run_key] = {}

        # Update collected_data with data from this production run
        collected_data = _collect_data(collected_data, subresults, run_key)

        # Print warning if significant deviations between target and actual T and p are observed
        _sanity_check_subresults(
            T_target=temperature,
            T_actual=float(np.mean(collected_data["T"])),
            p_target=p,
            ptot_actual=float(np.mean(collected_data["ptot"])),
            pxx_actual=float(np.mean(collected_data["pxx"])),
            pyy_actual=float(np.mean(collected_data["pyy"])),
            pzz_actual=float(np.mean(collected_data["pzz"])),
            T_key=T_key,
            run_key=run_key,
            logger=logger,
        )

        # Conversion from Pa*Ang^3 to eV
        PaAng3_to_eV = 6.2415e-12

        # compute CTE based on H-V fluctuations
        cte_data[run_key]["CTE_V"] = cte_from_npt_fluctuations(
            temperature=temperature,
            enthalpy=collected_data["E_tot"] + p * collected_data["V"] * PaAng3_to_eV,
            volume=collected_data["V"],
            use_running_mean=use_running_mean,
            N_points=N_points,
        )
        cte_data[run_key]["CTE_x"] = cte_from_npt_fluctuations(
            temperature=temperature,
            enthalpy=collected_data["E_tot"] + p * collected_data["Lx"] * PaAng3_to_eV,
            volume=collected_data["Lx"],
            use_running_mean=use_running_mean,
            N_points=N_points,
        )
        cte_data[run_key]["CTE_y"] = cte_from_npt_fluctuations(
            temperature=temperature,
            enthalpy=collected_data["E_tot"] + p * collected_data["Ly"] * PaAng3_to_eV,
            volume=collected_data["Ly"],
            use_running_mean=use_running_mean,
            N_points=N_points,
        )
        cte_data[run_key]["CTE_z"] = cte_from_npt_fluctuations(
            temperature=temperature,
            enthalpy=collected_data["E_tot"] + p * collected_data["Lz"] * PaAng3_to_eV,
            volume=collected_data["Lz"],
            use_running_mean=use_running_mean,
            N_points=N_points,
        )

        # Also keep other averaged properties to easily check for drift or convergence
        cte_data[run_key]["steps"] = collected_data["steps"]
        cte_data[run_key]["T"] = float(np.mean(collected_data["T"]))
        cte_data[run_key]["ptot"] = float(np.mean(collected_data["ptot"])) * 10**-9  # export in GPa
        cte_data[run_key]["pxx"] = float(np.mean(collected_data["pxx"])) * 10**-9
        cte_data[run_key]["pyy"] = float(np.mean(collected_data["pyy"])) * 10**-9
        cte_data[run_key]["pzz"] = float(np.mean(collected_data["pzz"])) * 10**-9
        cte_data[run_key]["E_tot"] = float(np.mean(collected_data["E_tot"]))
        cte_data[run_key]["V"] = float(np.mean(collected_data["V"]))
        cte_data[run_key]["Lx"] = float(np.mean(collected_data["Lx"]))
        cte_data[run_key]["Ly"] = float(np.mean(collected_data["Ly"]))
        cte_data[run_key]["Lz"] = float(np.mean(collected_data["Lz"]))
    return cte_data


def _is_converged(data: dict, criterion: float) -> bool:
    """Check for convergence of CTE values over multiple production runs.

    This function compares the CTE values computed with all data up to the previous production
    run to the CTE values computed with all data up to current production run. It computes the
    absolute difference and checks if it is below the specified convergence criterion.
    The check is performed for all CTE components (CTE_V, CTE_x, CTE_y, CTE_z) individually.

    Parameters
    ----------
    data : dict
        Dictionary containing CTE values from multiple production runs at the current temperature.
        Structure of the dictionary is assumed to be:
        {   "run01" : {"CTE_V" : ..., "CTE_x" : ..., "CTE_y" : ..., "CTE_z" : ..., ...},
            "run02" : {"CTE_V" : ..., "CTE_x" : ..., "CTE_y" : ..., "CTE_z" : ..., ...},
            ...
        }
    criterion : float
        Convergence criterion for the CTE values.

    Returns
    -------
    bool
        True if all the CTE values are converged within the specified criterion, False otherwise.

    """
    bools = []
    current_run_key = list(data.keys())[-1]
    previous_run_key = list(data.keys())[-2]
    for cte_key in ["CTE_V", "CTE_x", "CTE_y", "CTE_z"]:
        current_value = data[current_run_key][cte_key]
        previous_value = data[previous_run_key][cte_key]
        if cte_key == "CTE_V":
            if abs(current_value - previous_value) / 3 >= criterion:
                bools.append(False)
            else:
                bools.append(True)
        elif abs(current_value - previous_value) >= criterion:
            bools.append(False)
        else:
            bools.append(True)
    return all(bools)


def _input_checker(
    production_steps: int, max_production_runs: int, n_log: int, timestep: float, logger: logging.Logger
) -> tuple[int, int, int]:
    """Check and adjust input parameters for cte_simulation workflow."""
    # Minimum choices for a working CTE calculation. For reliable results, use considreably higher values!
    MIN_PRODUCTION_RUNS = 2
    AVERAGING_TIME_IN_PS = 10
    MIN_PRODUCTION_STEPS = int(2 * AVERAGING_TIME_IN_PS * 1000 / timestep)
    MIN_RUNNING_MEAN_POINTS = 1000

    if max_production_runs < MIN_PRODUCTION_RUNS:
        msg = "\n  At least 2 individual production runs are needed to check for CTE convergence."
        msg += f"\n  However, a value of {max_production_runs} was provided."
        msg += f"\n  Automatically setting max_production_runs to {MIN_PRODUCTION_RUNS} and continue."
        msg += "\n  Consider increasing max_production_runs even further if convergence is not reached."
        logger.warning(msg)
        max_production_runs = MIN_PRODUCTION_RUNS

    if production_steps < MIN_PRODUCTION_STEPS:
        msg = "\n  For calculating fluctuations based on running averages, sufficient data is needed."
        msg += f"\n  With currently averaging over {AVERAGING_TIME_IN_PS} ps, production runs are too short "
        msg += f"and we recommend at least {2 * AVERAGING_TIME_IN_PS} ps."
        msg += f"\n  Automatically setting the user-specified production_steps of {production_steps} "
        msg += f"to {MIN_PRODUCTION_STEPS} and continue."
        msg += "\n  Consider increasing production_steps even further to get more reliable results."
        logger.warning(msg)
        production_steps = MIN_PRODUCTION_STEPS

    N_for_averaging = int(AVERAGING_TIME_IN_PS * 1000 / n_log / timestep)
    if N_for_averaging < MIN_RUNNING_MEAN_POINTS:
        msg = "\n  Running mean values are most likely based on insufficient data points."
        msg += f"\n  We recommend averaging over at least {MIN_RUNNING_MEAN_POINTS} data points, "
        msg += f"but currently only {N_for_averaging} are used."
        msg += "\n  Consider decreasing n_log or change hard-coded AVERAGING_TIME_IN_PS variable."
        msg += "\n  Continuing regardless."
        logger.warning(msg)

    return production_steps, max_production_runs, N_for_averaging


def _set_initial_temperature(T_count: int, T: float, seed: int | None) -> None:
    """Determine initial temperature and seed for velocity initialization.

    Only the very first simulation should initialize the velocity field based on the target
    temperature and a random seed. All subsequent simulations should continue from the previous
    velocity field and therefore set initial_temperature to 0 and seed to None.

    Parameters
    ----------
    T_count : int
        Current temperature step count in the workflow.
    T : float
        Target temperature for the current simulation.
    seed : int | None
        Random seed for velocity initialization.

    Returns
    -------
    initial_temperature : float | int
        Initial temperature for velocity initialization. None if velocities should be initialized
        based on target temperature, 0 if previous velocities should be used.

    """
    if T_count == 1:
        return T, seed
    return 0, None


def _temperature_checker(temperature: float | list[int | float]) -> list[int | float]:
    """Check and prepare temperature list for cte_simulation workflow.

    Makes sure that all temperatures are positive. If only specified as a single value, it
    converts it to a list as required for the workflow.

    Parameters
    ----------
    temperature : float | list[int | float]
        Simulation temperature in Kelvin.

    Returns
    -------
    list[int | float]
        Prepared list of temperatures for the workflow.

    Raises
    ------
    ValueError
        If any temperature is non-positive.

    """
    if isinstance(temperature, (int, float)):
        temperature_sim = [temperature]

    for T in temperature_sim:
        if T <= 0:
            msg = "All Temperatures must be positive for CTE simulations."
            raise ValueError(msg)

    return temperature_sim


@job
def cte_simulation(
    structure: Atoms,
    potential: str,
    temperature: float | list[int | float] = 300,
    pressure: float = 1e-4,
    timestep: float = 1.0,
    equilibration_steps: int = 100_000,
    production_steps: int = 200_000,
    max_production_runs: int = 25,
    CTE_convergence_criterion: float = 1e-6,
    n_dump: int = 100000,
    n_log: int = 10,
    server_kwargs: dict[str, Any] | None = None,
    *,
    aniso: bool = False,
    seed: int | None = 12345,
    tmp_working_directory: str | Path | None = None,
) -> dict[Any, Any]:  # pylint: disable=too-many-positional-arguments
    """Perform a LAMMPS-based cte simulation protocol.

    This workflow equilibrates a structure at a target temperature and performs a
    production MD run to collect the instantaneous total energy, pressure and volume,
    needed to compute the CTE via H-V fluctuation analysis or V-T data, depending
    if only one temperature or multiple temperatures should be probed.

    The number of steps used here is only for testing purposes.
    It is assumed in this workflow that the given in structure is pre-quilibrated.

    Parameters
    ----------
    structure : Atoms
        Input structure (assumed pre-equilibrated).
    potential : str
        LAMMPS potential file.
    temperature : float | list[int | float], optional
        Simulation temperature in Kelvin (default 300 K).
    pressure : float, optional
        Target pressure in GPa (use pyiron units here!) for NPT simulations.
        (default 10-4 GPa = 10^5 Pa = 1 bar).
    timestep : float, optional
        MD integration timestep in femtoseconds (default 1.0 fs).
    equilibration_steps : int, optional
        Number of MD steps for the equilibration runs (default 100,000).
    production_steps : int, optional
        Number of MD steps for the production runs (default 200,000).
    max_production_runs : int, optional
        Maximum number of production runs to perform (default 10). If max number of
        production runs is reached without convergence, a warning is printed and the
        next temperature is started.
    CTE_convergence_criterion : float, optional
        Convergence criterion for the CTE value calculated based on H-V fluctuations
        calculated over subsequent production runs (default 1e-6).
    n_dump : int, optional
        Dump output frequency of the production runs (default 100,000).
    n_log : int, optional
        Log output frequency (default 10).
    server_kwargs : dict, optional
        Additional server configuration arguments for pyiron.
    aniso : bool, optional
        If false, an isotropic NPT calculation is performed and the simulation box is scaled uniformly.
        If True, anisotropic NPT calculation is performed and the simulation box can change shape and
        size independently along each axis (default False).
    seed : int, None, optional
        Random seed for velocity initialization (default 12345). If
    tmp_working_directory : str or Path, optional
        Temporary directory for job execution.

    Returns
    -------
    dict
        Nested dictionary containing collected output of the simulations. The main keys are the
        temperature steps in the format "01_300K", "02_400K", etc. Under each temperature key,
        the dictionary contains another dictionary with keys "run01", "run02", ... for each
        production run. Under each run key, the dictionary contains the parsed output from the
        production run as well as computed CTE values and other thermodynamic averages.
        Structure is:
        {   "01_300K" : { "run01" : { "CTE_V" : ..., "CTE_x" : ..., "CTE_y" : ..., "CTE_z" : ..., etc},
                          "run02" : { "CTE_V" : ..., "CTE_x" : ..., "CTE_y" : ..., "CTE_z" : ..., etc},
                          ...
                      },
            "02_400K" : { "run01" : { "CTE_V" : ..., "CTE_x" : ..., "CTE_y" : ..., "CTE_z" : ..., etc},
                          "run02" : { "CTE_V" : ..., "CTE_x" : ..., "CTE_y" : ..., "CTE_z" : ..., etc},
                          ...
                      },
            ...
        }
        On the lowest level, the structure is the same as returned by `_cte_fluctuation_workflow_analysis`.
        Additionally, on the run key level, the following entries are added for bookkeeping:
        "is_converged" : bool            # Whether convergence was reached within the max_production_runs
        "convergence_criterion" : float  # The convergence criterion
        "structure_final" : Atoms        # Final structure at this temperature

    Notes
    -----
    - For every temperature, the structure is first pre-equilibrated with short (10 ps) NVT.
    - Simulation settings for the NVT equilibration run are hard-coded
    - CTEs are calculated sequentially if a list of temperatures is provided. Alternatively, multiple
      jobs with independent temperatures can be submitted to achieve parallelization.

    """
    # Logging setup
    logger = _create_logger()

    # Check and adjust input parameters if needed
    production_steps, max_production_runs, N_for_averaging = _input_checker(
        production_steps, max_production_runs, n_log, timestep, logger
    )

    # Prepare temperature list
    temperature = _temperature_checker(temperature)

    # Set pressure to anisotropic if requested
    sim_pressure = [pressure, pressure, pressure, None, None, None] if aniso else pressure

    # Collect all results here
    cte_results = {}

    # initial structure used. Afterwards, it is updated after each temperature
    structure0 = structure.copy()

    # Loop over all temperatures to perform the cte simulation protocol
    for T_count, T in enumerate(temperature, start=1):
        T_key = f"{T_count:02d}_{int(T)}K"

        # Determine initial temperature and seed for velocity initialization
        initial_temperature, seed = _set_initial_temperature(T_count, T, seed)

        # Stage 1: Short equilibration in NVT at T for 10 ps
        structure1, _ = _run_lammps_md(
            structure=structure0,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=T,
            n_ionic_steps=10_000,
            timestep=timestep,
            n_dump=10_000,
            n_log=100,
            initial_temperature=initial_temperature,
            langevin=False,
            seed=seed,
            server_kwargs=server_kwargs,
        )

        # Stage 2: NPT equilibration runs at T,p.
        structure2, _ = _run_lammps_md(
            structure=structure1,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=T,
            pressure=sim_pressure,
            n_ionic_steps=equilibration_steps,
            timestep=timestep,
            n_dump=equilibration_steps,
            n_log=100,
            initial_temperature=0,
            langevin=True,
            server_kwargs=server_kwargs,
        )

        # Stage 3: NPT production runs at T,p.
        cte_results[T_key] = {}
        _results = {}
        counter_production_run = 1

        # Loop over production runs until convergence or max number of runs is reached
        while counter_production_run <= max_production_runs:
            run_key = f"run{counter_production_run:02d}"
            cte_results[T_key][run_key] = {}
            _results[run_key] = {}

            # to keep track of multiple production runs, print status message
            msg = f"Starting production {run_key} at {T_key}."
            logger.info(msg)

            # actual production run
            structure_production, parsed_output = _run_lammps_md(
                structure=structure2,
                potential=potential,
                tmp_working_directory=tmp_working_directory,
                temperature=T,
                pressure=sim_pressure,
                n_ionic_steps=production_steps,
                timestep=timestep,
                n_dump=n_dump,
                n_log=n_log,
                initial_temperature=0,
                langevin=True,
                server_kwargs=server_kwargs,
            )

            # Collect the intermediate results. kick out unneeded info to save memory
            for propkey in ["steps", "temperature", "energy_tot", "pressures", "volume"]:
                _results[run_key][propkey] = parsed_output.get("generic", None).get(propkey, None)
            _results[run_key].update(parsed_output.get("lammps", None))

            # Calculate and collect cte results
            cte_results[T_key] = _cte_fluctuation_workflow_analysis(
                subresults=_results,
                temperature=T,
                p_in_GPa=pressure,
                T_key=T_key,
                use_running_mean=True,
                N_points=N_for_averaging,
                logger=logger,
            )

            # check for convergence or if max number of production runs is reached
            if counter_production_run > 1:
                if _is_converged(cte_results[T_key], CTE_convergence_criterion):
                    cte_results[T_key]["is_converged"] = "True"
                    cte_results[T_key]["convergence_criterion"] = CTE_convergence_criterion
                    cte_results[T_key]["structure_final"] = structure_production.copy()
                    structure0 = structure_production.copy()
                    break
                if counter_production_run >= max_production_runs:
                    cte_results[T_key]["is_converged"] = "False"
                    cte_results[T_key]["convergence_criterion"] = CTE_convergence_criterion
                    cte_results[T_key]["structure_final"] = structure_production.copy()
                    structure0 = structure_production.copy()
                    break

            # continue with this in next production run
            counter_production_run += 1
            structure2 = structure_production.copy()

    return cte_results
