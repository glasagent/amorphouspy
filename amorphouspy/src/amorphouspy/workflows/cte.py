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
from lammpsparser.compatibility.file import lammps_file_interface_function

from amorphouspy.analysis.cte import cte_from_npt_fluctuations
from amorphouspy.io_utils import structure_from_parsed_output
from amorphouspy.workflows.shared import get_lammps_command


def _create_logger() -> logging.Logger:
    """Create and configure a logger for CTE workflow warnings."""
    # create logger
    logger = logging.getLogger("amorphouspy.workflows.cte")
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
    temperature: float,
    n_ionic_steps: int,
    timestep: float,
    n_dump: int,
    n_log: int,
    initial_temperature: float,
    pressure: float | list[int, float, None] | None = None,
    server_kwargs: dict | None = None,
    *,
    langevin: bool = False,
    seed: int | None = 12345,
    tmp_working_directory: str | Path | None = None,
) -> tuple[Atoms, dict]:  # pylint: disable=too-many-positional-arguments
    """Run a LAMMPS MD calculation with given parameters and return the final structure and parsed output.

    Args:
        structure: The atomic structure to simulate.
        potential: The potential file to be used for the simulation.
        temperature: The target temperature for the MD run. Can be a single value or a list [start, end].
        n_ionic_steps: Number of MD steps to run.
        timestep: Time step for integration in femtoseconds.
        n_dump: Frequency of dump output writing in simulation steps.
        n_log: Frequency of log output writing in simulation steps.
        initial_temperature: Initial temperature for velocity initialization. If None, the initial
            temperature will be twice the target temperature (which would go immediately down to the target temperature
            as described in equipartition theorem). If 0, the velocity field is not initialized (in which case the
            initial velocity given in structure will be used and seed to initialize velocities will be ignored).
        pressure: Target pressure. If None, NVT is used. If a float or int is provided, isotropic NPT is used. If a list
            of 6 values is provided, anisotropic or tricilinic NPT is used.
        server_kwargs: Additional keyword arguments for the server.
        langevin: Whether to use Langevin dynamics.
        seed: Random seed for velocity initialization (default is 12345). Ignored if `initial_temperature` is 0.
        tmp_working_directory: Specifies the location of the temporary directory to run the simulations.
            Per default (None), the directory is located in the operating systems location for temperary files.
            With the specification of tmp_working_directory, the temporary directory is created in the specified
            location. Therefore, tmp_working_directory needs to exist beforehand.

    Returns:
        A tuple containing:
            - structure_final: Final atomic structure from the simulation.
            - parsed_output: Parsed output dictionary returned by `lammps_function`.

    Notes:
        - Automatically manages a temporary working directory and cleans it after execution.
        - Uses `lammpsparser.compatibility.file.lammps_file_interface_function` as the backend.
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


def _initialize_datadict(*, with_CTE_keys: bool = False) -> dict:
    """Initialize dict to store data over multiple production runs."""
    init_dict = {
        "run_index": np.array([]),
        "steps": np.array([]),
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

    cte_dict = {"CTE_V": np.array([]), "CTE_x": np.array([]), "CTE_y": np.array([]), "CTE_z": np.array([])}

    if with_CTE_keys:
        init_dict.update(cte_dict)

    return init_dict


def _collect_sim_data(parsed_output: dict, counter_production_run: int) -> dict[str, Any]:
    """Extract the data of interest from the parsed output of a LAMMPS MD run.

    Parameters
    ----------
    ----------:
    parsed_output : dict
        Output dictionary as returned by `lammps_function`.
    counter_production_run : int
        Index of the current production run.

    Returns
    -------
    dict[str, Any]
        Parsed data dictionary with the relevant properties extracted and mapped to our keys:
        { "run_index" : ..., "steps" : ..., "T" : ..., "E_tot" : ..., "ptot" : ..., "pxx" : ...,
          "pyy" : ..., "pzz" : ..., "V" : ..., "Lx" : ..., "Ly" : ..., "Lz" : ... }

    """
    data = _initialize_datadict(with_CTE_keys=False)
    data["run_index"] = np.array([counter_production_run])

    # Map from the keys as given in lammps generic output to our keys
    key_map = {"steps": "steps", "temperature": "T", "energy_tot": "E_tot", "volume": "V"}
    for simkey, propkey in key_map.items():
        data[propkey] = np.array(parsed_output["generic"][simkey])

    # Pressure needs extra care
    pxx = np.array(parsed_output["generic"]["pressures"][:, 0, 0])  # in GPa
    pyy = np.array(parsed_output["generic"]["pressures"][:, 1, 1])  # in GPa
    pzz = np.array(parsed_output["generic"]["pressures"][:, 2, 2])  # in GPa
    p_tot = (pxx + pyy + pzz) / 3  # hydrostatic pressure in GPa
    data["ptot"] = p_tot
    data["pxx"] = pxx
    data["pyy"] = pyy
    data["pzz"] = pzz

    # Use the lammps output for box lengths from log file
    for key in ["Lx", "Ly", "Lz"]:
        data[key] = np.array(parsed_output["lammps"][key])

    return data


def _sanity_check_sim_data(
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


def _fluctuation_simulation_input_checker(
    production_steps: int,
    min_production_runs: int,
    max_production_runs: int,
    n_log: int,
    timestep: float,
    n_dump: int,
    logger: logging.Logger,
) -> tuple[int, int, int, int, int]:
    """Check and adjust input parameters for cte_from_fluctuations_simulation workflow."""
    # Minimum choices for a working CTE calculation. For reliable results, use considreably higher values!
    MIN_PRODUCTION_RUNS = 2
    AVERAGING_TIME_IN_PS = 10
    MIN_PRODUCTION_STEPS = int(2 * AVERAGING_TIME_IN_PS * 1000 / timestep)
    MIN_RUNNING_MEAN_POINTS = 1000

    if min_production_runs < MIN_PRODUCTION_RUNS:
        msg = "\n  At least 2 individual production runs are needed to check for CTE convergence."
        msg += f"\n  However, a value of {min_production_runs} was provided."
        msg += f"\n  Automatically setting min_production_runs to {MIN_PRODUCTION_RUNS} and continue."
        msg += "\n  Consider increasing min_production_runs even further if convergence is not reached."
        logger.warning(msg)
        min_production_runs = MIN_PRODUCTION_RUNS

    if max_production_runs < min_production_runs:
        msg = "\n  Maximum number of production runs needs to be at least the minimum number of production runs."
        msg += f"\n  However, min_production_runs is {min_production_runs} and {max_production_runs} was"
        msg += " provided for max_production_runs."
        msg += f"\n  Automatically setting max_production_runs to {min_production_runs} and continue."
        msg += "\n  Consider increasing max_production_runs even further if convergence is not reached."
        logger.warning(msg)
        max_production_runs = min_production_runs

    if production_steps < MIN_PRODUCTION_STEPS:
        msg = "\n  For calculating fluctuations based on running averages, sufficient data is needed."
        msg += f"\n  With currently averaging over {AVERAGING_TIME_IN_PS} ps, production runs are too short "
        msg += f"and we recommend at least {2 * AVERAGING_TIME_IN_PS} ps."
        msg += f"\n  Automatically re-setting the user-specified production_steps of {production_steps} "
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

    if n_dump > production_steps:
        msg = "\n  Dump frequency n_dump is larger than the total number of production steps."
        msg += f"\n  Currently, n_dump = {n_dump} and production_steps = {production_steps}."
        msg += f"\n  Automatically setting n_dump to production_steps ({production_steps}) and continue."
        logger.warning(msg)
        n_dump = production_steps

    return production_steps, min_production_runs, max_production_runs, N_for_averaging, n_dump


def _fluctuation_simulation_cte_calculation(
    sim_data: dict,
    temperature: float,
    p: float,
    N_points: int = 1000,
    *,
    use_running_mean: bool = False,
) -> dict[str, float]:
    """Parse the specific collected output from cte_simulation workflow.

    This function extracts and computes the "instantaneous enthalpy" and volume data from
    the different production runs and computes the isotropic CTE based on the H-V fluctuations.
    This is done in a fashion to be able to check for convergence in the workflow.
    E.g., if multiple production runs are available, the CTE is first calculated for only
    the first run, then for the combined data of the first two runs, etc.

    Parameters
    ----------
    sim_data : dict
        Parsed output dictionary from `cte_simulation` workflow for a specific temperature.
    temperature : float
        Target temperature in K.
    p : float
        Target pressure in GPa (for consistent usage of pyiron units).
    N_points : int, optional
        Window size if running mean approach is used, ignored otherwise. (default 1000)
    use_running_mean : bool, optional
        If False, fluctuations are calculated using the mean of the whole trajecotry.
        If True, fluctuations are calculated based on running mean values (helpful if non-stationary
        systems with drifts in energy and volume are observed). Note that this does not guarantee
        correct results for strongly non-equilibrium systems. (default False)

    Returns
    -------
    dict[str, float]
        Dictionary containing the calculated CTE values of the current production run:
        { "CTE_V": ..., "CTE_x": ..., "CTE_y": ..., "CTE_z": ... }

    Notes
    -----
        - Care needs to be taken to ensure the correct "enthalpy" is used. In Lammps, the enthalpy
        is calculated based on the instantaneous properties, H_inst = E_inst + p_inst * V_inst. However,
        the required property here is better reflected if the pressure that defines the ensemble is used:
        H_ens = E_inst + p_target * V_inst.
        - If isotropic NPT simulations are performed (see "iso" keyword in lammps), the apparent hydrostatic
        pressure = (pxx+pyy+pzz)/3 will breflect the user-specified pressure. Hoever, individual pxx, pyy
        and pzz components can deviate significantly if the structure is not fully relaxed or shows anisotropic
        features. In this case, it is not entirely clear if the hydrostatic pressure or the actual individual
        pressures should be used to calculate the 1D CTE components. Therefore, we perform anisotropic NPT
        simulations (see "aniso" keyword in lammps) per default and the user-specified pressure is applied to all
        individual components.

    """
    # collect final output here
    cte_data = {}

    # convert GPa to Pa, Pa*Ang^3 to eV, and compute enthalpy
    p_in_Pa = p * 1e9
    PaAng3_to_eV = 6.2415e-12
    H = sim_data["E_tot"] + p_in_Pa * sim_data["V"] * PaAng3_to_eV

    # compute CTE based on H-V fluctuations
    cte_data["CTE_V"] = cte_from_npt_fluctuations(
        temperature=temperature,
        enthalpy=H,
        volume=sim_data["V"],
        use_running_mean=use_running_mean,
        N_points=N_points,
    )
    cte_data["CTE_x"] = cte_from_npt_fluctuations(
        temperature=temperature,
        enthalpy=H,
        volume=sim_data["Lx"],
        use_running_mean=use_running_mean,
        N_points=N_points,
    )
    cte_data["CTE_y"] = cte_from_npt_fluctuations(
        temperature=temperature,
        enthalpy=H,
        volume=sim_data["Ly"],
        use_running_mean=use_running_mean,
        N_points=N_points,
    )
    cte_data["CTE_z"] = cte_from_npt_fluctuations(
        temperature=temperature,
        enthalpy=H,
        volume=sim_data["Lz"],
        use_running_mean=use_running_mean,
        N_points=N_points,
    )

    return cte_data


def _fluctuation_simulation_uncertainty_check(data: dict, criterion: float) -> tuple[bool, dict]:
    """Check for convergence of CTE values over multiple production runs.

    This function uses the CTE values that have been collected in all previous productions runs.
    It computes the current uncertainty as the standard deviation divided by the square root of
    the number of values. This is done individually for CTE_V, CTE_x, CTE_y, CTE_z. The uncertainty
    is then compared to the given criterion. The criterion of the volumetric CTE should be sqrt(3)
    times the user-specified criterion for an isotropic system with uncorrelated linear components.
    If all four values are within the specified tolerance, convergence is reached and True is
    returned along with a dict of the mean CTE values and their uncertainties. Otherwise, False is
    returned along with a dict of the mean CTE values and their uncertainties.

    Parameters
    ----------
    data : dict
        Dictionary with the collected data from the previous production runs containing
    criterion : float
        Convergence criterion for the uncertainty CTE values.

    Returns
    -------
    tuple[bool, dict]
        Tuple of False or True and dictionary with the mean CTE values and their uncertainties:
        { "CTE_V_mean": ..., "CTE_x_mean": ..., "CTE_y_mean": ..., "CTE_z_mean": ...,
        "CTE_V_uncertainty": ..., "CTE_x_uncertainty": ...,
        "CTE_y_uncertainty": ..., "CTE_z_uncertainty": ... }

    """

    def _uncertainty(values: np.ndarray) -> float:
        """Calculate the uncertainty of the given values."""
        return np.std(values, ddof=1) / np.sqrt(len(values))

    # Ensure data is not empty before calculating uncertainty or mean
    for cte_key in ["CTE_V", "CTE_x", "CTE_y", "CTE_z"]:
        if len(data[cte_key]) <= 1:
            msg = f"Data for '{cte_key}' must contain more than one value. Cannot calculate uncertainty from this."
            raise ValueError(msg)

    # Check for each CTE value if its uncertainty is within criterion
    bools = []

    # Volume CTE uses sqrt(3) times the criterion
    if _uncertainty(data["CTE_V"]) <= criterion * np.sqrt(3):
        bools.append(True)
    else:
        bools.append(False)

    # The linear CTEs use the given criterion
    for cte_key in ["CTE_x", "CTE_y", "CTE_z"]:
        if _uncertainty(data[cte_key]) <= criterion:
            bools.append(True)
        else:
            bools.append(False)

    # Return True/False if converged/not converged and the data
    return all(bools), {
        "CTE_V_mean": float(np.mean(data["CTE_V"])),
        "CTE_x_mean": float(np.mean(data["CTE_x"])),
        "CTE_y_mean": float(np.mean(data["CTE_y"])),
        "CTE_z_mean": float(np.mean(data["CTE_z"])),
        "CTE_V_uncertainty": float(_uncertainty(data["CTE_V"])),
        "CTE_x_uncertainty": float(_uncertainty(data["CTE_x"])),
        "CTE_y_uncertainty": float(_uncertainty(data["CTE_y"])),
        "CTE_z_uncertainty": float(_uncertainty(data["CTE_z"])),
    }


def _fluctuation_simulation_merge_results(
    previous_data: dict, new_sim_data: dict, new_cte_data: dict
) -> dict[str, Any]:
    """Merge the newly collected simulation data to the previously collected data.

    Parameters
    ----------
    previous_data : dict
        Dictionary containing previously collected results.
    new_sim_data : dict
        Dictionary containing new simulation data, whose averages will be added.
    new_cte_data : dict
        Dictionary containing new CTE data to be added.

    Returns
    -------
    dict[str, Any]
        Updated dictionary with averaged simulation data and CTE data.

    """
    merged_data = {}
    # Collect the average of the simulation data
    for key, values in new_sim_data.items():
        if key == "steps":
            merged_data[key] = np.append(previous_data[key], values[-1])
        elif key == "run_index":
            merged_data[key] = np.append(previous_data[key], values)
        else:
            merged_data[key] = np.append(previous_data[key], np.mean(values))
    # Collect all CTE data
    for key, values in new_cte_data.items():
        merged_data[key] = np.append(previous_data[key], values)

    return merged_data


def cte_from_fluctuations_simulation(
    structure: Atoms,
    potential: str,
    temperature: float | list[int | float] = 300,
    pressure: float = 1e-4,
    timestep: float = 1.0,
    equilibration_steps: int = 100_000,
    production_steps: int = 200_000,
    min_production_runs: int = 2,
    max_production_runs: int = 25,
    CTE_uncertainty_criterion: float = 1e-6,
    n_dump: int = 100000,
    n_log: int = 10,
    server_kwargs: dict[str, Any] | None = None,
    *,
    aniso: bool = False,
    seed: int | None = 12345,
    tmp_working_directory: str | Path | None = None,
) -> dict[str, Any]:  # pylint: disable=too-many-positional-arguments
    """Perform a LAMMPS-based cte simulation protocol based on fluctuations.

    This workflow equilibrates a structure at a target temperature and performs a
    production MD run to collect the instantaneous total energy, pressure and volume,
    needed to compute the CTE via H-V fluctuation analysis or V-T data, depending
    if only one temperature or multiple temperatures should be probed.

    The number of steps used here is only for testing purposes.
    It is assumed in this workflow that the given in structure is pre-equilibrated.

    Parameters
    ----------
    structure : Atoms
        Input structure (assumed pre-equilibrated).
    potential : str
        LAMMPS potential file.
    temperature : float | list[int | float]
        Simulation temperature in Kelvin (default 300 K).
    pressure : float
        Target pressure in GPa (use pyiron units here!) for NPT simulations (default 10-4 GPa = 10^5 Pa = 1 bar).
    timestep : float
        MD integration timestep in femtoseconds (default 1.0 fs).
    equilibration_steps : int
        Number of MD steps for the equilibration run (default 100,000).
    production_steps : int
        Number of MD steps for the production runs (default 200,000).
    min_production_runs : int
        Minimum number of production runs to perform before checking for convergence (default 2).
    max_production_runs : int
        Maximum number of production runs to perform before checking for convergence (default 10).
    CTE_uncertainty_criterion : float
        Convergence criterion for the uncertainty of the linear CTE (default 1e-6/K).
    n_dump : int
        Dump output frequency of the production runs (default 100,000).
    n_log : int
        Log output frequency (default 10).
    server_kwargs : dict[str, Any] | None
        Additional server configuration arguments for pyiron.
    aniso : bool
        If false, an isotropic NPT calculation is performed and the simulation box is
        scaled uniformly. If True, anisotropic NPT calculation is performed and the simulation
        box can change shape and size independently along each axis (default True).
    seed : int | None
        Random seed for velocity initialization (default 12345). If None, a random seed is used.
    tmp_working_directory : str | Path | None
        Temporary directory for job execution.

    Returns
    -------
    dict[str, Any]
        Nested dictionary containing the "summary" and "data" keys. In the "summary" section the CTE
        values and their uncertainties are returned together with info whether convergence was reached
        within the max_production_runs and the convergence criterion.
        The "data" holds the collected data from all individual production runs. "run_index" is to
        clearly identify which production run the data belongs to. "steps" contains the number of steps
        for each run. Thermodynamic and structural data are averaged over each production run and these
        averages are listed under the respective key. Finally, also the structure after the simulation
        has finished is stored for further use or analysis. Example:
        { 'summary' : {"CTE_V_mean" : ..., "CTE_x_mean" : ...,
                       "CTE_y_mean" : ..., "CTE_z_mean" : ...,
                       "CTE_V_uncertainty" : ..., "CTE_x_uncertainty" : ...,
                       "CTE_y_uncertainty" : ..., "CTE_z_uncertainty" : ...,
                       "is_converged" : "True" or "False",
                       "convergence_criterion" : float
                       },
          'data':  {"run_index" : [1, 2, 3, ...],
                    "steps" : [...],
                    "T" : [...],
                    "E_tot" : [...],
                    "ptot" : [...],
                    "pxx" : [...],
                    "pyy" : [...],
                    "pzz" : [...],
                    "V" : [...],
                    "Lx" : [...],
                    "Ly" : [...],
                    "Lz" : [...],
                    "CTE_V" : [...],
                    "CTE_x" : [...],
                    "CTE_y" : [...],
                    "CTE_z" : [...],
                    "structure_final" : Atoms
                    }
        }

    Notes
    -----
        - How to chose the uncertainty criterion for the volumetric CTE? Should it be the same as the defined
          CTE_uncertainty_criterion applied to the linear CTEs? The volumetric CTE is approximately
          the sum of the linear CTEs along x, y, and z. If three variables x, y and z were uncorrelated and
          have known uncertainties sigma_x, sigma_y, and sigma_z, the uncertainty of the sum of those them
          would be: sigma_V = sqrt( sigma_x**2 + sigma_y**2 + sigma_z**2 ). If we assume that the uncertainty
          criterion is reached at roughly the same time for all three variables, the uncertainty of the
          volumetric CTE can be approximated as sqrt(3)*CTE_uncertainty_criterion. However, x, y and z are
          typically not uncorrelated. If calculated from the actual simulation data, the uncertainty of CTE_V
          is found to be approximately the same as the individual uncertainties of the linear CTEs. Therefore,
          a sqrt(3)*CTE_uncertainty_criterion is likely to be on the safe side for the volumetric CTE, with
          the actual uncertainty of CTE_V being smaller than that.
        - The structure is first pre-equilibrated with a short, hard-coded 10 ps NVT run. Only then
          follows the user-defined NPT equilibration and production runs.

    """
    # Logging setup
    logger = _create_logger()

    # Check and adjust input parameters if necessary
    production_steps, min_production_runs, max_production_runs, N_for_averaging, n_dump = (
        _fluctuation_simulation_input_checker(
            production_steps, min_production_runs, max_production_runs, n_log, timestep, n_dump, logger
        )
    )

    # Set pressure to anisotropic if requested
    sim_pressure = [pressure, pressure, pressure, None, None, None] if aniso else pressure

    # initial structure used. Afterwards, it is updated after each temperature
    structure0 = structure

    logger.info("Starting 10 ps (hardcoded) NVT equilibration at %.2f K.", temperature)

    # Stage 1: Short equilibration in NVT at T for 10 ps
    structure1, _ = _run_lammps_md(
        structure=structure0,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=temperature,
        n_ionic_steps=10_000,
        timestep=timestep,
        n_dump=10_000,
        n_log=100,
        initial_temperature=temperature,
        langevin=False,
        seed=seed,
        server_kwargs=server_kwargs,
    )

    equilibration_time = equilibration_steps / timestep / 1000
    logger.info("Starting %.1f ps NVT equilibration at %.2f K and %.2e GPa.", equilibration_time, temperature, pressure)

    # Stage 2: NPT equilibration runs at T,p.
    structure2, _ = _run_lammps_md(
        structure=structure1,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=temperature,
        pressure=sim_pressure,
        n_ionic_steps=equilibration_steps,
        timestep=timestep,
        n_dump=equilibration_steps,
        n_log=100,
        initial_temperature=0,
        langevin=True,
        server_kwargs=server_kwargs,
    )

    # Stage 3: NPT production runs (loop) at T,p.
    results = _initialize_datadict(with_CTE_keys=True)
    counter_production_run = 1
    production_time = production_steps / timestep / 1000
    while counter_production_run <= max_production_runs:
        # to keep track of multiple production runs, print status message
        logger.info(
            "Starting %.1f ps NPT production run #%03d at %.2f K and %.2e GPa.",
            production_time,
            counter_production_run,
            temperature,
            pressure,
        )

        # actual production run
        structure_production, parsed_output = _run_lammps_md(
            structure=structure2,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature,
            pressure=sim_pressure,
            n_ionic_steps=production_steps,
            timestep=timestep,
            n_dump=n_dump,
            n_log=n_log,
            initial_temperature=0,
            langevin=True,
            server_kwargs=server_kwargs,
        )

        # parse and check the output of the production run
        _sim_data = _collect_sim_data(parsed_output, counter_production_run)
        _sanity_check_sim_data(sim_data=_sim_data, T_target=temperature, p_target=pressure, logger=logger)

        # Calculate cte based on the data of the current production run
        _cte_results = _fluctuation_simulation_cte_calculation(
            sim_data=_sim_data,
            temperature=temperature,
            p=pressure,
            use_running_mean=True,
            N_points=N_for_averaging,
        )

        # merge results to have the averages over all production runs so far
        results = _fluctuation_simulation_merge_results(
            previous_data=results, new_sim_data=_sim_data, new_cte_data=_cte_results
        )

        # start checking for convergence once the min number of production runs have been executed
        if counter_production_run >= min_production_runs:
            converge_bool, cte_summary = _fluctuation_simulation_uncertainty_check(results, CTE_uncertainty_criterion)
            logger.info(
                "Production run #%03d finished. Current CTE_V = %.4e +/- %.4e 1/K.",
                counter_production_run,
                cte_summary["CTE_V_mean"],
                cte_summary["CTE_V_uncertainty"],
            )

            # If converged, break the loop and update the summary accordingly
            if converge_bool:
                cte_summary.update({"is_converged": "True", "convergence_criterion": CTE_uncertainty_criterion})
                msg = f"All CTEs converged after production run #{counter_production_run:03d}."
                msg += "\nFINISHED SUCCESSFULLY."
                logger.info(msg)
                break

            # Also break the loop if max number of production runs is reached without convergence and update summary
            if counter_production_run == max_production_runs:
                cte_summary.update({"is_converged": "False", "convergence_criterion": CTE_uncertainty_criterion})
                msg = f"Maximum number of production runs ({max_production_runs}) reached without CTE convergence."
                msg += "\nFINISHED WITHOUT CONVERGENCE."
                logger.info(msg)
                break

        # In all other cases, continue with the next production run
        counter_production_run += 1
        structure2 = structure_production

    results.update({"structure_final": structure_production})
    return {"summary": cte_summary, "data": results}


def _temperature_scan_input_checker(
    temperature: list[int | float], production_steps: int, n_dump: int, logger: logging.Logger
) -> int:
    """Check and adjust input parameters for cte_from_temperature_scan_simulation workflow."""
    MIN_TEMP_ENTRIES = 2
    if len(set(temperature)) < MIN_TEMP_ENTRIES:
        msg = "\n  At least two different temperatures are needed to compute CTE from temperature scan."
        msg += "\n  Please provide at least two non identical temperatures and continue."
        logger.error(msg)
        raise ValueError(msg)

    if len(set(temperature)) != len(temperature):
        msg = "\n  There are duplicates in the provided temperatures. This can influence the results. "
        msg += "\n  I hope you know what you are doing. Continuing with the provided temperatures."
        logger.warning(msg)

    if n_dump > production_steps:
        msg = "\n  Dump frequency n_dump is larger than the total number of production steps."
        msg += f"\n  Currently, n_dump = {n_dump} and production_steps = {production_steps}."
        msg += f"\n  Automatically setting n_dump to production_steps ({production_steps}) and continue."
        logger.warning(msg)
        n_dump = production_steps

    return n_dump


def _temperature_scan_merge_results(previous_data: dict, new_sim_data: dict) -> dict[str, Any]:
    """Merge the newly collected simulation data to the previously collected data.

    Parameters
    ----------
    previous_data : dict[str, Any]
        Dictionary containing previously collected results.
    new_sim_data : dict[str, Any]
        Dictionary containing new simulation data, whose averages will be added.

    Returns
    -------
    dict[str, Any]
        Updated dictionary with averaged simulation data and CTE data.

    """
    merged_data = {}
    # Collect the average of the simulation data, except for steps and run_index
    for key, values in new_sim_data.items():
        if key == "steps":
            merged_data[key] = np.append(previous_data[key], values[-1])
        elif key == "run_index":
            merged_data[key] = np.append(previous_data[key], values)
        else:
            merged_data[key] = np.append(previous_data[key], np.mean(values))

    return merged_data


def temperature_scan_simulation(
    structure: Atoms,
    potential: str,
    temperature: list[int | float] | None = None,
    pressure: float = 1e-4,
    timestep: float = 1.0,
    equilibration_steps: int = 100_000,
    production_steps: int = 200_000,
    n_dump: int = 100000,
    n_log: int = 10,
    server_kwargs: dict[str, Any] | None = None,
    *,
    aniso: bool = True,
    seed: int | None = 12345,
    tmp_working_directory: str | Path | None = None,
) -> dict[Any, Any]:  # pylint: disable=too-many-positional-arguments
    """Perform a temperature scan and collect structural data.

    This workflow performs a temperature scan at the given list of temperatures. For each temperature, it
    equilibrates the structure at the target temperature and pressure and then performs a production MD
    run to collect the average volume and box lengths, needed to compute the CTE via V-T data.
    There differen CTEs often discussed, for example:
    - CTE20-300: Computed between solely as the slope from the two datapoints at 20°C and 300°C.
    - CTE20-600: Computed between solely as the slope from the two datapoints at 20°C and 600°C.
    - CTE over other arbitrary temperature range
    - CTE at a specific temperature based on the slope of the V-T curve at this temperature. This is often
      doen by fitting a linear model or higher polynomials fit to the V-T data and then taking the derivative
      at the temperature of interest.
    Because of the various options and methods to compute the CTE from V-T data, and because the actual CTE
    calculation is rather straightforward once the data is collected, we do not compute it directly in this
    workflow, but rather return the collected data for each temperature and leave the actual CTE calculation
    to the user.

    Args:
        structure: Input structure (assumed pre-equilibrated).
        potential: LAMMPS potential file.
        temperature: Simulation temperature in Kelvin (default 300 K).
        pressure: Target pressure in GPa for NPT simulations.
            (default 10-4 GPa = 10^5 Pa = 1 bar).
        timestep: MD integration timestep in femtoseconds (default 1.0 fs).
        equilibration_steps: Number of MD steps for the equilibration runs (default 100,000).
        production_steps: Number of MD steps for the production runs (default 200,000).
        max_production_runs: Maximum number of production runs to perform (default 10). If max number of
            production runs is reached without convergence, a warning is printed and the
            next temperature is started.
        CTE_convergence_criterion: Convergence criterion for the CTE value calculated based on H-V fluctuations
            calculated over subsequent production runs (default 1e-6).
        n_dump: Dump output frequency of the production runs (default 100,000).
        n_log: Log output frequency (default 10).
        server_kwargs: Additional server configuration arguments.
        aniso: If false, an isotropic NPT calculation is performed and the simulation box is scaled uniformly.
            If True, anisotropic NPT calculation is performed and the simulation box can change shape and
            size independently along each axis (default False).
        seed: Random seed for velocity initialization (default 12345).
        tmp_working_directory: Temporary directory for job execution.

    Returns:
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

    Notes:
        - For every temperature, the structure is first pre-equilibrated with short (10 ps) NVT.
        - Simulation settings for the NVT equilibration run are hard-coded
        - CTEs are calculated sequentially if a list of temperatures is provided. Alternatively, multiple
          jobs with independent temperatures can be submitted to achieve parallelization.

    Example:
        >>> result = cte_simulation(
        ...     structure=my_atoms,
        ...     potential=my_potential,
        ...     temperature=[300, 400, 500],
        ...     production_steps=500000
        ... )

    """
    # Logging setup
    logger = _create_logger()

    # Check and adjust input parameters if necessary
    n_dump = _temperature_scan_input_checker(temperature, production_steps, n_dump, logger)

    # Set pressure to anisotropic if requested
    sim_pressure = [pressure, pressure, pressure, None, None, None] if aniso else pressure

    # initial structure used. Afterwards, it is updated after each temperature
    structure0 = structure.copy()

    # Initialize results dictionary. CTE values will be calculated later
    results = _initialize_datadict(with_CTE_keys=False)

    # Loop over all temperatures
    for counter_run, T in enumerate(temperature, start=1):
        # Stage 1: Short equilibration in NVT at T for 10,000 steps
        nvt_equilibration_time = 10_000 / timestep / 1000
        msg = f"Starting {nvt_equilibration_time:.1f} ps (10,000 steps hardcoded) NVT equilibration at {T:.2f} K."
        logger.info(msg)

        structure1, _ = _run_lammps_md(
            structure=structure0,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=T,
            n_ionic_steps=10_000,
            timestep=timestep,
            n_dump=10_000,
            n_log=100,
            initial_temperature=T,
            langevin=False,
            seed=seed,
            server_kwargs=server_kwargs,
        )

        # Stage 2: NPT equilibration runs at T,p.
        equilibration_time = equilibration_steps / timestep / 1000
        msg = f"Starting {equilibration_time:.1f} ps NPT equilibration at {T:.2f} K and {pressure:.2e} GPa."
        logger.info(msg)

        structure2, _ = _run_lammps_md(
            structure=structure1,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=T,
            pressure=sim_pressure,
            n_ionic_steps=equilibration_steps,
            timestep=timestep,
            n_dump=equilibration_steps,
            n_log=n_log,
            initial_temperature=0,
            langevin=True,
            server_kwargs=server_kwargs,
        )

        # Stage 3: NPT production run
        production_time = production_steps / timestep / 1000
        msg = f"Starting {production_time:.1f} ps NPT production run at {T:.2f} K and {pressure:.2e} GPa."
        logger.info(msg)

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

        # parse and check the output of the production run
        _sim_data = _collect_sim_data(parsed_output, counter_run)

        _sanity_check_sim_data(sim_data=_sim_data, T_target=T, p_target=pressure, logger=logger)

        # Collect results
        results = _temperature_scan_merge_results(previous_data=results, new_sim_data=_sim_data)

        # Use this structure as starting point for next temperature
        structure0 = structure_production

    results.update({"structure_final": structure_production})
    msg = "FINISHED SUCCESSFULLY."
    logger.info(msg)
    return {"data": results}
