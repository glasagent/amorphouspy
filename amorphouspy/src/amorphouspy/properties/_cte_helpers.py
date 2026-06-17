"""Helper functions for CTE simulation workflows.

Implements helper functions needed to set up logger file, check (and if necessary modify) user input,
handle simulation data, and check for convergence of CTE values over multiple production runs for the workflow.

Author: Marcel Sadowski (github.com/Gitdowski)
"""

import logging
from typing import Any

import numpy as np

from amorphouspy.properties.cte_analysis import cte_from_npt_fluctuations


def _create_logger() -> logging.Logger:
    """Create and configure a logger for CTE workflow warnings."""
    # create logger
    logger = logging.getLogger("amorphouspy.properties.cte")
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

    Args:
        parsed_output: Output dictionary as returned by `lammps_function`.
        counter_production_run: Index of the current production run.

    Returns:
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


def _get_tail_data(sim_data: dict, n_points: int) -> dict[str, Any]:
    """Extract the last n_points of time-series data from a simulation data dictionary.

    This is used to carry over the tail of one run's data to the next run, so that the
    moving window used for running-mean fluctuation analysis does not cause data loss at
    run boundaries.

    Args:
        sim_data: Simulation data dictionary as returned by ``_collect_sim_data``.
        n_points: Number of trailing data points to extract.

    Returns:
        Dictionary with the same time-series keys, each trimmed to the last *n_points* entries.
        The ``run_index`` key is copied as-is.

    """
    tail: dict[str, Any] = {}
    for key, values in sim_data.items():
        if key == "run_index":
            tail[key] = values
        else:
            tail[key] = values[-n_points:]
    return tail


def _prepend_tail_data(sim_data: dict, tail_data: dict) -> dict[str, Any]:
    """Prepend tail data from a previous run to the current run's simulation data.

    The combined data array is used for the CTE fluctuation calculation so that the
    running-mean window can cover the boundary between runs without data loss.

    Args:
        sim_data: Current production run simulation data.
        tail_data: Tail data extracted from the previous run (or equilibration).

    Returns:
        New dictionary with time-series arrays that have *tail_data* prepended.
        The ``run_index`` key is taken from *sim_data* (current run).

    """
    combined: dict[str, Any] = {}
    for key, value in sim_data.items():
        if key == "run_index":
            combined[key] = value
        else:
            combined[key] = np.concatenate([tail_data[key], value])
    return combined


def _sanity_check_sim_data(
    sim_data: dict,
    T_target: float,
    p_target: float,
    logger: logging.Logger,
) -> None:
    """Perform sanity checks on actual vs target temperature and pressure values."""
    REL_TOL = 0.02  # 2 % relative tolerance
    ABS_TEMP_TOL = 5  # 5 K absolute tolerance
    ABS_PRESS_TOL = 0.05  # 0.05 GPa = 50 MPa absolute tolerance

    run_index = int(sim_data["run_index"][0])
    T_actual = np.mean(sim_data["T"])
    ptot_actual = np.mean(sim_data["ptot"])
    pxx_actual = np.mean(sim_data["pxx"])
    pyy_actual = np.mean(sim_data["pyy"])
    pzz_actual = np.mean(sim_data["pzz"])

    if abs(T_target - T_actual) > (ABS_TEMP_TOL + REL_TOL * T_target):
        msg = f"\n  Temperature differences (>5% and > 10 K) are observed during production run {run_index:02d}:\n"
        msg += f"  Specified target temperature = {T_target:.2f} K\n"
        msg += f"  Actual average temperature   = {T_actual:.2f} K.\n"
        msg += "  Using target temperature for CTE calculation."
        logger.warning(msg)

    if abs(p_target - ptot_actual) > (ABS_PRESS_TOL + REL_TOL * p_target):
        msg = (
            f"\n  Pressure (ptot) differences (>5% and > 10 MPa) are observed during production run {run_index:02d}:\n"
        )
        msg += f"  Specified target pressure = {p_target / 1e6:.2e} MPa\n"
        msg += f"  Actual average pressure   = {ptot_actual / 1e6:.2e} MPa.\n"
        msg += "  Using target pressure for CTE calculation."
        logger.warning(msg)

    if abs(p_target - pxx_actual) > (ABS_PRESS_TOL + REL_TOL * p_target):
        msg = f"\n  Pressure (pxx) differences (>5% and > 10 MPa) are observed during production run {run_index:02d}:\n"
        msg += f"  Specified target pressure = {p_target / 1e6:.2e} MPa\n"
        msg += f"  Actual average pressure   = {pxx_actual / 1e6:.2e} MPa.\n"
        msg += "  Using target pressure for CTE calculation."
        logger.warning(msg)

    if abs(p_target - pyy_actual) > (ABS_PRESS_TOL + REL_TOL * p_target):
        msg = f"\n  Pressure (pyy) differences (>5% and > 10 MPa) are observed during production run {run_index:02d}:\n"
        msg += f"  Specified target pressure = {p_target / 1e6:.2e} MPa\n"
        msg += f"  Actual average pressure   = {pyy_actual / 1e6:.2e} MPa.\n"
        msg += "  Using target pressure for CTE calculation."
        logger.warning(msg)

    if abs(p_target - pzz_actual) > (ABS_PRESS_TOL + REL_TOL * p_target):
        msg = f"\n  Pressure (pzz) differences (>5% and > 10 MPa) are observed during production run {run_index:02d}:\n"
        msg += f"  Specified target pressure = {p_target / 1e6:.2e} MPa\n"
        msg += f"  Actual average pressure   = {pzz_actual / 1e6:.2e} MPa.\n"
        msg += "  Using target pressure for CTE calculation."
        logger.warning(msg)


def _check_min_max_runs(
    min_production_runs: int,
    max_production_runs: int,
    logger: logging.Logger,
    min_allowed: int = 2,
) -> tuple[int, int]:
    """Ensure min/max production runs are sensible."""
    if min_production_runs < min_allowed:
        msg = "\n  At least 2 individual production runs are needed to check for CTE convergence."
        msg += f"\n  However, a value of {min_production_runs} was provided."
        msg += f"\n  Automatically setting min_production_runs to {min_allowed} and continue."
        msg += "\n  Consider increasing min_production_runs even further if convergence is not reached."
        logger.warning(msg)
        min_production_runs = min_allowed

    if max_production_runs < min_production_runs:
        msg = "\n  Maximum number of production runs needs to be at least the minimum number of production runs."
        msg += f"\n  However, min_production_runs is {min_production_runs} and {max_production_runs} was"
        msg += " provided for max_production_runs."
        msg += f"\n  Automatically setting max_production_runs to {min_production_runs} and continue."
        msg += "\n  Consider increasing max_production_runs even further if convergence is not reached."
        logger.warning(msg)
        max_production_runs = min_production_runs

    return min_production_runs, max_production_runs


def _ensure_prod_steps_minimum(
    production_steps: int,
    averaging_time_ps: int,
    timestep: float,
    logger: logging.Logger,
) -> int:
    """Ensure production steps are long enough for running-mean based fluctuation analysis."""
    min_production_steps = int(2 * averaging_time_ps * 1000 / timestep)
    if production_steps < min_production_steps:
        msg = "\n  For calculating fluctuations based on running averages, sufficient data is needed."
        msg += f"\n  With currently averaging over {averaging_time_ps} ps, production runs are too short "
        msg += f"and we recommend at least {2 * averaging_time_ps} ps."
        msg += f"\n  Automatically re-setting the user-specified production_steps of {production_steps} "
        msg += f"to {min_production_steps} and continue."
        msg += "\n  Consider increasing production_steps even further to get more reliable results."
        logger.warning(msg)
        production_steps = min_production_steps
    return production_steps


def _warn_running_mean_points(
    N_for_averaging: int,
    min_points: int,
    logger: logging.Logger,
) -> None:
    """Warn if too few data points are used for the running mean."""
    if N_for_averaging < min_points:
        msg = "\n  Running mean values are most likely based on insufficient data points."
        msg += f"\n  We recommend averaging over at least {min_points} data points, "
        msg += f"but currently only {N_for_averaging} are used."
        msg += "\n  Consider decreasing n_print_thermo or change hard-coded AVERAGING_TIME_IN_PS variable."
        msg += "\n  Continuing regardless."
        logger.warning(msg)


def _ensure_equilibration_steps(
    equilibration_steps: int,
    N_for_averaging: int,
    n_print_thermo: int,
    logger: logging.Logger,
) -> int:
    """Ensure equilibration produces at least one full window of data."""
    min_equilibration_steps = N_for_averaging * n_print_thermo
    if equilibration_steps < min_equilibration_steps:
        msg = "\n  The equilibration must produce at least one window width of data for the tail carry-over."
        msg += f"\n  The current window size requires {min_equilibration_steps} equilibration steps,"
        msg += f" but only {equilibration_steps} were specified."
        msg += f"\n  Automatically setting equilibration_steps to {min_equilibration_steps} and continue."
        logger.warning(msg)
        equilibration_steps = min_equilibration_steps
    return equilibration_steps


def _fluctuation_simulation_input_checker(
    production_steps: int,
    min_production_runs: int,
    max_production_runs: int,
    n_print_thermo: int,
    timestep: float,
    equilibration_steps: int,
    logger: logging.Logger,
) -> tuple[int, int, int, int, int]:
    """Check and adjust input parameters for cte_from_fluctuations_simulation workflow."""
    AVERAGING_TIME_IN_PS = 10
    MIN_RUNNING_MEAN_POINTS = 1000

    min_production_runs, max_production_runs = _check_min_max_runs(min_production_runs, max_production_runs, logger)
    production_steps = _ensure_prod_steps_minimum(production_steps, AVERAGING_TIME_IN_PS, timestep, logger)

    N_for_averaging = int(AVERAGING_TIME_IN_PS * 1000 / n_print_thermo / timestep)
    _warn_running_mean_points(N_for_averaging, MIN_RUNNING_MEAN_POINTS, logger)

    equilibration_steps = _ensure_equilibration_steps(equilibration_steps, N_for_averaging, n_print_thermo, logger)

    return production_steps, min_production_runs, max_production_runs, N_for_averaging, equilibration_steps


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

    Args:
        sim_data: Parsed output dictionary from `cte_simulation` workflow for a specific temperature.
        temperature: Target temperature in K.
        p: Target pressure in GPa (for consistent usage of pyiron units).
        N_points: Window size if running mean approach is used, ignored otherwise. (default 1000)
        use_running_mean: If False, fluctuations are calculated using the mean of the whole trajecotry.
            If True, fluctuations are calculated based on running mean values (helpful if non-stationary
            systems with drifts in energy and volume are observed). Note that this does not guarantee
            correct results for strongly non-equilibrium systems. (default False)

    Returns:
        Dictionary containing the calculated CTE values of the current production run:
        { "CTE_V": ..., "CTE_x": ..., "CTE_y": ..., "CTE_z": ... }

    Notes:
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


def _fluctuation_simulation_uncertainty_check(data: dict, criterion: float) -> tuple[bool, dict[str, float | str]]:
    """Check for convergence of CTE values over multiple production runs.

    This function uses the CTE values that have been collected in all previous productions runs.
    It computes the current uncertainty as the standard deviation divided by the square root of
    the number of values. This is done individually for CTE_V, CTE_x, CTE_y, CTE_z. The uncertainty
    is then compared to the given criterion. The criterion of the volumetric CTE should be sqrt(3)
    times the user-specified criterion for an isotropic system with uncorrelated linear components.
    If all four values are within the specified tolerance, convergence is reached and True is
    returned along with a dict of the mean CTE values and their uncertainties. Otherwise, False is
    returned along with a dict of the mean CTE values and their uncertainties.

    Args:
        data: Dictionary with the collected data from the previous production runs containing
        CTE values.
        criterion: Convergence criterion for the uncertainty CTE values.

    Returns:
        Tuple of one boolean (False or True) and dictionary with the mean CTE values and their
        uncertainties:
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

    Args:
        previous_data: Dictionary containing previously collected results.
        new_sim_data: Dictionary containing new simulation data, whose averages will be added.
        new_cte_data: Dictionary containing new CTE data to be added.

    Returns:
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


def _temperature_scan_input_checker(temperature: list[int | float], logger: logging.Logger) -> None:
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


def _temperature_scan_merge_results(previous_data: dict, new_sim_data: dict) -> dict[str, Any]:
    """Merge the newly collected simulation data to the previously collected data.

    Args:
        previous_data: Dictionary containing previously collected results.
        new_sim_data: Dictionary containing new simulation data, whose averages will be added.

    Returns:
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
