"""CTE simulation workflows for glass systems using LAMMPS.

Implements molecular dynamics workflows and post-processing utilities for
CTE calculations based on H-V fluctuations with NPT simulations.

Author
------
Marcel Sadowski (github.com/Gitdowski)
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from ase.atoms import Atoms
from pyiron_atomistics.lammps.lammps import lammps_function
from pyiron_base import job

from pyiron_glass.analysis.cte import CTE_from_NPT_fluctuations
from pyiron_glass.io_utils import structure_from_parsed_output


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
        _shell_output, parsed_output, _job_crashed = lammps_function(
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
            cutoff_radius=None,
            units="metal",
            bonds_kwargs={},
            server_kwargs=server_kwargs,
            enable_h5md=False,
            write_restart_file=False,
            read_restart_file=False,
            restart_file="restart.out",
            executable_path=None,
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


def _CTE_H_V_workflow_analysis(
    subresults: dict,
    T: float | None = None,
    p_in_GPa: float | None = None,
    N: int = 1000,
    *,
    running_mean: bool = False,
    sanity_check: bool = False,
) -> dict:
    """Parse the specific collected output from cte_simulation workflow.

    This function extracts and computes the "instantaneous enthalpy" and volume data from
    the different production runs and computes the isotropic CTE based on the H-V fluctuations.
    This is done in a fashion to be able to check for convergence in the workflow.
    E.g., if multiple production runs are available, the CTE is first calculated for only
    the first run, then for the combined data of the first two runs, etc.
    Care needs to be taken to ensure the correct "enthalpy" is used. In Lammps, the enthalpy
    is calculated based on the instantaneous properties, H_inst = E_inst + p_inst * V_inst. However,
    the required property here is H_ens = E_inst + p_target * V_average, where p_target can be either
    provided by the user or can be computed based on the data.

    Parameters
    ----------
    subresults : dict
        Parsed output dictionary from `cte_simulation` workflow for a specific temperature.
        Structure of the dictionary is:
        {   "run01" : {"pressure" : ..., "volume" : ..., "temperature" : ..., etc},
            "run02" : {"pressure" : ..., "volume" : ..., "temperature" : ..., etc},
            ...
        }
    T : int | float, optional
        Target temperature in K. If not specified the average temperature from the data is used.
    p_in_GPa : float, optional
        Target pressure in GPa (for consistent usage of pyiron units). Will be converted to Pa internally.
        If not specified, the average pressure from the data is used.
    N: int
        Window size for running mean calculation if running_mean is True. (default 1000)
    running_mean: bool
        Conventionally, fluctuations are calculated as difference from the mean of the whole trajectory. If
        running_mean is True, running mean values are used to determine fluctuations. This can be useful for
        non-stationary data where drift in volume and energy is still observed. (default False)
    sanity_check : bool
        If True, perform a sanity check to ensure that the specified temperature and pressure fit to the 
        actual data. If deviations larger than 5% are observed, the average values from the data are used
        instead of the user-provided ones. If False, the user specified values are used without modification. 
        As a fallback, averaged T or p_in_GPa are used if they are specified as None. (default False)
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
                        "V_mean" : ...,    # Average volume
                        "Lx_mean" : ...,   # Average cell length in x
                        "Ly_mean" : ...,   # Average cell length in y
                        "Lz_mean" : ...,   # Average cell length in z
                        },
            "run02" : {"CTE_V" : ..., "CTE_x" : ..., "CTE_y" : ..., "CTE_z" : ..., etc},
            ...
        }

    Notes
    -----
        - In lammps, instantaneous enthalphy is calculated based on the total energy and pressure at every
          time step. Ideally, however, the pressure that defines the ensemble should be used instead of the
          pressure at every individual time step.
        - If isotropic NPT simulations are performed (see "iso" keyword in lammps), the apparent hydrostatic 
          pressure = (pxx+pyy+pzz)/3 will be close to the user-specified pressure, but individual pxx, pyy 
          and pzz components can deviate significantly if the structure is not fully relaxed or shows anisotropic 
          features. In this case, it is not entirely clear if the hydrostatic pressure or the actual individual 
          pressures should be used to calculate the 1D CTE components. Therefore, we perform anisotropic NPT 
          simulations (see "aniso" keyword in lammps) per default and the user-specified pressure is applied to all
          individual components.
    """
    
    # collect output here
    cte_data = {}

    # convert GPa to Pa
    p = p_in_GPa * 1e9 if p_in_GPa is not None else None

    # collected data arrays over multiple production runs
    collected_steps = 0
    collected_T = np.array([])
    collected_Etot = np.array([])
    collected_ptot = np.array([])
    collected_px = np.array([])
    collected_py = np.array([])
    collected_pz = np.array([])
    collected_V = np.array([])
    collected_Lx = np.array([])
    collected_Ly = np.array([])
    collected_Lz = np.array([])

    for run_key in sorted(subresults.keys()):
        cte_data[run_key] = {}

        # collect T, E and p data of the currently looped production run
        collected_steps += int(subresults[run_key]["steps"][-1])
        collected_T = np.append(collected_T, subresults[run_key]["temperature"])
        collected_Etot = np.append(collected_Etot, subresults[run_key]["energy_tot"])
        pxx = subresults[run_key]["pressures"][:, 0, 0] * 1e9  # in Pa
        pyy = subresults[run_key]["pressures"][:, 1, 1] * 1e9  # in Pa
        pzz = subresults[run_key]["pressures"][:, 2, 2] * 1e9  # in Pa
        p_tot = (pxx + pyy + pzz) / 3  # hydrostatic pressure in Pa
        collected_ptot = np.append(collected_ptot, p_tot)
        collected_px = np.append(collected_px, pxx)
        collected_py = np.append(collected_py, pyy)
        collected_pz = np.append(collected_pz, pzz)

        # sanity check to ensure the specified temperature and pressure by the user makes sense
        # Ideally, the temperature and pressure that define the ensemble should be used
        if sanity_check == True:
            T_used = float(np.mean(collected_T)) if T is None or abs(T - np.mean(collected_T)) > 0.05 * T else float(T)
            ptot_used = (
                float(np.mean(collected_ptot)) if p is None or abs(p - np.mean(collected_ptot)) > 0.05 * p else float(p)
            )
            px_used = float(np.mean(collected_px)) if p is None or abs(p - np.mean(collected_px)) > 0.05 * p else float(p)
            py_used = float(np.mean(collected_py)) if p is None or abs(p - np.mean(collected_py)) > 0.05 * p else float(p)
            pz_used = float(np.mean(collected_pz)) if p is None or abs(p - np.mean(collected_pz)) > 0.05 * p else float(p)
        # use user-specified T and p values directly, or as a fallback the average from data 
        else:
            T_used = float(np.mean(collected_T)) if T is None else float(T)
            ptot_used = float(np.mean(collected_ptot)) if p is None else float(p)
            px_used = float(np.mean(collected_px)) if p is None else float(p)
            py_used = float(np.mean(collected_py)) if p is None else float(p)
            pz_used = float(np.mean(collected_pz)) if p is None else float(p)

        # collect structural data
        collected_V = np.append(collected_V, subresults[run_key]["volume"])  # in Ang^3
        collected_Lx = np.append(collected_Lx, subresults[run_key]["Lx"])
        collected_Ly = np.append(collected_Ly, subresults[run_key]["Ly"])
        collected_Lz = np.append(collected_Lz, subresults[run_key]["Lz"])

        # Conversion from Pa*Ang^3 to eV
        PaAng3_to_eV = 6.2415e-12

        # compute CTE based on H-V fluctuations
        cte_data[run_key]["CTE_V"] = CTE_from_NPT_fluctuations(
            T=T_used,
            H=collected_Etot + ptot_used * collected_V * PaAng3_to_eV,
            V=collected_V,
            running_mean=running_mean,
            N=N,
        )
        cte_data[run_key]["CTE_x"] = CTE_from_NPT_fluctuations(
            T=T_used,
            H=collected_Etot + px_used * collected_Lx * PaAng3_to_eV,
            V=collected_Lx,
            running_mean=running_mean,
            N=N,
        )
        cte_data[run_key]["CTE_y"] = CTE_from_NPT_fluctuations(
            T=T_used,
            H=collected_Etot + py_used * collected_Ly * PaAng3_to_eV,
            V=collected_Ly,
            running_mean=running_mean,
            N=N,
        )
        cte_data[run_key]["CTE_z"] = CTE_from_NPT_fluctuations(
            T=T_used,
            H=collected_Etot + pz_used * collected_Lz * PaAng3_to_eV,
            V=collected_Lz,
            running_mean=running_mean,
            N=N,
        )

        # Also keep other averaged properties to easily check for drift or convergence
        cte_data[run_key]["steps"] = collected_steps
        cte_data[run_key]["T"] = T_used
        cte_data[run_key]["ptot"] = ptot_used * 10**-9 # export in GPa
        cte_data[run_key]["pxx"] = px_used * 10**-9
        cte_data[run_key]["pyy"] = py_used * 10**-9
        cte_data[run_key]["pzz"] = pz_used * 10**-9
        cte_data[run_key]["E_tot"] = float(np.mean(collected_Etot))
        cte_data[run_key]["V_mean"] = float(np.mean(collected_V))
        cte_data[run_key]["Lx_mean"] = float(np.mean(collected_Lx))
        cte_data[run_key]["Ly_mean"] = float(np.mean(collected_Ly))
        cte_data[run_key]["Lz_mean"] = float(np.mean(collected_Lz))

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
        else:
            if abs(current_value - previous_value) >= criterion:
                bools.append(False)
            else:
                bools.append(True)
    return all(bools)


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
    compute_hysteresis: bool = False,
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
    compute_hysteresis : bool, optional
        If True, the temperature list is extended and reverts the specified temperatures after the last
        temperature is reached. This allows to compute hysteresis effects in the CTE calculation
        (default False).
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
        On the lowest level, the structure is the same as returned by `_CTE_H_V_workflow_analysis`.
        Additionally, on the temperature level, the following entries are added:
        "is_converged" : bool            # Whether convergence was reached within the max_production_runs 
        "convergence_criterion" : float  # The convergence criterion 
        "structure_final" : Atoms        # Final structure at this temperature

    Notes
    -----
    - For every temperature, the structure is first pre-equilibrated with short (10 ps) NVT.
    - Simulation settings for the NVT equilibration run are hard-coded
    - Dump frequency of the NPT equilibration run is hard-coded and set to 5 dumps over the run.
    - CTEs are calculated sequentially if a list of temperatures is provided. Alternatively, multiple
      jobs with independent temperatures can be submitted to achieve parallelization.

    """

    # Minimum choices for a working CTE calculation -> For reliable results, user should increase 
    # these values via input parameters to cte_simulation 
    MIN_PRODUCTION_RUNS = 2
    AVERAGING_TIME_IN_PS = 2
    MIN_PRODUCTION_STEPS = int(2 * AVERAGING_TIME_IN_PS * 1000 / timestep)
    MIN_RUNNING_MEAN_POINTS = 1000

    if max_production_runs < MIN_PRODUCTION_RUNS:
        msg = "WARNING: At least 2 individual production runs are needed to check for CTE convergence."
        msg += f"\n  However, a value of {max_production_runs} was provided."
        msg += f"\n  Automatically setting max_production_runs to {MIN_PRODUCTION_RUNS} and continue."
        msg += "\n  Consider increasing max_production_runs even further if convergence is not reached."
        print(msg)
        max_production_runs = MIN_PRODUCTION_RUNS

    if production_steps < MIN_PRODUCTION_STEPS:
        msg = f"WARNING: For calculating fluctuations based on running averages, sufficient data is needed."
        msg += f"\n  With currently averaging over {AVERAGING_TIME_IN_PS} ps, production runs are too short"
        msg += f"  and we recommend at least {2*AVERAGING_TIME_IN_PS} ps."
        msg += f"\n  Automatically setting the user-specified production_steps of {production_steps} to {MIN_PRODUCTION_STEPS} and continue."
        msg += f"\n  Consider increasing production_steps even further to get more reliable results."
        print(msg)
        production_steps = MIN_PRODUCTION_STEPS
    
    N_for_averaging = int(AVERAGING_TIME_IN_PS * 1000 / n_log / timestep)
    if N_for_averaging < MIN_RUNNING_MEAN_POINTS:
        msg = f"WARNING: Running mean values are most likely based on insufficient data points."
        msg += f"\n  We recommend averaging over at least {MIN_RUNNING_MEAN_POINTS} data points, but currently only {N_for_averaging} are used."
        msg += "\n  Consider decreasing n_log or change hard-coded AVERAGING_TIME_IN_PS variable."
        msg += "\n  Continuing regardless."
        print(msg)

    # Prepare temperature list, also for hysteresis calculations if requested
    if isinstance(temperature, (int, float)):
        temperature_sim = [temperature]
    if compute_hysteresis:
        temperature_sim += temperature_sim[-2::-1]

    # Set pressure to anisotropic if requested
    sim_pressure = [pressure, pressure, pressure, None, None, None] if aniso else pressure

    # Collect all results here
    cte_results = {}

    # initial structure used. Afterwards, it is updated after each temperature
    structure0 = structure.copy()

    # Loop over all temperatures to perform the cte simulation protocol
    for T_count, T in enumerate(temperature, start=1):
        if T <= 0:
            msg = "Temperature must be positive for CTE simulations."
            raise ValueError(msg)
        T_key = f"{T_count:02d}_{int(T)}K"

        # Only initialize velocities in the very first simulation.
        # For all subsequent runs, use velocities from previous runs.
        if T_count == 1:
            initial_temperature = T
        else:
            initial_temperature = 0
            seed = None
             
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
            n_dump=int(equilibration_steps / 5),
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
            cte_results[T_key] = _CTE_H_V_workflow_analysis(
                subresults=_results, T=T, p_in_GPa=pressure, running_mean=True, N=N_for_averaging
            )

            # check for convergence or if max number of production runs is reached
            if counter_production_run > 1:
                if _is_converged(cte_results[T_key], CTE_convergence_criterion):
                    cte_results[T_key]["is_converged"] = True
                    cte_results[T_key]["convergence_criterion"] = CTE_convergence_criterion
                    cte_results[T_key]["structure_final"] = structure_production.copy()
                    structure0 = structure_production.copy()
                    break
                if counter_production_run >= max_production_runs:
                    cte_results[T_key]["is_converged"] = False
                    cte_results[T_key]["convergence_criterion"] = CTE_convergence_criterion
                    cte_results[T_key]["structure_final"] = structure_production.copy()
                    structure0 = structure_production.copy()
                    break

            # continue with this in next production run
            counter_production_run += 1
            structure2 = structure_production.copy()

    return cte_results
