"""CTE simulation workflows for glass systems using LAMMPS.

Implements molecular dynamics workflows and post-processing utilities for
CTE calculations based on H-V fluctuations with NPT simulations.

Author: Marcel Sadowski (github.com/Gitdowski)
"""

import tempfile
from pathlib import Path
from typing import Any

from ase.atoms import Atoms
from lammpsparser.compatibility.file import lammps_file_interface_function

from amorphouspy.io_utils import structure_from_parsed_output
from amorphouspy.workflows.cte_helpers import (
    _collect_sim_data,
    _create_logger,
    _fluctuation_simulation_cte_calculation,
    _fluctuation_simulation_input_checker,
    _fluctuation_simulation_merge_results,
    _fluctuation_simulation_uncertainty_check,
    _initialize_datadict,
    _sanity_check_sim_data,
    _temperature_scan_input_checker,
    _temperature_scan_merge_results,
)
from amorphouspy.workflows.shared import get_lammps_command


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

    Args:
        structure: Input structure (assumed pre-equilibrated).
        potential: LAMMPS potential file.
        temperature: Simulation temperature in Kelvin (default 300 K).
        pressure: Target pressure in GPa (use pyiron units here!) for NPT simulations
            (default: 10-4 GPa = 10^5 Pa = 1 bar).
        timestep: MD integration timestep in femtoseconds (default 1.0 fs).
        equilibration_steps: Number of MD steps for the equilibration run (default 100,000).
        production_steps: Number of MD steps for the production runs (default 200,000).
        min_production_runs: Minimum number of production runs to perform before checking for
            convergence (default 2).
        max_production_runs: Maximum number of production runs to perform before checking for
            convergence (default 10).
        CTE_uncertainty_criterion: Convergence criterion for the uncertainty of the linear
            CTE (default 1e-6/K).
        n_dump: Dump output frequency of the production runs (default 100,000).
        n_log: Log output frequency (default 10).
        server_kwargs: Additional server configuration arguments for pyiron.
        aniso: If false, an isotropic NPT calculation is performed and the simulation box is
              scaled uniformly. If True, anisotropic NPT calculation is performed and the simulation
              box can change shape and size independently along each axis (default False).
        seed: Random seed for velocity initialization (default 12345). If None, a random seed is used.
        tmp_working_directory: Temporary directory for job execution.

    Returns:
        Nested dictionary containing the "summary" and "data" keys. In the "summary" section the CTE
        values and their uncertainties are returned together with info whether CTE_uncertainty_criterion was
        reached within the max_production_runs.
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

    Notes:
        - The structure is first pre-equilibrated with a short, hard-coded 10 ps NVT run. Only then
          follows the user-defined NPT equilibration and production runs.
        - The simulation is only marked as converged if the uncertainty criterion is reached for all four CTE
          values (CTE_V, CTE_x, CTE_y, CTE_z) at the same time. The CTE_uncertainty_criterion is applied to
          as-is to the linear CTEs. How this is treated for the CTE_V, see next point.
        - How to chose the uncertainty criterion for the volumetric CTE? Should it be the same as the defined
          CTE_uncertainty_criterion applied to the linear CTEs? Consider this: The volumetric CTE is approximately
          the sum of the linear CTEs along x, y, and z. If three variables x, y and z were uncorrelated and
          have known uncertainties sigma_x, sigma_y, and sigma_z, the uncertainty of the sum of them would
          be: sigma_V = sqrt( sigma_x**2 + sigma_y**2 + sigma_z**2 ). If we assume that the uncertainty
          criterion is reached at roughly the same time for all three variables, the uncertainty of the
          volumetric CTE can be approximated as sqrt(3)*CTE_uncertainty_criterion. However, x, y and z are
          typically not uncorrelated. If calculated from the actual simulation data, the uncertainty of CTE_V
          is found to be approximately the same as the individual uncertainties of the linear CTEs. To be not
          too strict here, we keep the sqrt(3)*CTE_uncertainty_criterion for CTE_V.


    Example:
        >>> result = cte_from_fluctuations_simulation(
        ...     structure=my_atoms,
        ...     potential=my_potential,
        ...     temperature=300,
        ...     equilibration_steps=500_000,
        ...     production_steps=200_000,
        ...     min_production_runs=10,
        ...     max_production_runs=50,
        ...     CTE_uncertainty_criterion=1e-6,
        ... )

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
    aniso: bool = False,
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
