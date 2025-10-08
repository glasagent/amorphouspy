"""Viscosity simulation workflows for glass systems using LAMMPS and the Green-Kubo method.

Implements molecular dynamics workflows and post-processing utilities for
viscosity calculations based on the stress autocorrelation function (SACF).

Author
------
Achraf Atila (achraf.atila@bam.de)
"""

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from ase.atoms import Atoms
from numpy.typing import ArrayLike, NDArray
from pyiron_atomistics.lammps.lammps import lammps_function
from pyiron_base import job
from scipy.optimize import curve_fit

from pyiron_glass.io_utils import structure_from_parsed_output

NPOINTS = 2


def _run_lammps_md(
    structure: Atoms,
    potential: str,
    temperature: float | list[float],
    n_ionic_steps: int,
    timestep: float,
    n_print: int,
    initial_temperature: float,
    pressure: float | None = None,
    server_kwargs: dict[str, Any] | None = None,
    *,
    langevin: bool = False,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
) -> tuple[Atoms, dict[str, Any]]:  # pylint: disable=too-many-positional-arguments
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

        # defines the temperature protocol
        temp_setting = temperature

        # Sets up the LAMMPS simulations
        _shell_output, parsed_output, _job_crashed = lammps_function(
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
                "dump_modify": f"1 every {n_ionic_steps} first yes",
                "thermo_style": "custom step temp density pe etotal pxx pxy pxz pyy pyz pzz vol",
                "thermo_modify": "flush yes",
            },
        )

        # Retrives the final structure from the parsed output
        new_structure = structure_from_parsed_output(initial_structure=structure, parsed_output=parsed_output)

    return new_structure, parsed_output


@job
def viscosity_simulation(
    structure: Atoms,
    potential: str,
    temperature_sim: float = 5000.0,
    timestep: float = 1.0,
    production_steps: int = 10_000_000,
    n_print: int = 1,
    server_kwargs: dict[str, Any] | None = None,
    *,
    langevin: bool = False,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
) -> dict[str, Any]:  # pylint: disable=too-many-positional-arguments
    """Perform a LAMMPS-based viscosity simulation via the Green-Kubo formalism.

    This workflow equilibrates a structure at a target temperature and performs a
    production MD run to collect the instantaneous off-diagonal stress tensor
    components required for viscosity computation.

    The number of steps used here is only for testing purposes.
    It is assumed in this workflow that the given in structure is pre-quilibrated.

    Parameters
    ----------
    structure : Atoms
        Input structure (assumed pre-equilibrated).
    potential : str
        LAMMPS potential file.
    temperature_sim : float, optional
        Simulation temperature in Kelvin (default 5000.0 K).
    timestep : float, optional
        MD integration timestep in femtoseconds (default 1.0 fs).
    production_steps : int, optional
        Number of MD steps for the production run (default 10,000,000).
    n_print : int, optional
        Thermodynamic output frequency (default 1).
    server_kwargs : dict, optional
        Additional server configuration arguments for pyiron.
    langevin : bool, optional
        Whether to use Langevin dynamics (default False).
    seed : int, optional
        Random seed for velocity initialization (default 12345).
    tmp_working_directory : str or Path, optional
        Temporary directory for job execution.

    Returns
    -------
    dict
        Dictionary containing the parsed `result` section from LAMMPS output.

    Raises
    ------
    KeyError
        If `"generic"` key is missing from the parsed output.

    Notes
    -----
    - The structure is first equilibrated with short NVT and Langevin stages.
    - The final production run provides stress tensors for Green-Kubo analysis.

    """
    # Stage 0: Langevin dynamics at T
    structure0, _ = _run_lammps_md(
        structure=structure,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=temperature_sim,
        n_ionic_steps=10_000,
        timestep=timestep,
        n_print=1000,
        initial_temperature=temperature_sim,
        langevin=True,
        seed=seed,
        server_kwargs=server_kwargs,
    )

    # Stage 1: Equilibration in NVT at T
    structure1, _ = _run_lammps_md(
        structure=structure0,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=temperature_sim,
        n_ionic_steps=10_000,
        timestep=timestep,
        n_print=1000,
        initial_temperature=temperature_sim,
        langevin=langevin,
        seed=seed,
        server_kwargs=server_kwargs,
    )

    # Stage 5: Production simulation for viscosity at T
    structure_final, parsed_output = _run_lammps_md(
        structure=structure1,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=temperature_sim,
        n_ionic_steps=production_steps,
        timestep=timestep,
        n_print=n_print,
        initial_temperature=0,
        langevin=langevin,
        server_kwargs=server_kwargs,
    )

    result = parsed_output.get("generic", None)

    if result is None:
        err_msg = "The 'generic' key is missing from parsed_output."
        raise KeyError(err_msg)
        result = {}

    return {"result": result}


def autocorrelation_fft(signal: ArrayLike, max_lag: int) -> NDArray[np.float64]:
    """Compute the autocorrelation function using FFT.

    Parameters
    ----------
    signal : array_like
        Input signal array.
    max_lag : int
        Maximum lag (number of time steps) to include.

    Returns
    -------
    acf : ndarray
        Autocorrelation function up to `max_lag`.

    Notes
    -----
    - Computation is O(N log N) using FFT convolution.
    - Output is normalized by the number of overlapping samples.

    """
    N = len(signal)
    f = np.fft.fft(signal, n=2 * N)
    acf = np.fft.ifft(f * np.conj(f))[:N].real
    norm = np.arange(N, 0, -1)
    acf /= norm
    return acf[:max_lag]


def cumulative_trapezoid(acf: ArrayLike, dt: float) -> NDArray[np.float64]:
    """Compute cumulative trapezoidal integral of an array.

    Parameters
    ----------
    acf : array_like
        Input array to integrate.
    dt : float
        Sampling interval.

    Returns
    -------
    out : ndarray
        Cumulative integral with the same shape as `acf`.

    """
    if len(acf) < NPOINTS:
        return np.zeros_like(acf)
    out = np.empty_like(acf)
    out[0] = 0.0
    out[1:] = np.cumsum(0.5 * (acf[1:] + acf[:-1]) * dt)
    return out


def auto_cutoff(  # noqa: C901, PLR0912, PLR0915
    sacf: ArrayLike,
    times: ArrayLike,
    method: str = "noise_threshold",
    epsilon: float = 0.01,
    min_cutoff: float | None = None,
    max_cutoff: float | None = None,
    min_points: int = 10,
    consecutive: int = 3,
    *,
    return_index: bool = False,
) -> float | tuple[float, int]:
    """Determine an optimal cutoff time for the symmetrized autocorrelation function (SACF).

    Parameters
    ----------
    sacf : array_like
        SACF values.
    times : array_like
        Corresponding time points (monotonically increasing).
    method : {'noise_threshold', 'cumulative_integral'}, optional
        Cutoff detection method (default 'noise_threshold').
    epsilon : float, optional
        Relative tolerance or threshold level (default 0.01).
    min_cutoff : float, optional
        Minimum allowed cutoff time.
    max_cutoff : float, optional
        Maximum allowed cutoff time.
    min_points : int, optional
        Minimum number of remaining points after applying max_cutoff.
    consecutive : int, optional
        Consecutive points below threshold required for stable cutoff (default 3).
    return_index : bool, optional
        If True, also return index of the cutoff point (default False).

    Returns
    -------
    t_cutoff : float
        Detected cutoff time.
    cutoff_idx : int, optional
        Index of cutoff (if `return_index=True`).

    Raises
    ------
    ValueError
        If input validation fails or no valid cutoff is detected.

    Notes
    -----
    - Supports both noise-based and cumulative integral saturation methods.
    - Falls back to last time point if no cutoff is detected.

    """
    sacf = np.asarray(sacf)
    times = np.asarray(times)

    if sacf.shape != times.shape:
        err_msg = "sacf and times must have the same shape"
        raise ValueError(err_msg)
    if len(sacf) < NPOINTS:
        err_msg = "Need at least 2 data points"
        raise ValueError(err_msg)
    if not np.all(np.diff(times) > 0):
        err_msg = "Times must be strictly increasing"
        raise ValueError(err_msg)

    # Apply max_cutoff if given
    if max_cutoff is not None:
        mask = times <= max_cutoff
        if np.sum(mask) < min_points:
            err_msg = f"Fewer than {min_points} time points within max_cutoff"
            raise ValueError(err_msg)
        sacf = sacf[mask]
        times = times[mask]

    # --- Noise threshold method ---
    if method == "noise_threshold":
        max_val = np.max(np.abs(sacf))
        if np.isclose(max_val, 0.0):
            return (times[-1], len(times) - 1) if return_index else times[-1]

        threshold = epsilon * max_val
        below = np.where(np.abs(sacf) < threshold)[0]

        if len(below) > 0:
            cutoff_idx = None
            for i in range(len(below) - (consecutive - 1)):
                if below[i + consecutive - 1] - below[i] == consecutive - 1:
                    cutoff_idx = below[i]
                    break
            if cutoff_idx is None:
                cutoff_idx = below[0]
        else:
            cutoff_idx = len(times) - 1

        t_cutoff = times[cutoff_idx]

    # --- Cumulative integral method ---
    elif method == "cumulative_integral":
        dt = np.diff(times)
        cum_integral = np.cumsum(0.5 * (sacf[1:] + sacf[:-1]) * dt)

        if len(cum_integral) == 0 or np.isclose(cum_integral[-1], 0.0):
            return (times[-1], len(times) - 1) if return_index else times[-1]

        normalized = cum_integral / cum_integral[-1]
        target = 1.0 - epsilon
        above_target = np.where(normalized >= target)[0]

        if len(above_target) > 0:
            cutoff_idx = above_target[0] + 1
        else:
            dnorm = np.diff(normalized)
            below = np.where(np.abs(dnorm) < epsilon)[0]
            cutoff_idx = below[0] + 2 if len(below) else len(times) - 1

        t_cutoff = times[cutoff_idx]

    else:
        err_msg = f"Unknown method '{method}'. Use 'noise_threshold' or 'cumulative_integral'"
        raise ValueError(err_msg)

    # Apply min/max cutoff constraints
    if min_cutoff is not None:
        t_cutoff = max(t_cutoff, min_cutoff)
    if max_cutoff is not None:
        t_cutoff = min(t_cutoff, max_cutoff)

    return (t_cutoff, cutoff_idx) if return_index else t_cutoff


def vft_model(T: ArrayLike, log10_eta0: float, B: float, T0: float) -> NDArray[np.float64]:
    """Vogel-Fulcher-Tammann (VFT) model for viscosity.

    Parameters
    ----------
    T : array_like
        Temperatures in Kelvin.
    log10_eta0 : float
        Log10 of pre-exponential viscosity factor.
    B : float
        Activation parameter in Kelvin.
    T0 : float
        Vogel temperature (Kelvin).

    Returns
    -------
    log10_eta_vft : ndarray
        log10(viscosity) evaluated at T.

    """
    return log10_eta0 + (B / (T - T0)) * np.log10(np.e)


def fit_vft(
    T_data: ArrayLike, log10_eta_data: ArrayLike, initial_guess: tuple[float, float, float] = (-3, 1000, 200)
) -> tuple[tuple[float, float, float], NDArray[np.float64]]:
    """Fit viscosity data to the Vogel-Fulcher-Tammann (VFT) model.

    Parameters
    ----------
    T_data : array_like
        Temperatures in Kelvin.
    log10_eta_data : array_like
        log10(viscosity) values.
    initial_guess : tuple, optional
        Initial guess for (log10_eta0, B, T0). Default is (-3, 1000, 200).

    Returns
    -------
    popt : tuple
        Best-fit parameters (log10_eta0, B, T0).
    pcov : ndarray
        Covariance matrix of the fit.

    """
    popt, pcov = curve_fit(vft_model, T_data, log10_eta_data, p0=initial_guess, maxfev=10000)
    return popt, pcov


def get_closest_divisor(target: int, y: int) -> int:
    """Return the divisor of `target` closest to a reference integer `y`.

    Parameters
    ----------
    target : int
        Positive integer whose divisors are considered.
    y : int
        Reference integer.

    Returns
    -------
    int
        Divisor of `target` closest to `y`.

    Raises
    ------
    ValueError
        If `target` is not a positive integer.

    """
    if target <= 0:
        err_msg = "Target must be a positive integer."
        raise ValueError(err_msg)

    closest = None
    min_diff = float("inf")

    # Iterate up to sqrt(target) and generate divisors in pairs
    for i in range(1, math.isqrt(target) + 1):
        if target % i == 0:
            for d in (i, target // i):
                diff = abs(d - y)
                if diff < min_diff or (diff == min_diff and d < closest):
                    closest = d
                    min_diff = diff

    return closest


def get_viscosity(
    result: dict[str, Any],
    timestep: float = 1.0,
    max_lag: int | None = 1_000_000,
) -> dict[str, float]:
    """Compute viscosity using the Green-Kubo formalism from MD stress data.

    Parameters
    ----------
    result : dict
        Parsed output dictionary from `viscosity_simulation`, containing stress tensor, volume, and temperature.
    timestep : float, optional
        MD integration time step in femtoseconds (default 1.0 fs).
    max_lag : int, optional
        Maximum correlation lag (number of steps). Default 1,000,000.

    Returns
    -------
    dict
        Dictionary with:
        - 'temperature' : float — average temperature (K),
        - 'viscosity' : float — viscosity (Pa·s),
        - 'max_lag' : float — effective cutoff time (ps).

    Notes
    -----
    - Uses the off-diagonal stress components (pxy, pxz, pyz).
    - Autocorrelation functions are computed using FFT.
    - Integration of SACF is performed using trapezoidal rule.
    - Automatically estimates a cutoff using `auto_cutoff`.
    - Assumes input units from LAMMPS 'metal' style except for timestep to be given in fs.

    """
    kB = 1.380649e-23  # m2 kg s-2 K-1
    A2m = 1e-10  # Angstroms to meter

    pxy = result["result"]["pressures"][:, 0, 1] * 1e9  # in Pa
    pxz = result["result"]["pressures"][:, 0, 2] * 1e9  # in Pa
    pyz = result["result"]["pressures"][:, 1, 2] * 1e9  # in Pa
    volume = np.mean(result["result"]["volume"]) * A2m**3  # volume in m3
    temperature = np.mean(result["result"]["temperature"])  # in K
    dt_s = timestep * 1e-15  # in seconds

    scale = volume / (kB * temperature)

    if max_lag is None:
        max_lag = len(pxy)

    acfxy = autocorrelation_fft(pxy, max_lag)
    acfxz = autocorrelation_fft(pxz, max_lag)
    acfyz = autocorrelation_fft(pyz, max_lag)

    lag_time_s = np.arange(max_lag) * dt_s
    lag_time_ps = lag_time_s * 1e12

    max_lag_1 = auto_cutoff(acfxy / acfxy[0], lag_time_ps, method="noise_threshold", epsilon=0.0001)

    max_lag_1 = get_closest_divisor(int(len(pxy) * timestep / 1000), int(max_lag_1)) * 2

    max_lag = int(max_lag_1 / (timestep / 1000))

    acfxy = autocorrelation_fft(pxy, max_lag)
    acfxz = autocorrelation_fft(pxz, max_lag)
    acfyz = autocorrelation_fft(pyz, max_lag)

    eta_xy_running = scale * cumulative_trapezoid(acfxy, dt_s)
    eta_xz_running = scale * cumulative_trapezoid(acfxz, dt_s)
    eta_yz_running = scale * cumulative_trapezoid(acfyz, dt_s)
    eta_avg = (eta_xy_running + eta_xz_running + eta_yz_running) / 3

    viscosity = eta_avg[-1]

    return {"temperature": temperature, "viscosity": viscosity, "max_lag": max_lag_1}
