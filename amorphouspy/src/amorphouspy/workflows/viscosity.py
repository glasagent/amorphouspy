"""Viscosity simulation workflows for glass systems using LAMMPS and the Green-Kubo method.

Implements molecular dynamics workflows and post-processing utilities for
viscosity calculations based on the stress autocorrelation function (SACF).

Author
------
Achraf Atila (achraf.atila@bam.de)
"""

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase.atoms import Atoms
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit

from amorphouspy.workflows.shared import _run_lammps_md

NPOINTS = 2


def _cutoff_time_ps(c: float | tuple[float, int]) -> float:
    return float(c[0]) if isinstance(c, tuple) else float(c)


def viscosity_simulation(
    structure: Atoms,
    potential: pd.DataFrame,
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

    Equilibrate a structure at a target temperature and perform a production MD run
    to collect the instantaneous off-diagonal stress tensor components required for
    viscosity computation.

    Args:
        structure: Input structure (assumed pre-equilibrated).
        potential: LAMMPS potential file.
        temperature_sim: Simulation temperature in Kelvin.
        timestep: MD timestep in fs.
        production_steps: Number of MD steps for the production run.
        n_print: Thermodynamic output frequency.
        server_kwargs: Additional server arguments.
        langevin: Whether to use Langevin dynamics.
        seed: Random seed for velocity initialization.
        tmp_working_directory: Temporary directory.

    Returns:
        A dictionary containing the parsed generic results from LAMMPS output.

    Raises:
        KeyError: If `"generic"` key is missing from the parsed output.

    Example:
        >>> result = viscosity_simulation(
        ...     structure=my_atoms,
        ...     potential=my_potential_df,
        ...     temperature_sim=2000.0,
        ...     production_steps=1000000
        ... )

    """
    if potential.empty:
        msg = "No matching potential found for the given configuration."
        raise ValueError(msg)
    potential_name = potential.loc[0, "Name"]

    if potential_name.lower() == "shik":
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

    # Stage 2: Production simulation for viscosity at T
    _structure_final, parsed_output = _run_lammps_md(
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

    return {"result": result}


def autocorrelation_fft(signal: ArrayLike, max_lag: int) -> NDArray[np.float64]:
    """Compute the autocorrelation function using FFT.

    Args:
        signal: Input signal array.
        max_lag: Maximum lag (number of time steps) to include.

    Returns:
        Autocorrelation function up to `max_lag`.

    Example:
        >>> signal = np.random.randn(100)
        >>> acf = autocorrelation_fft(signal, max_lag=50)

    """
    signal_arr = np.asarray(signal)
    N = len(signal_arr)
    f = np.fft.fft(signal_arr, n=2 * N)
    acf = np.fft.ifft(f * np.conj(f))[:N].real
    norm = np.arange(N, 0, -1)
    acf /= norm
    return acf[:max_lag]


def cumulative_trapezoid(acf: ArrayLike, dt: float) -> NDArray[np.float64]:
    """Compute cumulative trapezoidal integral of an array.

    Args:
        acf: Input array to integrate.
        dt: Sampling interval.

    Returns:
        Cumulative integral with the same shape as `acf`.

    Example:
        >>> data = np.array([0, 1, 2, 3])
        >>> integral = cumulative_trapezoid(data, dt=0.1)

    """
    acf_arr = np.asarray(acf)
    if len(acf_arr) < NPOINTS:
        return np.zeros_like(acf_arr)
    out = np.empty_like(acf_arr)
    out[0] = 0.0
    out[1:] = np.cumsum(0.5 * (acf_arr[1:] + acf_arr[:-1]) * dt)
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

    Args:
        sacf: SACF values.
        times: Corresponding time points (monotonically increasing).
        method: Cutoff detection method. Defaults to 'noise_threshold'.
        epsilon: Relative tolerance or threshold level. Defaults to 0.01.
        min_cutoff: Minimum allowed cutoff time.
        max_cutoff: Maximum allowed cutoff time.
        min_points: Minimum number of remaining points after applying max_cutoff.
        consecutive: Consecutive points below threshold required for stable cutoff. Defaults to 3.
        return_index: If True, also return index of the cutoff point.

    Returns:
        Detected cutoff time, and optionally the cutoff index if return_index is True.

    Raises:
        ValueError: If input validation fails or no valid cutoff is detected.

    Example:
        >>> t_cut = auto_cutoff(sacf_data, time_array, method='noise_threshold', epsilon=0.01)

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

    Args:
        T: Temperatures in Kelvin.
        log10_eta0: Log10 of pre-exponential viscosity factor.
        B: Activation parameter in Kelvin.
        T0: Vogel temperature (Kelvin).

    Returns:
        log10(viscosity) evaluated at T.

    """
    T_arr = np.asarray(T, dtype=np.float64)
    return log10_eta0 + (B / (T_arr - T0)) * np.log10(np.e)


def fit_vft(
    T_data: ArrayLike, log10_eta_data: ArrayLike, initial_guess: tuple[float, float, float] = (-3, 1000, 200)
) -> tuple[tuple[float, float, float], NDArray[np.float64]]:
    """Fits viscosity data to the Vogel-Fulcher-Tammann (VFT) model.

    Args:
        T_data: Temperatures in Kelvin.
        log10_eta_data: log10(viscosity) values.
        initial_guess: Initial guess for (log10_eta0, B, T0).

    Returns:
        A tuple containing the best-fit parameters and the covariance matrix.

    Example:
        >>> params, cov = fit_vft(temps, log_visc_data)
        >>> log10_eta0, B, T0 = params

    """
    popt, pcov = curve_fit(vft_model, T_data, log10_eta_data, p0=initial_guess, maxfev=1000000)
    return popt, pcov


def get_closest_divisor(target: int, y: int) -> int:
    """Return the divisor of `target` closest to a reference integer `y`.

    Args:
        target: Positive integer whose divisors are considered.
        y: Reference integer.

    Returns:
        Divisor of `target` closest to `y`.

    Raises:
        ValueError: If `target` is not a positive integer.

    Example:
        >>> d = get_closest_divisor(100, 33)
        >>> print(d)
        25

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
                if closest is None or diff < min_diff or (diff == min_diff and d < closest):
                    closest = d
                    min_diff = diff

    assert closest is not None
    return closest


def get_viscosity(
    result: dict[str, Any],
    timestep: float = 1.0,  # fs (MD integration timestep)
    output_frequency: int = 1,  # number of MD steps between stored frames
    max_lag: int | None = 1_000_000,
) -> dict[str, float]:
    """Compute viscosity using the Green-Kubo formalism from MD stress data.

    Args:
        result: Parsed output dictionary from `viscosity_simulation`.
        timestep: MD integration time step in femtoseconds.
        output_frequency: Number of MD steps between stored frames in the output.
        max_lag: Maximum correlation lag (number of steps). Defaults to 1,000,000.

    Returns:
        A dictionary containing:
            - temperature: Mean simulation temperature (K)
            - viscosity: Computed viscosity (Pa·s)
            - max_lag: Cutoff time per stress component (ps)
            - lag_time_ps: Array of lag times in picoseconds
            - sacf: Averaged normalized stress autocorrelation function
            - viscosity_integral: Cumulative viscosity integral (Pa·s) vs lag time

    Example:
        >>> result_dict = get_viscosity(simulation_output, timestep=1.0, output_frequency=1, max_lag=1000000)
        >>> print(result_dict['viscosity'])
        >>> print(result_dict['sacf'])
        >>> print(result_dict['viscosity_integral'])

    """
    kB = 1.380649e-23  # m^2 kg s^-2 K^-1
    A2m = 1e-10  # Å → m

    # --- Extract data ---
    pressures = result["result"]["pressures"]
    pxy = pressures[:, 0, 1] * 1e9  # Pa
    pxz = pressures[:, 0, 2] * 1e9  # Pa
    pyz = pressures[:, 1, 2] * 1e9  # Pa

    volume = np.mean(result["result"]["volume"]) * A2m**3  # m^3
    temperature = np.mean(result["result"]["temperature"])  # K

    # --- Correct effective timestep ---
    dt_s = timestep * output_frequency * 1e-15  # seconds

    scale = volume / (kB * temperature)

    if max_lag is None:
        max_lag = len(pxy)

    # --- First ACF pass ---
    acfxy = autocorrelation_fft(pxy, max_lag)
    acfxz = autocorrelation_fft(pxz, max_lag)
    acfyz = autocorrelation_fft(pyz, max_lag)

    # enforce consistent length
    n = min(len(acfxy), len(acfxz), len(acfyz))
    acfxy = acfxy[:n]
    acfxz = acfxz[:n]
    acfyz = acfyz[:n]

    lag_time_s = np.arange(n) * dt_s
    lag_time_ps = lag_time_s * 1e12

    # --- Auto cutoff detection ---
    cut_times_ps = [
        auto_cutoff(acfxy / acfxy[0], lag_time_ps, method="noise_threshold", epsilon=1e-4),
        auto_cutoff(acfxz / acfxz[0], lag_time_ps, method="noise_threshold", epsilon=1e-4),
        auto_cutoff(acfyz / acfyz[0], lag_time_ps, method="noise_threshold", epsilon=1e-4),
    ]

    # --- Convert cutoff times to lag indices ---
    total_time_ps = len(pxy) * timestep * output_frequency / 1000

    max_lag_1 = [
        get_closest_divisor(int(total_time_ps), int(_cutoff_time_ps(cut_times_ps[0]))) * 2,
        get_closest_divisor(int(total_time_ps), int(_cutoff_time_ps(cut_times_ps[1]))) * 2,
        get_closest_divisor(int(total_time_ps), int(_cutoff_time_ps(cut_times_ps[2]))) * 2,
    ]

    # convert ps → steps
    max_lag = int(np.max(max_lag_1) / (timestep * output_frequency / 1000))

    # --- Second ACF pass (final integration window) ---
    acfxy = autocorrelation_fft(pxy, max_lag)
    acfxz = autocorrelation_fft(pxz, max_lag)
    acfyz = autocorrelation_fft(pyz, max_lag)

    n = min(len(acfxy), len(acfxz), len(acfyz))
    acfxy = acfxy[:n]
    acfxz = acfxz[:n]
    acfyz = acfyz[:n]

    lag_time_s = np.arange(n) * dt_s
    lag_time_ps = lag_time_s * 1e12

    # --- Green-Kubo integration ---
    eta_xy = scale * cumulative_trapezoid(acfxy, dt=dt_s)
    eta_xz = scale * cumulative_trapezoid(acfxz, dt=dt_s)
    eta_yz = scale * cumulative_trapezoid(acfyz, dt=dt_s)

    eta_avg = (eta_xy + eta_xz + eta_yz) / 3
    viscosity = eta_avg[-1]

    # --- Normalized SACF ---
    sacf_xy = acfxy / acfxy[0]
    sacf_xz = acfxz / acfxz[0]
    sacf_yz = acfyz / acfyz[0]
    sacf_avg = ((sacf_xy + sacf_xz + sacf_yz) / 3).tolist()

    return {
        "temperature": temperature,
        "viscosity": viscosity,
        "max_lag": int(np.max(max_lag_1)),
        "lag_time_ps": lag_time_ps.tolist(),
        "sacf": sacf_avg,
        "viscosity_integral": eta_avg.tolist(),
    }
