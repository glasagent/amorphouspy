"""Viscosity simulation workflows for glass systems using LAMMPS and the Green-Kubo method.

Implements molecular dynamics workflows and post-processing utilities for
viscosity calculations based on the stress autocorrelation function (SACF).

Author
------
Achraf Atila (achraf.atila@bam.de)
"""

import json
import math
import warnings
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
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


def _viscosity_simulation(
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
        n_ionic_steps=100_000,
        timestep=timestep,
        pressure=0.0,
        n_print=1000,
        initial_temperature=temperature_sim,
        langevin=langevin,
        seed=seed,
        server_kwargs=server_kwargs,
    )

    # Stage 2: Production simulation for viscosity at T
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

    return {"result": result, "structure": structure_final}


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
    c: float = 5.0,
    *,
    return_index: bool = False,
) -> float | tuple[float, int]:
    """Determine an optimal cutoff time for the symmetrized autocorrelation function (SACF).

    Args:
        sacf: SACF values (should be normalised so that sacf[0] = 1).
        times: Corresponding time points (monotonically increasing).
        method: Cutoff detection method. One of ``'noise_threshold'``,
            ``'cumulative_integral'``, or ``'self_consistent'``.
            Defaults to ``'noise_threshold'``.
        epsilon: Relative tolerance or threshold level (used by
            ``'noise_threshold'`` and ``'cumulative_integral'``). Defaults to 0.01.
        min_cutoff: Minimum allowed cutoff time.
        max_cutoff: Maximum allowed cutoff time.
        min_points: Minimum number of remaining points after applying max_cutoff.
        consecutive: Consecutive points below threshold required for stable cutoff
            (``'noise_threshold'`` only). Defaults to 3.
        c: Self-consistency factor for ``'self_consistent'`` method. The cutoff T
            is accepted when ``tau_acf(T) < T / c``. Larger values are more
            conservative. Defaults to 5.0.
        return_index: If True, also return index of the cutoff point.

    Returns:
        Detected cutoff time, and optionally the cutoff index if return_index is True.

    Raises:
        ValueError: If input validation fails or no valid cutoff is detected.

    Example:
        >>> t_cut = auto_cutoff(sacf_data, time_array, method='self_consistent', c=5.0)

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

    # --- Self-consistent integration method (Chodera) ---
    elif method == "self_consistent":
        cutoff_idx = len(times) - 1  # default: use full data if no crossing found
        for i in range(1, len(times)):
            tau = float(np.trapezoid(sacf[:i], times[:i]))
            T = float(times[i])
            if tau > 0 and tau < T / c:
                cutoff_idx = i
                break
        t_cutoff = times[cutoff_idx]

    else:
        err_msg = f"Unknown method '{method}'. Use 'noise_threshold', 'cumulative_integral', or 'self_consistent'"
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


def get_viscosity(  # noqa: PLR0915
    result: dict[str, Any],
    timestep: float = 1.0,  # fs (MD integration timestep)
    output_frequency: int = 1,  # number of MD steps between stored frames
    max_lag: int | None = 1_000_000,
    cutoff_method: str = "noise_threshold",
) -> dict[str, float]:
    """Compute viscosity using the Green-Kubo formalism from MD stress data.

    Args:
        result: Parsed output dictionary from `viscosity_simulation`.
        timestep: MD integration time step in femtoseconds.
        output_frequency: Number of MD steps between stored frames in the output.
        max_lag: Maximum correlation lag (number of steps). Defaults to 1,000,000.
        cutoff_method: Method passed to ``auto_cutoff`` for τ_acf detection. One of
            ``'noise_threshold'``, ``'cumulative_integral'``, or ``'self_consistent'``.
            Defaults to ``'noise_threshold'``.

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
    pxy = pressures[::output_frequency, 0, 1] * 1e9  # Pa
    pxz = pressures[::output_frequency, 0, 2] * 1e9  # Pa
    pyz = pressures[::output_frequency, 1, 2] * 1e9  # Pa

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

    # --- Auto cutoff detection (with per-component fallback) ---
    cut_times_ps: list[float | tuple[float, int]] = []
    _detection_succeeded: list[bool] = []
    for _acf_norm, _label in [
        (acfxy / acfxy[0], "pxy"),
        (acfxz / acfxz[0], "pxz"),
        (acfyz / acfyz[0], "pyz"),
    ]:
        try:
            _c = auto_cutoff(_acf_norm, lag_time_ps, method=cutoff_method, epsilon=1e-4, min_cutoff=10)
            cut_times_ps.append(_c)
            _detection_succeeded.append(True)
        except ValueError:
            cut_times_ps.append(lag_time_ps[-1])  # fall back to last lag time
            _detection_succeeded.append(False)

    if not any(_detection_succeeded):
        msg = "tau_acf detection failed for all stress components"
        raise ValueError(msg)

    _successful_cut_times = [
        _cutoff_time_ps(c) for c, ok in zip(cut_times_ps, _detection_succeeded, strict=False) if ok
    ]
    tau_acf_ps = float(np.max(_successful_cut_times))

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
        "tau_acf_ps": tau_acf_ps,
    }


def _extract_md_data(result: dict[str, Any]) -> dict[str, Any]:
    """Normalise result dicts from both ``_viscosity_simulation`` and ``viscosity_simulation``.

    ``_viscosity_simulation`` returns ``{"result": {...}, "structure": ...}``.
    ``viscosity_simulation`` returns ``{"result": acc, "viscosity_data": ..., ...}``
    where ``acc`` already contains ``pressures``, ``volume``, ``temperature`` directly.
    Both are handled by checking for the ``pressures`` key one level down.
    """
    inner = result.get("result", result)
    if "pressures" not in inner:
        msg = (
            "Could not find 'pressures' in result dict. "
            "Pass the output of viscosity_simulation or _viscosity_simulation directly."
        )
        raise KeyError(msg)
    return inner


def helfand_viscosity(
    result: dict[str, Any],
    timestep: float = 1.0,
    output_frequency: int = 1,
    max_lag: int | None = None,
) -> dict[str, Any]:
    """Compute viscosity via the Helfand moment method.

    Integrates the stress tensor directly to form Helfand moments χ(t), then
    extracts viscosity from the slope of the mean-square displacement of χ.
    This avoids SACF cutoff detection entirely and is robust when τ_acf exceeds
    the trajectory length.

    Args:
        result: Parsed output dictionary from ``_viscosity_simulation``.
        timestep: MD integration timestep in femtoseconds.
        output_frequency: Number of MD steps between stored frames.
        max_lag: Maximum lag (frames) for MSD calculation. Defaults to N // 2.

    Returns:
        A dictionary containing:
            - viscosity: Computed viscosity (Pa·s)
            - viscosity_std: 1-sigma uncertainty from linear fit residuals (Pa·s)
            - temperature: Mean simulation temperature (K)
            - method: ``"helfand"``
            - msd: MSD of averaged Helfand moments (m²) vs lag
            - lag_time_ps: Lag times in picoseconds
            - slope: Linear fit slope (m²/s)

    Example:
        >>> out = helfand_viscosity(simulation_result, timestep=1.0)
        >>> print(out["viscosity"])

    """
    kB = 1.380649e-23
    A2m = 1e-10

    md = _extract_md_data(result)
    pressures = md["pressures"]
    pxy = pressures[::output_frequency, 0, 1] * 1e9  # Pa
    pxz = pressures[::output_frequency, 0, 2] * 1e9
    pyz = pressures[::output_frequency, 1, 2] * 1e9
    pxx = pressures[::output_frequency, 0, 0] * 1e9  # Pa — diagonal components for bulk viscosity
    pyy = pressures[::output_frequency, 1, 1] * 1e9
    pzz = pressures[::output_frequency, 2, 2] * 1e9

    volume = float(np.mean(np.asarray(md["volume"]))) * A2m**3
    temperature = float(np.mean(np.asarray(md["temperature"])))
    dt_s = timestep * output_frequency * 1e-15

    n = len(pxy)
    if max_lag is None:
        max_lag = n // 2

    # --- Helfand moments: cumulative stress integral ---
    chi_xy = np.cumsum(pxy) * dt_s
    chi_xz = np.cumsum(pxz) * dt_s
    chi_yz = np.cumsum(pyz) * dt_s

    # --- MSD via FFT: MSD[lag] ≈ 2*(C[0] - C[lag]) where C is unnorm. ACF of chi ---
    # This avoids an O(n^2) Python loop; valid for large n (stationary approximation).
    def _msd_fft(chi: NDArray[np.float64], max_lag: int) -> NDArray[np.float64]:
        acf = autocorrelation_fft(chi, max_lag + 1)
        return 2.0 * (acf[0] - acf[1 : max_lag + 1])

    msd_xy = _msd_fft(chi_xy, max_lag)
    msd_xz = _msd_fft(chi_xz, max_lag)
    msd_yz = _msd_fft(chi_yz, max_lag)
    msd_avg = (msd_xy + msd_xz + msd_yz) / 3.0

    lag_times_s = np.arange(1, max_lag + 1) * dt_s
    lag_time_ps = lag_times_s * 1e12

    # --- Linear fit over the diffusive regime ---
    # The MSD of chi(t) is sub-diffusive (MSD ~ t^2) at t << tau_stress and
    # linear (MSD ~ t) at t >> tau_stress.  We must start the fit well into
    # the diffusive regime.
    #
    # We estimate tau_stress from the PRESSURE ACF (not chi's ACF — chi is a
    # running integral whose ACF decays far more slowly than P itself).
    # A safe fit start is 5 * tau_stress, capped at 50% of max_lag so we
    # always retain at least half the window.
    probe = min(max_lag, 500_000)
    acf_p = autocorrelation_fft(pxy, probe)
    tau_idx = max(1, int(np.searchsorted(-(acf_p / acf_p[0]), -0.5)))
    start = min(int(5 * tau_idx), max(1, int(0.50 * max_lag)))
    slope, _ = np.polyfit(lag_times_s[start:], msd_avg[start:], 1)

    viscosity = slope * volume / (2.0 * kB * temperature)

    fit_vals = slope * lag_times_s[start:] + _
    residual_std = float(np.std(msd_avg[start:] - fit_vals))
    viscosity_std = residual_std * volume / (2.0 * kB * temperature)

    # --- Infinite-frequency shear modulus G_inf ---
    # G_inf = V / (kB T) * <P_ab^2>  (zero-lag SACF = variance for zero-mean signal)
    # Averaged over the three independent off-diagonal components.
    g_inf = volume / (kB * temperature) * float((np.var(pxy) + np.var(pxz) + np.var(pyz)) / 3.0)
    maxwell_relaxation_time_ps = float(viscosity / g_inf) * 1e12 if g_inf > 0.0 else float("nan")

    # --- Bulk viscosity ---
    # eta_bulk = V / (9 kB T) * integral_0^inf <dP_tr(0) dP_tr(t)> dt
    # where dP_tr = Pxx + Pyy + Pzz - 3*<P> is the isotropic pressure deviation.
    mean_pressure_pa = float(np.mean(pxx + pyy + pzz) / 3.0)
    p_trace_dev = (pxx + pyy + pzz) - 3.0 * mean_pressure_pa
    acf_bulk = autocorrelation_fft(p_trace_dev, max_lag)
    bulk_viscosity = volume / (9.0 * kB * temperature) * float(np.trapezoid(acf_bulk, dx=dt_s))

    return {
        "viscosity": float(viscosity),
        "viscosity_std": float(viscosity_std),
        "temperature": float(temperature),
        "method": "helfand",
        "msd": msd_avg.tolist(),
        "lag_time_ps": lag_time_ps.tolist(),
        "slope": float(slope),
        "shear_modulus_inf": float(g_inf),
        "maxwell_relaxation_time_ps": maxwell_relaxation_time_ps,
        "bulk_viscosity": float(bulk_viscosity),
        "mean_pressure_gpa": mean_pressure_pa / 1e9,
    }


def _viscosity_plateaued(
    viscosity_integral: NDArray[np.float64],
    tail_fraction: float = 0.2,
    rel_slope_tol: float = 0.05,
) -> bool:
    """Return True if the tail of the viscosity integral is flat.

    Fits a linear trend to the last ``tail_fraction`` of the integral and checks
    whether the absolute slope, normalised by the mean value in that window, is
    below ``rel_slope_tol`` per index step.  Used as a fallback convergence
    criterion when τ_acf detection fails.

    Args:
        viscosity_integral: Cumulative Green-Kubo integral (Pa·s) vs lag index.
        tail_fraction: Fraction of the array to use as the tail window.
        rel_slope_tol: Convergence threshold for |slope| / |mean(tail)|.

    Returns:
        True if the integral is considered plateaued.

    """
    n = len(viscosity_integral)
    tail_start = int(n * (1.0 - tail_fraction))
    tail = viscosity_integral[tail_start:]
    if len(tail) < NPOINTS:
        return False
    x = np.arange(len(tail), dtype=np.float64)
    slope = np.polyfit(x, tail, 1)[0]
    mean_val = np.mean(np.abs(tail))
    if np.isclose(mean_val, 0.0):
        return True
    return abs(slope) / mean_val < rel_slope_tol


def viscosity_simulation(
    structure: Atoms,
    potential: pd.DataFrame,
    temperature_sim: float = 5000.0,
    timestep: float = 1.0,
    initial_production_steps: int = 10_000_000,
    n_print: int = 1,
    max_total_time_ns: float = 50.0,
    max_iterations: int = 40,
    eta_rel_tol: float = 0.05,
    eta_stable_iters: int = 3,
    server_kwargs: dict[str, Any] | None = None,
    *,
    langevin: bool = False,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Perform an autonomous viscosity simulation using the Helfand moment method.

    Runs an initial NVT production MD (default 10 ns) and iteratively extends
    it by 100 ps until viscosity converges or limits are reached.

    Convergence requires both conditions to be true simultaneously:

    1. **eta-stability** - ``|eta_new - eta_prev| / eta_prev < eta_rel_tol``
       for ``eta_stable_iters`` consecutive iterations.
    2. **MSD linearity** - the local slope of the Helfand moment MSD is flat
       in the last 30 % of the lag window (``_viscosity_plateaued`` check),
       confirming that the diffusive regime has been reached.

    Extensions stop when either ``max_total_time_ns`` or ``max_iterations``
    is exhausted.  ``get_viscosity`` is retained as a legacy cross-check in
    the return dict.

    Args:
        structure: Input structure (assumed pre-equilibrated).
        potential: LAMMPS potential DataFrame.
        temperature_sim: Simulation temperature in Kelvin.
        timestep: MD timestep in fs.
        initial_production_steps: Steps for the first production run.
            Default 10,000,000 (= 10 ns at 1 fs timestep).
        n_print: Thermodynamic output frequency (must equal ``output_frequency``
            passed to analysis functions).
        max_total_time_ns: Maximum total production time in nanoseconds.
            Default 50 ns.
        max_iterations: Maximum number of 100 ps extension iterations.
            Default 40.
        eta_rel_tol: Relative change threshold for eta-stability check.
            Default 0.05 (5 %).
        eta_stable_iters: Consecutive stable iterations required.
            Default 3.
        server_kwargs: Additional server arguments forwarded to LAMMPS.
        langevin: Whether to use Langevin dynamics.
        seed: Random seed for velocity initialization.
        tmp_working_directory: Temporary directory for LAMMPS runs.

    Returns:
        A dictionary containing:
            - viscosity_data: Output of ``helfand_viscosity`` from the final iteration.
            - result: Accumulated raw MD arrays (pressures, volume, temperature).
            - structure: Final ASE ``Atoms`` object.
            - total_production_steps: Total production steps completed.
            - iterations: Number of iterations run.
            - converged: True if both convergence criteria were satisfied.

    Raises:
        ValueError: If the potential DataFrame is empty.

    Example:
        >>> result = viscosity_simulation(
        ...     structure=my_atoms,
        ...     potential=my_potential_df,
        ...     temperature_sim=3000.0,
        ... )
        >>> print(result["viscosity_data"]["viscosity"])

    """
    max_steps = int(max_total_time_ns * 1e6 / timestep)
    ext_steps = int(100_000.0 / timestep)  # 100 ps per extension

    warnings.warn(
        f"viscosity_simulation: starting initial run "
        f"({initial_production_steps:,} steps, T={temperature_sim} K, "
        f"timestep={timestep} fs)",
        stacklevel=2,
    )

    sim_result = _viscosity_simulation(
        structure=structure,
        potential=potential,
        temperature_sim=temperature_sim,
        timestep=timestep,
        production_steps=initial_production_steps,
        n_print=n_print,
        server_kwargs=server_kwargs,
        langevin=langevin,
        seed=seed,
        tmp_working_directory=tmp_working_directory,
    )

    acc: dict[str, Any] = {
        "pressures": sim_result["result"]["pressures"],
        "volume": sim_result["result"]["volume"],
        "temperature": sim_result["result"]["temperature"],
    }
    current_structure: Atoms = sim_result["structure"]
    total_production_steps: int = initial_production_steps
    iterations: int = 1
    converged: bool = False
    helfand_data: dict[str, Any] = {}
    eta_prev: float | None = None
    stable_count: int = 0

    while iterations <= max_iterations:
        t_total_ps = total_production_steps * timestep / 1000.0

        helfand_data = helfand_viscosity(
            {"result": acc},
            timestep=timestep,
            output_frequency=n_print,
        )
        eta_curr = float(helfand_data["viscosity"])

        # Secondary check: MSD must be in the linear diffusive regime.
        # np.diff of the MSD gives the local slope; if that slope is itself
        # flat (_viscosity_plateaued), the MSD is linear.
        msd_arr = np.asarray(helfand_data["msd"])
        msd_linear = _viscosity_plateaued(
            np.diff(msd_arr),
            tail_fraction=0.3,
            rel_slope_tol=0.05,
        )

        if eta_prev is not None and eta_prev != 0.0:
            rel_change = abs(eta_curr - eta_prev) / abs(eta_prev)
            if rel_change < eta_rel_tol:
                stable_count += 1
            else:
                stable_count = 0

            warnings.warn(
                f"viscosity_simulation: iter {iterations}, "
                f"eta={eta_curr:.4e} Pa.s, rel_change={rel_change:.3f} "
                f"(tol={eta_rel_tol}), stable={stable_count}/{eta_stable_iters}, "
                f"msd_linear={msd_linear}, t={t_total_ps:.1f} ps",
                stacklevel=2,
            )

            if stable_count >= eta_stable_iters and msd_linear:
                warnings.warn(
                    f"viscosity_simulation: CONVERGED at iteration {iterations} "
                    f"(eta_rel={rel_change:.3f}, msd_linear={msd_linear})",
                    stacklevel=2,
                )
                converged = True
                break
        else:
            warnings.warn(
                f"viscosity_simulation: iter {iterations}, "
                f"eta={eta_curr:.4e} Pa.s (first estimate), t={t_total_ps:.1f} ps",
                stacklevel=2,
            )

        eta_prev = eta_curr

        if total_production_steps >= max_steps:
            warnings.warn(
                f"viscosity_simulation: time budget exhausted ({t_total_ps:.1f} ps).",
                stacklevel=2,
            )
            break

        _ext = min(ext_steps, max_steps - total_production_steps)

        warnings.warn(
            f"viscosity_simulation: extending by {_ext:,} steps ({_ext * timestep / 1000:.0f} ps).",
            stacklevel=2,
        )

        current_structure, ext_parsed = _run_lammps_md(
            structure=current_structure,
            potential=potential,
            tmp_working_directory=tmp_working_directory,
            temperature=temperature_sim,
            n_ionic_steps=_ext,
            timestep=timestep,
            n_print=n_print,
            initial_temperature=temperature_sim,
            langevin=langevin,
            server_kwargs=server_kwargs,
        )

        ext_result = ext_parsed.get("generic", None)
        acc["pressures"] = np.concatenate([acc["pressures"], ext_result["pressures"]])
        acc["volume"] = np.concatenate([acc["volume"], ext_result["volume"]])
        acc["temperature"] = np.concatenate([acc["temperature"], ext_result["temperature"]])

        total_production_steps += _ext
        iterations += 1

    warnings.warn(
        f"viscosity_simulation: finished. converged={converged}, "
        f"iterations={iterations}, total_steps={total_production_steps:,}",
        stacklevel=2,
    )

    return {
        "viscosity_data": helfand_data,
        "result": acc,
        "structure": current_structure,
        "total_production_steps": total_production_steps,
        "iterations": iterations,
        "converged": converged,
    }


def viscosity_ensemble(  # noqa: C901
    structure: Atoms,
    potential: pd.DataFrame,
    n_replicas: int = 3,
    seeds: list[int] | None = None,
    base_seed: int = 12345,
    temperature_sim: float = 5000.0,
    timestep: float = 1.0,
    initial_production_steps: int = 10_000_000,
    n_print: int = 1,
    max_total_time_ns: float = 50.0,
    max_iterations: int = 40,
    eta_rel_tol: float = 0.05,
    eta_stable_iters: int = 3,
    server_kwargs: dict[str, Any] | None = None,
    *,
    langevin: bool = False,
    parallel: bool = False,
    executor: Executor | None = None,
    tmp_working_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Run multiple independent viscosity simulations and return ensemble-averaged results.

    Each replica starts from the same structure but uses a different random seed for
    velocity initialisation, producing statistically independent trajectories.  The
    viscosity (and other properties) are averaged across replicas and the inter-replica
    standard deviation is reported as the uncertainty.

    **Execution modes** (mutually exclusive; ``executor`` takes priority):

    * ``executor`` provided — each replica is submitted via the executor's ``.submit()``
      method.  Pass any executorlib executor (``SlurmJobExecutor``, ``SlurmClusterExecutor``,
      ``SingleNodeExecutor``, …) to run replicas on an HPC cluster.  The executor's
      ``resource_dict`` is automatically populated with the MPI core count from
      ``server_kwargs["cores"]``.  The executor's lifecycle is managed by the caller.
    * ``parallel=True`` — replicas run simultaneously on the local machine using
      ``ThreadPoolExecutor``.  Requires ``n_replicas * server_kwargs["cores"]`` total
      cores to be available.
    * default (sequential) — replicas run one after another on the local machine.

    Seeds are written to ``tmp_working_directory/viscosity_ensemble_seeds.json``
    immediately after generation so they are preserved even if the run is interrupted.

    Args:
        structure: Input structure (assumed pre-equilibrated).
        potential: LAMMPS potential DataFrame.
        n_replicas: Number of independent replicas to run. Default 3.
        seeds: Explicit list of integer seeds, one per replica.  If ``None``, seeds are
            auto-generated from ``base_seed`` using a NumPy RNG.
        base_seed: Master seed used to generate per-replica seeds when ``seeds`` is
            ``None``.  Default 12345.
        temperature_sim: Simulation temperature in Kelvin.
        timestep: MD timestep in fs.
        initial_production_steps: Steps for the first production run per replica.
        n_print: Thermodynamic output frequency.
        max_total_time_ns: Maximum total production time per replica in nanoseconds.
        max_iterations: Maximum number of 100 ps extension iterations per replica.
        eta_rel_tol: Relative change threshold for eta-stability check.
        eta_stable_iters: Consecutive stable iterations required per replica.
        server_kwargs: Additional server arguments forwarded to LAMMPS (e.g.
            ``{"cores": 6}`` sets the MPI process count per replica).
        langevin: Whether to use Langevin dynamics.
        parallel: If ``True``, run all replicas simultaneously using threads.
            Requires enough cores for all replicas at once. Default ``False``.
            Ignored when ``executor`` is provided.
        executor: Optional executorlib (or compatible) executor.  When provided,
            each replica is submitted as ``executor.submit(_run_replica, i, seed,
            resource_dict={"cores": N})``.  Example::

                from executorlib import SlurmJobExecutor
                with SlurmJobExecutor(max_workers=100) as exe:
                    result = viscosity_ensemble(..., executor=exe)

        tmp_working_directory: Directory for LAMMPS runs and seed file.

    Returns:
        A dictionary containing:
            - viscosity: Mean shear viscosity across replicas (Pa·s)
            - viscosity_std: Sample standard deviation across replicas (Pa·s, ddof=1)
            - viscosity_sem: Standard error of the mean (Pa·s)
            - shear_modulus_inf: Mean infinite-frequency shear modulus (Pa)
            - bulk_viscosity: Mean bulk viscosity (Pa·s)
            - maxwell_relaxation_time_ps: Mean Maxwell relaxation time (ps)
            - mean_pressure_gpa: Mean pressure averaged across replicas (GPa)
            - temperature: Mean temperature across all replicas (K)
            - n_replicas: Number of replicas run
            - seeds: List of seeds actually used
            - viscosities: Per-replica shear viscosity values (Pa·s)
            - converged: Per-replica convergence flags
            - results: Full ``viscosity_simulation`` output dicts per replica

    Example:
        >>> out = viscosity_ensemble(
        ...     structure=my_atoms,
        ...     potential=my_potential_df,
        ...     n_replicas=3,
        ...     temperature_sim=4000.0,
        ...     n_print=10,
        ...     server_kwargs={"cores": 4},
        ... )
        >>> print(out["viscosity"], "±", out["viscosity_sem"], "Pa·s")

    """
    if seeds is None:
        rng = np.random.default_rng(base_seed)
        seeds_list: list[int] = rng.integers(1, 900_000_000, size=n_replicas).tolist()
    else:
        seeds_list = list(seeds)

    if len(seeds_list) != n_replicas:
        msg = f"len(seeds)={len(seeds_list)} does not match n_replicas={n_replicas}"
        raise ValueError(msg)

    if tmp_working_directory is not None:
        seeds_path = Path(tmp_working_directory) / "viscosity_ensemble_seeds.json"
        seeds_path.parent.mkdir(parents=True, exist_ok=True)
        seeds_path.write_text(json.dumps({"base_seed": base_seed, "seeds": seeds_list}, indent=2))
        warnings.warn(
            f"viscosity_ensemble: seeds saved to {seeds_path}",
            stacklevel=2,
        )

    def _run_replica(i: int, seed: int) -> dict[str, Any]:
        warnings.warn(
            f"viscosity_ensemble: starting replica {i + 1}/{n_replicas}, seed={seed}",
            stacklevel=6,
        )
        if tmp_working_directory is not None:
            replica_workdir: Path | None = Path(tmp_working_directory) / f"replica_{i}"
            replica_workdir.mkdir(parents=True, exist_ok=True)
        else:
            replica_workdir = None
        result_i = viscosity_simulation(
            structure=structure,
            potential=potential,
            temperature_sim=temperature_sim,
            timestep=timestep,
            initial_production_steps=initial_production_steps,
            n_print=n_print,
            max_total_time_ns=max_total_time_ns,
            max_iterations=max_iterations,
            eta_rel_tol=eta_rel_tol,
            eta_stable_iters=eta_stable_iters,
            server_kwargs=server_kwargs,
            langevin=langevin,
            seed=int(seed),
            tmp_working_directory=replica_workdir,
        )
        result_i["seed"] = int(seed)
        return result_i

    replica_results: list[dict[str, Any]] = []
    if executor is not None:
        # Submit each replica as a separate job via the provided executor.
        # The executor (e.g. SlurmJobExecutor) handles SLURM script generation and
        # job submission automatically.  Configure resource requirements (cores,
        # partition, memory) on the executor itself at construction time.
        ordered_results: list[dict[str, Any]] = [{}] * n_replicas
        replica_futures = {executor.submit(_run_replica, i, seed): i for i, seed in enumerate(seeds_list)}
        for completed_future in as_completed(replica_futures):
            replica_index = replica_futures[completed_future]
            ordered_results[replica_index] = completed_future.result()
        replica_results = ordered_results
    elif parallel:
        # Run all replicas simultaneously on the local machine using threads.
        # Each LAMMPS process is MPI-parallel and releases the GIL, so threads
        # do not compete for the Python interpreter.
        ordered_results = [{}] * n_replicas
        with ThreadPoolExecutor(max_workers=n_replicas) as thread_pool:
            replica_futures = {thread_pool.submit(_run_replica, i, seed): i for i, seed in enumerate(seeds_list)}
            for completed_future in as_completed(replica_futures):
                replica_index = replica_futures[completed_future]
                ordered_results[replica_index] = completed_future.result()
        replica_results = ordered_results
    else:
        replica_results = [_run_replica(i, seed) for i, seed in enumerate(seeds_list)]

    per_replica_viscosity_data = [r["viscosity_data"] for r in replica_results]

    shear_viscosities = np.array([data["viscosity"] for data in per_replica_viscosity_data])
    mean_viscosity = float(np.mean(shear_viscosities))
    std_viscosity = float(np.std(shear_viscosities, ddof=1)) if n_replicas > 1 else 0.0
    sem_viscosity = std_viscosity / math.sqrt(n_replicas) if n_replicas > 1 else 0.0

    def _mean_across_replicas(key: str) -> float:
        return float(np.mean([data[key] for data in per_replica_viscosity_data]))

    return {
        "viscosity": mean_viscosity,
        "viscosity_std": std_viscosity,
        "viscosity_sem": sem_viscosity,
        "shear_modulus_inf": _mean_across_replicas("shear_modulus_inf"),
        "bulk_viscosity": _mean_across_replicas("bulk_viscosity"),
        "maxwell_relaxation_time_ps": _mean_across_replicas("maxwell_relaxation_time_ps"),
        "mean_pressure_gpa": _mean_across_replicas("mean_pressure_gpa"),
        "temperature": _mean_across_replicas("temperature"),
        "n_replicas": n_replicas,
        "seeds": seeds_list,
        "viscosities": shear_viscosities.tolist(),
        "converged": [r["converged"] for r in replica_results],
        "results": replica_results,
    }
