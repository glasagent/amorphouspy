"""Self-diffusion analysis for glass systems: mean-squared displacement and transport coefficients.

Implements a molecular-dynamics driver and post-processing utilities for atomic self-diffusion.
The primary mean-squared displacement (MSD) is computed by the multi-time-origin Fast Correlation
Algorithm (FFT, Calandrini et al. 2011) on LAMMPS unwrapped coordinates, giving a well-averaged MSD
from a single trajectory. For non-uniformly sampled trajectories (e.g. the log-then-linear dump
schedule used by :func:`diffusion_simulation`), where the FFT estimator does not apply, a
single-origin MSD on the same exact-unwrapped coordinates is used instead.

Note:
    A LAMMPS-side ``compute msd`` was evaluated for the single-origin path but cannot be wired
    through the ``lammpsparser`` file interface: its ``input_control`` mapping is keyed by the
    leading command token (so per-species ``group``/``compute`` lines collide) and injected commands
    are appended after the ``thermo_style``/``dump`` lines that would reference them. The
    single-origin MSD is therefore computed in Python from LAMMPS' own image-flag-unwrapped
    coordinates, which is the same exact unwrapping ``compute msd`` would use.

Author
------
Achraf Atila (achraf.atila@bam.de)
"""

import gzip
import warnings
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
import pandas as pd
from ase.atoms import Atoms
from numpy.typing import NDArray

from amorphouspy.atoms.shared import type_to_dict
from amorphouspy.lammps.io import frames_from_melt_quench_result
from amorphouspy.lammps.runner import _run_lammps_md

# Physical constants (CODATA 2018).
_KB_J = 1.380649e-23  # Boltzmann constant, J/K
_KB_EV = 8.617333262e-5  # Boltzmann constant, eV/K
_E_CHARGE = 1.602176634e-19  # elementary charge, C

# Unit conversion for diffusion coefficients: 1 Angstrom^2/ps = 1e-4 cm^2/s = 1e-8 m^2/s.
_A2_PS_TO_CM2_S = 1e-4
_A2_PS_TO_M2_S = 1e-8

_MIN_POINTS = 2  # minimum frames/fit points/data pairs
_DIFFUSIVE_SLOPE_TOL = 0.25  # allowed |log-log MSD slope - 1| before warning


def _unwrap_trajectory(wrapped: NDArray[np.float64], cells: NDArray[np.float64]) -> NDArray[np.float64]:
    n_frames = wrapped.shape[0]
    unwrapped = np.empty_like(wrapped)
    unwrapped[0] = wrapped[0]
    for t in range(1, n_frames):
        delta = wrapped[t] - wrapped[t - 1]
        inv_cell = np.linalg.inv(cells[t])
        frac = delta @ inv_cell
        frac -= np.round(frac)
        unwrapped[t] = unwrapped[t - 1] + frac @ cells[t]
    return unwrapped


def _build_trajectory(
    frames: list[Atoms], *, remove_com_drift: bool
) -> tuple[NDArray[np.float64], NDArray[np.int_], float]:
    n_frames = len(frames)
    if n_frames < _MIN_POINTS:
        msg = "At least two frames are required to compute an MSD."
        raise ValueError(msg)

    have_unwrapped = all("unwrapped_positions" in frame.arrays for frame in frames)
    if have_unwrapped:
        positions = np.stack([np.asarray(frame.arrays["unwrapped_positions"], dtype=float) for frame in frames])
    else:
        wrapped = np.stack([frame.get_positions() for frame in frames])
        cells = np.stack([np.asarray(frame.get_cell(), dtype=float) for frame in frames])
        positions = _unwrap_trajectory(wrapped, cells)

    if remove_com_drift:
        masses = frames[0].get_masses()
        com = (positions * masses[None, :, None]).sum(axis=1) / masses.sum()
        positions = positions - com[:, None, :]

    numbers = frames[0].get_atomic_numbers()
    temps = [float(frame.info["temperature"]) for frame in frames if "temperature" in frame.info]
    temperature = float(np.mean(temps)) if temps else float("nan")
    return positions, numbers, temperature


def _msd_fft_multi(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mean MSD over a group via the multi-origin Fast Correlation Algorithm.

    Args:
        positions: Unwrapped coordinates with shape ``(n_frames, n_group_atoms, 3)``.

    Returns:
        Group-averaged MSD per lag, length ``n_frames`` (lag 0 is exactly 0).
    """
    n_frames = positions.shape[0]
    n_atoms = positions.shape[1]
    if n_atoms == 0:
        return np.zeros(n_frames)

    # S2: autocorrelation of the position vector, summed over the 3 components, per atom.
    flat = positions.reshape(n_frames, n_atoms * 3)
    transformed = np.fft.fft(flat, n=2 * n_frames, axis=0)
    acf = np.fft.ifft(transformed * np.conj(transformed), axis=0)[:n_frames].real
    acf /= np.arange(n_frames, 0, -1)[:, None]
    s2 = acf.reshape(n_frames, n_atoms, 3).sum(axis=2)

    # S1: recursive sum-of-squares term, normalised by the number of origins (n_frames - lag).
    sq_norm = (positions**2).sum(axis=2)
    sq_norm = np.vstack([sq_norm, np.zeros((1, n_atoms))])
    running = 2.0 * sq_norm.sum(axis=0)
    s1 = np.zeros((n_frames, n_atoms))
    for lag in range(n_frames):
        running = running - sq_norm[lag - 1] - sq_norm[n_frames - lag]
        s1[lag] = running / (n_frames - lag)

    return (s1 - 2.0 * s2).mean(axis=1)


def _msd_single_origin(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    displacement = positions - positions[0][None, :, :]
    return (displacement**2).sum(axis=2).mean(axis=1)


def _msd_brute_force(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    n_frames = positions.shape[0]
    msd = np.zeros(n_frames)
    for lag in range(1, n_frames):
        delta = positions[lag:] - positions[:-lag]
        msd[lag] = (delta**2).sum(axis=2).mean()
    return msd


def log_linear_dump_steps(
    total_steps: int,
    *,
    crossover_steps: int,
    points_per_decade: int = 9,
    linear_interval_steps: int,
    first_step: int = 0,
) -> list[int]:
    """Return the timesteps of a log-then-linear dump schedule.

    Below *crossover_steps* the spacing is decade-logarithmic (the LAMMPS ``logfreq`` pattern): with
    the default ``points_per_decade=9`` the steps are 1, 2, ..., 9, 10, 20, ..., 90, 100, 200, ...; at
    and above the crossover the spacing is uniform (*linear_interval_steps*). This reproduces,
    bit-for-bit, the schedule realised in LAMMPS by :func:`_log_linear_dump_variable`, so it can be
    used to recover the exact per-frame times of a log-dumped trajectory.

    Args:
        total_steps: Number of production MD steps.
        crossover_steps: Step count corresponding to the log/linear crossover.
        points_per_decade: Log-spaced points per decade below the crossover (9 gives 1..9, 10..90, ...).
        linear_interval_steps: Uniform dump interval (in steps) at and above the crossover.
        first_step: First dumped step (usually 0).

    Returns:
        Sorted list of dump timesteps (inclusive of *first_step*, up to *total_steps*).

    Example:
        >>> log_linear_dump_steps(100000, crossover_steps=100000, points_per_decade=9, linear_interval_steps=1000)[:12]
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
    """
    steps = [int(first_step)]
    current = first_step
    while current < total_steps:
        if current < crossover_steps:
            base = 10 ** (len(str(max(current, 1))) - 1)
            increment = max(1, int(base * 9 / points_per_decade + 0.5))
            nxt = current + increment
        else:
            nxt = current + linear_interval_steps
        current = int(nxt)
        if current <= total_steps:
            steps.append(current)
    return steps


def _log_linear_dump_variable(crossover_steps: int, points_per_decade: int, linear_interval_steps: int) -> str:
    """LAMMPS ``variable`` line whose value is the next log-then-linear dump step (for ``dump_modify every``)."""
    # log increment = round(10^floor(log10(step)) * 9 / points_per_decade); the (step<1) guard avoids
    # log(0) at the first-yes dump of step 0. Matches log_linear_dump_steps exactly on the dumped steps.
    log_increment = f"floor(10^floor(log(step+(step<1))+0.000000001)*9/{points_per_decade}+0.5)"
    formula = f"step+(step<{crossover_steps})*{log_increment}+(step>={crossover_steps})*{linear_interval_steps}"
    return f'variable amx_dump equal "{formula}"'


def _frame_times_ps(frames: list[Atoms], *, timestep: float, output_frequency: int) -> NDArray[np.float64]:
    if all("step" in frame.info for frame in frames):
        return np.array([float(frame.info["step"]) for frame in frames]) * timestep / 1000.0
    return np.arange(len(frames), dtype=np.float64) * (timestep * output_frequency / 1000.0)


def compute_msd(
    frames: list[Atoms],
    *,
    timestep: float = 1.0,
    output_frequency: int = 1,
    max_lag: int | None = None,
    remove_com_drift: bool = True,
    per_species: bool = True,
    method: str = "fft",
) -> dict[str, Any]:
    """Compute the mean-squared displacement from a trajectory.

    With ``method="fft"`` (default) the multi-time-origin Fast Correlation Algorithm is used, which
    requires uniformly-spaced frames and gives the best statistics. With ``method="single_origin"``
    the MSD is ``<|r(t) - r(0)|^2>`` referenced to the first frame; this is noisier but works on a
    **non-uniformly** sampled trajectory (e.g. a log-then-linear dump), using each frame's actual time
    from ``atoms.info["step"]`` when present. If every frame carries an ``"unwrapped_positions"`` array
    (LAMMPS image-flag unwrapping) it is used directly; otherwise the trajectory is unwrapped on the
    fly from wrapped positions and per-frame cells.

    Args:
        frames: Trajectory as a list of :class:`ase.Atoms`, atoms in a consistent order across frames
            (LAMMPS dumps must use ``dump_modify ... sort id``).
        timestep: MD integration timestep in femtoseconds.
        output_frequency: MD steps between successive frames (used for the time axis when frames carry
            no ``info["step"]``).
        max_lag: Largest lag (in frames) to report. Defaults to ``len(frames) - 1``.
        remove_com_drift: Subtract the per-frame mass-weighted centre-of-mass displacement.
        per_species: Also report MSD resolved per chemical element.
        method: ``"fft"`` (multi-origin, uniform sampling) or ``"single_origin"`` (non-uniform-safe).

    Returns:
        Dict with ``"lag_time_ps"`` and ``"msd_total"`` (Angstrom^2) plus ``"temperature"`` (K). When
        *per_species* is set, also ``"msd_per_species"`` and ``"n_atoms_per_species"``.

    Example:
        >>> from amorphouspy import diffusion_simulation  # doctest: +SKIP
        >>> out = diffusion_simulation(structure, potential)  # doctest: +SKIP
        >>> msd = compute_msd(out["frames"], timestep=1.0, output_frequency=100)  # doctest: +SKIP
        >>> msd["msd_total"][0]  # doctest: +SKIP
        0.0
    """
    positions, numbers, temperature = _build_trajectory(frames, remove_com_drift=remove_com_drift)
    n_frames = positions.shape[0]
    lag_limit = n_frames - 1 if max_lag is None else int(min(max_lag, n_frames - 1))

    if method == "single_origin":
        times_ps = _frame_times_ps(frames, timestep=timestep, output_frequency=output_frequency)
        lag_time_ps = (times_ps - times_ps[0])[: lag_limit + 1]

        def per_group(group: NDArray[np.float64]) -> NDArray[np.float64]:
            return _msd_single_origin(group)[: lag_limit + 1]
    elif method == "fft":
        lag_time_ps = np.arange(lag_limit + 1) * (timestep * output_frequency / 1000.0)

        def per_group(group: NDArray[np.float64]) -> NDArray[np.float64]:
            return _msd_fft_multi(group)[: lag_limit + 1]
    else:
        msg = f"Unknown method {method!r}; use 'fft' or 'single_origin'."
        raise ValueError(msg)

    result: dict[str, Any] = {
        "lag_time_ps": lag_time_ps.tolist(),
        "msd_total": per_group(positions).tolist(),
        "temperature": temperature,
    }

    if per_species:
        msd_per_species: dict[str, list[float]] = {}
        n_atoms_per_species: dict[str, int] = {}
        for atomic_number, symbol in sorted(type_to_dict(numbers).items()):
            mask = numbers == atomic_number
            msd_per_species[symbol] = per_group(positions[:, mask, :]).tolist()
            n_atoms_per_species[symbol] = int(mask.sum())
        result["msd_per_species"] = msd_per_species
        result["n_atoms_per_species"] = n_atoms_per_species

    return result


def _fit_diffusion(
    lag_time_ps: NDArray[np.float64],
    msd: NDArray[np.float64],
    *,
    fit_start_frac: float,
    fit_end_frac: float,
    dimensionality: int,
) -> dict[str, Any]:
    n_points = len(msd)
    start = max(1, int(fit_start_frac * n_points))
    stop = min(n_points, max(start + 2, int(fit_end_frac * n_points)))
    times = np.asarray(lag_time_ps[start:stop], dtype=float)
    values = np.asarray(msd[start:stop], dtype=float)
    if times.size < _MIN_POINTS:
        msg = "Not enough points in the fit window to estimate a diffusion coefficient."
        raise ValueError(msg)

    slope, intercept = np.polyfit(times, values, 1)
    diffusion_a2_ps = slope / (2.0 * dimensionality)
    residual_std = float(np.std(values - (slope * times + intercept)))

    finite = (times > 0) & (values > 0)
    loglog_slope = (
        float(np.polyfit(np.log(times[finite]), np.log(values[finite]), 1)[0])
        if finite.sum() >= _MIN_POINTS
        else float("nan")
    )
    if np.isfinite(loglog_slope) and abs(loglog_slope - 1.0) > _DIFFUSIVE_SLOPE_TOL:
        warnings.warn(
            f"MSD log-log slope {loglog_slope:.2f} over the fit window deviates from 1; the trajectory "
            "may not be in the diffusive regime (extend the run or shift the fit window).",
            stacklevel=3,
        )

    return {
        "diffusion_cm2_s": float(diffusion_a2_ps * _A2_PS_TO_CM2_S),
        "diffusion_m2_s": float(diffusion_a2_ps * _A2_PS_TO_M2_S),
        "slope_a2_ps": float(slope),
        "fit_residual_cm2_s": float(residual_std / (2.0 * dimensionality) * _A2_PS_TO_CM2_S),
        "fit_window_ps": (float(times[0]), float(times[-1])),
        "loglog_slope": loglog_slope,
    }


def get_diffusion(
    msd_result: dict[str, Any],
    *,
    fit_start_frac: float = 0.2,
    fit_end_frac: float = 0.8,
    dimensionality: int = 3,
) -> dict[str, Any]:
    """Estimate self-diffusion coefficients from an MSD via the Einstein relation.

    Fits ``MSD = 2 * d * D * t`` over the diffusive window and reports ``D`` in cm^2/s and m^2/s. A
    warning is emitted if the log-log MSD slope in the window deviates from 1 (sub-diffusive).

    Args:
        msd_result: Output of :func:`compute_msd`.
        fit_start_frac: Start of the fit window as a fraction of the number of lags.
        fit_end_frac: End of the fit window as a fraction of the number of lags.
        dimensionality: Spatial dimensionality ``d`` (3 for bulk glasses).

    Returns:
        Dict with ``"total"`` diffusion data, ``"temperature"`` (K), and ``"per_species"`` when the
        input was species-resolved. Each entry holds ``diffusion_cm2_s``, ``diffusion_m2_s``,
        ``slope_a2_ps``, ``fit_residual_cm2_s``, ``fit_window_ps`` and ``loglog_slope``.

    Example:
        >>> msd = {"lag_time_ps": [0, 1, 2, 3, 4], "msd_total": [0.0, 6.0, 12.0, 18.0, 24.0]}
        >>> out = get_diffusion(msd, fit_start_frac=0.0, fit_end_frac=1.0)
        >>> round(out["total"]["diffusion_cm2_s"], 6)
        0.0001
    """
    lag_time_ps = np.asarray(msd_result["lag_time_ps"], dtype=float)

    def _one(msd: list[float]) -> dict[str, Any]:
        return _fit_diffusion(
            lag_time_ps,
            np.asarray(msd, dtype=float),
            fit_start_frac=fit_start_frac,
            fit_end_frac=fit_end_frac,
            dimensionality=dimensionality,
        )

    output: dict[str, Any] = {
        "temperature": msd_result.get("temperature"),
        "total": _one(msd_result["msd_total"]),
    }
    if "msd_per_species" in msd_result:
        output["per_species"] = {symbol: _one(msd) for symbol, msd in msd_result["msd_per_species"].items()}
    return output


def fit_arrhenius(
    temperatures: NDArray[np.float64] | list[float],
    diffusivities: NDArray[np.float64] | list[float],
) -> tuple[tuple[float, float], NDArray[np.float64]]:
    """Fit an Arrhenius law ``D(T) = D0 * exp(-Ea / (kB T))`` to diffusion data.

    Performs a linear fit of ``ln D`` versus ``1/T``.

    Args:
        temperatures: Temperatures in Kelvin (length >= 2).
        diffusivities: Diffusion coefficients at each temperature (any consistent unit; ``D0`` is
            returned in that same unit). Must be strictly positive.

    Returns:
        Tuple ``((D0, Ea_eV), covariance)`` where ``Ea_eV`` is the activation energy in eV and
        ``covariance`` is the 2x2 covariance matrix of the linear fit.

    Example:
        >>> import numpy as np
        >>> kb = 8.617333262e-5
        >>> temps = np.array([2000.0, 2500.0, 3000.0, 3500.0])
        >>> d0_true, ea_true = 1e-3, 1.2
        >>> d_vals = d0_true * np.exp(-ea_true / (kb * temps))
        >>> (d0, ea), _ = fit_arrhenius(temps, d_vals)
        >>> bool(np.isclose(ea, ea_true, atol=1e-6))
        True
    """
    temperature = np.asarray(temperatures, dtype=float)
    diffusion = np.asarray(diffusivities, dtype=float)
    if temperature.size < _MIN_POINTS or diffusion.size != temperature.size:
        msg = "fit_arrhenius requires at least two matching (temperature, diffusivity) pairs."
        raise ValueError(msg)
    if np.any(diffusion <= 0) or np.any(temperature <= 0):
        msg = "Temperatures and diffusivities must be strictly positive for an Arrhenius fit."
        raise ValueError(msg)

    (slope, intercept), covariance = np.polyfit(1.0 / temperature, np.log(diffusion), 1, cov=True)
    activation_energy_ev = -slope * _KB_EV
    prefactor = float(np.exp(intercept))
    return (prefactor, float(activation_energy_ev)), covariance


def nernst_einstein_conductivity(
    diffusion_per_species: dict[str, float],
    charges: dict[str, float],
    *,
    n_per_species: dict[str, int],
    volume_a3: float,
    temperature: float,
    haven_ratio: float = 1.0,
) -> dict[str, Any]:
    """Estimate ionic conductivity from self-diffusion via the Nernst-Einstein relation.

    Computes ``sigma = (1 / H_R) * sum_i (N_i / V) * (z_i e)^2 * D_i / (kB T)``. The Haven ratio
    ``H_R`` defaults to 1 (no cross-correlations), so this is an upper-bound estimate that ignores
    correlated ionic motion.

    Args:
        diffusion_per_species: Self-diffusion coefficient per element in m^2/s.
        charges: Formal charge per element in units of the elementary charge (e.g. ``{"Na": 1}``).
        n_per_species: Number of atoms of each element in the cell.
        volume_a3: Cell volume in cubic Angstrom.
        temperature: Temperature in Kelvin.
        haven_ratio: Haven ratio ``H_R``.

    Returns:
        Dict with total ``"conductivity_S_m"`` (S/m), the ``"per_species_contribution"`` mapping,
        ``"haven_ratio"`` and ``"temperature"``.

    Example:
        >>> out = nernst_einstein_conductivity(
        ...     {"Na": 1e-9},
        ...     {"Na": 1.0},
        ...     n_per_species={"Na": 100},
        ...     volume_a3=1.0e5,
        ...     temperature=1000.0,
        ... )
        >>> out["conductivity_S_m"] > 0
        True
    """
    if temperature <= 0 or volume_a3 <= 0 or haven_ratio <= 0:
        msg = "temperature, volume_a3 and haven_ratio must be strictly positive."
        raise ValueError(msg)

    volume_m3 = volume_a3 * 1e-30
    contributions: dict[str, float] = {}
    total = 0.0
    for symbol, diffusion in diffusion_per_species.items():
        charge = charges.get(symbol)
        count = n_per_species.get(symbol)
        if charge is None or count is None or charge == 0:
            continue
        sigma_i = (count / volume_m3) * (charge * _E_CHARGE) ** 2 * diffusion / (_KB_J * temperature) / haven_ratio
        contributions[symbol] = float(sigma_i)
        total += sigma_i

    return {
        "conductivity_S_m": float(total),
        "per_species_contribution": contributions,
        "haven_ratio": haven_ratio,
        "temperature": temperature,
    }


def save_frames(frames: list[Atoms], path: str | Path) -> Path:
    """Write a trajectory to extended XYZ (gzipped when *path* ends in ``.gz``).

    Per-atom arrays (including ``unwrapped_positions``) and per-frame ``info`` (``step``,
    ``temperature``) are preserved, so the file round-trips back into :func:`compute_msd`.

    Args:
        frames: Trajectory as a list of :class:`ase.Atoms`.
        path: Output path; a ``.gz`` suffix triggers gzip compression.

    Returns:
        The path written.

    Example:
        >>> save_frames(out["frames"], "trajectory.xyz.gz")  # doctest: +SKIP
    """
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "wt") as handle:
            ase.io.write(handle, frames, format="extxyz")
    else:
        ase.io.write(str(path), frames, format="extxyz")
    return path


def diffusion_simulation(
    structure: Atoms,
    potential: pd.DataFrame,
    temperature_sim: float = 3000.0,
    timestep: float = 1.0,
    production_steps: int = 1_000_000,
    n_print_thermo: int = 100,
    server_kwargs: dict[str, Any] | None = None,
    *,
    equilibration_steps: int = 100_000,
    langevin: bool = False,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
    max_lag: int | None = None,
    remove_com_drift: bool = True,
    crossover_ps: float = 100.0,
    points_per_decade: int = 9,
    linear_interval_ps: float = 10.0,
    save_trajectory: str | Path | None = None,
) -> dict[str, Any]:
    """Run an MD production trajectory and compute self-diffusion from the MSD.

    Equilibrates the structure (Langevin then NVT) and runs an NVT production stage that dumps on a
    **log-then-linear** schedule to keep the trajectory small: decade-log below *crossover_ps* (with
    the default ``points_per_decade=9`` the dumped steps are 1, 2, ..., 9, 10, 20, ..., 90, 100, ...)
    and uniform every *linear_interval_ps* above it. The MSD is computed single-origin (the non-uniform
    grid cannot use a multi-time-origin FFT estimator).

    Equilibration and production are run at constant volume, so the simulation keeps the density of
    the input structure. Pass a structure pre-equilibrated at the desired density/pressure to control it.

    Args:
        structure: Input structure (a melt or pre-equilibrated glass); its cell sets the run density.
        potential: LAMMPS potential as a pandas DataFrame.
        temperature_sim: Production temperature in Kelvin.
        timestep: MD timestep in femtoseconds.
        production_steps: Number of MD steps in the production stage.
        n_print_thermo: Thermo/log output interval in MD steps.
        server_kwargs: Additional arguments for the LAMMPS server (e.g. ``{"cores": 4}``).
        equilibration_steps: Length of the constant-volume equilibration stage in MD steps
            (use a small value for quick demos, the default for production-quality runs).
        langevin: Use Langevin dynamics during equilibration/production.
        seed: Random seed for velocity initialization.
        tmp_working_directory: Directory for temporary simulation files.
        max_lag: Largest lag (frames) reported in the MSD.
        remove_com_drift: Subtract the centre-of-mass drift before computing the MSD.
        crossover_ps: Log/linear crossover time in ps.
        points_per_decade: Decade-log dump points per decade below the crossover (9 → 1..9, 10..90, ...).
        linear_interval_ps: Uniform dump interval in ps above the crossover.
        save_trajectory: If given, write the trajectory to this path (gzipped extXYZ when it ends in
            ``.gz``) via :func:`save_frames`.

    Returns:
        Dict with ``"result"`` (parsed LAMMPS production output), ``"structure"`` (final Atoms),
        ``"frames"`` (trajectory), ``"msd"`` (:func:`compute_msd` output), ``"diffusion"``
        (:func:`get_diffusion` output) and ``"trajectory_path"`` (or ``None``).

    Example:
        >>> out = diffusion_simulation(structure, potential, temperature_sim=3000.0)  # doctest: +SKIP
        >>> out["diffusion"]["per_species"]["Na"]["diffusion_cm2_s"]  # doctest: +SKIP
    """
    if potential.empty:
        msg = "No matching potential found for the given configuration."
        raise ValueError(msg)

    structure0, _ = _run_lammps_md(
        structure=structure,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=temperature_sim,
        n_ionic_steps=10_000,
        timestep=timestep,
        n_print_thermo=1000,
        initial_temperature=temperature_sim,
        langevin=True,
        seed=seed,
        server_kwargs=server_kwargs,
    )

    # NVT (constant volume) equilibration: it preserves the input density. A zero-pressure NPT stage
    # is deliberately avoided here because at high melt temperatures it lets the box expand without
    # bound (vaporisation), leaving the production stage with non-interacting, free-flying atoms and a
    # spurious ballistic MSD. To run at a relaxed density, pass a pre-equilibrated structure instead.
    structure1, _ = _run_lammps_md(
        structure=structure0,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=temperature_sim,
        n_ionic_steps=equilibration_steps,
        timestep=timestep,
        n_print_thermo=1000,
        initial_temperature=temperature_sim,
        langevin=langevin,
        seed=seed,
        server_kwargs=server_kwargs,
    )

    crossover_steps = round(crossover_ps * 1000.0 / timestep)
    linear_interval_steps = max(1, round(linear_interval_ps * 1000.0 / timestep))
    schedule = log_linear_dump_steps(
        production_steps,
        crossover_steps=crossover_steps,
        points_per_decade=points_per_decade,
        linear_interval_steps=linear_interval_steps,
    )
    variable_line = _log_linear_dump_variable(crossover_steps, points_per_decade, linear_interval_steps)
    # The variable is placed first in the Config (before LAMMPS' dump line) and reset_timestep last,
    # so the schedule references a production timestep starting at 0.
    production_potential = potential.copy()
    production_potential["Config"] = production_potential["Config"].apply(
        lambda lines: [variable_line, *list(lines), "reset_timestep 0"]
    )

    structure_final, parsed_output = _run_lammps_md(
        structure=structure1,
        potential=production_potential,
        tmp_working_directory=tmp_working_directory,
        temperature=temperature_sim,
        n_ionic_steps=production_steps,
        timestep=timestep,
        n_print_thermo=n_print_thermo,
        initial_temperature=0,
        langevin=langevin,
        server_kwargs=server_kwargs,
        input_control_file={"dump_modify": "1 every v_amx_dump first yes sort id"},
    )

    result = parsed_output.get("generic", None)
    frames = frames_from_melt_quench_result({"structure": structure_final, "result": [result]}, structure)

    # The parser overwrites the dump-frame steps with thermo steps, so assign the exact dump times from
    # the (bit-identical) Python schedule. Align defensively if the frame count differs.
    n_assign = min(len(frames), len(schedule))
    if len(frames) != len(schedule):
        warnings.warn(
            f"log_linear dump produced {len(frames)} frames but the schedule has {len(schedule)}; "
            "using the overlapping frames.",
            stacklevel=2,
        )
    frames = frames[:n_assign]
    for frame, step in zip(frames, schedule[:n_assign], strict=False):
        frame.info["step"] = step

    msd = compute_msd(
        frames, timestep=timestep, max_lag=max_lag, remove_com_drift=remove_com_drift, method="single_origin"
    )
    diffusion = get_diffusion(msd)

    trajectory_path = str(save_frames(frames, save_trajectory)) if save_trajectory is not None else None

    return {
        "result": result,
        "structure": structure_final,
        "frames": frames,
        "msd": msd,
        "diffusion": diffusion,
        "trajectory_path": trajectory_path,
    }
