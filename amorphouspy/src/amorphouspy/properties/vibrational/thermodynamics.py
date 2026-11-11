"""Harmonic vibrational thermodynamics from a VDOS."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.constants import Boltzmann, e, h

from amorphouspy.properties.vibrational._common import _MIN_GRID_POINTS, _is_real_number
from amorphouspy.properties.vibrational.units import _THZ_TO_UNIT

# h nu / k_B in K for nu = 1 THz (about 48 K), and k_B T in meV for T = 1 K.
_THZ_TO_K = h * 1e12 / Boltzmann
_K_TO_MEV = Boltzmann / e * 1e3
# Cap of x = h nu / (k_B T): exp(-x / 2) is exactly 0 from x = 1491 on, so every term of a capped mode is 0.
_MAX_REDUCED_FREQUENCY = 1500.0


def _validate_thermodynamics_inputs(
    frequencies_thz: NDArray, vdos: NDArray, temperatures_k: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    frequencies_thz = np.asarray(frequencies_thz, dtype=np.float64)
    vdos = np.asarray(vdos, dtype=np.float64)
    temperatures_k = _temperatures_array(temperatures_k)
    if frequencies_thz.ndim != 1 or vdos.shape != frequencies_thz.shape:
        msg = (
            "frequencies_thz and vdos must be 1-D arrays of the same length, "
            f"got shapes {frequencies_thz.shape} and {vdos.shape}."
        )
        raise ValueError(msg)
    if frequencies_thz.size < _MIN_GRID_POINTS:
        msg = f"At least 2 frequencies are needed, got {frequencies_thz.size}."
        raise ValueError(msg)
    if not (np.all(np.isfinite(frequencies_thz)) and np.all(np.isfinite(vdos))):
        msg = "frequencies_thz and vdos must not contain NaN or inf."
        raise ValueError(msg)
    if np.any(vdos < 0):
        msg = "vdos must not be negative."
        raise ValueError(msg)
    if np.any(np.diff(frequencies_thz) <= 0):
        msg = "frequencies_thz must be strictly increasing."
        raise ValueError(msg)
    if not np.any(frequencies_thz > 0):
        msg = "frequencies_thz must contain at least one frequency above 0 THz."
        raise ValueError(msg)
    _check_temperatures(temperatures_k)
    return frequencies_thz, vdos, temperatures_k


def _temperatures_array(temperatures_k: ArrayLike) -> NDArray[np.float64]:
    """Return ``temperatures_k`` as a float64 array of at least 1-D; raise for booleans, which would cast to 0 and 1."""
    if np.asarray(temperatures_k).dtype == np.bool_:
        msg = "temperatures_k must be a scalar or a non-empty 1-D array of finite, non-negative values, not booleans."
        raise ValueError(msg)
    return np.array(temperatures_k, dtype=np.float64, ndmin=1)


def _check_temperatures(temperatures_k: NDArray[np.float64]) -> None:
    """Raise unless ``temperatures_k`` (float64, at least 1-D) is a non-empty 1-D array of finite values >= 0."""
    valid_temperatures = np.all(np.isfinite(temperatures_k) & (temperatures_k >= 0))
    if temperatures_k.ndim != 1 or temperatures_k.size == 0 or not valid_temperatures:
        msg = "temperatures_k must be a scalar or a non-empty 1-D array of finite, non-negative values."
        raise ValueError(msg)


def compute_vibrational_thermodynamics(
    frequencies_thz: NDArray,
    vdos: NDArray,
    temperatures_k: ArrayLike,
    *,
    lowest_frequency_thz: float | None = None,
) -> dict[str, Any]:
    r"""Compute the harmonic heat capacity, entropy and energies from a VDOS, with a lower temperature bound.

    Each mode of frequency $\nu$ is a quantum harmonic oscillator. With $x = h \nu / (k_B T)$ it contributes

    $$C_V = k_B \frac{x^2 e^x}{(e^x - 1)^2}, \quad
    S = k_B \left[\frac{x}{e^x - 1} - \ln\left(1 - e^{-x}\right)\right],$$

    $$U = \frac{h \nu}{2} + \frac{h \nu}{e^x - 1}, \quad F = \frac{h \nu}{2} + k_B T \ln\left(1 - e^{-x}\right),
    \quad E_\mathrm{ZP} = \frac{h \nu}{2},$$

    and a quantity per atom is $Q(T) = 3 \int g(\nu)\, q(\nu, T)\, d\nu$ (three modes per atom), integrated with the
    trapezoid rule on the given grid, as the VDOS functions of this module normalise. The VDOS must therefore be
    normalised to 1 (``normalization="unit"`` in :func:`compute_vdos_from_velocities`); this is not enforced, and
    ``"vdos_integral"`` lets the caller check it. A partial VDOS from :func:`compute_partial_vdos` with ``"unit"`` gives
    that group's additive contribution per atom of the whole system, so the partials sum to the total. At $T = 0$ the
    exact limits are returned: $C_V = S = 0$ and $U = F = E_\mathrm{ZP}$. Large $x$ is evaluated in forms that cannot
    overflow (with ``np.expm1`` and ``np.log1p(-np.exp(-x))``).

    $S$ and $F$ diverge as $\nu \to 0$, so only $\nu > 0$ is integrated: the integrand is set to zero at every grid
    point with $\nu \le 0$ (the zero-frequency bin of the VACF route, and imaginary harmonic modes stored as negative
    frequencies) and the trapezoid rule runs over the full grid. Nothing is dropped silently: ``"vdos_integral"`` is the
    integral of $g$ over the full grid, and ``"excluded_weight"`` is that integral minus the integral of $g$ with the
    points at $\nu \le 0$ set to zero, i.e. the share of modes left out. At high $T$, $C_V$ tends to
    $3 \, (\mathrm{vdos\_integral} - \mathrm{excluded\_weight})$ k$_B$ per atom.

    At low temperature the values depend on the low-frequency VDOS, which a finite box or a short trajectory does not
    resolve. Below $T_\mathrm{min} = h \nu_\mathrm{low} / k_B$ (1 THz corresponds to 48 K) the values are flagged as
    unreliable, not removed. Usual choices of $\nu_\mathrm{low}$ (``lowest_frequency_thz``) are
    ``debye["lowest_box_frequency_thz"]`` from :func:`compute_debye_level` (the lowest acoustic mode of the box), the
    frequency resolution $1 / (N \Delta t)$ of the VACF route, or the lowest non-zero mode of the harmonic route. With
    ``None`` the first positive grid frequency is used, which is only the resolution of the grid, not the box limit, so
    the bound is then too optimistic for a small box. $T = 0$ is always flagged.

    Args:
        frequencies_thz: Frequencies in THz, 1-D and strictly increasing. Zero and negative values are allowed and
            excluded from the integrals.
        vdos: VDOS on ``frequencies_thz``, 1-D of the same length, non-negative, in 1/THz and normalised to 1.
        temperatures_k: Temperatures in K, a scalar, a list or a 1-D array of finite values $\ge 0$.
        lowest_frequency_thz: $\nu_\mathrm{low}$ in THz for the temperature bound, positive and finite, or ``None``
            (default) for the first positive grid frequency.

    Returns:
        A dict with:
            ``"temperatures_k"``: the temperatures as a float64 array (length 1 for a scalar).
            ``"heat_capacity_kb_per_atom"``: $C_V$ in k$_B$ per atom, an array of the same length.
            ``"entropy_kb_per_atom"``: $S$ in k$_B$ per atom.
            ``"internal_energy_mev_per_atom"``: $U$ in meV per atom, including the zero-point energy.
            ``"free_energy_mev_per_atom"``: $F$ in meV per atom, including the zero-point energy.
            ``"zero_point_energy_mev_per_atom"``: $E_\mathrm{ZP}$ in meV per atom, a float.
            ``"temperature_bound_k"``: $T_\mathrm{min}$ in K, a float.
            ``"reliable"``: boolean array, True where $T \ge T_\mathrm{min}$.
            ``"lowest_frequency_thz"``: $\nu_\mathrm{low}$ in THz.
            ``"lowest_frequency_source"``: ``"caller"`` or ``"grid"``.
            ``"vdos_integral"``: trapezoid integral of ``vdos`` over the full grid.
            ``"excluded_weight"``: the part of ``"vdos_integral"`` at $\nu \le 0$ that is left out.

    Raises:
        ValueError: If ``frequencies_thz`` and ``vdos`` are not 1-D arrays of the same length with at least 2 points,
            contain NaN or inf, ``vdos`` is negative anywhere, the frequencies are not strictly increasing or none is
            above 0, a temperature is negative or not finite, or ``lowest_frequency_thz`` is not positive and finite.

    Example:
        ```pycon
        >>> frequencies_thz = np.linspace(0.0, 10.0, 1001)
        >>> vdos = 3 * frequencies_thz**2 / 10.0**3  # Debye VDOS with nu_D = 10 THz (theta_D = 480 K)
        >>> thermo = compute_vibrational_thermodynamics(
        ...     frequencies_thz, vdos, [0.0, 300.0, 3000.0], lowest_frequency_thz=1.0
        ... )
        >>> thermo["heat_capacity_kb_per_atom"].round(3)
        array([0.   , 2.649, 2.996])
        >>> round(thermo["temperature_bound_k"], 1), thermo["reliable"].tolist()
        (48.0, [False, True, True])
        >>> round(thermo["zero_point_energy_mev_per_atom"], 2)  # 9/8 h nu_D
        46.53

        ```
    """
    frequencies_thz, vdos, temperatures_k = _validate_thermodynamics_inputs(frequencies_thz, vdos, temperatures_k)
    positive = frequencies_thz > 0
    if lowest_frequency_thz is None:
        lowest_thz = float(frequencies_thz[positive][0])
        source = "grid"
    else:
        is_number = _is_real_number(lowest_frequency_thz)
        if not (is_number and np.isfinite(lowest_frequency_thz) and lowest_frequency_thz > 0):
            msg = f"lowest_frequency_thz must be positive and finite, got {lowest_frequency_thz}."
            raise ValueError(msg)
        lowest_thz = float(lowest_frequency_thz)
        source = "caller"

    # Trapezoid weights of the full grid, so that sum(weights * f) is np.trapezoid(f, frequencies_thz). Keeping only the
    # points with nu > 0 integrates the integrand set to zero at nu <= 0. Three modes per atom.
    spacing = np.diff(frequencies_thz)
    weights = np.zeros_like(frequencies_thz)
    weights[:-1] += spacing / 2
    weights[1:] += spacing / 2
    mode_weights = 3 * (weights * vdos)[positive]
    mode_k = frequencies_thz[positive] * _THZ_TO_K
    mode_mev = frequencies_thz[positive] * _THZ_TO_UNIT["meV"]
    vdos_integral = float(np.trapezoid(vdos, frequencies_thz))
    positive_integral = float(np.trapezoid(np.where(positive, vdos, 0.0), frequencies_thz))

    zero_point = float(mode_weights @ mode_mev) / 2
    heat_capacity = np.zeros_like(temperatures_k)
    entropy = np.zeros_like(temperatures_k)
    internal_energy = np.full_like(temperatures_k, zero_point)
    free_energy = np.full_like(temperatures_k, zero_point)
    for i in np.flatnonzero(temperatures_k > 0):
        # Capped where exp(-x / 2) is already exactly 0, so an overflow of x at a subnormal T cannot give inf * 0.
        with np.errstate(over="ignore"):
            x = np.minimum(mode_k / temperatures_k[i], _MAX_REDUCED_FREQUENCY)
        exp_neg = np.exp(-x)
        one_minus_exp_neg = -np.expm1(-x)
        occupation = exp_neg / one_minus_exp_neg  # 1 / (e^x - 1) without overflow
        log_term = np.log1p(-exp_neg)  # ln(1 - e^-x)
        # x^2 e^x / (e^x - 1)^2 written so that neither e^x nor x^2 is formed.
        heat_capacity[i] = mode_weights @ (x * np.exp(-x / 2) / one_minus_exp_neg) ** 2
        entropy[i] = mode_weights @ (x * occupation - log_term)
        internal_energy[i] += mode_weights @ (mode_mev * occupation)
        free_energy[i] += _K_TO_MEV * temperatures_k[i] * (mode_weights @ log_term)

    temperature_bound_k = lowest_thz * _THZ_TO_K
    return {
        "temperatures_k": temperatures_k,
        "heat_capacity_kb_per_atom": heat_capacity,
        "entropy_kb_per_atom": entropy,
        "internal_energy_mev_per_atom": internal_energy,
        "free_energy_mev_per_atom": free_energy,
        "zero_point_energy_mev_per_atom": zero_point,
        "temperature_bound_k": temperature_bound_k,
        "reliable": temperatures_k >= temperature_bound_k,
        "lowest_frequency_thz": lowest_thz,
        "lowest_frequency_source": source,
        "vdos_integral": vdos_integral,
        "excluded_weight": vdos_integral - positive_integral,
    }
