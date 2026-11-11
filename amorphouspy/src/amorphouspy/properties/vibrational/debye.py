"""Debye level and reduced VDOS g(nu) / nu^2."""

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.constants import Boltzmann, atomic_mass, h

from amorphouspy.atoms.neighbors import cell_perpendicular_heights
from amorphouspy.properties.vibrational._common import VdosNormalization

if TYPE_CHECKING:
    from ase import Atoms

# 1 GPa / (1 g/cm^3) = 1e9 Pa / 1e3 kg/m^3, in m^2/s^2.
_GPA_PER_G_CM3_TO_M2_S2 = 1e6


def compute_debye_level(
    structure: "Atoms",
    bulk_modulus_gpa: float,
    shear_modulus_gpa: float,
    *,
    normalization: VdosNormalization = "unit",
) -> dict[str, float]:
    r"""Compute the Debye level of the reduced VDOS from elastic moduli and the density of a structure.

    The isotropic elastic continuum with bulk modulus $B$, shear modulus $G$ and mass density $\rho$ has the
    longitudinal and transverse sound velocities

    $$v_L = \sqrt{\frac{B + 4G/3}{\rho}}, \quad v_T = \sqrt{\frac{G}{\rho}},$$

    in m/s for $B$, $G$ in GPa and $\rho$ in g/cm$^3$ (1 GPa / (1 g/cm$^3$) = $10^6$ m$^2$/s$^2$). With the number
    density $n$, the Debye frequency and temperature are

    $$\nu_D^3 = \frac{9 n}{4 \pi \left(1 / v_L^3 + 2 / v_T^3\right)}, \quad \theta_D = \frac{h \nu_D}{k_B},$$

    and the Debye VDOS $g_D(\nu) = 3 \nu^2 / \nu_D^3$ ($\nu \le \nu_D$, normalised to 1) gives the Debye level
    $g_D(\nu) / \nu^2 = 3 / \nu_D^3$ in THz$^{-3}$, the low-frequency limit of the reduced VDOS of an isotropic solid.
    For ``"3N"`` the level is $3N$ times larger, matching a VDOS that integrates to $3N$.

    The moduli are taken as given: no fit of the VDOS is made, so without moduli there is no Debye level. Use moduli of
    the same model and state as the VDOS, e.g. ``elastic_simulation(...)["moduli"]["B"]`` and ``["G"]``, which are in
    GPa. Only the masses (``structure.get_masses()``, in amu) and the cell volume enter $\rho$ and $n$; the positions
    are not used.

    The lowest frequency the periodic box supports is $\nu_\mathrm{min} = v_T / L_\mathrm{max}$, the transverse
    acoustic mode whose wavelength is the largest perpendicular cell height $L_\mathrm{max}$ (the edge $L$ of a cubic
    box). The box has no acoustic modes below it, so the reduced VDOS there does not represent the bulk glass. Taking
    $L_\mathrm{max}$ as the longest wavelength is exact for orthogonal boxes and assumes a cell that is not strongly
    skewed otherwise.

    Args:
        structure: Periodic structure as an ASE object, with a 3-D cell of positive volume.
        bulk_modulus_gpa: Bulk modulus $B$ in GPa.
        shear_modulus_gpa: Shear modulus $G$ in GPa.
        normalization: ``"unit"`` (default) or ``"3N"``, the normalisation of the VDOS the level is compared with, as
            in :func:`compute_vdos_from_velocities`. ``"none"`` is rejected: a power spectral density has no Debye
            level.

    Returns:
        A dict of floats:
            ``"longitudinal_velocity_m_s"`` and ``"transverse_velocity_m_s"``: $v_L$ and $v_T$ in m/s.
            ``"debye_frequency_thz"``: $\nu_D$ in THz.
            ``"debye_temperature_k"``: $\theta_D$ in K.
            ``"debye_level_thz-3"``: $3 / \nu_D^3$ (``"unit"``) or $9N / \nu_D^3$ (``"3N"``) in THz$^{-3}$.
            ``"lowest_box_frequency_thz"``: $\nu_\mathrm{min} = v_T / L_\mathrm{max}$ in THz.
            ``"density_g_cm3"``: $\rho$ in g/cm$^3$.
            ``"number_density_per_a3"``: $n$ in atoms/Å$^3$.

    Raises:
        ValueError: If a modulus is not positive and finite, ``normalization`` is not ``"unit"`` or ``"3N"``,
            ``structure`` has no atoms, is not periodic in all three directions, or its cell volume is not positive.

    Example:
        ```pycon
        >>> from ase import Atoms
        >>> silica = Atoms("Si1000O2000", cell=[35.661] * 3, pbc=True)  # 2.2 g/cm^3
        >>> debye = compute_debye_level(silica, bulk_modulus_gpa=36.7, shear_modulus_gpa=31.2)
        >>> round(debye["density_g_cm3"], 3), round(debye["transverse_velocity_m_s"])
        (2.2, 3766)
        >>> round(debye["debye_frequency_thz"], 2), round(debye["debye_temperature_k"])
        (10.4, 499)
        >>> round(debye["lowest_box_frequency_thz"], 3)
        1.056

        ```
    """
    for name, modulus in (("bulk_modulus_gpa", bulk_modulus_gpa), ("shear_modulus_gpa", shear_modulus_gpa)):
        is_number = isinstance(modulus, int | float | np.integer | np.floating) and not isinstance(modulus, bool)
        if not (is_number and np.isfinite(modulus) and modulus > 0):
            msg = f"{name} must be positive and finite, got {modulus}."
            raise ValueError(msg)
    if normalization not in ("unit", "3N"):
        msg = (
            f"Unknown normalization {normalization!r}; allowed values are 'unit' and '3N' "
            "(a power spectral density has no Debye level)."
        )
        raise ValueError(msg)
    n_atoms = len(structure)
    if n_atoms == 0:
        msg = "structure contains no atoms."
        raise ValueError(msg)
    if not np.all(structure.pbc):
        msg = "structure must be periodic in all three directions."
        raise ValueError(msg)
    volume_a3 = float(structure.cell.volume)
    if not volume_a3 > 0:
        msg = "structure must have a 3-D cell with positive volume."
        raise ValueError(msg)

    # amu -> g and A^3 -> cm^3.
    density_g_cm3 = float(structure.get_masses().sum()) * atomic_mass * 1e3 / (volume_a3 * 1e-24)
    number_density_per_a3 = n_atoms / volume_a3
    longitudinal_modulus_gpa = bulk_modulus_gpa + 4 * shear_modulus_gpa / 3
    v_longitudinal = float(np.sqrt(longitudinal_modulus_gpa / density_g_cm3 * _GPA_PER_G_CM3_TO_M2_S2))
    v_transverse = float(np.sqrt(shear_modulus_gpa / density_g_cm3 * _GPA_PER_G_CM3_TO_M2_S2))
    # Number density in m^-3 and velocities in m/s give the Debye frequency in Hz.
    debye_frequency_hz = (
        9 * number_density_per_a3 * 1e30 / (4 * np.pi * (1 / v_longitudinal**3 + 2 / v_transverse**3))
    ) ** (1 / 3)
    debye_frequency_thz = debye_frequency_hz * 1e-12
    scale = 1.0 if normalization == "unit" else 3.0 * n_atoms
    largest_height_a = float(cell_perpendicular_heights(structure.cell.array).max())
    return {
        "longitudinal_velocity_m_s": v_longitudinal,
        "transverse_velocity_m_s": v_transverse,
        "debye_frequency_thz": debye_frequency_thz,
        "debye_temperature_k": h * debye_frequency_hz / Boltzmann,
        "debye_level_thz-3": scale * 3 / debye_frequency_thz**3,
        "lowest_box_frequency_thz": v_transverse / (largest_height_a * 1e-10) * 1e-12,
        "density_g_cm3": density_g_cm3,
        "number_density_per_a3": number_density_per_a3,
    }


def compute_reduced_vdos(
    frequencies_thz: NDArray,
    vdos: NDArray,
    *,
    peak_range_thz: tuple[float, float] = (0.5, 5.0),
    debye: dict[str, float] | None = None,
) -> dict[str, Any]:
    r"""Compute the reduced VDOS $g(\nu) / \nu^2$, the boson-peak position and its height over the Debye level.

    The boson peak is the raw maximum of $g(\nu) / \nu^2$ within ``peak_range_thz``, with no smoothing or fit, so pass
    a block-averaged VDOS (e.g. a Hann-windowed mean over blocks); the maximum of a single noisy spectrum is mostly
    noise. The default range of 0.5 to 5 THz is a generic choice for oxide glasses (silica peaks near 1 THz); set it
    for the system at hand. A maximum on the first or last bin of the range (``"peak_at_range_edge"``) means no peak
    was found inside it, e.g. a reduced VDOS that rises steeply towards zero frequency.

    With ``debye`` (the output of :func:`compute_debye_level`), the peak height is also divided by the Debye level, and
    every frequency below the lowest one the box supports is marked unreliable. The level is only compared, never
    fitted: without ``debye`` none of these keys is returned. The VDOS and ``debye`` must use the same normalisation
    (``"unit"`` or ``"3N"``).

    Any VDOS works, including a partial VDOS, but the Debye comparison is meaningful for the total VDOS only.

    Args:
        frequencies_thz: Frequencies in THz, 1-D, non-negative and strictly increasing, e.g. from
            :func:`compute_vdos_from_velocities`.
        vdos: VDOS on ``frequencies_thz``, 1-D of the same length, in 1/THz.
        peak_range_thz: ``(low, high)`` in THz, both ends included, in which the boson peak is searched. The
            frequency 0 is never part of the range.
        debye: The dict returned by :func:`compute_debye_level`, or ``None`` (default).

    Returns:
        A dict with:
            ``"reduced_vdos"``: $g(\nu) / \nu^2$ in THz$^{-3}$, a float64 array of the input length, NaN at
            $\nu = 0$.
            ``"boson_peak_thz"``: frequency of the maximum within ``peak_range_thz``, in THz.
            ``"boson_peak_height"``: reduced VDOS at that frequency, in THz$^{-3}$.
            ``"peak_at_range_edge"``: True if the maximum is on the first or last bin of the range.
        With ``debye`` also:
            ``"debye_level_thz-3"``: the Debye level from ``debye``.
            ``"boson_peak_over_debye"``: ``boson_peak_height`` divided by the Debye level.
            ``"reliable"``: boolean array, True where the frequency is at or above the lowest box frequency.
            ``"peak_below_box_limit"``: True if the boson peak lies below the lowest box frequency.

    Raises:
        ValueError: If ``frequencies_thz`` and ``vdos`` are not 1-D arrays of the same length, contain NaN or inf,
            a frequency is negative, the frequencies are not strictly increasing, ``peak_range_thz`` does not satisfy
            ``low < high``, or the range contains no frequency above 0.

    Example:
        ```pycon
        >>> from ase import Atoms
        >>> silica = Atoms("Si1000O2000", cell=[35.661] * 3, pbc=True)  # 2.2 g/cm^3
        >>> debye = compute_debye_level(silica, bulk_modulus_gpa=36.7, shear_modulus_gpa=31.2)
        >>> frequencies_thz = np.linspace(0.0, 10.0, 201)
        >>> excess = 1 + np.exp(-0.5 * ((frequencies_thz - 1.2) / 0.4) ** 2)
        >>> vdos = debye["debye_level_thz-3"] * frequencies_thz**2 * excess  # Debye VDOS with an excess at 1.2 THz
        >>> result = compute_reduced_vdos(frequencies_thz, vdos, debye=debye)
        >>> round(result["boson_peak_thz"], 3), result["peak_at_range_edge"], round(result["boson_peak_over_debye"], 3)
        (1.2, False, 2.0)
        >>> bool(np.isnan(result["reduced_vdos"][0])), result["reduced_vdos"].shape
        (True, (201,))
        >>> round(float(frequencies_thz[result["reliable"]][0]), 3), result["peak_below_box_limit"]
        (1.1, False)

        ```
    """
    frequencies_thz = np.asarray(frequencies_thz, dtype=np.float64)
    vdos = np.asarray(vdos, dtype=np.float64)
    if frequencies_thz.ndim != 1 or vdos.shape != frequencies_thz.shape:
        msg = (
            "frequencies_thz and vdos must be 1-D arrays of the same length, "
            f"got shapes {frequencies_thz.shape} and {vdos.shape}."
        )
        raise ValueError(msg)
    if not (np.all(np.isfinite(frequencies_thz)) and np.all(np.isfinite(vdos))):
        msg = "frequencies_thz and vdos must not contain NaN or inf."
        raise ValueError(msg)
    if np.any(frequencies_thz < 0):
        msg = "frequencies_thz must not be negative."
        raise ValueError(msg)
    if np.any(np.diff(frequencies_thz) <= 0):
        msg = "frequencies_thz must be strictly increasing."
        raise ValueError(msg)
    low, high = peak_range_thz
    if not low < high:
        msg = f"peak_range_thz must satisfy low < high, got {peak_range_thz}."
        raise ValueError(msg)

    positive = frequencies_thz > 0
    reduced = np.full_like(vdos, np.nan)
    np.divide(vdos, frequencies_thz**2, out=reduced, where=positive)
    in_range = np.flatnonzero(positive & (frequencies_thz >= low) & (frequencies_thz <= high))
    if in_range.size == 0:
        msg = f"peak_range_thz {peak_range_thz} contains no frequency bins above 0 THz."
        raise ValueError(msg)
    peak = in_range[np.argmax(reduced[in_range])]
    result: dict[str, Any] = {
        "reduced_vdos": reduced,
        "boson_peak_thz": float(frequencies_thz[peak]),
        "boson_peak_height": float(reduced[peak]),
        "peak_at_range_edge": bool(peak in (in_range[0], in_range[-1])),
    }
    if debye is not None:
        level = float(debye["debye_level_thz-3"])
        lowest_thz = float(debye["lowest_box_frequency_thz"])
        result["debye_level_thz-3"] = level
        result["boson_peak_over_debye"] = result["boson_peak_height"] / level
        result["reliable"] = frequencies_thz >= lowest_thz
        result["peak_below_box_limit"] = bool(frequencies_thz[peak] < lowest_thz)
    return result
