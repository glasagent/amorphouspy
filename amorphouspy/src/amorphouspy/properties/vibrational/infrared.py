"""Infrared response: oscillator strengths, dielectric function and optical constants."""

import math
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.fft
import scipy.signal
from numpy.typing import ArrayLike, NDArray
from scipy.constants import atomic_mass, e, epsilon_0

from amorphouspy.lammps.potentials.potential import charges_from_potential
from amorphouspy.properties.vibrational import _common
from amorphouspy.properties.vibrational._common import (
    _MATRIX_NDIM,
    _N_COMPONENTS,
    _group_selections,
    _is_real_number,
    _membership,
)
from amorphouspy.properties.vibrational._trajectory import (
    _atoms_per_chunk,
    _trajectory_statistics,
    _validate_trajectory,
)
from amorphouspy.properties.vibrational.harmonic import _harmonic_grid, _validate_frequency_grid
from amorphouspy.properties.vibrational.units import convert_frequency
from amorphouspy.properties.vibrational.vdos import VdosWindow, _validate_window

if TYPE_CHECKING:
    from ase import Atoms

_SCALAR_CHARGE_NDIM = 1
# The acoustic sum rule (zero total charge) is taken as violated above this fraction of sum |charges|.
_ACOUSTIC_SUM_RTOL = 1e-6
# e^2 / (4 pi^2 epsilon_0 amu A^3) in THz^2: the susceptibility of strength 1 e^2/amu in 1 A^3 at 1 THz, about 44224.5.
_IR_THZ2_A3 = e**2 / (4 * math.pi**2 * epsilon_0 * atomic_mass * 1e-30) * 1e-24
# e^2 ps / (amu A^3) in S/m: the conductivity of a current response of 1 e^2 ps / amu in 1 A^3, about 1.5457e7.
_S_PER_M_A3 = e**2 * 1e-12 / (atomic_mass * 1e-30)
_MIN_LAG_FRAMES = 2
# Time points per frame on which the correlation function is interpolated for its sine transform.
_LAG_OVERSAMPLING = 16


def _validate_charges(charges: ArrayLike, n_atoms: int) -> NDArray[np.float64]:
    """Return the charges as float64, shape (N,) or (N, 3, 3); warn when their sum over the atoms is not zero."""
    charges = np.asarray(charges, dtype=np.float64)
    if charges.shape not in ((n_atoms,), (n_atoms, _N_COMPONENTS, _N_COMPONENTS)) or not np.all(np.isfinite(charges)):
        msg = (
            f"charges must be finite with shape ({n_atoms},) or ({n_atoms}, 3, 3), one scalar charge or one Born "
            f"effective charge tensor per atom, got shape {charges.shape}."
        )
        raise ValueError(msg)
    total = float(np.abs(charges.sum(axis=0)).max())
    if total > _ACOUSTIC_SUM_RTOL * float(np.abs(charges).sum()):
        warnings.warn(
            f"The charges do not sum to zero over the atoms (largest component of the sum: {total:.4g} e): the "
            "acoustic sum rule is violated and a rigid translation would carry a dipole.",
            UserWarning,
            stacklevel=3,
        )
    return charges


def _validate_modes(
    eigenvectors: NDArray[np.float64], masses: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Check the eigenvectors and masses; return the eigenvectors, the masses as float64 and N."""
    eigenvectors = np.asarray(eigenvectors)
    if eigenvectors.ndim != _MATRIX_NDIM or eigenvectors.dtype != np.float64:
        msg = (
            "eigenvectors must be a 2-D float64 array (a cast would copy it), "
            f"got {eigenvectors.ndim}-D {eigenvectors.dtype}."
        )
        raise ValueError(msg)
    n_rows, n_modes = eigenvectors.shape
    if n_rows == 0 or n_rows % _N_COMPONENTS or n_modes == 0:
        msg = f"eigenvectors must have 3N rows (N >= 1) and at least one column, got shape {eigenvectors.shape}."
        raise ValueError(msg)
    n_atoms = n_rows // _N_COMPONENTS
    masses = np.asarray(masses, dtype=np.float64)
    if masses.shape != (n_atoms,) or not np.all(np.isfinite(masses) & (masses > 0)):
        msg = f"masses must be {n_atoms} positive finite values, one per atom, got shape {masses.shape}."
        raise ValueError(msg)
    return eigenvectors, masses, n_atoms


def compute_ir_oscillator_strengths(
    eigenvectors: NDArray[np.float64],
    masses: ArrayLike,
    charges: ArrayLike,
    *,
    groups: dict[str, dict[str, NDArray]] | None = None,
) -> dict[str, Any]:
    r"""Compute the infrared oscillator strength of every mode from its eigenvector and the atomic charges.

    A mode $m$ with mass-weighted eigenvector $e_{3i+\beta,m}$ displaces atom $i$ by $e_{3i+\beta,m} / \sqrt{m_i}$ per
    unit normal coordinate. With the charge tensor $Z_{i,\alpha\beta}$ of atom $i$ (the dipole along $\alpha$ created
    by a unit displacement along $\beta$), the dipole per unit normal coordinate, called the mode effective charge, is

    $$\tilde Z_{m,\alpha} = \sum_i \sum_\beta \frac{Z_{i,\alpha\beta}\, e_{3i+\beta,m}}{\sqrt{m_i}},$$

    and the isotropic oscillator strength, the orientational average of $\tilde Z_{m,\alpha} \tilde Z_{m,\beta}$, is

    $$f_m = \tfrac{1}{3} \sum_\alpha \tilde Z_{m,\alpha}^2 .$$

    Scalar charges $q_i$ stand for $Z_{i,\alpha\beta} = q_i \delta_{\alpha\beta}$: the rigid-ion model, in which the
    fixed charges of the potential move with the atoms and nothing polarises. Its peak positions are those of the
    potential, but its intensities are qualitative; Born effective charge tensors from an electronic-structure
    calculation can be passed instead.

    For a full orthonormal set of $3N$ modes the strengths obey the sum rule
    $\sum_m \tilde Z_{m,\alpha} \tilde Z_{m,\beta} = \sum_i (Z_i Z_i^T)_{\alpha\beta} / m_i$; the three rigid
    translations hold $(\sum_i Z_i)(\sum_i Z_i)^T / \sum_i m_i$ of it, which is zero for a neutral cell.

    With ``groups`` the sum over atoms is restricted to each group $G$, giving $\tilde Z^G_m$, and every pair of labels
    of one grouping gets the strength

    $$f^{GG'}_m = \frac{2 - \delta_{GG'}}{3} \sum_\alpha \tilde Z^G_{m,\alpha} \tilde Z^{G'}_{m,\alpha},$$

    whose cross terms ($G \neq G'$) can be negative. Over the unordered pairs of a grouping that holds every atom once,
    such as ``"element"`` of :func:`classify_vibrational_groups`, they sum to $f_m$.

    Args:
        eigenvectors: The $(3N, K)$ float64 eigenvectors of the mass-weighted dynamical matrix, one orthonormal mode
            per column with row $3i + \alpha$ for atom $i$ and direction $\alpha$, e.g. ``"eigenvectors"`` of
            :func:`diagonalize_dynamical_matrix` (any memory order; no cast or copy is made).
        masses: Atomic masses in amu, shape $(N,)$, in the order of the rows, e.g. ``minimised.get_masses()``.
        charges: Atomic charges in units of the elementary charge, in the order of the rows: shape $(N,)$ for scalar
            charges or $(N, 3, 3)$ for charge tensors $Z_{i,\alpha\beta}$.
        groups: Atom groups as ``{grouping: {label: indices}}`` with 0-based positions, e.g. from
            :func:`classify_vibrational_groups`.

    Returns:
        A dictionary with:
            ``"mode_charges"``: The $(K, 3)$ mode effective charges $\tilde Z_{m,\alpha}$ in $e/\sqrt{\mathrm{amu}}$.
            ``"strengths"``: The $(K,)$ isotropic oscillator strengths $f_m$ in $e^2/\mathrm{amu}$.
            ``"partial_mode_charges"``: ``{grouping: {label: (K, 3) array}}``, only with ``groups``.
            ``"partial_strengths"``: ``{grouping: {"A-B": (K,) array}}`` for every unordered pair of labels, in the
                order of the labels, only with ``groups``.

    Raises:
        ValueError: If ``eigenvectors`` is not a 2-D float64 array, its row count is not a positive multiple of 3, it
            has no columns or it contains NaN or inf; if ``masses`` does not hold $N$ positive finite values; if
            ``charges`` is not finite with shape $(N,)$ or $(N, 3, 3)$; if ``groups`` has no labels or a label has
            invalid positions.

    Warns:
        UserWarning: If the charges do not sum to zero over the atoms.

    Example:
        ```pycon
        >>> modes = diagonalize_dynamical_matrix(matrix, masses=minimised.get_masses())  # doctest: +SKIP
        >>> infrared = compute_ir_oscillator_strengths(  # doctest: +SKIP
        ...     modes["eigenvectors"], minimised.get_masses(), charges
        ... )
        >>> infrared["strengths"]  # doctest: +SKIP

        ```
    """
    eigenvectors, masses, n_atoms = _validate_modes(eigenvectors, masses)
    n_rows, n_modes = eigenvectors.shape
    charges = _validate_charges(charges, n_atoms)

    # Column 0 sums over every atom (the total); with groups, one column per group label follows.
    membership = np.ones((n_atoms, 1))
    keys: list[tuple[str, str]] = []
    if groups is not None:
        keys, selections = _group_selections(groups)
        membership = np.hstack([membership, _membership(selections, n_atoms)])
    scalar = charges.ndim == _SCALAR_CHARGE_NDIM
    # Charge per unit mass-weighted displacement: Z_i / sqrt(m_i).
    weights = charges / np.sqrt(masses).reshape(-1, *([1] * (charges.ndim - 1)))
    if scalar:
        membership = membership * weights[:, None]

    mode_charges = np.empty((membership.shape[1], _N_COMPONENTS, n_modes))
    step = max(1, _common._CHUNK_BYTES // (n_rows * eigenvectors.itemsize))  # noqa: SLF001
    for start in range(0, n_modes, step):
        columns = slice(start, min(start + step, n_modes))
        block = eigenvectors[:, columns].reshape(n_atoms, _N_COMPONENTS, -1)
        if not scalar:
            block = np.einsum("iab,ibk->iak", weights, block)
        mode_charges[:, :, columns] = np.einsum("ig,iak->gak", membership, block)
    if not np.all(np.isfinite(mode_charges)):
        msg = "eigenvectors contain NaN or inf."
        raise ValueError(msg)
    mode_charges = np.ascontiguousarray(mode_charges.transpose(0, 2, 1))  # (columns, modes, 3)

    result: dict[str, Any] = {
        "mode_charges": mode_charges[0],
        "strengths": np.einsum("ka,ka->k", mode_charges[0], mode_charges[0]) / _N_COMPONENTS,
    }
    if groups is not None:
        column_of = {key: column for column, key in enumerate(keys, start=1)}
        partial_charges: dict[str, dict[str, NDArray[np.float64]]] = {}
        partial_strengths: dict[str, dict[str, NDArray[np.float64]]] = {}
        for grouping, labels in groups.items():
            names = list(labels)
            partial_charges[grouping] = {name: mode_charges[column_of[grouping, name]] for name in names}
            partial_strengths[grouping] = {
                f"{first}-{second}": np.einsum(
                    "ka,ka->k", partial_charges[grouping][first], partial_charges[grouping][second]
                )
                * ((1 if first == second else 2) / _N_COMPONENTS)
                for position, first in enumerate(names)
                for second in names[position:]
            }
        result["partial_mode_charges"] = partial_charges
        result["partial_strengths"] = partial_strengths
    return result


def _validate_positive(name: str, value: float) -> None:
    """Raise unless ``value`` is a positive finite real number."""
    if not _is_real_number(value) or not np.isfinite(value) or value <= 0:
        msg = f"{name} must be positive and finite, got {value!r}."
        raise ValueError(msg)


def compute_dielectric_function(
    frequencies_thz: ArrayLike,
    strengths: ArrayLike,
    volume_a3: float,
    *,
    frequency_grid_thz: ArrayLike | None = None,
    eps_inf: float = 1.0,
    damping_thz: float = 0.5,
) -> dict[str, Any]:
    r"""Compute the isotropic dielectric function of damped harmonic modes from their oscillator strengths.

    Every mode of frequency $\nu_m$ and oscillator strength $f_m$ (:func:`compute_ir_oscillator_strengths`) responds
    to a field as a damped oscillator, and the modes of a cell of volume $V$ add up to

    $$\varepsilon(\nu) = \varepsilon_\infty + \frac{e^2}{4\pi^2 \varepsilon_0 V}
    \sum_m \frac{f_m}{\nu_m^2 - \nu^2 - i \gamma \nu},$$

    with the time dependence $e^{-i 2\pi\nu t}$, so that absorption is $\operatorname{Im} \varepsilon \ge 0$. The
    prefactor is $44224.5\ \mathrm{THz^2\,Å^3\,amu}/e^2$. The peaks of $\operatorname{Im} \varepsilon$ are the
    transverse optical modes; those of the loss function of :func:`compute_optical_response` the longitudinal ones.

    $\varepsilon_\infty$ is the electronic contribution, which a model of rigid ions does not have: 1 leaves it out, and
    the square of the optical refractive index (about 2.1 for silica) adds the measured one. The damping $\gamma$ is
    the full width at half maximum of the peaks of $\operatorname{Im} \varepsilon$; in a harmonic calculation it is a
    numerical broadening that joins the discrete modes of a finite cell into a spectrum, not a physical linewidth.

    Modes with $\nu_m \le 0$ (imaginary modes, reported as negative frequencies) are left out and counted.

    Args:
        frequencies_thz: The mode frequencies in THz, shape $(M,)$, e.g. ``"frequencies_thz"`` of
            :func:`diagonalize_dynamical_matrix`.
        strengths: The oscillator strengths in $e^2/\mathrm{amu}$, shape $(M,)$, e.g. ``"strengths"`` of
            :func:`compute_ir_oscillator_strengths`. They may be negative: a pair strength of its
            ``"partial_strengths"`` gives that pair's share of $\varepsilon - \varepsilon_\infty$ once ``eps_inf``
            is subtracted from the result.
        volume_a3: The volume of the cell in Å$^3$.
        frequency_grid_thz: The frequencies in THz at which to evaluate, 1-D and strictly increasing. Defaults to
            a grid from 0 to the highest mode plus $5\gamma$ in steps of $\gamma / 5$.
        eps_inf: The high-frequency dielectric constant $\varepsilon_\infty$.
        damping_thz: The damping $\gamma$ in THz.

    Returns:
        A dictionary with:
            ``"frequencies_thz"``: The grid in THz.
            ``"epsilon"``: The complex dielectric function on the grid.
            ``"epsilon_static"``: The undamped static limit
                $\varepsilon_\infty + (e^2 / 4\pi^2\varepsilon_0 V) \sum_m f_m / \nu_m^2$.
            ``"n_modes_excluded"``: The number of modes with $\nu_m \le 0$ left out.
            ``"eps_inf"``, ``"damping_thz"``: The values used.

    Raises:
        ValueError: If ``frequencies_thz`` is not a non-empty finite 1-D array, ``strengths`` is not finite with the
            same shape, ``volume_a3``, ``eps_inf`` or ``damping_thz`` is not positive and finite, or the grid is not
            1-D, finite and strictly increasing with at least 2 points.

    Example:
        ```pycon
        >>> infrared = compute_ir_oscillator_strengths(  # doctest: +SKIP
        ...     modes["eigenvectors"], minimised.get_masses(), charges
        ... )
        >>> dielectric = compute_dielectric_function(  # doctest: +SKIP
        ...     modes["frequencies_thz"], infrared["strengths"], minimised.cell.volume, eps_inf=2.1
        ... )
        >>> dielectric["epsilon"].imag  # doctest: +SKIP

        ```
    """
    frequencies = np.asarray(frequencies_thz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0 or not np.all(np.isfinite(frequencies)):
        msg = f"frequencies_thz must be a non-empty 1-D array of finite values, got shape {frequencies.shape}."
        raise ValueError(msg)
    strengths = np.asarray(strengths, dtype=np.float64)
    if strengths.shape != frequencies.shape or not np.all(np.isfinite(strengths)):
        msg = f"strengths must hold {frequencies.size} finite values, one per mode, got shape {strengths.shape}."
        raise ValueError(msg)
    _validate_positive("volume_a3", volume_a3)
    _validate_positive("eps_inf", eps_inf)
    _validate_positive("damping_thz", damping_thz)
    if frequency_grid_thz is None:
        grid = _harmonic_grid(None, frequencies, damping_thz)
    else:
        grid = _validate_frequency_grid(frequency_grid_thz)

    kept = frequencies > 0
    squares, kept_strengths = frequencies[kept] ** 2, strengths[kept]
    scale = _IR_THZ2_A3 / volume_a3
    shift = grid**2 + 1j * damping_thz * grid
    epsilon = np.full(grid.size, eps_inf, dtype=np.complex128)
    step = max(1, _common._CHUNK_BYTES // (grid.size * epsilon.itemsize))  # noqa: SLF001
    for start in range(0, squares.size, step):
        block = slice(start, start + step)
        epsilon += scale * ((1 / (squares[None, block] - shift[:, None])) @ kept_strengths[block])
    return {
        "frequencies_thz": grid,
        "epsilon": epsilon,
        "epsilon_static": float(eps_inf + scale * np.sum(kept_strengths / squares)),
        "n_modes_excluded": int(frequencies.size - np.count_nonzero(kept)),
        "eps_inf": float(eps_inf),
        "damping_thz": float(damping_thz),
    }


def compute_optical_response(frequencies_thz: ArrayLike, epsilon: ArrayLike) -> dict[str, Any]:
    r"""Compute the optical constants, absorption coefficient, reflectivity and loss function from $\varepsilon(\nu)$.

    The complex refractive index is $n + i k = \sqrt{\varepsilon}$ (the root with $n \ge 0$). It gives the
    absorption coefficient of the intensity, $I(x) = I_0 e^{-\alpha x}$, the reflectivity of a thick sample at normal
    incidence in vacuum, and the energy loss function, whose peaks mark the longitudinal optical modes:

    $$\alpha = 4\pi \tilde\nu k, \qquad
    R = \left|\frac{\sqrt{\varepsilon} - 1}{\sqrt{\varepsilon} + 1}\right|^2, \qquad
    L = \operatorname{Im}\left(-\frac{1}{\varepsilon}\right),$$

    with the wavenumber $\tilde\nu = \nu / c$ in cm$^{-1}$. NaN in ``epsilon`` (the zero-frequency bin of
    :func:`compute_dielectric_from_velocities`) gives NaN at the same points.

    Args:
        frequencies_thz: The frequencies in THz, 1-D, finite and non-negative.
        epsilon: The complex dielectric function at these frequencies, e.g. ``"epsilon"`` of
            :func:`compute_dielectric_function`.

    Returns:
        A dictionary with one array per frequency:
            ``"refractive_index"``: $n$.
            ``"extinction"``: $k$.
            ``"absorption_cm"``: $\alpha$ in cm$^{-1}$.
            ``"reflectivity"``: $R$.
            ``"loss_function"``: $L$.

    Raises:
        ValueError: If ``frequencies_thz`` is not a non-empty 1-D array of finite non-negative values, or
            ``epsilon`` does not have the same shape or contains inf.

    Example:
        ```pycon
        >>> optics = compute_optical_response(dielectric["frequencies_thz"], dielectric["epsilon"])  # doctest: +SKIP
        >>> optics["reflectivity"]  # doctest: +SKIP

        ```
    """
    frequencies = np.asarray(frequencies_thz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0 or not np.all(np.isfinite(frequencies) & (frequencies >= 0)):
        msg = (
            "frequencies_thz must be a non-empty 1-D array of finite non-negative values, "
            f"got shape {frequencies.shape}."
        )
        raise ValueError(msg)
    epsilon = np.asarray(epsilon, dtype=np.complex128)
    if epsilon.shape != frequencies.shape or np.any(np.isinf(epsilon)):
        msg = f"epsilon must hold {frequencies.size} values without inf, one per frequency, got shape {epsilon.shape}."
        raise ValueError(msg)
    root = np.sqrt(epsilon)
    with np.errstate(invalid="ignore"):  # NaN in epsilon stays NaN
        reflectivity = np.abs((root - 1) / (root + 1)) ** 2
        loss_function = (-1 / epsilon).imag
    return {
        "refractive_index": root.real,
        "extinction": root.imag,
        "absorption_cm": 4 * math.pi * convert_frequency(frequencies, "THz", "cm-1") * root.imag,
        "reflectivity": reflectivity,
        "loss_function": loss_function,
    }


def _charge_currents(
    velocities: NDArray,
    charges: NDArray[np.float64],
    com_velocity: NDArray[np.float64] | None,
    membership: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Charge current of every membership column per frame, sum_i Z_i (v_i - v_com): shape (n_frames, n_cols, 3)."""
    n_frames, n_atoms, _ = velocities.shape
    scalar = charges.ndim == _SCALAR_CHARGE_NDIM
    currents = np.zeros((n_frames, membership.shape[1], _N_COMPONENTS))
    chunk = _atoms_per_chunk(n_frames, velocities.itemsize)
    for start in range(0, n_atoms, chunk):
        atoms = slice(start, start + chunk)
        if scalar:
            currents += np.einsum(
                "tia,ig->tga", velocities[:, atoms], membership[atoms] * charges[atoms, None], dtype=np.float64
            )
        else:
            currents += np.einsum(
                "tib,iab,ig->tga", velocities[:, atoms], charges[atoms], membership[atoms], dtype=np.float64
            )
    if com_velocity is not None:
        tensors = charges[:, None, None] * np.eye(_N_COMPONENTS) if scalar else charges
        currents -= np.einsum("tb,gab->tga", com_velocity, np.einsum("ig,iab->gab", membership, tensors))
    return currents


def compute_dielectric_from_velocities(
    velocities: NDArray,
    masses: NDArray,
    charges: ArrayLike,
    dt_fs: float,
    volume_a3: float,
    *,
    groups: dict[str, dict[str, NDArray]] | None = None,
    eps_inf: float = 1.0,
    window: VdosWindow = "hann",
    n_lag_frames: int | None = None,
    remove_com: bool = True,
) -> dict[str, Any]:
    r"""Compute the isotropic dielectric function from the charge-current fluctuations of an MD trajectory.

    The charges $Z_i$ moving with the velocities $v_i$ carry the current $J(t) = \sum_i Z_i\, v_i(t)$, the time
    derivative of the dipole moment of the cell (which, unlike the dipole itself, is well defined in a periodic
    box). Linear response gives the conductivity of the cell of volume $V$ from the equilibrium fluctuations of $J$,

    $$\sigma(\nu) = \frac{1}{3 V k_B T} \int_0^\infty \langle J(0) \cdot J(t) \rangle\, e^{i 2\pi\nu t}\, dt,
    \qquad \varepsilon(\nu) = \varepsilon_\infty + \frac{i\, \sigma(\nu)}{2\pi\nu\, \varepsilon_0},$$

    with the same sign convention as :func:`compute_dielectric_function` ($\operatorname{Im} \varepsilon \ge 0$ for
    absorption). The transform is one-sided, so the real and imaginary parts obey the Kramers-Kronig relations.

    The temperature is taken from the trajectory itself, $k_B T = \langle \sum_i m_i |v_i|^2 \rangle / n_\mathrm{dof}$
    with $n_\mathrm{dof} = 3N - 3$ when the centre-of-mass velocity is removed and $3N$ otherwise, so the unit of
    the velocities cancels and no temperature is passed. For a harmonic solid at equipartition the result is that of
    :func:`compute_dielectric_function` with the strengths of :func:`compute_ir_oscillator_strengths`: each mode
    contributes $f_m$ to $\langle J \cdot J \rangle / 3 k_B T$, and because the response of a harmonic oscillator is
    the same in classical and quantum mechanics, no quantum correction applies. Anharmonicity enters through the
    dynamics, as the temperature-dependent position and width of the peaks.

    The correlation function is the unbiased estimate over all time origins for the lags
    $0 \ldots n_\mathrm{lag} - 1$. It is multiplied by the decaying half of ``window`` (1 at zero lag), which
    smooths the spectrum over about $1 / (n_\mathrm{lag}\, \Delta t)$, and transformed on the grid
    ``np.fft.rfftfreq(2 * n_lag_frames, dt_fs * 1e-3)``: the cosine part with the trapezoid rule over the frames,
    the sine part over the band-limited interpolation on a 16 times finer time grid, without which
    $\operatorname{Re} \varepsilon$ below a mode at $\nu_0$ would be too large by about
    $(2\pi\nu_0 \Delta t)^2 / 12$. The integral of $\operatorname{Re} \sigma$ is fixed by the zero-lag value alone
    (the f-sum rule), whatever the window. A window with negative side lobes, and above all ``None``, can make
    $\operatorname{Im} \varepsilon$ slightly negative next to a sharp peak.

    At $\nu = 0$ the dielectric function is NaN: there $\sigma$ is the DC ionic conductivity, which is finite when
    modifier ions diffuse during the run, and $\varepsilon$ diverges.

    With ``groups`` the current of every group $G$ gives, for each pair of labels of one grouping, the part of
    $\varepsilon - \varepsilon_\infty$ that comes from $\langle J_G(0) \cdot J_{G'}(t) \rangle$ and
    $\langle J_{G'}(0) \cdot J_G(t) \rangle$. The cross terms can be negative. Over the unordered pairs of a
    grouping that holds every atom once they sum to $\varepsilon - \varepsilon_\infty$.

    Args:
        velocities: Velocities, shape $(n_\mathrm{frames}, N, 3)$, in any unit, sampled every ``dt_fs``.
        masses: Atomic masses, shape $(N,)$, in the order of the atoms.
        charges: Atomic charges in units of the elementary charge: shape $(N,)$ for scalar charges or
            $(N, 3, 3)$ for charge tensors $Z_{i,\alpha\beta}$.
        dt_fs: Time between frames in fs.
        volume_a3: The volume of the cell in Å$^3$.
        groups: Atom groups as ``{grouping: {label: indices}}`` with 0-based positions, e.g. from
            :func:`classify_vibrational_groups`.
        eps_inf: The high-frequency dielectric constant $\varepsilon_\infty$; see
            :func:`compute_dielectric_function`.
        window: The window whose decaying half multiplies the correlation function: ``"hann"``, ``"hamming"``,
            ``"blackmanharris"``, ``("gaussian", std)`` with ``std`` in frames, or ``None`` for a sharp cut.
        n_lag_frames: The number of lags kept. Defaults to ``n_frames // 2``.
        remove_com: Whether to subtract the centre-of-mass velocity of every frame.

    Returns:
        A dictionary with:
            ``"frequencies_thz"``: The $n_\mathrm{lag} + 1$ frequencies in THz.
            ``"epsilon"``: The complex dielectric function, NaN at zero frequency.
            ``"conductivity_s_per_m"``: The complex conductivity $\sigma(\nu)$ in S/m.
            ``"partial_susceptibility"``: ``{grouping: {"A-B": complex array}}`` for every unordered pair of
                labels, in the order of the labels, only with ``groups``.
            ``"n_lag_frames"``, ``"eps_inf"``: The values used.

    Raises:
        ValueError: If the trajectory, masses or ``dt_fs`` are invalid (see :func:`compute_vdos_from_velocities`),
            the velocities are all zero, ``charges`` is not finite with shape $(N,)$ or $(N, 3, 3)$,
            ``volume_a3`` or ``eps_inf`` is not positive and finite, ``window`` is unknown, ``n_lag_frames`` is
            not an integer in $[2, n_\mathrm{frames} - 1]$, or ``groups`` has no labels or invalid positions.

    Warns:
        UserWarning: If the charges do not sum to zero over the atoms.

    Example:
        ```pycon
        >>> dielectric = compute_dielectric_from_velocities(  # doctest: +SKIP
        ...     velocities, structure.get_masses(), charges, dt_fs=5.0, volume_a3=structure.cell.volume
        ... )
        >>> dielectric["frequencies_thz"], dielectric["epsilon"].imag  # doctest: +SKIP

        ```
    """
    velocities, masses = _validate_trajectory(velocities, masses, dt_fs)
    n_frames, n_atoms, _ = velocities.shape
    charges = _validate_charges(charges, n_atoms)
    _validate_positive("volume_a3", volume_a3)
    _validate_positive("eps_inf", eps_inf)
    _validate_window(window)
    if n_lag_frames is None:
        n_lag_frames = n_frames // 2
    if isinstance(n_lag_frames, bool) or not (
        isinstance(n_lag_frames, int | np.integer) and _MIN_LAG_FRAMES <= n_lag_frames <= n_frames - 1
    ):
        msg = f"n_lag_frames must be an integer in [{_MIN_LAG_FRAMES}, {n_frames - 1}], got {n_lag_frames!r}."
        raise ValueError(msg)

    # Column 0 sums over every atom (the total); with groups, one column per group label follows.
    membership = np.ones((n_atoms, 1))
    keys: list[tuple[str, str]] = []
    if groups is not None:
        keys, selections = _group_selections(groups)
        membership = np.hstack([membership, _membership(selections, n_atoms)])
    com_velocity, _, kinetic_sum = _trajectory_statistics(velocities, masses, None, remove_com=remove_com)
    thermal = kinetic_sum / (n_frames * _N_COMPONENTS * (n_atoms - 1 if remove_com else n_atoms))  # k_B T
    currents = _charge_currents(velocities, charges, com_velocity, membership)

    # Zero padding to 2N makes the correlations linear, as in the VACF.
    spectrum = scipy.fft.rfft(currents, n=2 * n_frames, axis=0)
    dt_ps = dt_fs * 1e-3
    frequencies = np.fft.rfftfreq(2 * n_lag_frames, d=dt_ps)
    origins = n_frames - np.arange(n_lag_frames)
    taper = 1.0 if window is None else scipy.signal.get_window(window, 2 * n_lag_frames)[n_lag_frames:]

    def response(first: int, second: int) -> NDArray[np.complex128]:
        """One-sided transform of <J_first(0) . J_second(t)> + <J_second(0) . J_first(t)>, halved, over 3 k_B T."""
        power = np.einsum("fa,fa->f", spectrum[:, first].real, spectrum[:, second].real)
        power += np.einsum("fa,fa->f", spectrum[:, first].imag, spectrum[:, second].imag)
        correlation = scipy.fft.irfft(power, n=2 * n_frames)[:n_lag_frames] / origins
        correlation *= taper / (_N_COMPONENTS * thermal)
        # Real part: half the spectrum of the even extension, exact for the sampled correlation (trapezoid rule).
        even = np.concatenate([correlation, [0.0], correlation[:0:-1]])
        real = 0.5 * dt_ps * scipy.fft.rfft(even).real
        # Imaginary part: the sine transform of the band-limited interpolation on a finer time grid. On the frames
        # themselves the sum would replace 1 / x by (dt / 2) cot(x dt / 2) and overestimate the response below a
        # mode at nu0 by a factor of about 1 + (2 pi nu0 dt)^2 / 12.
        fine = scipy.signal.resample(even, even.size * _LAG_OVERSAMPLING)[: n_lag_frames * _LAG_OVERSAMPLING]
        imaginary = -scipy.fft.rfft(fine, n=even.size * _LAG_OVERSAMPLING).imag[: n_lag_frames + 1]
        return real + 1j * (dt_ps / _LAG_OVERSAMPLING) * imaginary

    def susceptibility(current_response: NDArray[np.complex128]) -> NDArray[np.complex128]:
        with np.errstate(divide="ignore", invalid="ignore"):
            chi = (2j * math.pi * _IR_THZ2_A3 / volume_a3) * current_response / frequencies
        chi[0] = np.nan
        return chi

    total = response(0, 0)
    result: dict[str, Any] = {
        "frequencies_thz": frequencies,
        "epsilon": eps_inf + susceptibility(total),
        "conductivity_s_per_m": total * (_S_PER_M_A3 / volume_a3),
    }
    if groups is not None:
        column_of = {key: column for column, key in enumerate(keys, start=1)}
        result["partial_susceptibility"] = {
            grouping: {
                f"{first}-{second}": susceptibility(
                    response(column_of[grouping, first], column_of[grouping, second]) * (1 if first == second else 2)
                )
                for position, first in enumerate(labels)
                for second in list(labels)[position:]
            }
            for grouping, labels in groups.items()
        }
    result.update({"n_lag_frames": int(n_lag_frames), "eps_inf": float(eps_inf)})
    return result


def _infrared_settings(
    charges: ArrayLike | None, eps_inf: float, potential: pd.DataFrame, structure: "Atoms"
) -> dict[str, Any]:
    """Validate the infrared arguments of a workflow; return the charges (from the potential if None) and eps_inf."""
    _validate_positive("eps_inf", eps_inf)
    if charges is None:
        charges = charges_from_potential(potential, structure)
    return {"charges": _validate_charges(charges, len(structure)), "eps_inf": float(eps_inf)}


def _harmonic_infrared(
    frequencies_thz: NDArray[np.float64],
    eigenvectors: NDArray[np.float64],
    masses: NDArray[np.float64],
    groups: dict[str, dict[str, NDArray]],
    volume_a3: float,
    frequency_grid_thz: NDArray[np.float64],
    damping_thz: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Infrared entry of :func:`harmonic_vdos_simulation`: strengths, dielectric function and optical response."""
    charges, eps_inf = settings["charges"], settings["eps_inf"]
    strengths = compute_ir_oscillator_strengths(eigenvectors, masses, charges, groups=groups)

    def dielectric(mode_strengths: NDArray[np.float64]) -> dict[str, Any]:
        return compute_dielectric_function(
            frequencies_thz,
            mode_strengths,
            volume_a3,
            frequency_grid_thz=frequency_grid_thz,
            eps_inf=eps_inf,
            damping_thz=damping_thz,
        )

    total = dielectric(strengths["strengths"])
    return {
        "frequencies_thz": total["frequencies_thz"],
        "mode_charges": strengths["mode_charges"],
        "strengths": strengths["strengths"],
        "partial_strengths": strengths["partial_strengths"],
        "epsilon": total["epsilon"],
        "epsilon_static": total["epsilon_static"],
        "partial_susceptibility": {
            grouping: {pair: dielectric(values)["epsilon"] - eps_inf for pair, values in pairs.items()}
            for grouping, pairs in strengths["partial_strengths"].items()
        },
        **compute_optical_response(total["frequencies_thz"], total["epsilon"]),
        "n_modes_excluded": total["n_modes_excluded"],
        "charges": charges,
        "volume_a3": float(volume_a3),
        "eps_inf": eps_inf,
        "damping_thz": total["damping_thz"],
    }


def _real_imag(values: Any) -> Any:  # noqa: ANN401
    """Split every complex array of nested dicts into ``{"real": ..., "imag": ...}``, for block statistics."""
    if isinstance(values, dict):
        return {key: _real_imag(value) for key, value in values.items()}
    return {"real": np.ascontiguousarray(values.real), "imag": np.ascontiguousarray(values.imag)}


def _complex(values: dict[str, Any]) -> Any:  # noqa: ANN401
    """Inverse of :func:`_real_imag`."""
    if set(values) == {"real", "imag"} and not isinstance(values["real"], dict):
        return values["real"] + 1j * values["imag"]
    return {key: _complex(value) for key, value in values.items()}


def _block_infrared(
    velocities: NDArray,
    masses: NDArray[np.float64],
    dt_fs: float,
    window: VdosWindow,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Infrared curves of one NVE block of :func:`vdos_simulation`, complex arrays split into real and imaginary."""
    dielectric = compute_dielectric_from_velocities(
        velocities,
        masses,
        settings["charges"],
        dt_fs,
        settings["volume_a3"],
        groups=settings["groups"],
        eps_inf=settings["eps_inf"],
        window=window,
    )
    curves = {key: dielectric[key] for key in ("epsilon", "conductivity_s_per_m", "partial_susceptibility")}
    return {
        "frequencies_thz": dielectric["frequencies_thz"],
        "n_lag_frames": dielectric["n_lag_frames"],
        "curves": _real_imag(curves),
    }


def _averaged_infrared(
    blocks: list[dict[str, Any]], mean: dict[str, Any], sem: dict[str, Any], settings: dict[str, Any]
) -> dict[str, Any]:
    """Infrared entry of :func:`vdos_simulation` from the block mean and standard error of the split curves."""
    mean, sem = _complex(mean), _complex(sem)
    frequencies = blocks[0]["frequencies_thz"]
    return {
        "frequencies_thz": frequencies,
        "epsilon": mean["epsilon"],
        "epsilon_sem": sem["epsilon"],
        "conductivity_s_per_m": mean["conductivity_s_per_m"],
        "conductivity_sem_s_per_m": sem["conductivity_s_per_m"],
        "partial_susceptibility": mean["partial_susceptibility"],
        "partial_susceptibility_sem": sem["partial_susceptibility"],
        **compute_optical_response(frequencies, mean["epsilon"]),
        "n_lag_frames": blocks[0]["n_lag_frames"],
        "charges": settings["charges"],
        "volume_a3": settings["volume_a3"],
        "eps_inf": settings["eps_inf"],
    }
