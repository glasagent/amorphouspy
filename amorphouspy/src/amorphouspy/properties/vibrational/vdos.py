"""Vibrational density of states from velocity trajectories, total and per group."""

from typing import Literal

import numpy as np
import scipy.signal
from numpy.typing import NDArray

from amorphouspy.properties.vibrational._common import (
    VdosNormalization,
    _columns_by_group,
    _group_selections,
    _membership,
)
from amorphouspy.properties.vibrational._trajectory import (
    _accumulate_power,
    _trajectory_statistics,
    _validate_trajectory,
)

VdosWindow = Literal["hann", "hamming", "blackmanharris"] | tuple[Literal["gaussian"], float] | None
_WINDOWS = ("hann", "hamming", "blackmanharris")
_GAUSSIAN_LEN = 2  # ("gaussian", std)


def _validate_window(window: object) -> None:
    gaussian = (
        isinstance(window, tuple)
        and len(window) == _GAUSSIAN_LEN
        and window[0] == "gaussian"
        and isinstance(window[1], int | float | np.integer | np.floating)
        and not isinstance(window[1], bool)
        and np.isfinite(window[1])
        and window[1] > 0
    )
    if not (gaussian or window is None or (isinstance(window, str) and window in _WINDOWS)):
        msg = (
            f"Unknown window {window!r}; allowed values are 'hann', 'hamming', 'blackmanharris', "
            "('gaussian', std) with std > 0 in frames, and None."
        )
        raise ValueError(msg)


def _validate_vdos_inputs(
    velocities: NDArray, masses: NDArray, dt_fs: float, normalization: str, window: object
) -> tuple[NDArray, NDArray[np.float64]]:
    velocities, masses = _validate_trajectory(velocities, masses, dt_fs)
    if normalization not in ("unit", "3N", "none"):
        msg = f"Unknown normalization {normalization!r}; allowed values are 'unit', '3N' and 'none'."
        raise ValueError(msg)
    _validate_window(window)
    return velocities, masses


def _vdos_columns(
    velocities: NDArray,
    masses: NDArray,
    dt_fs: float,
    selections: dict[str, NDArray] | None,
    normalization: str,
    *,
    remove_com: bool,
    window: VdosWindow,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Shared implementation: VDOS of the whole system (``selections=None``) or of each named selection, as columns."""
    velocities, masses = _validate_vdos_inputs(velocities, masses, dt_fs, normalization, window)
    n_frames, n_atoms, _ = velocities.shape
    membership = _membership(selections, n_atoms)
    # get_window builds the periodic (DFT-even) form of each window.
    weights = None if window is None else np.asarray(scipy.signal.get_window(window, n_frames), dtype=np.float64)
    com_velocity, parseval_sum, _ = _trajectory_statistics(velocities, masses, weights, remove_com=remove_com)
    power = _accumulate_power(velocities, masses, com_velocity, membership, weights)

    dt_ps = dt_fs * 1e-3
    frequencies_thz = np.asarray(np.fft.rfftfreq(n_frames, d=dt_ps), dtype=np.float64)
    if normalization == "none":
        if weights is None:
            return frequencies_thz, power * (2 * dt_ps / n_frames)
        return frequencies_thz, power * (2 * dt_ps / float(np.sum(weights**2)))
    if not parseval_sum > 0:
        msg = "The windowed velocity spectrum is zero (the window removes all motion)."
        raise ValueError(msg)
    reference_integral = 0.5 * float(frequencies_thz[1]) * parseval_sum
    scale = 1.0 if normalization == "unit" else 3.0 * n_atoms
    return frequencies_thz, power * (scale / reference_integral)


def compute_vdos_from_velocities(
    velocities: NDArray,
    masses: NDArray,
    dt_fs: float,
    *,
    indices: NDArray | None = None,
    normalization: VdosNormalization = "unit",
    remove_com: bool = True,
    window: VdosWindow = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Compute the vibrational density of states from an MD velocity trajectory.

    The VDOS is the mass-weighted velocity power spectrum

    $$g(\nu_k) \propto \sum_{i} m_i \sum_{\alpha} |V_{i\alpha}(\nu_k)|^2,
    \quad V_{i\alpha}(\nu_k) = \sum_{t=0}^{N-1} v_{i\alpha}(t)\, e^{-2\pi \mathrm{i} k t / N},$$

    the unnormalised forward DFT along time (the ``np.fft.rfft`` convention; computed with ``scipy.fft.rfft``), on the
    one-sided grid
    $\nu_k = k / (N \Delta t)$, $k = 0, \dots, \lfloor N/2 \rfloor$, in THz. The frequency resolution is
    $1 / (N \Delta t)$ and the highest frequency is the Nyquist frequency $1 / (2 \Delta t)$. No zero padding or
    per-atom mean subtraction is applied, and no window unless ``window`` is set.

    With a window, the (centre-of-mass-free) velocities are multiplied by $w_t$, $t = 0, \dots, N-1$, before the
    transform. All windows are periodic (``scipy.signal.get_window(window, N)``):

    - ``"hann"``: $w_t = \tfrac{1}{2}\left(1 - \cos(2\pi t / N)\right)$.
    - ``"hamming"``: $w_t = 0.54 - 0.46 \cos(2\pi t / N)$. Lower first sidelobe than Hann, but the edges are not
      zero (0.08), so leakage far from a peak decays more slowly.
    - ``"blackmanharris"``: the 4-term Blackman-Harris window $w_t = \sum_{j=0}^{3} (-1)^j a_j \cos(2\pi j t / N)$,
      $a = (0.35875, 0.48829, 0.14128, 0.01168)$, with sidelobes below $-92$ dB.
    - ``("gaussian", std)``: $w_t = \exp\left(-\tfrac{1}{2}\left((t - N/2) / \sigma\right)^2\right)$ with
      $\sigma$ = ``std`` in frames, so its width relative to the block depends on $N$. It is truncated at the edges
      (edge value $e^{-8} \approx 3 \times 10^{-4}$ for $\sigma = N/8$). A narrow window ($\sigma \ll N/8$) widens
      every peak and can spread power further than no window at all.

    A window suppresses spectral leakage from the finite trajectory length at the cost of resolution. The effective
    resolution is $\mathrm{ENBW} / (N \Delta t)$ with the equivalent noise bandwidth
    $\mathrm{ENBW} = N \sum_t w_t^2 / (\sum_t w_t)^2$ bins: 1.5 (Hann), 1.36 (Hamming), 2.0 (Blackman-Harris) and
    about $N / (2 \sqrt{\pi} \sigma)$ (Gaussian, $\sigma \ll N$). For 10 ps blocks that is 0.15, 0.14 and 0.20 THz,
    and 0.23 THz for a Gaussian with $\sigma = N/8$. With ``window=None`` (default) the full $1 / (N \Delta t)$
    resolution is kept, and a peak between grid points leaks into distant bins.

    With ``remove_com=True`` (default), the instantaneous mass-weighted centre-of-mass velocity of the whole system,
    $\mathbf{v}_\mathrm{COM}(t) = \sum_i m_i \mathbf{v}_i(t) / \sum_i m_i$, is subtracted from every atom in every
    frame before the transform, also when ``indices`` selects a subset. With ``remove_com=False`` the velocities are
    transformed as given, and any centre-of-mass drift appears at and near zero frequency.

    Normalisation (all integrals are trapezoidal over the returned grid):

    - ``"unit"`` (default): the total VDOS integrates to 1 (units 1/THz).
    - ``"3N"``: the total VDOS integrates to ``3 * n_atoms`` (the number of degrees of freedom, including the 3
      removed with the centre of mass).
    - ``"none"``: the power spectral density $S(\nu_k) = (2 \Delta t / N) \sum_i m_i \sum_\alpha |V_{i\alpha}|^2$ in
      amu (velocity unit)$^2$ ps. For an even number of frames its integral equals the time average of
      $\sum_i m_i |\mathbf{v}_i|^2$ (twice the mean kinetic energy) exactly, by Parseval's theorem; for an odd number
      it is lower by half the last bin. With a window, $V_{i\alpha}$ is the transform of the windowed velocities and
      the prefactor is $2 \Delta t / \sum_t w_t^2$ instead of $2 \Delta t / N$; the integral is then the
      $w^2$-weighted time average $\sum_t w_t^2 \sum_i m_i |\mathbf{v}_i(t)|^2 / \sum_t w_t^2$ (exact for an even
      number of frames).

    With a window, ``"unit"`` and ``"3N"`` are applied to the windowed total VDOS, which still integrates to exactly 1
    and ``3 * n_atoms``.

    A partial VDOS (``indices`` given) is the same spectrum summed over the selected atoms only and is scaled by the
    total VDOS's normalisation, so partials over disjoint selections add up to the total. Its integral is the
    selection's share of the total kinetic energy, which is close to its share of atoms by equipartition: about
    ``3 * len(indices)`` for ``"3N"``. The integral of the total spectrum comes from Parseval's theorem in the time
    domain, so a partial only transforms the selected atoms. Use :func:`compute_partial_vdos` for many groups.

    Memory: the trajectory is read twice in chunks of atoms (about 64 MB each), once for the centre-of-mass velocity,
    the normalisation and the finiteness check, and once for the transforms. Besides the input, peak memory is about
    two chunks. float32 input is transformed in single precision (no float64 copy); the spectrum is accumulated in
    float64. The input is never modified, and it may be a read-only ``np.memmap`` (e.g. ``np.load(path,
    mmap_mode="r")``), in which case only the chunks being processed are read from disk.

    Args:
        velocities: Velocities of shape ``(n_frames, n_atoms, 3)`` in any unit; ``"unit"`` and ``"3N"`` remove the
            scale.
        masses: Atomic masses of shape ``(n_atoms,)`` in amu.
        dt_fs: Time between consecutive frames in fs (the dump interval, not the MD timestep).
        indices: 0-based positions of the atoms in a partial VDOS, e.g. one entry of
            :func:`classify_vibrational_groups`. ``None`` (default) gives the total VDOS.
        normalization: ``"unit"``, ``"3N"`` or ``"none"``, see above.
        remove_com: Subtract the centre-of-mass velocity of the whole system from every frame.
        window: ``"hann"``, ``"hamming"``, ``"blackmanharris"``, ``("gaussian", std)`` with ``std > 0`` in frames,
            or ``None`` (default) for no window, see above.

    Returns:
        A tuple ``(frequencies_thz, vdos)`` of float64 arrays of length ``n_frames // 2 + 1``: the frequencies in THz
        from ``np.fft.rfftfreq(n_frames, d=dt_fs * 1e-3)`` and the VDOS in 1/THz (``"unit"``, ``"3N"``) or the power
        spectral density (``"none"``).

    Raises:
        ValueError: If ``velocities`` is not ``(n_frames, n_atoms, 3)`` with at least one atom, ``masses`` does not
            have shape ``(n_atoms,)``, there are fewer than 2 frames, ``dt_fs`` is not a positive finite number, any
            mass is not positive and finite, ``velocities`` is not real numeric or contains NaN or inf,
            ``normalization`` or ``window`` is unknown, ``indices`` is empty, not 1-D integer, out of range or has
            duplicates, or the spectrum vanishes (all velocities zero, no motion left after centre-of-mass removal,
            e.g. every atom moves with the same velocity, or, with ``"unit"`` or ``"3N"``, no motion left after the
            window).

    Example:
        ```pycon
        >>> from amorphouspy import get_atomic_mass
        >>> dt_fs, n_frames = 5.0, 2000
        >>> t_ps = np.arange(n_frames) * dt_fs * 1e-3
        >>> velocities = np.zeros((n_frames, 2, 3))
        >>> velocities[:, 0, 0] = np.cos(2 * np.pi * 10.0 * t_ps)
        >>> velocities[:, 1, 0] = -velocities[:, 0, 0]
        >>> masses = np.full(2, get_atomic_mass("Si"))
        >>> frequencies_thz, vdos = compute_vdos_from_velocities(velocities, masses, dt_fs)
        >>> float(frequencies_thz[np.argmax(vdos)])
        10.0
        >>> round(float(np.trapezoid(vdos, frequencies_thz)), 6)
        1.0
        >>> _, partial = compute_vdos_from_velocities(
        ...     velocities, masses, dt_fs, indices=np.array([0]), normalization="3N"
        ... )
        >>> round(float(np.trapezoid(partial, frequencies_thz)), 6)
        3.0
        >>> _, vdos_hann = compute_vdos_from_velocities(velocities, masses, dt_fs, window="hann")
        >>> float(frequencies_thz[np.argmax(vdos_hann)]), round(float(np.trapezoid(vdos_hann, frequencies_thz)), 6)
        (10.0, 1.0)

        ```
    """
    selections = None if indices is None else {"indices": indices}
    frequencies_thz, columns = _vdos_columns(
        velocities, masses, dt_fs, selections, normalization, remove_com=remove_com, window=window
    )
    return frequencies_thz, np.ascontiguousarray(columns[:, 0])


def compute_partial_vdos(
    velocities: NDArray,
    masses: NDArray,
    dt_fs: float,
    groups: dict[str, dict[str, NDArray]],
    *,
    normalization: VdosNormalization = "unit",
    remove_com: bool = True,
    window: VdosWindow = None,
) -> tuple[NDArray[np.float64], dict[str, dict[str, NDArray[np.float64]]]]:
    """Compute partial VDOS for many atom groups with one transform per atom.

    Each partial is exactly what :func:`compute_vdos_from_velocities` returns with ``indices`` set to the group
    (same centre-of-mass treatment, window, normalisation and units), but every atom is transformed once, however many
    groups it belongs to. ``groups`` has the layout returned by :func:`classify_vibrational_groups`: classify once, then
    reuse the indices for any number of trajectories or blocks. Within one grouping that covers every atom once (e.g.
    ``"element"``), the partials add up to the total VDOS computed with the same window and normalisation.

    Args:
        velocities: Velocities of shape ``(n_frames, n_atoms, 3)`` in any unit.
        masses: Atomic masses of shape ``(n_atoms,)`` in amu.
        dt_fs: Time between consecutive frames in fs (the dump interval, not the MD timestep).
        groups: ``{grouping: {label: indices}}`` with 0-based atom positions, e.g. the output of
            :func:`classify_vibrational_groups`. Atoms may appear in several groupings.
        normalization: ``"unit"``, ``"3N"`` or ``"none"``, as in :func:`compute_vdos_from_velocities`.
        remove_com: Subtract the centre-of-mass velocity of the whole system from every frame.
        window: ``"hann"``, ``"hamming"``, ``"blackmanharris"``, ``("gaussian", std)`` or ``None`` (default), as in
            :func:`compute_vdos_from_velocities`.

    Returns:
        A tuple ``(frequencies_thz, partial)``: the frequencies in THz and ``{grouping: {label: vdos}}`` with the
        same keys as ``groups``, each a float64 array of length ``n_frames // 2 + 1``.

    Raises:
        ValueError: As :func:`compute_vdos_from_velocities` (including an unknown ``window``), for any group's
            indices, or if ``groups`` has no labels.

    Example:
        ```pycon
        >>> rng = np.random.default_rng(0)
        >>> velocities = rng.normal(size=(100, 3, 3))
        >>> masses = np.array([28.085, 15.999, 15.999])
        >>> groups = {"element": {"Si": np.array([0]), "O": np.array([1, 2])}}
        >>> frequencies_thz, partial = compute_partial_vdos(velocities, masses, 5.0, groups)
        >>> _, total = compute_vdos_from_velocities(velocities, masses, 5.0)
        >>> bool(np.allclose(partial["element"]["Si"] + partial["element"]["O"], total))
        True

        ```
    """
    keys, selections = _group_selections(groups)
    frequencies_thz, columns = _vdos_columns(
        velocities, masses, dt_fs, selections, normalization, remove_com=remove_com, window=window
    )
    return frequencies_thz, _columns_by_group(groups, keys, columns)
