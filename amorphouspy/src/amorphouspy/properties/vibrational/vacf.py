"""Velocity autocorrelation function from velocity trajectories, total and per group."""

from typing import Literal

import numpy as np
import scipy.fft
from numpy.typing import NDArray

from amorphouspy.properties.vibrational._common import _columns_by_group, _group_selections, _membership
from amorphouspy.properties.vibrational._trajectory import (
    _accumulate_power,
    _trajectory_statistics,
    _validate_trajectory,
)

VacfNormalization = Literal["unit", "none"]


def _vacf_columns(
    velocities: NDArray,
    masses: NDArray,
    dt_fs: float,
    selections: dict[str, NDArray] | None,
    max_lag_frames: int | None,
    normalization: str,
    *,
    remove_com: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Shared implementation: VACF of the whole system (``selections=None``) or of each named selection, as columns."""
    velocities, masses = _validate_trajectory(velocities, masses, dt_fs)
    if normalization not in ("unit", "none"):
        msg = f"Unknown normalization {normalization!r}; allowed values are 'unit' and 'none'."
        raise ValueError(msg)
    n_frames, n_atoms, _ = velocities.shape
    if max_lag_frames is None:
        max_lag_frames = n_frames // 2
    elif isinstance(max_lag_frames, bool) or not (
        isinstance(max_lag_frames, int | np.integer) and 1 <= max_lag_frames <= n_frames - 1
    ):
        msg = f"max_lag_frames must be an integer in [1, {n_frames - 1}], got {max_lag_frames!r}."
        raise ValueError(msg)
    membership = _membership(selections, n_atoms)
    com_velocity, _, kinetic_sum = _trajectory_statistics(velocities, masses, None, remove_com=remove_com)
    # Zero padding to 2N makes the correlation linear. The inverse transform is linear, so it is applied once to the
    # mass-weighted power of each column instead of once per atom.
    power = _accumulate_power(velocities, masses, com_velocity, membership, None, n_fft=2 * n_frames)
    n_lags = max_lag_frames + 1
    origins = n_frames - np.arange(n_lags)
    vacf = scipy.fft.irfft(power, n=2 * n_frames, axis=0)[:n_lags] / origins[:, None]
    if normalization == "unit":
        vacf /= kinetic_sum / n_frames
    lag_ps = np.arange(n_lags) * dt_fs * 1e-3
    return lag_ps, vacf


def compute_vacf_from_velocities(
    velocities: NDArray,
    masses: NDArray,
    dt_fs: float,
    *,
    indices: NDArray | None = None,
    max_lag_frames: int | None = None,
    normalization: VacfNormalization = "unit",
    remove_com: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Compute the mass-weighted velocity autocorrelation function (VACF) from an MD velocity trajectory.

    The VACF is

    $$C(t_k) = \frac{\sum_i m_i \langle \mathbf{v}_i(0) \cdot \mathbf{v}_i(t_k) \rangle}
    {\sum_i m_i \langle |\mathbf{v}_i|^2 \rangle}, \quad t_k = k \Delta t,$$

    where $\langle \cdot \rangle$ is the average over all time origins with the unbiased estimator: lag $k$ averages
    the $N - k$ available products,

    $$\sum_i m_i \langle \mathbf{v}_i(0) \cdot \mathbf{v}_i(t_k) \rangle
    = \frac{1}{N - k} \sum_{t=0}^{N-1-k} \sum_i m_i\, \mathbf{v}_i(t) \cdot \mathbf{v}_i(t + k),$$

    for $k = 0, \dots,$ ``max_lag_frames``, with $N$ frames. Fewer origins enter at larger lags, so the statistical
    noise grows with the lag; the default ``max_lag_frames = N // 2`` keeps at least $N/2$ origins at every lag.

    The sums over origins are computed with FFTs (``scipy.fft``). Every velocity component of every atom is
    zero-padded to $2N$ frames before the forward transform, so the correlation is linear, not circular (no
    wrap-around from the end of the trajectory to its start). The squared transforms are weighted by the masses and
    summed over the atoms, and the sum is transformed back once, which is exact because the inverse transform is
    linear. No window is applied: the VACF always uses the unwindowed velocities.

    With ``remove_com=True`` (default), the instantaneous mass-weighted centre-of-mass velocity of the whole system,
    $\mathbf{v}_\mathrm{COM}(t) = \sum_i m_i \mathbf{v}_i(t) / \sum_i m_i$, is subtracted from every atom in every
    frame first, also when ``indices`` selects a subset, exactly as in :func:`compute_vdos_from_velocities`. With
    ``remove_com=False`` the velocities are used as given; a constant centre-of-mass velocity $\mathbf{V}$ then adds
    $\sum_i m_i |\mathbf{V}|^2$ to the unnormalised total VACF at every lag.

    Normalisation:

    - ``"unit"`` (default): divided by the total VACF at zero lag, the time average of
      $\sum_i m_i |\mathbf{v}_i - \mathbf{v}_\mathrm{COM}|^2$ (of $\sum_i m_i |\mathbf{v}_i|^2$ with
      ``remove_com=False``), so the total $C(0) = 1$ to rounding. Dimensionless.
    - ``"none"``: the numerator $\sum_i m_i \langle \mathbf{v}_i(0) \cdot \mathbf{v}_i(t_k) \rangle$ in amu
      (velocity unit)$^2$. Its value at zero lag is the time average of
      $\sum_i m_i |\mathbf{v}_i - \mathbf{v}_\mathrm{COM}|^2$ (twice the mean internal kinetic energy).

    A partial VACF (``indices`` given) sums the numerator over the selected atoms only and, with ``"unit"``, is divided
    by the same denominator as the total, so partials over disjoint selections add up to the total VACF, and a
    partial's $C(0)$ is the selection's share of the centred kinetic sum. The denominator comes from the
    statistics read over all atoms, so a partial only transforms the selected atoms. Use
    :func:`compute_partial_vacf` for many groups.

    Relation to the VDOS: by the Wiener-Khinchin theorem the VDOS is the Fourier transform of the unnormalised VACF.
    For a finite trajectory, the ``"none"`` spectral density of :func:`compute_vdos_from_velocities` (no window, same
    ``remove_com``) on its grid $\nu_j = j / (N \Delta t)$ is exactly

    $$S(\nu_j) = 2 \Delta t \left[C(0) + 2 \sum_{k=1}^{N-1} \left(1 - \frac{k}{N}\right) C(t_k)
    \cos\left(\frac{2 \pi j k}{N}\right)\right],$$

    with $C$ the ``"none"`` VACF up to ``max_lag_frames = N - 1`` and $\Delta t$ in ps: the transform over all lags
    with a triangular weight. The library computes the VDOS directly from the velocity spectrum, not from this
    function; transforming a VACF truncated at a shorter lag gives a lower-resolution spectrum.

    Memory: as in :func:`compute_vdos_from_velocities`, the trajectory is read twice in chunks of atoms, once for the
    centre-of-mass velocity, the normalisation and the finiteness check, and once for the transforms, with chunks
    sized for the padded length $2N$. float32 input is transformed in single precision (no float64 copy); the
    correlation is accumulated in float64. The input is never modified, and it may be a read-only ``np.memmap``.

    Args:
        velocities: Velocities of shape ``(n_frames, n_atoms, 3)`` in any unit; ``"unit"`` removes the scale.
        masses: Atomic masses of shape ``(n_atoms,)`` in amu.
        dt_fs: Time between consecutive frames in fs (the dump interval, not the MD timestep).
        indices: 0-based positions of the atoms in a partial VACF, e.g. one entry of
            :func:`classify_vibrational_groups`. ``None`` (default) gives the total VACF.
        max_lag_frames: Largest lag in frames, from 1 to ``n_frames - 1``. ``None`` (default) means
            ``n_frames // 2``.
        normalization: ``"unit"`` or ``"none"``, see above.
        remove_com: Subtract the centre-of-mass velocity of the whole system from every frame.

    Returns:
        A tuple ``(lag_ps, vacf)`` of float64 arrays of length ``max_lag_frames + 1``: the lags
        ``np.arange(max_lag_frames + 1) * dt_fs * 1e-3`` in ps and the VACF (dimensionless for ``"unit"``, amu
        (velocity unit)$^2$ for ``"none"``).

    Raises:
        ValueError: If ``velocities`` is not ``(n_frames, n_atoms, 3)`` with at least one atom, ``masses`` does not
            have shape ``(n_atoms,)``, there are fewer than 2 frames, ``dt_fs`` is not a positive finite number, any
            mass is not positive and finite, ``velocities`` is not real numeric or contains NaN or inf,
            ``normalization`` is unknown, ``max_lag_frames`` is not an integer from 1 to ``n_frames - 1``,
            ``indices`` is empty, not 1-D integer, out of range or has duplicates, or there is no motion (all
            velocities zero, or none left after centre-of-mass removal).

    Example:
        ```pycon
        >>> from amorphouspy import get_atomic_mass
        >>> dt_fs, n_frames = 5.0, 2000
        >>> t_ps = np.arange(n_frames) * dt_fs * 1e-3
        >>> velocities = np.zeros((n_frames, 2, 3))
        >>> velocities[:, 0, 0] = np.cos(2 * np.pi * 10.0 * t_ps)
        >>> velocities[:, 1, 0] = -velocities[:, 0, 0]
        >>> masses = np.full(2, get_atomic_mass("Si"))
        >>> lag_ps, vacf = compute_vacf_from_velocities(velocities, masses, dt_fs)
        >>> lag_ps.shape, round(float(lag_ps[10]), 6)
        ((1001,), 0.05)
        >>> round(float(vacf[0]), 6), round(float(vacf[10]), 6)  # half a 10 THz period after 0.05 ps
        (1.0, -1.0)
        >>> _, partial = compute_vacf_from_velocities(velocities, masses, dt_fs, indices=np.array([0]))
        >>> round(float(partial[0]), 6)
        0.5

        ```
    """
    selections = None if indices is None else {"indices": indices}
    lag_ps, columns = _vacf_columns(
        velocities, masses, dt_fs, selections, max_lag_frames, normalization, remove_com=remove_com
    )
    return lag_ps, np.ascontiguousarray(columns[:, 0])


def compute_partial_vacf(
    velocities: NDArray,
    masses: NDArray,
    dt_fs: float,
    groups: dict[str, dict[str, NDArray]],
    *,
    max_lag_frames: int | None = None,
    normalization: VacfNormalization = "unit",
    remove_com: bool = True,
) -> tuple[NDArray[np.float64], dict[str, dict[str, NDArray[np.float64]]]]:
    """Compute partial VACFs for many atom groups with one transform per atom.

    Each partial is exactly what :func:`compute_vacf_from_velocities` returns with ``indices`` set to the group
    (same centre-of-mass treatment, lags, normalisation and units), but every atom is transformed once, however many
    groups it belongs to. ``groups`` has the layout returned by :func:`classify_vibrational_groups`. Within one
    grouping that covers every atom once (e.g. ``"element"``), the partials add up to the total VACF.

    Args:
        velocities: Velocities of shape ``(n_frames, n_atoms, 3)`` in any unit.
        masses: Atomic masses of shape ``(n_atoms,)`` in amu.
        dt_fs: Time between consecutive frames in fs (the dump interval, not the MD timestep).
        groups: ``{grouping: {label: indices}}`` with 0-based atom positions, e.g. the output of
            :func:`classify_vibrational_groups`. Atoms may appear in several groupings.
        max_lag_frames: Largest lag in frames, as in :func:`compute_vacf_from_velocities`.
        normalization: ``"unit"`` or ``"none"``, as in :func:`compute_vacf_from_velocities`.
        remove_com: Subtract the centre-of-mass velocity of the whole system from every frame.

    Returns:
        A tuple ``(lag_ps, partial)``: the lags in ps and ``{grouping: {label: vacf}}`` with the same keys as
        ``groups``, each a float64 array of length ``max_lag_frames + 1``.

    Raises:
        ValueError: As :func:`compute_vacf_from_velocities`, for any group's indices, or if ``groups`` has no labels.

    Example:
        ```pycon
        >>> rng = np.random.default_rng(0)
        >>> velocities = rng.normal(size=(100, 3, 3))
        >>> masses = np.array([28.085, 15.999, 15.999])
        >>> groups = {"element": {"Si": np.array([0]), "O": np.array([1, 2])}}
        >>> lag_ps, partial = compute_partial_vacf(velocities, masses, 5.0, groups)
        >>> _, total = compute_vacf_from_velocities(velocities, masses, 5.0)
        >>> bool(np.allclose(partial["element"]["Si"] + partial["element"]["O"], total))
        True

        ```
    """
    keys, selections = _group_selections(groups)
    lag_ps, columns = _vacf_columns(
        velocities, masses, dt_fs, selections, max_lag_frames, normalization, remove_com=remove_com
    )
    return lag_ps, _columns_by_group(groups, keys, columns)
