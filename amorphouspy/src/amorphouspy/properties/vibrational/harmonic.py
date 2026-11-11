"""Harmonic route: VDOS, partials and participation ratio from eigenvectors."""

import math
import warnings
from collections.abc import Iterator
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from amorphouspy.properties.vibrational import _common
from amorphouspy.properties.vibrational._common import (
    _MATRIX_NDIM,
    _MIN_GRID_POINTS,
    _N_COMPONENTS,
    _N_TRANSLATIONS,
    _columns_by_group,
    _group_selections,
    _is_real_number,
    _membership,
)

HarmonicNormalization = Literal["unit", "3N"]
# Default grid of compute_harmonic_vdos: step sigma / 5, from 0 to the highest mode plus 5 sigma.
_GAUSSIAN_STEPS_PER_SIGMA = 5
_GAUSSIAN_TAIL_SIGMAS = 5.0
# A caller grid warns when a spacing exceeds sigma / 2 by more than this; np.fft.rfftfreq spacings carry ulp noise.
_GRID_SPACING_RTOL = 1e-9


def _mode_atom_weights(eigenvectors: NDArray[np.float64], n_atoms: int) -> Iterator[tuple[slice, NDArray[np.float64]]]:
    """Yield ``(columns, a)`` over blocks of modes: a[i, k] = sum_alpha e[3i + alpha, k]^2, shape (n_atoms, block).

    Each block holds about ``_CHUNK_BYTES`` of eigenvectors. The per-atom reshape of a column block is a view for C- and
    F-ordered input, so only ``a`` (a third of the block) is allocated: never a squared copy of the eigenvectors or an
    (n_atoms, n_modes) array. :func:`compute_participation_ratio` reuses this code, not the weights of a
    :func:`compute_harmonic_vdos` call: its second pass over the eigenvectors costs memory bandwidth, not memory.
    """
    n_modes = eigenvectors.shape[1]
    step = max(1, _common._CHUNK_BYTES // (eigenvectors.shape[0] * eigenvectors.itemsize))  # noqa: SLF001
    for start in range(0, n_modes, step):
        columns = slice(start, min(start + step, n_modes))
        block = eigenvectors[:, columns].reshape(n_atoms, _N_COMPONENTS, -1)
        yield columns, np.einsum("iak,iak->ik", block, block)


def _validate_harmonic_inputs(
    frequencies_thz: ArrayLike,
    eigenvectors: NDArray | None,
    groups: dict[str, dict[str, NDArray]] | None,
    broadening_thz: float,
    normalization: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, int]:
    """Check the inputs; return the frequencies as float64, the eigenvectors and N (rows / 3, else (M + 3) // 3)."""
    frequencies = np.asarray(frequencies_thz, dtype=np.float64)
    if frequencies.ndim != 1 or frequencies.size == 0 or not np.all(np.isfinite(frequencies)):
        msg = f"frequencies_thz must be a non-empty 1-D array of finite values, got shape {frequencies.shape}."
        raise ValueError(msg)
    n_modes = frequencies.size
    _validate_broadening_options(broadening_thz, normalization)
    if eigenvectors is None:
        if groups is not None:
            msg = "groups need eigenvectors: the group weights come from the eigenvector components."
            raise ValueError(msg)
        if normalization == "3N" and n_modes % _N_COMPONENTS:
            msg = f"normalization='3N' without eigenvectors needs 3N - 3 frequencies; {n_modes} is not a multiple of 3."
            raise ValueError(msg)
        return frequencies, None, (n_modes + _N_TRANSLATIONS) // _N_COMPONENTS
    eigenvectors = np.asarray(eigenvectors)
    if eigenvectors.ndim != _MATRIX_NDIM or eigenvectors.dtype != np.float64:
        msg = (
            "eigenvectors must be a 2-D float64 array (a cast would copy it), "
            f"got {eigenvectors.ndim}-D {eigenvectors.dtype}."
        )
        raise ValueError(msg)
    if n_modes % _N_COMPONENTS or eigenvectors.shape != (n_modes + _N_TRANSLATIONS, n_modes):
        msg = f"eigenvectors must have shape (3N, 3N - 3) to match {n_modes} frequencies, got {eigenvectors.shape}."
        raise ValueError(msg)
    return frequencies, eigenvectors, eigenvectors.shape[0] // _N_COMPONENTS


def _validate_broadening_options(broadening_thz: float, normalization: str) -> None:
    """Raise for an unknown normalisation or a broadening that is not positive and finite."""
    if normalization not in ("unit", "3N"):
        msg = f"Unknown normalization {normalization!r}; allowed values are 'unit' and '3N'."
        raise ValueError(msg)
    if not _is_real_number(broadening_thz) or not np.isfinite(broadening_thz) or broadening_thz <= 0:
        msg = f"broadening_thz must be positive and finite, got {broadening_thz!r}."
        raise ValueError(msg)


def _validate_frequency_grid(frequency_grid_thz: ArrayLike) -> NDArray[np.float64]:
    """Return a caller's grid as float64; raise unless it is 1-D, finite, strictly increasing, with >= 2 points."""
    grid = np.asarray(frequency_grid_thz, dtype=np.float64)
    if grid.ndim != 1 or grid.size < _MIN_GRID_POINTS or not np.all(np.isfinite(grid)) or not np.all(np.diff(grid) > 0):
        msg = "frequency_grid_thz must be a 1-D, finite, strictly increasing array with at least 2 points."
        raise ValueError(msg)
    return grid


def _harmonic_grid(
    frequency_grid_thz: ArrayLike | None, frequencies: NDArray[np.float64], broadening_thz: float
) -> NDArray[np.float64]:
    """The caller's grid after checks, or the default grid with step sigma / 5 whose last point is >= max + 5 sigma."""
    if frequency_grid_thz is None:
        step = broadening_thz / _GAUSSIAN_STEPS_PER_SIGMA
        top = max(float(frequencies.max()), 0.0) + _GAUSSIAN_TAIL_SIGMAS * broadening_thz
        return np.arange(math.ceil(top / step) + 1) * step
    grid = _validate_frequency_grid(frequency_grid_thz)
    spacing = float(np.diff(grid).max())
    if spacing > 0.5 * broadening_thz * (1 + _GRID_SPACING_RTOL):
        warnings.warn(
            f"The largest grid spacing, {spacing:.4g} THz, exceeds half the broadening ({broadening_thz:.4g} THz): "
            "the Gaussians are undersampled and the trapezoid integral is inaccurate.",
            UserWarning,
            stacklevel=3,
        )
    return grid


def compute_harmonic_vdos(
    frequencies_thz: ArrayLike,
    eigenvectors: NDArray[np.float64] | None = None,
    *,
    groups: dict[str, dict[str, NDArray]] | None = None,
    frequency_grid_thz: ArrayLike | None = None,
    broadening_thz: float = 0.2,
    normalization: HarmonicNormalization = "unit",
) -> dict[str, Any]:
    r"""Compute the harmonic VDOS and its group partials from the modes of :func:`diagonalize_dynamical_matrix`.

    The input is the $M = 3N - 3$ kept modes: frequencies $\nu_k$ in THz (negative for imaginary modes) and the
    mass-weighted eigenvectors $e_k$, orthonormal columns with row $3i + \alpha$ for atom $i$ and direction $\alpha$.
    The weight of atom $i$ in mode $k$ and of a group $G$ are

    $$a_{ik} = \sum_\alpha e_{3i+\alpha,k}^2, \quad \sum_i a_{ik} = 1, \qquad w_{kG} = \sum_{i \in G} a_{ik},$$

    and the VDOS and its partials are sums of unit-area Gaussians of standard deviation $\sigma$ (``broadening_thz``),

    $$g(\nu) = c \sum_{k:\, \nu_k \ge 0} G_\sigma(\nu - \nu_k), \quad
    g_G(\nu) = c \sum_{k:\, \nu_k \ge 0} w_{kG}\, G_\sigma(\nu - \nu_k), \quad
    G_\sigma(x) = \frac{e^{-x^2 / 2\sigma^2}}{\sigma \sqrt{2\pi}}.$$

    Partials within a grouping that covers every atom once (e.g. ``"element"``) sum to the total, and the classes of
    a grouping that covers a subset sum to that subset's partial (e.g. the ``"oxygen"`` classes to ``"O"``).

    Normalisation:

    - ``"unit"`` (default): $c = 1/M$, so the total integrates to 1 over all modes.
    - ``"3N"``: $c = 3N/M$, so the total integrates to $3N$, as with the VACF route, although the three translations
      were dropped: each kept mode carries $3N/M$ instead of 1.

    What is not counted:

    - Imaginary modes ($\nu_k < 0$, any negative value, unlike the ``imaginary_threshold_thz`` of
      :func:`diagonalize_dynamical_matrix`) are excluded from all sums and counted in ``"n_imaginary_excluded"``. They
      stay in $M$, so the integral falls short of 1 (or $3N$) by their share, ``"imaginary_weight"`` $= n/M$.
    - Gaussian weight below or above the grid is not renormalised either: the default grid starts at 0, so modes within
      a few $\sigma$ of 0 lose part of their weight. ``"grid_integral"`` is the trapezoid integral of the total on the
      grid. A grid spacing up to $\sigma/2$ integrates a Gaussian to better than $10^{-10}$.

    Completeness: summed over all $3N$ modes, $\sum_k w_{kG} = 3 N_G$. The translations are the mass-weighted rigid
    shifts $\sqrt{m_i / m_\mathrm{tot}}$, so over the kept modes it is $3 N_G - 3 m_G / m_\mathrm{tot}$, with $N_G$
    the atom count and $m_G$ the mass of the group.

    Relation to the VACF route: in mode $k$, atom $i$ carries the share $a_{ik}$ of the mode's kinetic energy, since
    its mass-weighted velocity is proportional to $e_{3i+\alpha,k}$. A partial of
    :func:`compute_partial_vdos` is the group's share of the mass-weighted velocity power, so with every mode at the
    same mean kinetic energy (classical equipartition in a harmonic solid) the two routes use the same partial
    definition, and the same normalisations.

    The eigenvectors are not checked for orthonormality, which would cost a full $(M, M)$ matrix product; columns that
    are not orthonormal break $\sum_i a_{ik} = 1$ and the sums above.

    Memory: the eigenvectors are read in blocks of columns of ``_CHUNK_BYTES`` (64 MB), each reduced at once to the
    groups, and the Gaussians are evaluated in blocks of modes of the same size. Neither a squared copy of the
    eigenvectors nor an $(N, M)$ weight array is built.

    Args:
        frequencies_thz: The $M = 3N - 3$ mode frequencies in THz, negative for imaginary modes, e.g.
            ``"frequencies_thz"`` of :func:`diagonalize_dynamical_matrix`.
        eigenvectors: The $(3N, M)$ float64 eigenvectors with orthonormal columns in the order of
            ``frequencies_thz``, e.g. ``"eigenvectors"`` of :func:`diagonalize_dynamical_matrix` (any memory order;
            no cast or copy is made). Only needed for ``groups``, and for ``"3N"`` it gives $N$.
        groups: ``{grouping: {label: indices}}`` with 0-based atom positions in the order of the matrix rows, e.g. the
            output of :func:`classify_vibrational_groups` on the minimised structure. Needs ``eigenvectors``.
        frequency_grid_thz: The frequencies in THz at which the VDOS is evaluated: 1-D, finite, strictly increasing,
            at least 2 points. ``None`` (default): from 0 in steps of exactly $\sigma/5$ up to the first point at or
            above $\max(\nu_k, 0) + 5\sigma$.
        broadening_thz: The standard deviation $\sigma$ of the Gaussians in THz (not the FWHM, which is
            $2.355\sigma$).
        normalization: ``"unit"`` or ``"3N"``, see above.

    Returns:
        A dictionary with:
            ``"frequencies_thz"``: the grid in THz.
            ``"vdos"``: ``{"total": g}`` and, with ``groups``, ``{grouping: {label: g_G}}`` (the layout of
            :func:`vdos_simulation`), each a float64 array on the grid in 1/THz.
            ``"mode_weights"``: ``{grouping: {label: w}}`` with the $(M,)$ weights $w_{kG}$ of every mode,
            imaginary ones included; only with ``groups``.
            ``"n_modes"``: $M$.
            ``"n_imaginary_excluded"``: the number of modes with $\nu_k < 0$.
            ``"imaginary_weight"``: their share of the normalisation, ``n_imaginary_excluded / n_modes``.
            ``"grid_integral"``: the trapezoid integral of the total on the grid.
            ``"broadening_thz"``: $\sigma$ in THz.
            ``"normalization"``: ``"unit"`` or ``"3N"``.

    Raises:
        ValueError: If ``frequencies_thz`` is empty, not 1-D or not finite; ``eigenvectors`` is not 2-D float64 or
            not $(M + 3, M)$ with $M$ a multiple of 3; ``broadening_thz`` is not positive and finite;
            ``normalization`` is unknown, or ``"3N"`` without eigenvectors and $M$ not a multiple of 3;
            ``groups`` is given without ``eigenvectors``, has no labels or has invalid indices;
            ``frequency_grid_thz`` is not 1-D, finite and strictly increasing with at least 2 points.

    Warns:
        UserWarning: If the largest spacing of ``frequency_grid_thz`` exceeds $\sigma/2$.

    Example:
        ```pycon
        >>> minimised, summary = _run_lammps_dynamical_matrix(glass, potential, "dynmat.bin")  # doctest: +SKIP
        >>> matrix = read_dynamical_matrix(summary["matrix_path"], summary["n_atoms"])  # doctest: +SKIP
        >>> modes = diagonalize_dynamical_matrix(  # doctest: +SKIP
        ...     matrix, masses=minimised.get_masses(), overwrite_matrix=True
        ... )
        >>> groups = classify_vibrational_groups(minimised)  # doctest: +SKIP
        >>> vdos = compute_harmonic_vdos(  # doctest: +SKIP
        ...     modes["frequencies_thz"], modes["eigenvectors"], groups=groups, broadening_thz=0.1
        ... )
        >>> vdos["vdos"]["oxygen"]["O_NBO"], vdos["grid_integral"], vdos["n_imaginary_excluded"]  # doctest: +SKIP

        ```
    """
    frequencies, eigenvectors, n_atoms = _validate_harmonic_inputs(
        frequencies_thz, eigenvectors, groups, broadening_thz, normalization
    )
    grid = _harmonic_grid(frequency_grid_thz, frequencies, broadening_thz)
    n_modes = frequencies.size
    kept = frequencies >= 0

    # Column 0 weighs every mode by 1 (the total); with groups, one column per group label follows.
    weights = np.ones((n_modes, 1))
    keys: list[tuple[str, str]] = []
    if groups is not None and eigenvectors is not None:
        keys, selections = _group_selections(groups)
        membership = _membership(selections, n_atoms)
        weights = np.ones((n_modes, 1 + membership.shape[1]))
        for columns, atom_weights in _mode_atom_weights(eigenvectors, n_atoms):
            weights[columns, 1:] = atom_weights.T @ membership

    centres, kept_weights = frequencies[kept], weights[kept]
    spectra = np.zeros((grid.size, weights.shape[1]))
    step = max(1, _common._CHUNK_BYTES // (grid.size * grid.itemsize))  # noqa: SLF001
    buffer = np.empty((grid.size, min(step, centres.size)))  # one block, reused for every chunk of modes
    for start in range(0, centres.size, step):
        gaussians = buffer[:, : min(step, centres.size - start)]
        np.subtract(grid[:, None], centres[None, start : start + step], out=gaussians)
        gaussians /= broadening_thz
        np.square(gaussians, out=gaussians)
        gaussians *= -0.5
        np.exp(gaussians, out=gaussians)
        spectra += gaussians @ kept_weights[start : start + step]
    spectra *= (1.0 if normalization == "unit" else 3.0 * n_atoms) / (n_modes * broadening_thz * math.sqrt(2 * math.pi))

    total = np.ascontiguousarray(spectra[:, 0])
    result: dict[str, Any] = {"frequencies_thz": grid, "vdos": {"total": total}}
    if groups is not None:
        result["vdos"] = {"total": total, **_columns_by_group(groups, keys, spectra[:, 1:])}
        result["mode_weights"] = _columns_by_group(groups, keys, weights[:, 1:])
    n_imaginary = n_modes - int(np.count_nonzero(kept))
    result.update(
        {
            "n_modes": n_modes,
            "n_imaginary_excluded": n_imaginary,
            "imaginary_weight": n_imaginary / n_modes,
            "grid_integral": float(np.trapezoid(total, grid)),
            "broadening_thz": float(broadening_thz),
            "normalization": normalization,
        }
    )
    return result


def compute_participation_ratio(eigenvectors: NDArray[np.float64], masses: ArrayLike) -> NDArray[np.float64]:
    r"""Compute the participation ratio of every mode from the displacements of its mass-weighted eigenvector.

    The eigenvectors $e_{3i+\alpha,k}$ of the mass-weighted dynamical matrix (atom $i$, direction $\alpha$, mode $k$)
    give the atomic displacements $u_{3i+\alpha,k} \propto e_{3i+\alpha,k} / \sqrt{m_i}$. With the squared displacement
    of atom $i$, the participation ratio is (Taraskin and Elliott, Phys. Rev. B 56, 8605 (1997), Eq. 16, a definition
    they attribute to Bell and Dean)

    $$b_{ik} = \frac{1}{m_i} \sum_\alpha e_{3i+\alpha,k}^2, \qquad
    P_k = \frac{\left(\sum_i b_{ik}\right)^2}{N \sum_i b_{ik}^2}.$$

    The ratio form makes $P_k$ independent of the scale and sign of each column, so the displacements need no
    normalisation. $P_k = 1$ when every atom moves by the same amount (a rigid translation, whatever the masses),
    $1/N$ for a mode on one atom and $n/N$ for a mode shared equally by $n$ atoms. The per-atom weight is summed over
    the three directions before squaring, so $P_k$ does not depend on the orientation of the axes. Unlike the
    kinetic-energy share $m_i b_{ik}$ used by :func:`compute_harmonic_vdos`, $b_{ik}$ counts atoms by amplitude; the
    two coincide for a single element.

    An extended mode has $P$ of order 1 at any size, but a mode on a fixed number of atoms has $P \propto 1/N$, and in a
    small box the two can be hard to tell apart: a claim that a mode is localised needs the same band compared across
    box sizes.


    Args:
        eigenvectors: The $(3N, K)$ float64 eigenvectors of the mass-weighted dynamical matrix, one mode per column
            with row $3i + \alpha$ for atom $i$ and direction $\alpha$, e.g. ``"eigenvectors"`` of
            :func:`diagonalize_dynamical_matrix` (any memory order; no cast or copy is made). Columns need not be
            normalised.
        masses: Atomic masses, shape $(N,)$, in the order of the rows, e.g. ``minimised.get_masses()``. Only their
            ratios matter.

    Returns:
        The $(K,)$ float64 participation ratios, in the order of the columns, between $1/N$ and 1.

    Raises:
        ValueError: If ``eigenvectors`` is not a 2-D float64 array, its row count is not a positive multiple of 3,
            it has no columns, or a column is zero or contains NaN or inf; if ``masses`` does not hold $N$ positive
            finite values.

    Example:
        ```pycon
        >>> minimised, summary = _run_lammps_dynamical_matrix(glass, potential, "dynmat.bin")  # doctest: +SKIP
        >>> matrix = read_dynamical_matrix(summary["matrix_path"], summary["n_atoms"])  # doctest: +SKIP
        >>> modes = diagonalize_dynamical_matrix(  # doctest: +SKIP
        ...     matrix, masses=minimised.get_masses(), overwrite_matrix=True
        ... )
        >>> ratios = compute_participation_ratio(modes["eigenvectors"], minimised.get_masses())  # doctest: +SKIP
        >>> bands = np.digitize(modes["frequencies_thz"], [8.0, 20.0, 28.0])  # doctest: +SKIP
        >>> np.bincount(bands, weights=ratios) / np.bincount(bands)  # mean P per band  # doctest: +SKIP

        ```
    """
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
    ratios = np.empty(n_modes)
    for columns, atom_weights in _mode_atom_weights(eigenvectors, n_atoms):
        np.divide(atom_weights, masses[:, None], out=atom_weights)  # squared displacements, in place in our block
        # The weights are non-negative, so NaN or inf anywhere in a column makes its sum of squares non-finite.
        squares = np.einsum("ik,ik->k", atom_weights, atom_weights)
        if not np.all(np.isfinite(squares)):
            msg = f"eigenvectors contain NaN or inf (mode {columns.start + int(np.argmin(np.isfinite(squares)))})."
            raise ValueError(msg)
        if np.any(squares == 0):
            msg = f"eigenvectors has a zero column (mode {columns.start + int(np.argmax(squares == 0))})."
            raise ValueError(msg)
        ratios[columns] = atom_weights.sum(axis=0) ** 2 / (n_atoms * squares)
    return ratios
