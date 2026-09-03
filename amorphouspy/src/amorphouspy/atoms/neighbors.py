"""Neighbor list module for multicomponent glass systems.

Supports orthogonal and triclinic boxes with any combination of periodic and
non-periodic directions.

The cell-list builders and the compiled kernels are deliberately pbc-unaware:
they always apply the minimum-image convention along all three lattice
vectors. Periodicity is a property of the (cell, coords) pair handed to them.
``_pad_nonperiodic`` stretches every non-periodic lattice vector so that it
exceeds the atom cloud by ``2 * cutoff``, after which no wrapped image can fall
inside the cutoff and the periodic kernels return the exact non-periodic
answer. Do not add per-axis wrapping logic to a kernel: it would be applied
twice.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
from ase import Atoms
from ase.geometry import complete_cell, minkowski_reduce

# Try to import numba for maximum performance
try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def jit(*args: Any, **kwargs: Any) -> Callable:  # noqa: ANN401, ARG001
        """No-op decorator replacing numba.jit when numba is unavailable."""

        def decorator(func: Callable) -> Callable:
            return func

        return decorator

    prange = range  # type: ignore[ty:invalid-assignment]

# Precompute 3D shift grid for neighbor cells (used in NumPy path)
SHIFT_GRID_3D = np.stack(
    np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"),
    axis=-1,
).reshape(-1, 3)

_MIN_VOLUME: float = 1e-10
# Cap on the cell-list grid. Beyond this the int32 flat cell ids would overflow
# for large sparse systems; halving the grid keeps the stencil correct, only slower.
_MAX_CELLS: int = 1_000_000
# Squared cutoff standing in for "this pair never bonds". No squared distance is negative,
# so `dist_sq <= _EXCLUDED_PAIR_SQ` is false for every pair, including coincident atoms.
_EXCLUDED_PAIR_SQ: float = -1.0
# Width of the neighbour-cell stencil along one dimension: offsets -1, 0, +1.
_STENCIL_WIDTH: int = 3


# ============================================================================
# Per-pair cutoff helpers
# ============================================================================


def _parse_cutoff(
    cutoff: float | dict[tuple[int, int], float],
    types: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, bool]:
    """Parse a scalar or per-pair cutoff into a flat lookup table for Numba.

    Numba cannot accept Python dicts, so per-pair cutoffs are encoded as two
    parallel arrays (pair_types, pair_cutoffs_sq) that the kernel searches
    linearly. For typical glass systems the number of unique pairs is small
    (<=10), making linear search faster than a hash map in nopython mode.

    Args:
        cutoff: Either a scalar cutoff in Angstrom applied to all pairs, or a
                dict mapping (atomic_number_i, atomic_number_j) to a
                pair-specific cutoff in Angstrom. The dict is symmetric:
                (8, 14) and (14, 8) are treated as the same pair.
                Any pair not explicitly listed defaults to the maximum
                cutoff value in the dict. A non-positive per-pair value
                marks that pair as never bonded (the convention used by
                generate_bond_length_dict's default_cutoff=0.0); it is
                encoded as a negative squared cutoff, which no squared
                distance can satisfy — so an excluded pair stays unbonded
                even for coincident atoms, which squaring the value would not.
        types:  Integer array of atomic numbers for all atoms.

    Returns:
        max_cutoff:       Largest cutoff across all pairs (used for cell-list).
        pair_types:       (M, 2) int32 array of unique type pairs.
        pair_cutoffs_sq:  (M,) float64 array of squared cutoffs per pair;
                          _EXCLUDED_PAIR_SQ for pairs marked as never bonded.
        use_pair_cutoffs: True when per-pair mode is active.

    Raises:
        ValueError: If a scalar cutoff is not positive, the dict is empty, or
                    no pair in the dict has a positive cutoff.
    """
    if isinstance(cutoff, (int, float)):
        if float(cutoff) <= 0.0:
            msg = f"cutoff must be a positive distance in Angstrom, got {float(cutoff)}"
            raise ValueError(msg)
        return float(cutoff), np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float64), False

    if not cutoff:
        msg = "cutoff is an empty dict; pass a positive float or a non-empty {(Z_i, Z_j): r_c} mapping"
        raise ValueError(msg)

    unique_types = np.unique(types).tolist()
    max_cutoff = float(max(cutoff.values()))
    if max_cutoff <= 0.0:
        msg = (
            "no pair in the cutoff dict has a positive distance, so no atom could ever be a neighbour; "
            f"got {dict(cutoff)!r}"
        )
        raise ValueError(msg)

    # Build a dict covering all ordered pairs, defaulting to max_cutoff
    pair_dict: dict[tuple[int, int], float] = {}
    for type_i in unique_types:
        for type_j in unique_types:
            pair_dict[(type_i, type_j)] = max_cutoff

    # Override with user-specified values (both orderings)
    for (type_i, type_j), rc in cutoff.items():
        pair_dict[(int(type_i), int(type_j))] = float(rc)
        pair_dict[(int(type_j), int(type_i))] = float(rc)

    pairs = list(pair_dict.keys())
    # A non-positive cutoff means "these two species never bond". Squaring it would turn
    # -1.0 A into a 1.0 A cutoff, which only looks like "no bond" because no real pair sits
    # that close; _EXCLUDED_PAIR_SQ makes the exclusion exact.
    per_pair_cutoffs_sq = [pair_dict[p] ** 2 if pair_dict[p] > 0.0 else _EXCLUDED_PAIR_SQ for p in pairs]

    pair_types = np.array(pairs, dtype=np.int32)
    pair_cutoffs_sq = np.array(per_pair_cutoffs_sq, dtype=np.float64)
    return max_cutoff, pair_types, pair_cutoffs_sq, True


@jit(nopython=True, fastmath=True, cache=True)
def _lookup_cutoff_sq(
    type_i: int,
    type_j: int,
    pair_types: np.ndarray,
    pair_cutoffs_sq: np.ndarray,
) -> float:
    """Return squared cutoff for (type_i, type_j) via linear search through pair table.

    Linear search is optimal here: the number of unique type pairs in a
    typical glass is <=10, the entire table fits in L1 cache, and hashing
    is not available in Numba nopython mode.
    """
    for k in range(len(pair_types)):
        if pair_types[k, 0] == type_i and pair_types[k, 1] == type_j:
            return pair_cutoffs_sq[k]
    return pair_cutoffs_sq[0]  # fallback, should not be reached


def _get_pair_cutoff_sq_python(
    type_i: int,
    type_j: int,
    pair_types: np.ndarray,
    pair_cutoffs_sq: np.ndarray,
) -> float:
    """Python equivalent of _lookup_cutoff_sq for the NumPy fallback path."""
    for k in range(len(pair_types)):
        if pair_types[k, 0] == type_i and pair_types[k, 1] == type_j:
            return float(pair_cutoffs_sq[k])
    return float(pair_cutoffs_sq[0])


# ============================================================================
# Triclinic geometry helpers
# ============================================================================


def cell_perpendicular_heights(cell: np.ndarray) -> np.ndarray:
    """Compute the perpendicular height of each cell face.

    For a triclinic cell with lattice vectors a, b, c (rows of `cell`),
    the perpendicular height along axis i is h_i = V / |b_j x b_k|.

    Args:
        cell: (3, 3) array with lattice vectors as rows.

    Returns:
        heights: (3,) perpendicular heights h_a, h_b, h_c in Angstrom.
    """
    a, b, c = cell[0], cell[1], cell[2]
    volume = abs(np.dot(a, np.cross(b, c)))
    height_a = volume / np.linalg.norm(np.cross(b, c))
    height_b = volume / np.linalg.norm(np.cross(a, c))
    height_c = volume / np.linalg.norm(np.cross(a, b))
    return np.array([height_a, height_b, height_c])


def _half_min_height(cell: np.ndarray) -> float:
    """Return half the smallest perpendicular cell height.

    Any displacement shorter than this bound is guaranteed to be the minimum
    image: every non-zero lattice vector is at least as long as the smallest
    perpendicular height, so a competing image cannot be closer.

    Args:
        cell: (3, 3) array with lattice vectors as rows.

    Returns:
        half_height: Half the smallest perpendicular height in Angstrom.
    """
    return float(cell_perpendicular_heights(cell).min()) / 2.0


def _image_search_bound(cell: np.ndarray, n_cells: np.ndarray) -> float | None:
    """Return the squared bound for the exact minimum-image test, or None if it cannot fire.

    Candidate pairs only ever come from the 3x3x3 block of cells around an atom,
    which caps each rounded fractional component at min(0.5, 2 / n_cells). The
    longest Cartesian displacement reachable inside that fractional box sits at
    one of its eight corners. When even that is within half the smallest
    perpendicular height, per-component rounding is provably the nearest image
    for every pair the kernel can see, and the guard is pure overhead — a branch
    inside the innermost loop costs real time whether or not it is ever taken.

    Args:
        cell: (3, 3) array with lattice vectors as rows.
        n_cells: (3,) number of cell-list cells along each lattice vector.

    Returns:
        bound_sq: Squared half-height to test against, or None when no candidate
                  pair can possibly need the image search.
    """
    half_height = _half_min_height(cell)
    component_bounds = np.minimum(0.5, 2.0 / np.asarray(n_cells, dtype=np.float64))
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    longest = float(np.linalg.norm((corners * component_bounds) @ cell, axis=1).max())
    if longest <= half_height:
        return None
    return half_height**2


def _min_periodic_lattice_vector(cell: np.ndarray, pbc: np.ndarray) -> float:
    """Return the length of the shortest non-zero lattice vector of the periodic sub-lattice.

    The minimum-image convention is exact iff the cutoff is below half this
    length. Perpendicular heights are only a lower bound on it, so they would
    make the warning fire spuriously on skewed cells and on every slab.

    Args:
        cell: (3, 3) array with lattice vectors as rows.
        pbc: (3,) bool array, one flag per lattice vector.

    Returns:
        Shortest periodic lattice vector in Angstrom; inf when nothing is periodic.
    """
    if not pbc.any():
        return np.inf
    reduced, _ = minkowski_reduce(cell, pbc=pbc)
    return float(np.linalg.norm(reduced[pbc], axis=1).min())


def _warn_if_cutoff_exceeds_minimum_image(cell: np.ndarray, cutoff: float, pbc: np.ndarray) -> None:
    """Warn when the cutoff is too large for the minimum-image convention.

    Beyond half the shortest periodic lattice vector a pair can have more than
    one periodic image inside the cutoff, so a minimum-image neighbor list is
    approximate no matter how it is built.

    Args:
        cell: (3, 3) array with lattice vectors as rows.
        cutoff: Requested cutoff in Angstrom.
        pbc: (3,) bool array, one flag per lattice vector.
    """
    half_shortest = _min_periodic_lattice_vector(cell, pbc) / 2.0
    if cutoff > half_shortest:
        msg = (
            f"cutoff {cutoff:.3f} A exceeds half the shortest periodic lattice vector "
            f"({half_shortest:.3f} A). The minimum-image convention is not valid in this regime: "
            "only the nearest periodic image of each pair is reported, so neighbor counts are "
            "underestimated. Use a larger cell or a smaller cutoff."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _pad_nonperiodic(
    cell: np.ndarray,
    coords: np.ndarray,
    pbc: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stretch every non-periodic lattice vector so no wrapped image can reach the cutoff.

    Along lattice vector d the fractional coordinate equals the signed distance
    to the face plane divided by the perpendicular height h_d, and the other two
    vectors contribute nothing along that normal. Scaling a_d therefore changes
    h_d alone: the periodic sub-lattice is untouched and several non-periodic
    axes compose independently. With the padded height set to exactly
    E_d + 2 * cutoff (E_d the extent of the atom cloud), every image with a
    non-zero coefficient along d is at least 2 * cutoff away, so the periodic
    kernels can never accept one. The factor 2 is load-bearing: with 1 * cutoff
    a crystalline slab whose surface layers sit in registry produces spurious
    cross-vacuum bonds at exactly the cutoff.

    Args:
        cell: (3, 3) array with lattice vectors as rows. Zero rows are completed.
        coords: (N, 3) Cartesian coordinates, wrapped or not.
        pbc: (3,) bool array, one flag per lattice vector.
        cutoff: Largest cutoff in Angstrom the neighbor search will use.

    Returns:
        (cell, coords): unchanged objects when fully periodic; otherwise a padded
        cell and coordinates rigidly translated so the cloud starts at the origin
        along each non-periodic vector.
    """
    if pbc.all():
        return cell, coords
    cell = np.array(complete_cell(cell), dtype=np.float64)
    frac = coords @ np.linalg.inv(cell)
    heights = cell_perpendicular_heights(cell)
    for axis in np.flatnonzero(~pbc):
        lowest = frac[:, axis].min()
        stretch = frac[:, axis].max() - lowest + 2.0 * cutoff / heights[axis]
        frac[:, axis] = (frac[:, axis] - lowest) / stretch
        cell[axis] *= stretch
    return cell, frac @ cell


def _validate_coords(coords: np.ndarray) -> None:
    """Reject non-finite coordinates; under fastmath they would silently vanish from the kernels."""
    if not np.isfinite(coords).all():
        msg = "coordinates contain NaN or inf"
        raise ValueError(msg)


def _validate_cell(cell: np.ndarray) -> None:
    """Reject a zero-volume cell; fractional coordinates are undefined for it."""
    if abs(np.linalg.det(cell)) < _MIN_VOLUME:
        msg = "cell is degenerate (zero volume): every periodic direction needs a non-zero, independent lattice vector"
        raise ValueError(msg)


def _normalize_type_filter(value: int | list[int] | None, name: str) -> list[int] | None:
    """Normalize a target/neighbor type argument into a list of atomic numbers.

    A bare atomic number is accepted and wrapped, so ``target_types=14`` behaves
    like ``target_types=[14]`` instead of failing inside the compiled kernel.

    Args:
        value: None, a single atomic number, or a sequence of atomic numbers.
        name: Argument name, used in the error message.

    Returns:
        The normalized list, or None when no filter was requested.

    Raises:
        TypeError: If the value is neither None, an integer, nor a sequence of integers.
    """
    if value is None:
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return [int(value)]
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            return [int(v) for v in value]
        except (TypeError, ValueError) as exc:
            msg = f"{name} must contain atomic numbers (integers), got {value!r}"
            raise TypeError(msg) from exc
    msg = f"{name} must be None, an atomic number, or a list of atomic numbers, got {type(value).__name__}"
    raise TypeError(msg)


# ============================================================================
# Cell list construction
# ============================================================================


def _cell_offsets(n_cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stencil (start, count) per dimension so that each neighbour cell is visited exactly once.

    With three or more cells along a dimension the usual -1, 0, +1 offsets are
    all distinct. With fewer, some of them wrap onto the same cell, so the
    stencil is simply every cell along that dimension: offsets 0 .. n_cells - 1.

    Args:
        n_cells: (3,) number of cells along each lattice vector.

    Returns:
        start: (3,) int32 first offset per dimension (-1 or 0).
        count: (3,) int32 number of offsets per dimension (1, 2 or 3).
    """
    n_cells = np.asarray(n_cells)
    start = np.where(n_cells >= _STENCIL_WIDTH, -1, 0).astype(np.int32)
    count = np.minimum(n_cells, _STENCIL_WIDTH).astype(np.int32)
    return start, count


def _stencil_grid(n_cells: np.ndarray) -> np.ndarray:
    """Return the (K, 3) stencil offsets of _cell_offsets as one array for the NumPy path."""
    start, count = _cell_offsets(n_cells)
    axes = [np.arange(start[d], start[d] + count[d]) for d in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _clamp_cell_count(n_cells: np.ndarray) -> np.ndarray:
    """Halve the cell grid until it holds at most _MAX_CELLS cells.

    Thicker cells are always correct — the stencil still covers the cutoff
    sphere — only slower.
    """
    n_cells = np.asarray(n_cells).astype(np.int64)
    while np.prod(n_cells) > _MAX_CELLS:
        n_cells = np.maximum(1, n_cells // 2)
    return n_cells.astype(np.int32)


def _csr_cell_list(atom_cells: np.ndarray, n_cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort atoms by flat cell id into CSR form: (cell_start, order)."""
    n_total = int(np.prod(n_cells, dtype=np.int64))
    strides = np.array([int(n_cells[1]) * int(n_cells[2]), int(n_cells[2]), 1], dtype=np.int64)
    flat_cell_ids = atom_cells @ strides
    order = np.argsort(flat_cell_ids, kind="stable").astype(np.int32)
    cell_start = np.zeros(n_total + 1, dtype=np.int32)
    cell_start[1:] = np.bincount(flat_cell_ids, minlength=n_total)
    np.cumsum(cell_start, out=cell_start)
    return cell_start, order


def compute_cell_list_orthogonal(
    coords: np.ndarray,
    box_size: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct a flat CSR-style cell list for orthogonal boxes.

    Args:
        coords: Cartesian coordinates of atoms, wrapped or not.
        box_size: Lengths of the simulation box.
        cutoff: Cell list cutoff.

    Returns:
        A tuple of (atom_cells, n_cells, cell_start, order).
    """
    n_cells = _clamp_cell_count(np.maximum(1, np.floor(box_size / cutoff)))
    atom_cells = (np.floor(coords * (n_cells / box_size)).astype(np.int64) % n_cells).astype(np.int32)
    cell_start, order = _csr_cell_list(atom_cells, n_cells)
    return atom_cells, n_cells, cell_start, order


def compute_cell_list_triclinic(
    coords: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct a flat CSR-style cell list for triclinic boxes (fractional coords).

    Args:
        coords: Cartesian coordinates of atoms, wrapped or not.
        cell: Lattice vector matrix.
        cutoff: Cell list cutoff.

    Returns:
        A tuple of (coords_frac, atom_cells, n_cells, cell_start, order).
    """
    # Lattice vectors are rows, so fractional coordinates are coords @ inv(cell).
    coords_frac = (coords @ np.linalg.inv(cell)) % 1.0
    heights = cell_perpendicular_heights(cell)
    n_cells = _clamp_cell_count(np.maximum(1, np.floor(heights / cutoff)))
    atom_cells = (np.floor(coords_frac * n_cells).astype(np.int64) % n_cells).astype(np.int32)
    cell_start, order = _csr_cell_list(atom_cells, n_cells)
    return coords_frac, atom_cells, n_cells, cell_start, order


def _build_cell_list(
    coords: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the cell list on the orthogonal or the triclinic path, whichever the cell needs.

    Args:
        coords: Cartesian coordinates of atoms, wrapped or not.
        cell: (3, 3) array with lattice vectors as rows.
        cutoff: Cell list cutoff in Angstrom.

    Returns:
        is_orthogonal: True when the cell is diagonal and the orthogonal path was taken.
        box_size: (3,) box lengths on the orthogonal path, None on the triclinic one.
        coords_frac: (N, 3) fractional coordinates on the triclinic path, None otherwise.
        atom_cells: (N, 3) int32 cell index per atom.
        n_cells: (3,) int32 number of cells along each lattice vector.
        cell_start: (n_total + 1,) int32 CSR offsets into cell_atoms.
        cell_atoms: (N,) int32 atom indices sorted by cell.
    """
    is_orthogonal = bool(np.allclose(cell - np.diag(np.diag(cell)), 0.0, atol=1e-10))
    if is_orthogonal:
        box_size = np.diag(cell)
        atom_cells, n_cells, cell_start, cell_atoms = compute_cell_list_orthogonal(coords, box_size, cutoff)
        return is_orthogonal, box_size, None, atom_cells, n_cells, cell_start, cell_atoms
    coords_frac, atom_cells, n_cells, cell_start, cell_atoms = compute_cell_list_triclinic(coords, cell, cutoff)
    return is_orthogonal, None, coords_frac, atom_cells, n_cells, cell_start, cell_atoms


# ============================================================================
# Numba distance kernels — return vector + squared distance in one call
# ============================================================================


@jit(nopython=True, fastmath=True, cache=True)
def _dist_and_vec_ortho(
    ci: np.ndarray,
    cj: np.ndarray,
    box: np.ndarray,
) -> tuple[float, float, float, float]:
    """Minimum-image displacement vector and squared distance, orthogonal box.

    Returns:
        (dx, dy, dz, dist_sq) — displacement r_i - r_j and its squared length.
        Note the sign: this points from j to i, opposite to ASE's "D".
    """
    dx = ci[0] - cj[0]
    dx -= box[0] * round(dx / box[0])
    dy = ci[1] - cj[1]
    dy -= box[1] * round(dy / box[1])
    dz = ci[2] - cj[2]
    dz -= box[2] * round(dz / box[2])
    return dx, dy, dz, dx * dx + dy * dy + dz * dz


@jit(nopython=True, fastmath=True, cache=True)
def _dist_and_vec_tri(
    frac_i: np.ndarray,
    frac_j: np.ndarray,
    cell: np.ndarray,
) -> tuple[float, float, float, float]:
    """Nearest-integer displacement vector and squared distance, triclinic box.

    Per-component rounding is exact whenever the result is shorter than half the
    smallest perpendicular cell height; beyond that bound a skewed cell can have
    a closer image. Use _dist_and_vec_tri_exact when that guarantee is needed.

    Returns:
        (dx, dy, dz, dist_sq) — Cartesian displacement r_i - r_j and its squared length.
    """
    delta_frac_x = frac_i[0] - frac_j[0]
    delta_frac_x -= round(delta_frac_x)
    delta_frac_y = frac_i[1] - frac_j[1]
    delta_frac_y -= round(delta_frac_y)
    delta_frac_z = frac_i[2] - frac_j[2]
    delta_frac_z -= round(delta_frac_z)
    dx = delta_frac_x * cell[0, 0] + delta_frac_y * cell[1, 0] + delta_frac_z * cell[2, 0]
    dy = delta_frac_x * cell[0, 1] + delta_frac_y * cell[1, 1] + delta_frac_z * cell[2, 1]
    dz = delta_frac_x * cell[0, 2] + delta_frac_y * cell[1, 2] + delta_frac_z * cell[2, 2]
    return dx, dy, dz, dx * dx + dy * dy + dz * dz


@jit(nopython=True, fastmath=True, cache=True)
def _search_mic_images_tri(
    base_x: float,
    base_y: float,
    base_z: float,
    cell: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    dist_sq: float,
) -> tuple[float, float, float, float]:
    """Search the 27 images around a rounded fractional displacement for the nearest one.

    Args:
        base_x: Rounded fractional displacement along a.
        base_y: Rounded fractional displacement along b.
        base_z: Rounded fractional displacement along c.
        cell: (3, 3) lattice vectors as rows.
        dx: Cartesian displacement of the rounded image along x.
        dy: Cartesian displacement of the rounded image along y.
        dz: Cartesian displacement of the rounded image along z.
        dist_sq: Squared length of the rounded image.

    Returns:
        (dx, dy, dz, dist_sq) for the shortest image found.
    """
    for shift_x in range(-1, 2):
        trial_x = base_x + shift_x
        for shift_y in range(-1, 2):
            trial_y = base_y + shift_y
            for shift_z in range(-1, 2):
                trial_z = base_z + shift_z
                trial_dx = trial_x * cell[0, 0] + trial_y * cell[1, 0] + trial_z * cell[2, 0]
                trial_dy = trial_x * cell[0, 1] + trial_y * cell[1, 1] + trial_z * cell[2, 1]
                trial_dz = trial_x * cell[0, 2] + trial_y * cell[1, 2] + trial_z * cell[2, 2]
                trial_sq = trial_dx * trial_dx + trial_dy * trial_dy + trial_dz * trial_dz
                if trial_sq < dist_sq:
                    dx = trial_dx
                    dy = trial_dy
                    dz = trial_dz
                    dist_sq = trial_sq

    return dx, dy, dz, dist_sq


@jit(nopython=True, fastmath=True, cache=True)
def _dist_and_vec_tri_exact(
    frac_i: np.ndarray,
    frac_j: np.ndarray,
    cell: np.ndarray,
    half_height_sq: float,
) -> tuple[float, float, float, float]:
    """Exact minimum-image displacement vector and squared distance, triclinic box.

    Takes the nearest-integer result first. Anything shorter than half the
    smallest perpendicular height is provably minimal and returned as is; only
    longer displacements call into _search_mic_images_tri. For typical
    cutoff-to-box ratios that search never runs, leaving one extra float
    comparison relative to _dist_and_vec_tri.

    Args:
        frac_i: Fractional coordinates of atom i.
        frac_j: Fractional coordinates of atom j.
        cell: (3, 3) lattice vectors as rows.
        half_height_sq: Squared half of the smallest perpendicular cell height.

    Returns:
        (dx, dy, dz, dist_sq) — Cartesian displacement r_i - r_j and its squared length.
    """
    base_x = frac_i[0] - frac_j[0]
    base_x -= round(base_x)
    base_y = frac_i[1] - frac_j[1]
    base_y -= round(base_y)
    base_z = frac_i[2] - frac_j[2]
    base_z -= round(base_z)
    dx = base_x * cell[0, 0] + base_y * cell[1, 0] + base_z * cell[2, 0]
    dy = base_x * cell[0, 1] + base_y * cell[1, 1] + base_z * cell[2, 1]
    dz = base_x * cell[0, 2] + base_y * cell[1, 2] + base_z * cell[2, 2]
    dist_sq = dx * dx + dy * dy + dz * dz
    if dist_sq <= half_height_sq:
        return dx, dy, dz, dist_sq
    return _search_mic_images_tri(base_x, base_y, base_z, cell, dx, dy, dz, dist_sq)


# ============================================================================
# Numba kernel — orthogonal box
# ============================================================================


@jit(nopython=True, parallel=True, cache=True)
def _build_nl_ortho_numba(
    coords: np.ndarray,
    types: np.ndarray,
    box_size: np.ndarray,
    atom_cells: np.ndarray,
    n_cells: np.ndarray,
    cell_start: np.ndarray,
    cell_atoms: np.ndarray,
    stencil_start: np.ndarray,
    stencil_count: np.ndarray,
    cutoff_sq: float,
    target_types: np.ndarray,
    neighbor_types: np.ndarray,
    use_target_filter: bool,  # noqa: FBT001
    use_neighbor_filter: bool,  # noqa: FBT001
    max_neighbors: int,
    pair_types: np.ndarray,
    pair_cutoffs_sq: np.ndarray,
    use_pair_cutoffs: bool,  # noqa: FBT001
    return_vectors: bool,  # noqa: FBT001
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build neighbor list for an orthogonal box using Numba.

    Supports both a global scalar cutoff and per-pair cutoffs, and
    optionally accumulates the minimum-image bond vector for each pair.

    Returns:
        neighbor_list:   (N, max_neighbors) int32 — neighbor array indices.
        neighbor_counts: (N,) int32 — actual neighbor count per atom.
        vector_list:     (N, max_neighbors, 3) float32 — bond vectors i->j.
                         All zeros when return_vectors=False.
    """
    n_atoms = len(coords)
    n_cells_y = n_cells[1]
    n_cells_z = n_cells[2]

    # Only the first neighbor_counts[i] slots of each row are ever read, so the
    # buffers are left uninitialised rather than paying for a memset.
    neighbor_list = np.empty((n_atoms, max_neighbors), dtype=np.int32)
    neighbor_counts = np.zeros(n_atoms, dtype=np.int32)
    if return_vectors:
        vector_list = np.empty((n_atoms, max_neighbors, 3), dtype=np.float32)
    else:
        vector_list = np.empty((0, 0, 3), dtype=np.float32)

    for i in prange(n_atoms):  # type: ignore[ty:not-iterable]
        type_i = types[i]
        if use_target_filter:
            is_target_type = False
            for atom_type in target_types:
                if type_i == atom_type:
                    is_target_type = True
                    break
            if not is_target_type:
                continue

        cell_idx_i = atom_cells[i]
        count = 0

        for offset_x in range(stencil_count[0]):
            neighbor_cell_x = (cell_idx_i[0] + stencil_start[0] + offset_x) % n_cells[0]
            for offset_y in range(stencil_count[1]):
                neighbor_cell_y = (cell_idx_i[1] + stencil_start[1] + offset_y) % n_cells[1]
                for offset_z in range(stencil_count[2]):
                    neighbor_cell_z = (cell_idx_i[2] + stencil_start[2] + offset_z) % n_cells[2]
                    flat_cell_idx = (
                        neighbor_cell_x * n_cells_y * n_cells_z + neighbor_cell_y * n_cells_z + neighbor_cell_z
                    )
                    start = cell_start[flat_cell_idx]
                    end = cell_start[flat_cell_idx + 1]

                    for k in range(start, end):
                        j = cell_atoms[k]
                        if j == i:
                            continue
                        type_j = int(types[j])

                        if use_neighbor_filter:
                            is_valid_neighbor_type = False
                            for atom_type in neighbor_types:
                                if type_j == atom_type:
                                    is_valid_neighbor_type = True
                                    break
                            if not is_valid_neighbor_type:
                                continue

                        pair_cutoff_sq = (
                            _lookup_cutoff_sq(type_i, type_j, pair_types, pair_cutoffs_sq)
                            if use_pair_cutoffs
                            else cutoff_sq
                        )

                        dx, dy, dz, dist_sq = _dist_and_vec_ortho(coords[i], coords[j], box_size)

                        if dist_sq <= pair_cutoff_sq:
                            if count < max_neighbors:
                                neighbor_list[i, count] = j
                                if return_vectors:
                                    vector_list[i, count, 0] = dx
                                    vector_list[i, count, 1] = dy
                                    vector_list[i, count, 2] = dz
                            count += 1

        neighbor_counts[i] = count

    return neighbor_list, neighbor_counts, vector_list


# ============================================================================
# Numba kernel — triclinic box
# ============================================================================


@jit(nopython=True, parallel=True, cache=True)
def _build_nl_tri_numba(  # noqa: PLR0915
    coords_frac: np.ndarray,
    types: np.ndarray,
    cell: np.ndarray,
    atom_cells: np.ndarray,
    n_cells: np.ndarray,
    cell_start: np.ndarray,
    cell_atoms: np.ndarray,
    stencil_start: np.ndarray,
    stencil_count: np.ndarray,
    cutoff_sq: float,
    target_types: np.ndarray,
    neighbor_types: np.ndarray,
    use_target_filter: bool,  # noqa: FBT001
    use_neighbor_filter: bool,  # noqa: FBT001
    max_neighbors: int,
    pair_types: np.ndarray,
    pair_cutoffs_sq: np.ndarray,
    use_pair_cutoffs: bool,  # noqa: FBT001
    return_vectors: bool,  # noqa: FBT001
    half_height_sq: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build neighbor list for a triclinic box using Numba.

    Returns:
        neighbor_list:   (N, max_neighbors) int32
        neighbor_counts: (N,) int32
        vector_list:     (N, max_neighbors, 3) float32 — Cartesian bond vectors i->j.
    """
    n_atoms = len(coords_frac)
    n_cells_y = n_cells[1]
    n_cells_z = n_cells[2]

    # Only the first neighbor_counts[i] slots of each row are ever read, so the
    # buffers are left uninitialised rather than paying for a memset.
    neighbor_list = np.empty((n_atoms, max_neighbors), dtype=np.int32)
    neighbor_counts = np.zeros(n_atoms, dtype=np.int32)
    if return_vectors:
        vector_list = np.empty((n_atoms, max_neighbors, 3), dtype=np.float32)
    else:
        vector_list = np.empty((0, 0, 3), dtype=np.float32)

    for i in prange(n_atoms):  # type: ignore[ty:not-iterable]
        type_i = types[i]
        if use_target_filter:
            is_target_type = False
            for atom_type in target_types:
                if type_i == atom_type:
                    is_target_type = True
                    break
            if not is_target_type:
                continue

        cell_idx_i = atom_cells[i]
        count = 0

        for offset_x in range(stencil_count[0]):
            neighbor_cell_x = (cell_idx_i[0] + stencil_start[0] + offset_x) % n_cells[0]
            for offset_y in range(stencil_count[1]):
                neighbor_cell_y = (cell_idx_i[1] + stencil_start[1] + offset_y) % n_cells[1]
                for offset_z in range(stencil_count[2]):
                    neighbor_cell_z = (cell_idx_i[2] + stencil_start[2] + offset_z) % n_cells[2]
                    flat_cell_idx = (
                        neighbor_cell_x * n_cells_y * n_cells_z + neighbor_cell_y * n_cells_z + neighbor_cell_z
                    )
                    start = cell_start[flat_cell_idx]
                    end = cell_start[flat_cell_idx + 1]

                    for k in range(start, end):
                        j = cell_atoms[k]
                        if j == i:
                            continue
                        type_j = int(types[j])

                        if use_neighbor_filter:
                            is_valid_neighbor_type = False
                            for atom_type in neighbor_types:
                                if type_j == atom_type:
                                    is_valid_neighbor_type = True
                                    break
                            if not is_valid_neighbor_type:
                                continue

                        pair_cutoff_sq = (
                            _lookup_cutoff_sq(type_i, type_j, pair_types, pair_cutoffs_sq)
                            if use_pair_cutoffs
                            else cutoff_sq
                        )

                        # half_height_sq is None when the cell list geometry rules the
                        # image search out; Numba prunes this branch at compile time.
                        if half_height_sq is None:
                            dx, dy, dz, dist_sq = _dist_and_vec_tri(coords_frac[i], coords_frac[j], cell)
                        else:
                            dx, dy, dz, dist_sq = _dist_and_vec_tri_exact(
                                coords_frac[i], coords_frac[j], cell, half_height_sq
                            )

                        if dist_sq <= pair_cutoff_sq:
                            if count < max_neighbors:
                                neighbor_list[i, count] = j
                                if return_vectors:
                                    vector_list[i, count, 0] = dx
                                    vector_list[i, count, 1] = dy
                                    vector_list[i, count, 2] = dz
                            count += 1

        neighbor_counts[i] = count

    return neighbor_list, neighbor_counts, vector_list


# ============================================================================
# Numba half-pair distance kernels
# ============================================================================


@jit(nopython=True, parallel=True, cache=True)
def _build_distances_numba(  # pragma: no cover
    coords: np.ndarray,
    box_size: np.ndarray,
    types: np.ndarray,
    atom_cells: np.ndarray,
    n_cells: np.ndarray,
    cell_start: np.ndarray,
    cell_atoms: np.ndarray,
    stencil_start: np.ndarray,
    stencil_count: np.ndarray,
    r_max_sq: float,
    max_pairs: int,
    pair_types: np.ndarray,
    use_type_filter: bool,  # noqa: FBT001
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect half-pair (j > i) distances within r_max_sq for an orthogonal box.

    When use_type_filter is True, only pairs whose unordered types match one of
    the rows in pair_types (shape (M, 2), each row sorted ascending) are kept.

    Returns:
        dist_buf: (N, max_pairs) float64 — distances per central atom.
        j_buf:    (N, max_pairs) int32   — j indices per central atom.
        counts:   (N,) int32             — valid entries per row.
    """
    n = len(coords)
    n_cells_x = n_cells[0]
    n_cells_y = n_cells[1]
    n_cells_z = n_cells[2]
    dist_buf = np.empty((n, max_pairs), dtype=np.float64)
    j_buf = np.empty((n, max_pairs), dtype=np.int32)
    counts = np.zeros(n, dtype=np.int32)

    for i in prange(n):  # type: ignore[ty:not-iterable]
        ti = types[i] if use_type_filter else np.int32(0)

        ci = atom_cells[i]
        k = 0
        for dix in range(stencil_count[0]):
            cjx = (ci[0] + stencil_start[0] + dix) % n_cells_x
            for diy in range(stencil_count[1]):
                cjy = (ci[1] + stencil_start[1] + diy) % n_cells_y
                for diz in range(stencil_count[2]):
                    cjz = (ci[2] + stencil_start[2] + diz) % n_cells_z
                    flat = cjx * n_cells_y * n_cells_z + cjy * n_cells_z + cjz
                    for p in range(cell_start[flat], cell_start[flat + 1]):
                        j = cell_atoms[p]
                        if j <= i:
                            continue
                        if use_type_filter:
                            tj = types[j]
                            lo = min(ti, tj)
                            hi = max(tj, ti)
                            pair_ok = False
                            for m in range(len(pair_types)):
                                if pair_types[m, 0] == lo and pair_types[m, 1] == hi:
                                    pair_ok = True
                                    break
                            if not pair_ok:
                                continue
                        _, _, _, dsq = _dist_and_vec_ortho(coords[i], coords[j], box_size)
                        if dsq <= r_max_sq:
                            if k < max_pairs:
                                dist_buf[i, k] = dsq**0.5
                                j_buf[i, k] = j
                            k += 1
        counts[i] = k

    return dist_buf, j_buf, counts


@jit(nopython=True, parallel=True, cache=True)
def _build_distances_numba_tri(  # pragma: no cover
    coords_frac: np.ndarray,
    cell: np.ndarray,
    types: np.ndarray,
    atom_cells: np.ndarray,
    n_cells: np.ndarray,
    cell_start: np.ndarray,
    cell_atoms: np.ndarray,
    stencil_start: np.ndarray,
    stencil_count: np.ndarray,
    r_max_sq: float,
    max_pairs: int,
    pair_types: np.ndarray,
    use_type_filter: bool,  # noqa: FBT001
    half_height_sq: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect half-pair (j > i) distances within r_max_sq for a triclinic box.

    When use_type_filter is True, only pairs whose unordered types match one of
    the rows in pair_types (shape (M, 2), each row sorted ascending) are kept.

    Returns:
        dist_buf: (N, max_pairs) float64 — distances per central atom.
        j_buf:    (N, max_pairs) int32   — j indices per central atom.
        counts:   (N,) int32             — valid entries per row.
    """
    n = len(coords_frac)
    n_cells_x = n_cells[0]
    n_cells_y = n_cells[1]
    n_cells_z = n_cells[2]
    dist_buf = np.empty((n, max_pairs), dtype=np.float64)
    j_buf = np.empty((n, max_pairs), dtype=np.int32)
    counts = np.zeros(n, dtype=np.int32)

    for i in prange(n):  # type: ignore[ty:not-iterable]
        ti = types[i] if use_type_filter else np.int32(0)

        ci = atom_cells[i]
        k = 0
        for dix in range(stencil_count[0]):
            cjx = (ci[0] + stencil_start[0] + dix) % n_cells_x
            for diy in range(stencil_count[1]):
                cjy = (ci[1] + stencil_start[1] + diy) % n_cells_y
                for diz in range(stencil_count[2]):
                    cjz = (ci[2] + stencil_start[2] + diz) % n_cells_z
                    flat = cjx * n_cells_y * n_cells_z + cjy * n_cells_z + cjz
                    for p in range(cell_start[flat], cell_start[flat + 1]):
                        j = cell_atoms[p]
                        if j <= i:
                            continue
                        if use_type_filter:
                            tj = types[j]
                            lo = min(ti, tj)
                            hi = max(tj, ti)
                            pair_ok = False
                            for m in range(len(pair_types)):
                                if pair_types[m, 0] == lo and pair_types[m, 1] == hi:
                                    pair_ok = True
                                    break
                            if not pair_ok:
                                continue
                        # See _build_nl_tri_numba: None prunes the guard at compile time.
                        if half_height_sq is None:
                            _, _, _, dsq = _dist_and_vec_tri(coords_frac[i], coords_frac[j], cell)
                        else:
                            _, _, _, dsq = _dist_and_vec_tri_exact(coords_frac[i], coords_frac[j], cell, half_height_sq)
                        if dsq <= r_max_sq:
                            if k < max_pairs:
                                dist_buf[i, k] = dsq**0.5
                                j_buf[i, k] = j
                            k += 1
        counts[i] = k

    return dist_buf, j_buf, counts


def _valid_slots(counts: np.ndarray, width: int) -> np.ndarray:
    """(N, width) bool mask of the filled slots in a padded (N, width) kernel buffer."""
    return np.arange(width)[np.newaxis, :] < counts[:, np.newaxis]


def _flatten_distance_buffers(
    dist_buf: np.ndarray,
    j_buf: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten (N, max_pairs) Numba output buffers into flat (M,) arrays."""
    valid = _valid_slots(counts, dist_buf.shape[1])
    dist_out = dist_buf[valid].astype(np.float64)
    j_out = j_buf[valid].astype(np.int32)
    i_out = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    return dist_out, i_out, j_out


def _build_distances_numpy(
    coords: np.ndarray,
    cell: np.ndarray,
    r_max: float,
    types: np.ndarray,
    pair_types: np.ndarray,
    use_type_filter: bool,  # noqa: FBT001
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Half-pair distances without Numba, derived from the shared _numpy_fallback.

    The fallback returns every (i, j) within r_max; keeping j > i halves it and
    the requested unordered type pairs are applied afterwards.
    """
    n_atoms = len(coords)
    is_orthogonal, box_size, coords_frac, atom_cells, n_cells, _cell_start, _cell_atoms = _build_cell_list(
        coords, cell, r_max
    )
    flat_j, counts, flat_vecs = _numpy_fallback(
        coords=coords,
        coords_frac=coords_frac,
        types=types,
        cell=cell,
        box_size=box_size,
        atom_cells=atom_cells,
        n_cells=n_cells,
        cutoff_sq=r_max * r_max,
        pair_types=np.empty((0, 2), dtype=np.int32),
        pair_cutoffs_sq=np.empty(0, dtype=np.float64),
        use_pair_cutoffs=False,
        use_target_filter=False,
        use_neighbor_filter=False,
        target_types=None,
        neighbor_types=None,
        n_atoms=n_atoms,
        return_vectors=True,
        is_orthogonal=is_orthogonal,
    )
    i_idx = np.repeat(np.arange(n_atoms, dtype=np.int32), counts)
    keep = flat_j > i_idx
    if use_type_filter:
        if len(pair_types) == 0:
            keep[:] = False
        else:
            n_types = max(int(types.max()), int(pair_types.max())) + 1
            allowed = np.zeros((n_types, n_types), dtype=bool)
            allowed[pair_types[:, 0], pair_types[:, 1]] = True
            allowed |= allowed.T
            keep &= allowed[types[i_idx], types[flat_j]]
    dists = np.linalg.norm(flat_vecs[keep], axis=1)
    return dists, i_idx[keep], flat_j[keep]


def build_distances(
    structure_wrapped: Atoms,
    r_max: float,
    types: np.ndarray | None = None,
    unordered_pairs: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (distances, i_indices, j_indices) for half-pairs within r_max.

    When types and unordered_pairs are given, only pairs whose types match a
    requested canonical pair are collected — skipping irrelevant atoms entirely
    inside the compiled kernel. ``structure_wrapped.pbc`` is honoured per
    lattice vector; coordinates need not actually be wrapped.

    Args:
        structure_wrapped: ASE Atoms object.
        r_max: Maximum distance cutoff in Angstroms.
        types: Atomic-number array aligned with structure positions.
        unordered_pairs: Canonical (min, max) type pairs to restrict to.

    Returns:
        distances: (M,) float64 array of pairwise distances.
        i_indices: (M,) int32 array of first atom indices.
        j_indices: (M,) int32 array of second atom indices (always > i).
    """
    coords = structure_wrapped.get_positions()
    cell = np.array(structure_wrapped.get_cell().array, dtype=np.float64)
    pbc = np.asarray(structure_wrapped.pbc, dtype=bool)
    n_atoms = len(coords)
    if n_atoms == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    _validate_coords(coords)

    use_type_filter = types is not None and unordered_pairs is not None
    if use_type_filter and unordered_pairs is not None and types is not None:
        # Build a sorted (M, 2) pair_types array for exact pair matching in the kernel.
        pair_rows = sorted({(min(a, b), max(a, b)) for a, b in unordered_pairs})
        pair_types_arr = np.array(pair_rows, dtype=np.int32).reshape(-1, 2)
        types_arr = types.astype(np.int32)
    else:
        pair_types_arr = np.empty((0, 2), dtype=np.int32)
        types_arr = np.empty(0, dtype=np.int32)

    # Initial max_pairs estimate on the real cell, before padding inflates the
    # volume: 4/3 pi r_max^3 x number_density x 1.5.
    volume = float(abs(np.linalg.det(cell)))
    max_pairs = max(32, int(4.0 / 3.0 * np.pi * r_max**3 * (n_atoms / volume) * 1.5)) if volume >= _MIN_VOLUME else 200

    cell, coords = _pad_nonperiodic(cell, coords, pbc, r_max)
    _validate_cell(cell)
    _warn_if_cutoff_exceeds_minimum_image(cell, r_max, pbc)

    if not NUMBA_AVAILABLE:
        types_full = types_arr if use_type_filter else np.zeros(n_atoms, dtype=np.int32)
        return _build_distances_numpy(coords, cell, r_max, types_full, pair_types_arr, use_type_filter)

    is_orthogonal, box_size, coords_frac, atom_cells, n_cells, cell_start, cell_atoms = _build_cell_list(
        coords, cell, r_max
    )
    stencil_start, stencil_count = _cell_offsets(n_cells)
    kwargs: dict[str, Any] = {
        "types": types_arr,
        "atom_cells": atom_cells,
        "n_cells": n_cells,
        "cell_start": cell_start,
        "cell_atoms": cell_atoms,
        "stencil_start": stencil_start,
        "stencil_count": stencil_count,
        "r_max_sq": r_max * r_max,
        "max_pairs": max_pairs,
        "pair_types": pair_types_arr,
        "use_type_filter": use_type_filter,
    }
    if is_orthogonal:
        build_fn = _build_distances_numba
        kwargs.update(coords=coords, box_size=box_size)
    else:
        build_fn = _build_distances_numba_tri
        kwargs.update(
            coords_frac=coords_frac,
            cell=cell,
            half_height_sq=_image_search_bound(cell, n_cells),
        )

    # The kernels count past the buffer instead of truncating, so counts holds
    # the true value even on overflow and one retry always suffices.
    while True:
        dist_buf, j_buf, counts = build_fn(**kwargs)
        if int(counts.max()) <= kwargs["max_pairs"]:
            break
        kwargs["max_pairs"] = int(counts.max() * 1.2) + 1

    return _flatten_distance_buffers(dist_buf, j_buf, counts)


# ============================================================================
# Numba output converter
# ============================================================================


def _grow_until_fits(
    build_fn: Callable,
    build_kwargs: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run a neighbor-list kernel, enlarging max_neighbors until every atom's neighbors fit.

    The kernels count past the buffer instead of truncating, so neighbor_counts
    holds the true count even on overflow and one retry always suffices.
    """
    neighbor_list, neighbor_counts, vector_list = build_fn(**build_kwargs)
    overflow = int(neighbor_counts.max())
    while overflow > build_kwargs["max_neighbors"]:
        build_kwargs["max_neighbors"] = int(overflow * 1.2) + 1
        neighbor_list, neighbor_counts, vector_list = build_fn(**build_kwargs)
        overflow = int(neighbor_counts.max())
    return neighbor_list, neighbor_counts, vector_list


def _padded_to_csr(
    neighbor_list: np.ndarray,
    neighbor_counts: np.ndarray,
    vector_list: np.ndarray,
    return_vectors: bool,  # noqa: FBT001
) -> tuple[np.ndarray, np.ndarray]:
    """Compact padded (N, max_neighbors) kernel buffers into flat CSR arrays (flat_j, flat_vecs)."""
    valid = _valid_slots(neighbor_counts, neighbor_list.shape[1])
    flat_j = neighbor_list[valid]
    flat_vecs = vector_list[valid].astype(np.float64) if return_vectors else np.empty((0, 3), dtype=np.float64)
    return flat_j, flat_vecs


def _build_neighbor_output(
    flat_j: np.ndarray,
    counts: np.ndarray,
    flat_vecs: np.ndarray,
    atom_ids: np.ndarray,
    return_vectors: bool,  # noqa: FBT001
) -> list[tuple]:
    """Assemble the public [(central_id, [neighbor_ids], ...)] list from CSR arrays.

    Everything is converted to Python objects once, up front; the per-atom loop
    then only slices Python lists. Per-row ndarray.tolist() calls and per-id
    int() conversions were the dominant cost of get_neighbors before this.
    """
    flat_ids = atom_ids[flat_j].tolist()
    offsets = np.concatenate(([0], np.cumsum(counts))).tolist()
    ids_list = atom_ids.tolist()
    rows = zip(ids_list, offsets[:-1], offsets[1:], strict=True)
    if return_vectors:
        return [(cid, flat_ids[a:b], flat_vecs[a:b]) for cid, a, b in rows]
    return [(cid, flat_ids[a:b]) for cid, a, b in rows]


# ============================================================================
# Vectorized NumPy distance functions
# ============================================================================


def _dist_vec_ortho(coord_i: np.ndarray, coords_j: np.ndarray, box_size: np.ndarray) -> tuple:
    """Vectorised minimum-image displacements and squared distances, orthogonal box."""
    rij = coord_i - coords_j
    rij -= box_size * np.round(rij / box_size)
    return np.einsum("ij,ij->i", rij, rij), rij


def _dist_vec_tri(frac_i: np.ndarray, frac_j: np.ndarray, cell: np.ndarray, half_height_sq: float) -> tuple:
    """Vectorised minimum-image displacements and squared distances, triclinic box.

    Mirrors _dist_and_vec_tri_exact: nearest-integer rounding first, then a
    search over the 27 surrounding images for the rows that are longer than
    half the smallest perpendicular height and so are not provably minimal.
    """
    delta_frac = frac_i - frac_j
    delta_frac -= np.round(delta_frac)
    rij = delta_frac @ cell
    dist_sq = np.einsum("ij,ij->i", rij, rij)

    needs_search = dist_sq > half_height_sq
    if np.any(needs_search):
        trial_frac = delta_frac[needs_search][:, np.newaxis, :] + SHIFT_GRID_3D[np.newaxis, :, :]  # (m, 27, 3)
        trial_rij = trial_frac @ cell
        trial_sq = np.einsum("ijk,ijk->ij", trial_rij, trial_rij)
        best = np.argmin(trial_sq, axis=1)
        rows = np.arange(len(best))
        rij[needs_search] = trial_rij[rows, best]
        dist_sq[needs_search] = trial_sq[rows, best]

    return dist_sq, rij


# ============================================================================
# NumPy fallback paths for environments without Numba
# ============================================================================


def _numpy_fallback(
    coords: np.ndarray,
    coords_frac: np.ndarray | None,
    types: np.ndarray,
    cell: np.ndarray,
    box_size: np.ndarray | None,
    atom_cells: np.ndarray,
    n_cells: np.ndarray,
    cutoff_sq: float,
    pair_types: np.ndarray,
    pair_cutoffs_sq: np.ndarray,
    use_pair_cutoffs: bool,  # noqa: FBT001
    use_target_filter: bool,  # noqa: FBT001
    use_neighbor_filter: bool,  # noqa: FBT001
    target_types: list[int] | None,
    neighbor_types: list[int] | None,
    n_atoms: int,
    return_vectors: bool,  # noqa: FBT001
    is_orthogonal: bool,  # noqa: FBT001
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared NumPy fallback for both orthogonal and triclinic boxes.

    Returns:
        flat_j: (M,) int32 neighbor array indices, grouped by central atom.
        counts: (N,) int32 neighbors per atom; zero for atoms the filters skip.
        flat_vecs: (M, 3) float64 bond vectors r_i - r_j, or (0, 3) when not requested.
    """
    half_height_sq = 0.0 if is_orthogonal else _half_min_height(cell) ** 2

    cells: defaultdict[tuple, list[int]] = defaultdict(list)
    for idx, c in enumerate(atom_cells):
        cells[tuple(c)].append(idx)

    target_set = set(target_types) if use_target_filter and target_types else None
    neighbor_set = set(neighbor_types) if use_neighbor_filter and neighbor_types else None

    if use_pair_cutoffs:
        _max_type = int(types.max()) + 1
        cutoff_matrix = np.zeros((_max_type, _max_type), dtype=np.float64)
        for _ti in range(_max_type):
            for _tj in range(_max_type):
                cutoff_matrix[_ti, _tj] = _get_pair_cutoff_sq_python(_ti, _tj, pair_types, pair_cutoffs_sq)
    else:
        cutoff_matrix = np.empty((0, 0), dtype=np.float64)

    stencil = _stencil_grid(n_cells)
    counts = np.zeros(n_atoms, dtype=np.int32)
    flat_j_parts: list[np.ndarray] = []
    vec_parts: list[np.ndarray] = []

    for i in range(n_atoms):
        type_i = int(types[i])
        if target_set is not None and type_i not in target_set:
            continue

        candidates: list[int] = []
        for neighbor_cell in (atom_cells[i] + stencil) % n_cells:
            candidates.extend(cells[tuple(neighbor_cell)])
        candidates = [j for j in candidates if j != i]
        if neighbor_set is not None:
            candidates = [j for j in candidates if int(types[j]) in neighbor_set]
        if not candidates:
            continue

        candidates_arr = np.array(candidates, dtype=np.int32)

        if is_orthogonal:
            assert box_size is not None
            dist_sq_arr, rij = _dist_vec_ortho(coords[i], coords[candidates_arr], box_size)
        else:
            assert coords_frac is not None
            dist_sq_arr, rij = _dist_vec_tri(coords_frac[i], coords_frac[candidates_arr], cell, half_height_sq)

        if use_pair_cutoffs:
            mask = dist_sq_arr <= cutoff_matrix[type_i, types[candidates_arr]]
        else:
            mask = dist_sq_arr <= cutoff_sq

        kept = candidates_arr[mask]
        counts[i] = len(kept)
        flat_j_parts.append(kept)
        if return_vectors:
            vec_parts.append(rij[mask])

    flat_j = np.concatenate(flat_j_parts) if flat_j_parts else np.empty(0, dtype=np.int32)
    flat_vecs = np.concatenate(vec_parts) if vec_parts else np.empty((0, 3), dtype=np.float64)
    return flat_j, counts, flat_vecs


# ============================================================================
# ID extraction helper
# ============================================================================


def _extract_atom_ids(atoms: Atoms | tuple[np.ndarray, ...]) -> np.ndarray:
    """Return the real atom IDs from an ASE Atoms object or tuple.

    Priority:
      1. atoms.arrays['id'] — present when read from LAMMPS/XYZ with id column
      2. 1-based sequential IDs as fallback (OVITO default)
    """
    if isinstance(atoms, Atoms):
        for key in ("id", "ID"):
            if key in atoms.arrays:
                return atoms.arrays[key].astype(np.int64)
        return np.arange(1, len(atoms) + 1, dtype=np.int64)
    coords, *_ = atoms
    n_atoms = len(np.asarray(coords))
    return np.arange(1, n_atoms + 1, dtype=np.int64)


# ============================================================================
# Main public function
# ============================================================================


def get_neighbors(
    atoms: Atoms | tuple[np.ndarray, np.ndarray, np.ndarray],
    cutoff: float | dict[tuple[int, int], float],
    target_types: int | list[int] | None = None,
    neighbor_types: int | list[int] | None = None,
    *,
    return_vectors: bool = False,
    use_numba: bool | None = None,
) -> list[tuple]:
    """Find all neighbors within cutoff for each atom.

    Returns a list of tuples where all IDs are the real atom IDs from the
    structure file (e.g. non-sequential LAMMPS/XYZ ids), not array indices.

    Args:
        atoms: Either an ASE Atoms object or a tuple (coords, types, cell_matrix)
               where cell_matrix is a (3,3) array with lattice vectors as rows.
               For an Atoms object ``atoms.pbc`` is honoured per lattice vector
               (slabs, wires and isolated molecules are handled); the tuple form
               is treated as fully periodic. Coordinates need not be wrapped.
        cutoff: Cutoff radius in Angstrom. Either:
                  - A single float applied uniformly to all pairs.
                  - A dict mapping (atomic_number_i, atomic_number_j) to a
                    pair-specific cutoff in Angstrom. Symmetric: (8, 14) and
                    (14, 8) are equivalent. Pairs not listed default to the
                    maximum cutoff in the dict. The cell list is built on the
                    maximum cutoff so only one build is needed.
        target_types: Atomic numbers of atoms to find neighbors for. A bare
                      atomic number is accepted and treated as a one-element
                      list. None means all atoms.
        neighbor_types: Atomic numbers that count as valid neighbors. A bare
                        atomic number is accepted. None means all types.
        return_vectors: If True, each output tuple gains a third element — a
                        (k, 3) float64 array of Cartesian minimum-image bond
                        vectors r_i - r_j in Angstrom, i.e. pointing from the
                        neighbour to the central atom (the opposite sign of
                        ASE's neighbor_list "D"). Scalar distances are
                        np.linalg.norm(vectors, axis=1).
        use_numba: Force Numba on/off. None = auto-detect.

    Returns:
        If return_vectors=False (default):
            [(central_id, [neighbor_ids]), ...]

        If return_vectors=True:
            [(central_id, [neighbor_ids], vectors_shape_k3), ...]

    Examples:
        Scalar cutoff (backward compatible)::

            >>> neighbors = get_neighbors(atoms, cutoff=3.5)
            >>> for central_id, nn_ids in neighbors:
            ...     print(central_id, nn_ids)

        Per-pair cutoffs for a Na2O-Al2O3-SiO2 glass::

            >>> cutoff = {(14, 8): 2.0, (13, 8): 1.9, (11, 8): 2.7}
            >>> neighbors = get_neighbors(atoms, cutoff=cutoff)

        With bond vectors for Steinhardt parameters::

            >>> result = get_neighbors(atoms, cutoff=3.5, return_vectors=True)
            >>> for central_id, nn_ids, vecs in result:
            ...     distances = np.linalg.norm(vecs, axis=1)
            ...     print(f"atom {central_id}: mean bond length = {distances.mean():.3f} A")

        Quick lookup by original atom ID::

            >>> nl = {cid: nn for cid, nn, *_ in get_neighbors(atoms, cutoff=3.5)}
            >>> nl[43586]
    """
    if use_numba is None:
        use_numba = NUMBA_AVAILABLE

    target_types = _normalize_type_filter(target_types, "target_types")
    neighbor_types = _normalize_type_filter(neighbor_types, "neighbor_types")

    # ------------------------------------------------------------------
    # Parse input
    # ------------------------------------------------------------------
    atom_ids = _extract_atom_ids(atoms)

    if not isinstance(atoms, Atoms):
        coords, types, cell = atoms
        coords = np.asarray(coords, dtype=np.float64)
        types = np.asarray(types, dtype=np.int32)
        cell = np.asarray(cell, dtype=np.float64)
        pbc = np.ones(3, dtype=bool)
    else:
        coords = atoms.get_positions()
        types = atoms.get_atomic_numbers().astype(np.int32)
        cell = np.array(atoms.get_cell().array, dtype=np.float64)
        pbc = np.asarray(atoms.pbc, dtype=bool)

    n_atoms = len(coords)
    _validate_coords(coords)

    # ------------------------------------------------------------------
    # Parse cutoff
    # ------------------------------------------------------------------
    max_cutoff, pair_types, pair_cutoffs_sq, use_pair_cutoffs = _parse_cutoff(cutoff, types)
    cutoff_sq = max_cutoff * max_cutoff
    if n_atoms == 0:
        return []

    target_arr = np.array(target_types, dtype=np.int32) if target_types is not None else np.empty(0, dtype=np.int32)
    neighbor_arr = (
        np.array(neighbor_types, dtype=np.int32) if neighbor_types is not None else np.empty(0, dtype=np.int32)
    )
    use_target_filter = target_types is not None
    use_neighbor_filter = neighbor_types is not None

    # ------------------------------------------------------------------
    # Geometry: estimate density on the real cell, then pad non-periodic axes
    # ------------------------------------------------------------------
    initial_max_neighbors = _estimate_max_neighbors(coords, cell, max_cutoff)
    cell, coords = _pad_nonperiodic(cell, coords, pbc, max_cutoff)
    _validate_cell(cell)
    _warn_if_cutoff_exceeds_minimum_image(cell, max_cutoff, pbc)
    is_orthogonal, box_size, coords_frac, atom_cells, n_cells, cell_start, cell_atoms = _build_cell_list(
        coords, cell, max_cutoff
    )

    # ------------------------------------------------------------------
    # Build neighbor list
    # ------------------------------------------------------------------
    if use_numba and NUMBA_AVAILABLE:
        stencil_start, stencil_count = _cell_offsets(n_cells)
        kwargs: dict[str, Any] = {
            "types": types,
            "atom_cells": atom_cells,
            "n_cells": n_cells,
            "cell_start": cell_start,
            "cell_atoms": cell_atoms,
            "stencil_start": stencil_start,
            "stencil_count": stencil_count,
            "cutoff_sq": cutoff_sq,
            "target_types": target_arr,
            "neighbor_types": neighbor_arr,
            "use_target_filter": use_target_filter,
            "use_neighbor_filter": use_neighbor_filter,
            "max_neighbors": initial_max_neighbors,
            "pair_types": pair_types,
            "pair_cutoffs_sq": pair_cutoffs_sq,
            "use_pair_cutoffs": use_pair_cutoffs,
            "return_vectors": return_vectors,
        }
        if is_orthogonal:
            build_fn = _build_nl_ortho_numba
            kwargs.update(coords=coords, box_size=box_size)
        else:
            build_fn = _build_nl_tri_numba
            kwargs.update(
                coords_frac=coords_frac,
                cell=cell,
                half_height_sq=_image_search_bound(cell, n_cells),
            )
        neighbor_list, counts, vector_list = _grow_until_fits(build_fn, kwargs)
        flat_j, flat_vecs = _padded_to_csr(neighbor_list, counts, vector_list, return_vectors)
    else:
        flat_j, counts, flat_vecs = _numpy_fallback(
            coords=coords,
            coords_frac=coords_frac,
            types=types,
            cell=cell,
            box_size=box_size,
            atom_cells=atom_cells,
            n_cells=n_cells,
            cutoff_sq=cutoff_sq,
            pair_types=pair_types,
            pair_cutoffs_sq=pair_cutoffs_sq,
            use_pair_cutoffs=use_pair_cutoffs,
            use_target_filter=use_target_filter,
            use_neighbor_filter=use_neighbor_filter,
            target_types=target_types,
            neighbor_types=neighbor_types,
            n_atoms=n_atoms,
            return_vectors=return_vectors,
            is_orthogonal=is_orthogonal,
        )

    return _build_neighbor_output(flat_j, counts, flat_vecs, atom_ids, return_vectors)


# ============================================================================
# Utilities
# ============================================================================


def _estimate_max_neighbors(coords: np.ndarray, cell: np.ndarray, cutoff: float) -> int:
    """Estimate a safe upper bound for the number of neighbors per atom.

    Args:
        coords: Cartesian coordinates.
        cell: Lattice vector matrix.
        cutoff: Neighbor cutoff distance.

    Returns:
        An estimated maximum number of neighbors.
    """
    n_atoms = len(coords)
    volume = abs(np.linalg.det(cell))
    if volume < _MIN_VOLUME or n_atoms == 0:
        return 200
    density = n_atoms / volume
    sphere_vol = (4.0 / 3.0) * np.pi * cutoff**3
    return max(int(density * sphere_vol * 3.0) + 32, 64)
