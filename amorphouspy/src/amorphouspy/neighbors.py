"""Neighbor list module for multicomponent glass systems.

Supports both orthogonal and triclinic boxes.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
from ase import Atoms

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

    prange = range

# Precompute 3D shift grid for neighbor cells (used in NumPy path)
SHIFT_GRID_3D = np.stack(
    np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"),
    axis=-1,
).reshape(-1, 3)

_MIN_VOLUME: float = 1e-10


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
                cutoff value in the dict.
        types:  Integer array of atomic numbers for all atoms.

    Returns:
        max_cutoff:       Largest cutoff across all pairs (used for cell-list).
        pair_types:       (M, 2) int32 array of unique type pairs.
        pair_cutoffs_sq:  (M,) float64 array of squared cutoffs per pair.
        use_pair_cutoffs: True when per-pair mode is active.
    """
    if isinstance(cutoff, (int, float)):
        return float(cutoff), np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float64), False

    unique_types = np.unique(types).tolist()
    max_rc = float(max(cutoff.values()))

    # Build a dict covering all ordered pairs, defaulting to max_rc
    pair_dict: dict[tuple[int, int], float] = {}
    for ti in unique_types:
        for tj in unique_types:
            pair_dict[(ti, tj)] = max_rc

    # Override with user-specified values (both orderings)
    for (ti, tj), rc in cutoff.items():
        pair_dict[(int(ti), int(tj))] = float(rc)
        pair_dict[(int(tj), int(ti))] = float(rc)

    pairs = list(pair_dict.keys())
    cutoffs_sq = [pair_dict[p] ** 2 for p in pairs]

    pair_types = np.array(pairs, dtype=np.int32)
    pair_cutoffs_sq = np.array(cutoffs_sq, dtype=np.float64)
    return max_rc, pair_types, pair_cutoffs_sq, True


@jit(nopython=True, fastmath=True, cache=True)
def _lookup_cutoff_sq(
    ti: int,
    tj: int,
    pair_types: np.ndarray,
    pair_cutoffs_sq: np.ndarray,
) -> float:
    """Return squared cutoff for (ti, tj) via linear search through pair table.

    Linear search is optimal here: the number of unique type pairs in a
    typical glass is <=10, the entire table fits in L1 cache, and hashing
    is not available in Numba nopython mode.
    """
    for k in range(len(pair_types)):
        if pair_types[k, 0] == ti and pair_types[k, 1] == tj:
            return pair_cutoffs_sq[k]
    return pair_cutoffs_sq[0]  # fallback, should not be reached


def _get_pair_cutoff_sq_python(
    ti: int,
    tj: int,
    pair_types: np.ndarray,
    pair_cutoffs_sq: np.ndarray,
) -> float:
    """Python equivalent of _lookup_cutoff_sq for the NumPy fallback path."""
    for k in range(len(pair_types)):
        if pair_types[k, 0] == ti and pair_types[k, 1] == tj:
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
    ha = volume / np.linalg.norm(np.cross(b, c))
    hb = volume / np.linalg.norm(np.cross(a, c))
    hc = volume / np.linalg.norm(np.cross(a, b))
    return np.array([ha, hb, hc])


# ============================================================================
# Cell list construction
# ============================================================================


def compute_cell_list_orthogonal(
    coords: np.ndarray,
    box_size: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct a flat CSR-style cell list for orthogonal boxes.

    Args:
        coords: Cartesian coordinates of atoms.
        box_size: Lengths of the simulation box.
        cutoff: Cell list cutoff.

    Returns:
        A tuple of (atom_cells, n_cells, cell_start, order).
    """
    n_cells = np.maximum(1, np.floor(box_size / cutoff)).astype(np.int32)
    inv_cell_size = n_cells / box_size
    atom_cells = np.floor(coords * inv_cell_size).astype(np.int64) % n_cells
    n_total = int(n_cells[0]) * int(n_cells[1]) * int(n_cells[2])
    flat_ids = (
        atom_cells[:, 0] * int(n_cells[1]) * int(n_cells[2])
        + atom_cells[:, 1] * int(n_cells[2])
        + atom_cells[:, 2]
    ).astype(np.int32)
    order = np.argsort(flat_ids, kind="stable")
    sorted_flat = flat_ids[order]
    cell_start = np.zeros(n_total + 1, dtype=np.int32)
    np.add.at(cell_start[1:], sorted_flat, 1)
    np.cumsum(cell_start, out=cell_start)
    return atom_cells.astype(np.int32), n_cells, cell_start, order.astype(np.int32)


def compute_cell_list_triclinic(
    coords: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct a flat CSR-style cell list for triclinic boxes (fractional coords).

    Args:
        coords: Cartesian coordinates of atoms.
        cell: Lattice vector matrix.
        cutoff: Cell list cutoff.

    Returns:
        A tuple of (coords_frac, atom_cells, n_cells, cell_start, order).
    """
    inv_cell = np.linalg.inv(cell)
    coords_frac = (inv_cell @ coords.T).T % 1.0
    heights = cell_perpendicular_heights(cell)
    n_cells = np.maximum(1, np.floor(heights / cutoff)).astype(np.int32)
    n_total = int(n_cells[0]) * int(n_cells[1]) * int(n_cells[2])
    atom_cells = (np.floor(coords_frac * n_cells).astype(np.int64) % n_cells).astype(np.int32)
    flat_ids = (
        atom_cells[:, 0] * int(n_cells[1]) * int(n_cells[2])
        + atom_cells[:, 1] * int(n_cells[2])
        + atom_cells[:, 2]
    ).astype(np.int32)
    order = np.argsort(flat_ids, kind="stable")
    sorted_flat = flat_ids[order]
    cell_start = np.zeros(n_total + 1, dtype=np.int32)
    np.add.at(cell_start[1:], sorted_flat, 1)
    np.cumsum(cell_start, out=cell_start)
    return coords_frac, atom_cells, n_cells, cell_start, order.astype(np.int32)


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
        (dx, dy, dz, dist_sq) — displacement i->j and its squared length.
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
    fi: np.ndarray,
    fj: np.ndarray,
    cell: np.ndarray,
) -> tuple[float, float, float, float]:
    """Minimum-image displacement vector and squared distance, triclinic box.

    Returns:
        (dx, dy, dz, dist_sq) — Cartesian displacement i->j and its squared length.
    """
    dfx = fi[0] - fj[0]
    dfx -= round(dfx)
    dfy = fi[1] - fj[1]
    dfy -= round(dfy)
    dfz = fi[2] - fj[2]
    dfz -= round(dfz)
    dx = dfx * cell[0, 0] + dfy * cell[1, 0] + dfz * cell[2, 0]
    dy = dfx * cell[0, 1] + dfy * cell[1, 1] + dfz * cell[2, 1]
    dz = dfx * cell[0, 2] + dfy * cell[1, 2] + dfz * cell[2, 2]
    return dx, dy, dz, dx * dx + dy * dy + dz * dz


# ============================================================================
# Numba kernel — orthogonal box
# ============================================================================


@jit(nopython=True, parallel=True, cache=True)
def _build_nl_ortho_numba(  # noqa: PLR0912, C901
    coords: np.ndarray,
    types: np.ndarray,
    box_size: np.ndarray,
    atom_cells: np.ndarray,
    n_cells: np.ndarray,
    cell_start: np.ndarray,
    cell_atoms: np.ndarray,
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
    ny = n_cells[1]
    nz = n_cells[2]

    neighbor_list = np.full((n_atoms, max_neighbors), -1, dtype=np.int32)
    neighbor_counts = np.zeros(n_atoms, dtype=np.int32)
    vector_list = np.zeros((n_atoms, max_neighbors, 3), dtype=np.float32)

    for i in prange(n_atoms):
        ti = types[i]
        if use_target_filter:
            found = False
            for t in target_types:
                if ti == t:
                    found = True
                    break
            if not found:
                continue

        ci = atom_cells[i]
        count = 0

        for dix in range(-1, 2):
            cjx = (ci[0] + dix) % n_cells[0]
            for diy in range(-1, 2):
                cjy = (ci[1] + diy) % n_cells[1]
                for diz in range(-1, 2):
                    cjz = (ci[2] + diz) % n_cells[2]
                    flat_id = cjx * ny * nz + cjy * nz + cjz
                    start = cell_start[flat_id]
                    end = cell_start[flat_id + 1]

                    for k in range(start, end):
                        j = cell_atoms[k]
                        if j == i:
                            continue
                        tj = types[j]

                        if use_neighbor_filter:
                            ok = False
                            for t in neighbor_types:
                                if tj == t:
                                    ok = True
                                    break
                            if not ok:
                                continue

                        rc_sq = (
                            _lookup_cutoff_sq(ti, tj, pair_types, pair_cutoffs_sq)
                            if use_pair_cutoffs else cutoff_sq
                        )

                        dx, dy, dz, dist_sq = _dist_and_vec_ortho(coords[i], coords[j], box_size)

                        if dist_sq <= rc_sq:
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
def _build_nl_tri_numba(  # noqa: PLR0912, C901
    coords_frac: np.ndarray,
    types: np.ndarray,
    cell: np.ndarray,
    atom_cells: np.ndarray,
    n_cells: np.ndarray,
    cell_start: np.ndarray,
    cell_atoms: np.ndarray,
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
    """Build neighbor list for a triclinic box using Numba.

    Returns:
        neighbor_list:   (N, max_neighbors) int32
        neighbor_counts: (N,) int32
        vector_list:     (N, max_neighbors, 3) float32 — Cartesian bond vectors i->j.
    """
    n_atoms = len(coords_frac)
    ny = n_cells[1]
    nz = n_cells[2]

    neighbor_list = np.full((n_atoms, max_neighbors), -1, dtype=np.int32)
    neighbor_counts = np.zeros(n_atoms, dtype=np.int32)
    vector_list = np.zeros((n_atoms, max_neighbors, 3), dtype=np.float32)

    for i in prange(n_atoms):
        ti = types[i]
        if use_target_filter:
            found = False
            for t in target_types:
                if ti == t:
                    found = True
                    break
            if not found:
                continue

        ci = atom_cells[i]
        count = 0

        for dix in range(-1, 2):
            cjx = (ci[0] + dix) % n_cells[0]
            for diy in range(-1, 2):
                cjy = (ci[1] + diy) % n_cells[1]
                for diz in range(-1, 2):
                    cjz = (ci[2] + diz) % n_cells[2]
                    flat_id = cjx * ny * nz + cjy * nz + cjz
                    start = cell_start[flat_id]
                    end = cell_start[flat_id + 1]

                    for k in range(start, end):
                        j = cell_atoms[k]
                        if j == i:
                            continue
                        tj = types[j]

                        if use_neighbor_filter:
                            ok = False
                            for t in neighbor_types:
                                if tj == t:
                                    ok = True
                                    break
                            if not ok:
                                continue

                        rc_sq = (
                            _lookup_cutoff_sq(ti, tj, pair_types, pair_cutoffs_sq)
                            if use_pair_cutoffs else cutoff_sq
                        )

                        dx, dy, dz, dist_sq = _dist_and_vec_tri(coords_frac[i], coords_frac[j], cell)

                        if dist_sq <= rc_sq:
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
# Numba output converter
# ============================================================================


def _numba_to_list(
    neighbor_list: np.ndarray,
    neighbor_counts: np.ndarray,
    vector_list: np.ndarray,
    n_atoms: int,
    max_neighbors: int,
    build_fn: Callable,
    build_kwargs: dict[str, Any],
    return_vectors: bool,  # noqa: FBT001
) -> tuple[list[list[int]], list[np.ndarray]]:
    """Convert Numba output arrays to Python lists, retrying on buffer overflow."""
    overflow = int(neighbor_counts.max()) if n_atoms > 0 else 0

    while overflow > max_neighbors:
        max_neighbors = int(overflow * 1.2) + 1
        build_kwargs["max_neighbors"] = max_neighbors
        neighbor_list, neighbor_counts, vector_list = build_fn(**build_kwargs)
        overflow = int(neighbor_counts.max())

    idx_neighbors: list[list[int]] = []
    vec_neighbors: list[np.ndarray] = []

    for i in range(n_atoms):
        c = int(neighbor_counts[i])
        idx_neighbors.append(neighbor_list[i, :c].tolist())
        vec_neighbors.append(
            vector_list[i, :c].astype(np.float64) if return_vectors
            else np.empty((0, 3), dtype=np.float64)
        )

    return idx_neighbors, vec_neighbors


# ============================================================================
# Vectorized NumPy distance functions
# ============================================================================


def _dist_vec_ortho(coord_i: np.ndarray, coords_j: np.ndarray, box_size: np.ndarray) -> tuple:
    """Vectorised minimum-image displacements and squared distances, orthogonal box."""
    rij = coord_i - coords_j
    rij -= box_size * np.round(rij / box_size)
    return np.einsum("ij,ij->i", rij, rij), rij


def _dist_vec_tri(frac_i: np.ndarray, frac_j: np.ndarray, cell: np.ndarray) -> tuple:
    """Vectorised minimum-image displacements and squared distances, triclinic box."""
    delta_frac = frac_i - frac_j
    delta_frac -= np.round(delta_frac)
    rij = delta_frac @ cell
    return np.einsum("ij,ij->i", rij, rij), rij


def _get_neighbor_cells_numpy(ci: np.ndarray, n_cells: np.ndarray) -> np.ndarray:
    """Return the 27 neighbour cell indices (with periodic wrap) for cell ci."""
    return (ci + SHIFT_GRID_3D) % n_cells


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
    use_tf: bool,  # noqa: FBT001
    use_nf: bool,  # noqa: FBT001
    target_types: list[int] | None,
    neighbor_types: list[int] | None,
    n_atoms: int,
    return_vectors: bool,  # noqa: FBT001
    is_orthogonal: bool,  # noqa: FBT001
) -> tuple[list[list[int]], list[np.ndarray]]:
    """Shared NumPy fallback for both orthogonal and triclinic boxes."""
    cells: defaultdict[tuple, list[int]] = defaultdict(list)
    for idx, c in enumerate(atom_cells):
        cells[tuple(c)].append(idx)

    target_set = set(target_types) if use_tf and target_types else None
    neighbor_set = set(neighbor_types) if use_nf and neighbor_types else None

    idx_neighbors: list[list[int]] = [[] for _ in range(n_atoms)]
    vec_neighbors: list[np.ndarray] = [np.empty((0, 3), dtype=np.float64) for _ in range(n_atoms)]

    for i in range(n_atoms):
        ti = int(types[i])
        if target_set is not None and ti not in target_set:
            continue

        ci = atom_cells[i]
        candidates: list[int] = []
        for cj in _get_neighbor_cells_numpy(ci, n_cells):
            candidates.extend(cells[tuple(cj)])
        candidates = [j for j in candidates if j != i]
        if neighbor_set is not None:
            candidates = [j for j in candidates if int(types[j]) in neighbor_set]
        if not candidates:
            continue

        ca = np.array(candidates, dtype=np.int32)

        if is_orthogonal:
            dsq, rij = _dist_vec_ortho(coords[i], coords[ca], box_size)
        else:
            dsq, rij = _dist_vec_tri(coords_frac[i], coords_frac[ca], cell)

        if use_pair_cutoffs:
            rc_sq_arr = np.array(
                [_get_pair_cutoff_sq_python(ti, int(types[j]), pair_types, pair_cutoffs_sq) for j in ca],
                dtype=np.float64,
            )
            mask = dsq <= rc_sq_arr
        else:
            mask = dsq <= cutoff_sq

        idx_neighbors[i] = ca[mask].tolist()
        if return_vectors:
            vec_neighbors[i] = rij[mask]

    return idx_neighbors, vec_neighbors


# ============================================================================
# ID extraction helper
# ============================================================================


def _extract_atom_ids(atoms: Atoms | tuple) -> np.ndarray:
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
    n = len(np.asarray(coords))
    return np.arange(1, n + 1, dtype=np.int64)


# ============================================================================
# Main public function
# ============================================================================


def get_neighbors(
    atoms: Atoms | tuple[np.ndarray, np.ndarray, np.ndarray],
    cutoff: float | dict[tuple[int, int], float],
    target_types: list[int] | None = None,
    neighbor_types: list[int] | None = None,
    return_vectors: bool = False,  # noqa: FBT001
    use_numba: bool | None = None,
) -> list[tuple]:
    """Find all neighbors within cutoff for each atom.

    Returns a list of tuples where all IDs are the real atom IDs from the
    structure file (e.g. non-sequential LAMMPS/XYZ ids), not array indices.

    Args:
        atoms: Either an ASE Atoms object or a tuple (coords, types, cell_matrix)
               where cell_matrix is a (3,3) array with lattice vectors as rows.
        cutoff: Cutoff radius in Angstrom. Either:
                  - A single float applied uniformly to all pairs.
                  - A dict mapping (atomic_number_i, atomic_number_j) to a
                    pair-specific cutoff in Angstrom. Symmetric: (8, 14) and
                    (14, 8) are equivalent. Pairs not listed default to the
                    maximum cutoff in the dict. The cell list is built on the
                    maximum cutoff so only one build is needed.
        target_types: Atomic numbers of atoms to find neighbors for.
                      None means all atoms.
        neighbor_types: Atomic numbers that count as valid neighbors.
                        None means all types.
        return_vectors: If True, each output tuple gains a third element — a
                        (k, 3) float64 array of Cartesian minimum-image bond
                        vectors (i -> j) in Angstrom. Scalar distances are
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

    # ------------------------------------------------------------------
    # Parse input
    # ------------------------------------------------------------------
    atom_ids = _extract_atom_ids(atoms)

    if isinstance(atoms, Atoms):
        atoms_copy = atoms.copy()
        atoms_copy.wrap()
        coords = atoms_copy.get_positions()
        types = atoms_copy.get_atomic_numbers().astype(np.int32)
        cell = atoms_copy.get_cell().array
    else:
        coords, types, cell = atoms
        coords = np.asarray(coords, dtype=np.float64)
        types = np.asarray(types, dtype=np.int32)
        cell = np.asarray(cell, dtype=np.float64)

    n_atoms = len(coords)
    is_orthogonal = np.allclose(cell - np.diag(np.diag(cell)), 0.0, atol=1e-10)

    # ------------------------------------------------------------------
    # Parse cutoff
    # ------------------------------------------------------------------
    max_cutoff, pair_types, pair_cutoffs_sq, use_pair_cutoffs = _parse_cutoff(cutoff, types)
    cutoff_sq = max_cutoff * max_cutoff

    target_arr = np.array(target_types, dtype=np.int32) if target_types is not None else np.empty(0, dtype=np.int32)
    neigh_arr = np.array(neighbor_types, dtype=np.int32) if neighbor_types is not None else np.empty(0, dtype=np.int32)
    use_tf = target_types is not None
    use_nf = neighbor_types is not None

    _initial_max_neighbors = _estimate_max_neighbors(coords, cell, max_cutoff)

    # ------------------------------------------------------------------
    # Build neighbor list
    # ------------------------------------------------------------------
    if is_orthogonal:
        box_size = np.diag(cell)
        atom_cells, n_cells, cell_start, cell_atoms = compute_cell_list_orthogonal(
            coords, box_size, max_cutoff
        )
        if use_numba and NUMBA_AVAILABLE:
            kwargs = {
                "coords": coords, "types": types, "box_size": box_size,
                "atom_cells": atom_cells, "n_cells": n_cells,
                "cell_start": cell_start, "cell_atoms": cell_atoms,
                "cutoff_sq": cutoff_sq, "target_types": target_arr,
                "neighbor_types": neigh_arr, "use_target_filter": use_tf,
                "use_neighbor_filter": use_nf, "max_neighbors": _initial_max_neighbors,
                "pair_types": pair_types, "pair_cutoffs_sq": pair_cutoffs_sq,
                "use_pair_cutoffs": use_pair_cutoffs, "return_vectors": return_vectors,
            }
            nl, nc, vl = _build_nl_ortho_numba(**kwargs)
            idx_neighbors, vec_neighbors = _numba_to_list(
                nl, nc, vl, n_atoms, _initial_max_neighbors,
                _build_nl_ortho_numba, kwargs, return_vectors,
            )
        else:
            idx_neighbors, vec_neighbors = _numpy_fallback(
                coords=coords, coords_frac=None, types=types, cell=cell,
                box_size=box_size, atom_cells=atom_cells, n_cells=n_cells,
                cutoff_sq=cutoff_sq, pair_types=pair_types,
                pair_cutoffs_sq=pair_cutoffs_sq, use_pair_cutoffs=use_pair_cutoffs,
                use_tf=use_tf, use_nf=use_nf, target_types=target_types,
                neighbor_types=neighbor_types, n_atoms=n_atoms,
                return_vectors=return_vectors, is_orthogonal=True,
            )
    else:
        coords_frac, atom_cells, n_cells, cell_start, cell_atoms = compute_cell_list_triclinic(
            coords, cell, max_cutoff
        )
        if use_numba and NUMBA_AVAILABLE:
            kwargs = {
                "coords_frac": coords_frac, "types": types, "cell": cell,
                "atom_cells": atom_cells, "n_cells": n_cells,
                "cell_start": cell_start, "cell_atoms": cell_atoms,
                "cutoff_sq": cutoff_sq, "target_types": target_arr,
                "neighbor_types": neigh_arr, "use_target_filter": use_tf,
                "use_neighbor_filter": use_nf, "max_neighbors": _initial_max_neighbors,
                "pair_types": pair_types, "pair_cutoffs_sq": pair_cutoffs_sq,
                "use_pair_cutoffs": use_pair_cutoffs, "return_vectors": return_vectors,
            }
            nl, nc, vl = _build_nl_tri_numba(**kwargs)
            idx_neighbors, vec_neighbors = _numba_to_list(
                nl, nc, vl, n_atoms, _initial_max_neighbors,
                _build_nl_tri_numba, kwargs, return_vectors,
            )
        else:
            idx_neighbors, vec_neighbors = _numpy_fallback(
                coords=coords, coords_frac=coords_frac, types=types, cell=cell,
                box_size=None, atom_cells=atom_cells, n_cells=n_cells,
                cutoff_sq=cutoff_sq, pair_types=pair_types,
                pair_cutoffs_sq=pair_cutoffs_sq, use_pair_cutoffs=use_pair_cutoffs,
                use_tf=use_tf, use_nf=use_nf, target_types=target_types,
                neighbor_types=neighbor_types, n_atoms=n_atoms,
                return_vectors=return_vectors, is_orthogonal=False,
            )

    # ------------------------------------------------------------------
    # Translate array indices -> real atom IDs
    # ------------------------------------------------------------------
    if return_vectors:
        return [
            (int(atom_ids[i]), [int(atom_ids[j]) for j in idx_neighbors[i]], vec_neighbors[i])
            for i in range(n_atoms)
        ]
    return [
        (int(atom_ids[i]), [int(atom_ids[j]) for j in idx_neighbors[i]])
        for i in range(n_atoms)
    ]


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
