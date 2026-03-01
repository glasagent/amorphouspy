"""module to get neighbors within a cutoff distance.

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

# Threshold used to detect a near-zero cell volume
_MIN_VOLUME: float = 1e-10


# ============================================================================
# Triclinic geometry helpers
# ============================================================================


def cell_perpendicular_heights(cell: np.ndarray) -> np.ndarray:
    """Compute the perpendicular height of each cell face.

    For a triclinic cell with lattice vectors a, b, c (rows of `cell`),
    the perpendicular height along axis i is:

        h_i = V / |b_j x b_k|

    where V = |det(cell)| and b_j, b_k are the other two vectors.
    This is the physically correct minimum repeat distance along each axis
    and must be used instead of |a_i| when converting a Cartesian cutoff
    to fractional space.

    Args:
        cell: (3, 3) array with lattice vectors as rows.

    Returns:
        heights: (3,) perpendicular heights h_a, h_b, h_c.
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct a flat cell list for orthogonal boxes."""
    n_cells = np.maximum(1, np.floor(box_size / cutoff)).astype(np.int32)
    inv_cell_size = n_cells / box_size

    atom_cells = np.floor(coords * inv_cell_size).astype(np.int64) % n_cells  # (N,3)

    n_total = int(n_cells[0]) * int(n_cells[1]) * int(n_cells[2])
    flat_ids = (
        atom_cells[:, 0] * int(n_cells[1]) * int(n_cells[2]) + atom_cells[:, 1] * int(n_cells[2]) + atom_cells[:, 2]
    ).astype(np.int32)

    order = np.argsort(flat_ids, kind="stable")
    sorted_flat = flat_ids[order]

    cell_start = np.zeros(n_total + 1, dtype=np.int32)
    np.add.at(cell_start[1:], sorted_flat, 1)
    np.cumsum(cell_start, out=cell_start)

    cell_atoms = order.astype(np.int32)

    return atom_cells.astype(np.int32), n_cells, cell_start, cell_atoms


def compute_cell_list_triclinic(
    coords: np.ndarray,
    cell: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct a skewed cell list for triclinic boxes in fractional coordinates."""
    inv_cell = np.linalg.inv(cell)
    coords_frac = (inv_cell @ coords.T).T % 1.0

    heights = cell_perpendicular_heights(cell)
    cutoff_frac_per_axis = cutoff / heights
    n_cells = np.maximum(1, np.floor(1.0 / cutoff_frac_per_axis)).astype(np.int32)
    inv_cell_size = n_cells.astype(np.float64)

    n_total = int(n_cells[0]) * int(n_cells[1]) * int(n_cells[2])
    atom_cells = (np.floor(coords_frac * inv_cell_size).astype(np.int64) % n_cells).astype(np.int32)

    flat_ids = (
        atom_cells[:, 0] * int(n_cells[1]) * int(n_cells[2]) + atom_cells[:, 1] * int(n_cells[2]) + atom_cells[:, 2]
    ).astype(np.int32)

    order = np.argsort(flat_ids, kind="stable")
    sorted_flat = flat_ids[order]

    cell_start = np.zeros(n_total + 1, dtype=np.int32)
    np.add.at(cell_start[1:], sorted_flat, 1)
    np.cumsum(cell_start, out=cell_start)

    cell_atoms = order.astype(np.int32)

    return coords_frac, atom_cells, n_cells, cell_start, cell_atoms


# ============================================================================
# Numba-accelerated distance kernels
# ============================================================================


@jit(nopython=True, fastmath=True, cache=True)
def _dist_sq_ortho(ci: np.ndarray, cj: np.ndarray, box: np.ndarray) -> float:
    """Minimum-image squared distance, orthogonal box."""
    dx = ci[0] - cj[0]
    dx -= box[0] * round(dx / box[0])
    dy = ci[1] - cj[1]
    dy -= box[1] * round(dy / box[1])
    dz = ci[2] - cj[2]
    dz -= box[2] * round(dz / box[2])
    return dx * dx + dy * dy + dz * dz


@jit(nopython=True, fastmath=True, cache=True)
def _dist_sq_tri(fi: np.ndarray, fj: np.ndarray, cell: np.ndarray) -> float:
    """Minimum-image squared distance, triclinic box."""
    dfx = fi[0] - fj[0]
    dfx -= round(dfx)
    dfy = fi[1] - fj[1]
    dfy -= round(dfy)
    dfz = fi[2] - fj[2]
    dfz -= round(dfz)
    dx = dfx * cell[0, 0] + dfy * cell[1, 0] + dfz * cell[2, 0]
    dy = dfx * cell[0, 1] + dfy * cell[1, 1] + dfz * cell[2, 1]
    dz = dfx * cell[0, 2] + dfy * cell[1, 2] + dfz * cell[2, 2]
    return dx * dx + dy * dy + dz * dz


# ============================================================================
# Numba neighbor list builders
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
) -> tuple[np.ndarray, np.ndarray]:
    """Build neighbor list for an orthogonal box using Numba."""
    n_atoms = len(coords)
    ny = n_cells[1]
    nz = n_cells[2]

    neighbor_list = np.full((n_atoms, max_neighbors), -1, dtype=np.int32)
    neighbor_counts = np.zeros(n_atoms, dtype=np.int32)

    for i in prange(n_atoms):
        if use_target_filter:
            found = False
            for t in target_types:
                if types[i] == t:
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

                        if use_neighbor_filter:
                            ok = False
                            for t in neighbor_types:
                                if types[j] == t:
                                    ok = True
                                    break
                            if not ok:
                                continue

                        if _dist_sq_ortho(coords[i], coords[j], box_size) <= cutoff_sq:
                            if count < max_neighbors:
                                neighbor_list[i, count] = j
                            count += 1

        neighbor_counts[i] = count

    return neighbor_list, neighbor_counts


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
) -> tuple[np.ndarray, np.ndarray]:
    """Build neighbor list for a triclinic box using Numba."""
    n_atoms = len(coords_frac)
    ny = n_cells[1]
    nz = n_cells[2]

    neighbor_list = np.full((n_atoms, max_neighbors), -1, dtype=np.int32)
    neighbor_counts = np.zeros(n_atoms, dtype=np.int32)

    for i in prange(n_atoms):
        if use_target_filter:
            found = False
            for t in target_types:
                if types[i] == t:
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

                        if use_neighbor_filter:
                            ok = False
                            for t in neighbor_types:
                                if types[j] == t:
                                    ok = True
                                    break
                            if not ok:
                                continue

                        if _dist_sq_tri(coords_frac[i], coords_frac[j], cell) <= cutoff_sq:
                            if count < max_neighbors:
                                neighbor_list[i, count] = j
                            count += 1

        neighbor_counts[i] = count

    return neighbor_list, neighbor_counts


def _numba_to_list(
    neighbor_list: np.ndarray,
    neighbor_counts: np.ndarray,
    n_atoms: int,
    max_neighbors: int,
    build_fn: Callable,
    build_kwargs: dict[str, Any],
) -> list[list[int]]:
    """Convert Numba output arrays to Python list-of-lists, retrying on overflow."""
    overflow = int(neighbor_counts.max()) if n_atoms > 0 else 0

    while overflow > max_neighbors:
        max_neighbors = int(overflow * 1.2) + 1
        build_kwargs["max_neighbors"] = max_neighbors
        neighbor_list, neighbor_counts = build_fn(**build_kwargs)
        overflow = int(neighbor_counts.max())

    neighbors = []
    for i in range(n_atoms):
        c = int(neighbor_counts[i])
        neighbors.append(neighbor_list[i, :c].tolist())
    return neighbors


# ============================================================================
# Vectorized NumPy distance functions
# ============================================================================


def _dist_sq_ortho_vec(coord_i: np.ndarray, coords_j: np.ndarray, box_size: np.ndarray) -> np.ndarray:
    """Vectorised minimum-image squared distances for an orthogonal box."""
    rij = coord_i - coords_j
    rij -= box_size * np.round(rij / box_size)
    return np.einsum("ij,ij->i", rij, rij)


def _dist_sq_tri_vec(frac_i: np.ndarray, frac_j: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Vectorised minimum-image squared distances for a triclinic box."""
    delta_frac = frac_i - frac_j
    delta_frac -= np.round(delta_frac)
    rij = delta_frac @ cell
    return np.einsum("ij,ij->i", rij, rij)


def _get_neighbor_cells_numpy(ci: np.ndarray, n_cells: np.ndarray) -> np.ndarray:
    """Return the 27 neighbour cell indices (with periodic wrap) for cell *ci*."""
    return (ci + SHIFT_GRID_3D) % n_cells


# ============================================================================
# ID extraction helper
# ============================================================================


def _extract_atom_ids(atoms: Atoms | tuple) -> np.ndarray:
    """Return the real atom IDs from an ASE Atoms object or tuple.

    Priority for ASE Atoms:
      1. atoms.arrays['id']  — present when read from LAMMPS/XYZ with id column
      2. atoms.get_array('id') fallback
      3. 1-based sequential IDs as last resort (OVITO default)

    For a raw tuple input, sequential 1-based IDs are returned because
    there is no ID information available.
    """
    if isinstance(atoms, Atoms):
        for key in ("id", "ID"):
            if key in atoms.arrays:
                return atoms.arrays[key].astype(np.int64)
        # Fall back to 1-based sequential IDs matching OVITO convention
        return np.arange(1, len(atoms) + 1, dtype=np.int64)
    coords, *_ = atoms
    n = len(np.asarray(coords))
    return np.arange(1, n + 1, dtype=np.int64)


# ============================================================================
# Main function
# ============================================================================


def get_neighbors(  # noqa: C901, PLR0912, PLR0915
    atoms: Atoms | tuple[np.ndarray, np.ndarray, np.ndarray],
    cutoff: float,
    target_types: list[int] | None = None,
    neighbor_types: list[int] | None = None,
    use_numba: bool | None = None,
) -> list[tuple[int, list[int]]]:
    """Find all neighbors within *cutoff* for each atom.

    Returns a list of ``(central_atom_id, [neighbor_atom_ids])`` tuples
    where all IDs are the **real atom IDs** from the structure file
    (e.g. non-sequential LAMMPS/XYZ ids), not array indices.
    This makes it straightforward to cross-check results in OVITO.

    Args:
        atoms: Either an ASE ``Atoms`` object or a tuple
               ``(coords, types, cell_matrix)`` where *cell_matrix* is a
               (3, 3) array with **lattice vectors as rows** (ASE convention).
        cutoff: Cutoff radius in Å.
        target_types: Atomic numbers of atoms to find neighbors *for*.
                      ``None`` means all atoms.
        neighbor_types: Atomic numbers that count as valid neighbors.
                        ``None`` means all types.
        use_numba: Force Numba on/off.  ``None`` = auto-detect.

    Returns:
        List of ``(central_id, [neighbor_ids])`` — one entry per atom.
        Atoms excluded by *target_types* have an empty neighbor list but
        are still present in the output so index positions are preserved.

    Examples:
        >>> neighbors = get_neighbors(atoms, cutoff=3.5)
        >>> for central_id, nn_ids in neighbors:
        ...     print(central_id, nn_ids)

        >>> # Quick lookup by original atom ID:
        >>> nl_dict = {cid: nn for cid, nn in get_neighbors(atoms, cutoff=3.5)}
        >>> nl_dict[43586]   # neighbors of atom ID 43586
    """
    if use_numba is None:
        use_numba = NUMBA_AVAILABLE

    # ------------------------------------------------------------------
    # Parse input & extract real IDs
    # ------------------------------------------------------------------
    atom_ids = _extract_atom_ids(atoms)  # shape (N,), real IDs

    if isinstance(atoms, Atoms):
        atoms_copy = atoms.copy()
        atoms_copy.wrap()
        coords = atoms_copy.get_positions()
        types = atoms_copy.get_atomic_numbers()
        cell = atoms_copy.get_cell().array
    else:
        coords, types, cell = atoms
        coords = np.asarray(coords, dtype=np.float64)
        types = np.asarray(types, dtype=np.int32)
        cell = np.asarray(cell, dtype=np.float64)

    n_atoms = len(coords)
    cutoff_sq = cutoff * cutoff

    is_orthogonal = np.allclose(cell - np.diag(np.diag(cell)), 0.0, atol=1e-10)

    target_arr = np.array(target_types, dtype=np.int32) if target_types is not None else np.empty(0, dtype=np.int32)
    neigh_arr = np.array(neighbor_types, dtype=np.int32) if neighbor_types is not None else np.empty(0, dtype=np.int32)
    use_tf = target_types is not None
    use_nf = neighbor_types is not None

    _initial_max_neighbors = _estimate_max_neighbors(coords, cell, cutoff)

    # ------------------------------------------------------------------
    # Build raw index-based neighbor list (internal)
    # ------------------------------------------------------------------
    if is_orthogonal:
        box_size = np.diag(cell)
        atom_cells, n_cells, cell_start, cell_atoms = compute_cell_list_orthogonal(coords, box_size, cutoff)

        if use_numba and NUMBA_AVAILABLE:
            kwargs: dict[str, Any] = {
                "coords": coords,
                "types": types,
                "box_size": box_size,
                "atom_cells": atom_cells,
                "n_cells": n_cells,
                "cell_start": cell_start,
                "cell_atoms": cell_atoms,
                "cutoff_sq": cutoff_sq,
                "target_types": target_arr,
                "neighbor_types": neigh_arr,
                "use_target_filter": use_tf,
                "use_neighbor_filter": use_nf,
                "max_neighbors": _initial_max_neighbors,
            }
            nl, nc = _build_nl_ortho_numba(**kwargs)
            idx_neighbors = _numba_to_list(nl, nc, n_atoms, _initial_max_neighbors, _build_nl_ortho_numba, kwargs)
        else:
            cells: defaultdict[tuple, list[int]] = defaultdict(list)
            for idx, c in enumerate(atom_cells):
                cells[tuple(c)].append(idx)

            idx_neighbors = [[] for _ in range(n_atoms)]
            for i in range(n_atoms):
                if use_tf and types[i] not in target_types:
                    continue
                ci = atom_cells[i]
                candidates: list[int] = []
                for cj in _get_neighbor_cells_numpy(ci, n_cells):
                    candidates.extend(cells[tuple(cj)])
                candidates = [j for j in candidates if j != i]
                if use_nf:
                    candidates = [j for j in candidates if types[j] in neighbor_types]
                if not candidates:
                    continue
                ca = np.array(candidates, dtype=np.int32)
                mask = _dist_sq_ortho_vec(coords[i], coords[ca], box_size) <= cutoff_sq
                idx_neighbors[i] = ca[mask].tolist()

    else:
        coords_frac, atom_cells, n_cells, cell_start, cell_atoms = compute_cell_list_triclinic(coords, cell, cutoff)

        if use_numba and NUMBA_AVAILABLE:
            kwargs = {
                "coords_frac": coords_frac,
                "types": types,
                "cell": cell,
                "atom_cells": atom_cells,
                "n_cells": n_cells,
                "cell_start": cell_start,
                "cell_atoms": cell_atoms,
                "cutoff_sq": cutoff_sq,
                "target_types": target_arr,
                "neighbor_types": neigh_arr,
                "use_target_filter": use_tf,
                "use_neighbor_filter": use_nf,
                "max_neighbors": _initial_max_neighbors,
            }
            nl, nc = _build_nl_tri_numba(**kwargs)
            idx_neighbors = _numba_to_list(nl, nc, n_atoms, _initial_max_neighbors, _build_nl_tri_numba, kwargs)
        else:
            cells = defaultdict(list)
            for idx, c in enumerate(atom_cells):
                cells[tuple(c)].append(idx)

            idx_neighbors = [[] for _ in range(n_atoms)]
            for i in range(n_atoms):
                if use_tf and types[i] not in target_types:
                    continue
                ci = atom_cells[i]
                candidates = []
                for cj in _get_neighbor_cells_numpy(ci, n_cells):
                    candidates.extend(cells[tuple(cj)])
                candidates = [j for j in candidates if j != i]
                if use_nf:
                    candidates = [j for j in candidates if types[j] in neighbor_types]
                if not candidates:
                    continue
                ca = np.array(candidates, dtype=np.int32)
                mask = _dist_sq_tri_vec(coords_frac[i], coords_frac[ca], cell) <= cutoff_sq
                idx_neighbors[i] = ca[mask].tolist()

    # ------------------------------------------------------------------
    # Translate array indices → real atom IDs
    # ------------------------------------------------------------------
    return [(int(atom_ids[i]), [int(atom_ids[j]) for j in idx_neighbors[i]]) for i in range(n_atoms)]


def _estimate_max_neighbors(coords: np.ndarray, cell: np.ndarray, cutoff: float) -> int:
    """Estimate a safe upper bound for the number of neighbors per atom."""
    n_atoms = len(coords)
    volume = abs(np.linalg.det(cell))
    if volume < _MIN_VOLUME or n_atoms == 0:
        return 200
    density = n_atoms / volume
    sphere_vol = (4.0 / 3.0) * np.pi * cutoff**3
    estimate = int(density * sphere_vol * 3.0) + 32
    return max(estimate, 64)
