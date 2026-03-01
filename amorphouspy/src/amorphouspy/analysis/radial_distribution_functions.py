"""Structural analysis functions for multicomponent glass systems.

Author: Achraf Atila (achraf.atila@bam.de)

"""

from itertools import combinations_with_replacement

import numpy as np
from ase import Atoms
from numba import jit

from amorphouspy.neighbors import cell_perpendicular_heights, get_neighbors
from amorphouspy.shared import count_distribution


def compute_coordination(
    structure: Atoms,
    target_type: int,
    cutoff: float,
    neighbor_types: list[int] | None = None,
) -> tuple[dict[int, int], dict[int, int]]:
    """Compute coordination number for atoms of a target type.

    Args:
        structure: The atomic structure as ASE object.
        target_type: Atom type (atomic number) for which to compute coordination.
        cutoff: Cutoff radius in Å.
        neighbor_types: Valid neighbor atomic numbers. None means all types.

    Returns:
        Tuple containing:
            - coordination number distribution (dict): coordination number → count
            - per-atom coordination numbers (dict): atom_id → coordination number

    Example:
        >>> structure = read('glass.xyz')
        >>> dist, cn = compute_coordination(structure, target_type=14, cutoff=2.0)

    """
    neighbors = get_neighbors(
        structure,
        cutoff=cutoff,
        target_types=[target_type],
        neighbor_types=neighbor_types,
    )

    # get_neighbors now returns List[Tuple[central_id, List[neighbor_ids]]]
    # coord_numbers is keyed by real atom ID (matches OVITO IDs)
    coord_numbers = {
        central_id: len(nn_ids)
        for central_id, nn_ids in neighbors
        if len(nn_ids) > 0 or _atom_is_target(structure, central_id, target_type)
    }

    coord_numbers_distribution = count_distribution(coord_numbers)
    return dict(sorted(coord_numbers_distribution.items())), coord_numbers


def _atom_is_target(structure: Atoms, atom_id: int, target_type: int) -> bool:
    """Check whether a given atom ID belongs to the target type.

    Looks up the atom by its real ID from atoms.arrays['id'] if present,
    otherwise falls back to treating atom_id as a 1-based index.
    """
    types = structure.get_atomic_numbers()
    if "id" in structure.arrays:
        ids = structure.arrays["id"]
        matches = np.where(ids == atom_id)[0]
        if len(matches) == 0:
            return False
        return int(types[matches[0]]) == target_type
    # 1-based sequential fallback
    idx = atom_id - 1
    if idx < 0 or idx >= len(types):
        return False
    return int(types[idx]) == target_type


# ============================================================================
# Numba-accelerated distance kernel — supports triclinic via fractional coords
# ============================================================================


@jit(nopython=True, cache=True)
def _compute_distances_triclinic(
    coords_frac: np.ndarray,
    cell: np.ndarray,
    r_max: float,
) -> tuple:
    """Numba-accelerated all-pairs distance computation for triclinic boxes.

    Works in fractional coordinates so the minimum-image convention is a
    simple round() in every case (orthogonal and triclinic alike).

    Args:
        coords_frac: Fractional coordinates (N, 3).
        cell:        Cell matrix, lattice vectors as rows (3, 3).
        r_max:       Maximum distance to collect.

    Returns:
        distances:  1-D array of pair distances ≤ r_max.
        i_indices:  Corresponding i atom indices.
        j_indices:  Corresponding j atom indices.
    """
    n = len(coords_frac)
    r_max_sq = r_max * r_max

    max_pairs = n * (n - 1) // 2
    distances = np.empty(max_pairs, dtype=np.float64)
    i_indices = np.empty(max_pairs, dtype=np.int32)
    j_indices = np.empty(max_pairs, dtype=np.int32)
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            dfx = coords_frac[i, 0] - coords_frac[j, 0]
            dfy = coords_frac[i, 1] - coords_frac[j, 1]
            dfz = coords_frac[i, 2] - coords_frac[j, 2]

            dfx -= round(dfx)
            dfy -= round(dfy)
            dfz -= round(dfz)

            dx = dfx * cell[0, 0] + dfy * cell[1, 0] + dfz * cell[2, 0]
            dy = dfx * cell[0, 1] + dfy * cell[1, 1] + dfz * cell[2, 1]
            dz = dfx * cell[0, 2] + dfy * cell[1, 2] + dfz * cell[2, 2]

            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq <= r_max_sq:
                distances[count] = np.sqrt(dist_sq)
                i_indices[count] = i
                j_indices[count] = j
                count += 1

    return distances[:count], i_indices[:count], j_indices[:count]


def _compute_distances(structure: Atoms, r_max: float) -> tuple:
    """Wrap the Numba kernel: convert to fractional coords and call kernel."""
    structure_wrapped = structure.copy()
    structure_wrapped.wrap()

    cell = structure_wrapped.get_cell().array
    inv_cell = np.linalg.inv(cell)
    coords_frac = (inv_cell @ structure_wrapped.get_positions().T).T
    coords_frac = coords_frac % 1.0

    return _compute_distances_triclinic(
        coords_frac.astype(np.float64),
        cell.astype(np.float64),
        float(r_max),
    )


# ============================================================================
# RDF helpers
# ============================================================================


def _compute_rdf_histograms(
    unordered_pairs: list[tuple[int, int]],
    type_i: np.ndarray,
    type_j: np.ndarray,
    distances: np.ndarray,
    type_counts: dict[int, int],
    volume: float,
    bin_edges: np.ndarray,
    shell_volumes: np.ndarray,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray]]:
    """Compute normalised g(r) and directed histograms for each type pair."""
    rdfs: dict[tuple[int, int], np.ndarray] = {}
    hist_directed: dict[tuple[int, int], np.ndarray] = {}

    for t1, t2 in unordered_pairs:
        canonical = (min(t1, t2), max(t1, t2))
        if t1 == t2:
            mask = (type_i == t1) & (type_j == t2)
            hist = np.histogram(distances[mask], bins=bin_edges)[0]
            n = type_counts[t1]
            rho_excl = (n - 1) / volume
            rdfs[canonical] = (hist * 2) / (n * rho_excl * shell_volumes + 1e-10)
            hist_directed[canonical] = hist
        else:
            mask_fwd = (type_i == t1) & (type_j == t2)
            mask_rev = (type_i == t2) & (type_j == t1)
            hist_fwd = np.histogram(distances[mask_fwd], bins=bin_edges)[0]
            hist_rev = np.histogram(distances[mask_rev], bins=bin_edges)[0]
            hist_sym = hist_fwd + hist_rev
            n1, n2 = type_counts[t1], type_counts[t2]
            rho2 = n2 / volume
            rdfs[canonical] = hist_sym / (n1 * rho2 * shell_volumes + 1e-10)
            hist_directed[canonical] = hist_sym

    return rdfs, hist_directed


def _compute_cn_cumulative(
    requested_ordered: list[tuple[int, int]],
    hist_directed: dict[tuple[int, int], np.ndarray],
    type_counts: dict[int, int],
) -> dict[tuple[int, int], np.ndarray]:
    """Compute cumulative coordination numbers from directed histograms."""
    cn_cumulative: dict[tuple[int, int], np.ndarray] = {}
    for t1, t2 in requested_ordered:
        canonical = (min(t1, t2), max(t1, t2))
        hist = hist_directed[canonical]
        n_ref = type_counts[t1]
        factor = 2 if t1 == t2 else 1
        cn_cumulative[(t1, t2)] = np.cumsum(hist * factor) / n_ref
    return cn_cumulative


def compute_rdf(
    structure: Atoms,
    r_max: float = 10.0,
    n_bins: int = 500,
    type_pairs: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, dict, dict]:
    """Compute radial distribution functions (RDFs) and cumulative coordination numbers.

    Calculates the pair-wise radial distribution function g(r) for specified
    atom-type pairs under periodic boundary conditions (orthogonal and
    triclinic), along with the cumulative coordination number n(r).

    Args:
        structure:  ASE Atoms object.
        r_max:      Maximum distance in Å (default 10.0).
        n_bins:     Number of radial bins (default 500).
        type_pairs: List of (atomic_number_1, atomic_number_2) pairs.
                    None → all unique unordered combinations of present types
                    plus all same-type pairs.

    Returns:
        r (np.ndarray):
            Radial bin centres in Å, shape (n_bins,).
        rdfs (dict[(int,int), np.ndarray]):
            Normalised g(r) for each *unordered* type pair, shape (n_bins,).
        cn_cumulative (dict[(int,int), np.ndarray]):
            Mean number of neighbours of the *second* type within radius r
            around an atom of the *first* type, shape (n_bins,).

    Raises:
        ValueError: If r_max exceeds half the smallest perpendicular cell height.

    Notes:
        - This function operates on array indices internally (not atom IDs)
          because it only needs type information and distances, not ID lookup.
        - Periodic boundaries are handled via the minimum-image convention in
          fractional space, so both orthogonal and triclinic cells are correct.

    Example:
        >>> structure = read('glass.xyz')
        >>> r, rdfs, cn = compute_rdf(structure, r_max=10.0, n_bins=500)
        >>> g_SiO = rdfs[(8, 14)]
        >>> cn_SiO = cn[(8, 14)]   # O around Si
        >>> cn_OSi = cn[(14, 8)]   # Si around O

    """
    types = structure.get_atomic_numbers()
    unique_types = np.unique(types)
    type_counts = {int(t): int(np.sum(types == t)) for t in unique_types}
    cell = structure.get_cell().array
    volume = abs(np.linalg.det(cell))

    heights = cell_perpendicular_heights(cell)
    r_max_allowed = float(heights.min()) / 2.0
    if r_max > r_max_allowed:
        msg = (
            f"r_max={r_max:.4f} Å exceeds half the smallest perpendicular cell "
            f"height ({r_max_allowed:.4f} Å). The minimum-image convention "
            f"breaks down beyond this limit, producing incorrect RDF and CN "
            f"values. Reduce r_max or use a larger simulation box.\n"
            f"Perpendicular heights: {heights[0]:.4f}, {heights[1]:.4f}, "
            f"{heights[2]:.4f} Å  →  limits: {heights[0] / 2:.4f}, "
            f"{heights[1] / 2:.4f}, {heights[2] / 2:.4f} Å"
        )
        raise ValueError(msg)

    if type_pairs is None:
        unordered_pairs = [(int(a), int(b)) for a, b in combinations_with_replacement(unique_types, 2)]
        requested_ordered = []
        for a, b in unordered_pairs:
            requested_ordered.append((a, b))
            if a != b:
                requested_ordered.append((b, a))
    else:
        unordered_pairs = list({(min(a, b), max(a, b)) for a, b in type_pairs})
        requested_ordered = list(dict.fromkeys(type_pairs))

    bin_edges = np.linspace(0, r_max, n_bins + 1)
    r = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    dr = bin_edges[1] - bin_edges[0]
    shell_volumes = 4.0 * np.pi * r**2 * dr

    distances, i_idx, j_idx = _compute_distances(structure, r_max)
    type_i = types[i_idx]
    type_j = types[j_idx]

    rdfs, hist_directed = _compute_rdf_histograms(
        unordered_pairs,
        type_i,
        type_j,
        distances,
        type_counts,
        volume,
        bin_edges,
        shell_volumes,
    )
    cn_cumulative = _compute_cn_cumulative(requested_ordered, hist_directed, type_counts)

    return r, rdfs, cn_cumulative
