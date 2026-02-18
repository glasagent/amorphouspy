"""Structural analysis functions for multicomponent glass systems.

Author: Achraf Atila (achraf.atila@bam.de)

"""

import numpy as np
from ase import Atoms
from numba import jit

from amorphouspy.io_utils import get_properties_for_structure_analysis
from amorphouspy.neighbors import get_neighbors
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
        target_type: Atom type for which to compute coordination.
        cutoff: Cutoff radius.
        neighbor_types: Valid neighbor types.

    Returns:
        Tuple containing:
            - coordination number distribution (dict): coordination number → count
            - per-atom coordination numbers (dict): atom ID → coordination number

    Example:
        >>> structure = read('glass.xyz')
        >>> dist, cn = compute_coordination(structure, target_type=14, cutoff=2.0)

    """
    ids, types, coords, box_size = get_properties_for_structure_analysis(structure)
    neighbors = get_neighbors(
        coords,
        types,
        box_size,
        cutoff,
        target_type,
        neighbor_types,
    )
    coord_numbers = {ids[idx]: len(neighbors[idx]) for idx, atom_type in enumerate(types) if atom_type == target_type}
    coord_numbers_distribution = count_distribution(coord_numbers)
    return dict(sorted(coord_numbers_distribution.items())), coord_numbers


@jit(nopython=True)
def compute_distances(coords: np.ndarray, box_size: np.ndarray, r_max: float) -> tuple:
    """Simplified Numba-accelerated distance computation with PBC."""
    n = len(coords)
    distances = []
    i_indices = []
    j_indices = []

    for i in range(n):
        for j in range(i + 1, n):  # Avoid double-counting and self-pairs
            # Compute minimum image distance
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            dz = coords[i, 2] - coords[j, 2]

            dx -= box_size[0] * np.round(dx / box_size[0])
            dy -= box_size[1] * np.round(dy / box_size[1])
            dz -= box_size[2] * np.round(dz / box_size[2])

            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            if dist <= r_max:
                distances.append(dist)
                i_indices.append(i)
                j_indices.append(j)

    return np.array(distances), np.array(i_indices), np.array(j_indices)


def compute_rdf(
    structure: Atoms,
    r_max: float = 10.0,
    n_bins: int = 500,
    type_pairs: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, dict, dict]:
    """Compute radial distribution functions (RDFs) and cumulative coordination numbers.

    Calculates the pair-wise radial distribution function g(r) for specified
    atom-type pairs under periodic boundary conditions, along with the
    cumulative coordination number n(r), i.e., the average number of
    neighbors within radius r.

    Args:
        structure: ASE Atoms object containing atomic coordinates and types.
        r_max: Maximum distance to evaluate RDF (default is 10.0 Å).
        n_bins: Number of radial bins between 0 and r_max (default is 500).
        type_pairs: List of type index pairs to compute RDF for. If None, computes all
            combinations of present types.

    Returns:
        r (np.ndarray): Radial bin centers (Å).
        rdfs (dict[(int, int), np.ndarray]): Normalized RDF values g(r) for each type pair.
        cn_cumulative (dict[(int, int), np.ndarray]): Cumulative coordination number n(r), average count of neighbors
            within distance r for each reference type.

    Notes:
        - Periodic boundaries handled via minimum image convention.
        - Normalization accounts for shell volume and pair densities.
        - Cumulative coordination is normalized per reference particle.
        - If t1 == t2, self-correlation is corrected via density and count.

    Example:
        >>> structure = read('glass.xyz')
        >>> r, rdfs, cn = compute_rdf(structure, r_max=10.0, n_bins=500)

    """
    _ids, types, coords, box_size = get_properties_for_structure_analysis(structure)
    # Input validation and type conversion
    coords = np.asarray(coords, dtype=np.float64)
    types = np.asarray(types, dtype=np.int64)
    box_size = np.asarray(box_size, dtype=np.float64)

    assert len(coords) == len(types), "coords and types must match!"
    assert box_size.shape == (3,), "box_size must be (3,) array"

    # Set up bins
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    r = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    dr = bin_edges[1] - bin_edges[0]

    # Get unique types and determine type pairs
    unique_types = np.unique(types)
    if type_pairs is None:
        type_pairs = [(int(t1), int(t2)) for t1 in unique_types for t2 in unique_types]

    # Compute distances (Numba-accelerated)
    distances, i_indices, j_indices = compute_distances(coords, box_size, float(r_max))

    # Get types for each pair
    type_i = types[i_indices]
    type_j = types[j_indices]

    # Initialize results
    rdfs = {pair: np.zeros(n_bins) for pair in type_pairs}
    cn_cumulative = {pair: np.zeros(n_bins) for pair in type_pairs}

    # Compute histograms for each type pair
    for t1, t2 in type_pairs:
        if t1 == t2:
            pair_mask = (type_i == t1) & (type_j == t2)
        else:
            pair_mask = ((type_i == t1) & (type_j == t2)) | ((type_i == t2) & (type_j == t1))

        pair_dists = distances[pair_mask]
        hist, _ = np.histogram(pair_dists, bins=bin_edges)
        if t1 == t2:
            hist = hist * 2

        rdfs[(t1, t2)] = hist
        cn_cumulative[(t1, t2)] = np.cumsum(hist)

    # Normalization
    volume = np.prod(box_size)
    type_counts = {t: np.sum(types == t) for t in unique_types}

    for (t1, t2), hist in rdfs.items():
        n1 = type_counts[t1]
        n2 = type_counts[t2] if t1 != t2 else (n1 - 1)  # Exclude self for same type

        rho_pair = n2 / volume
        shell_volumes = 4 * np.pi * (r**2) * dr
        rdfs[(t1, t2)] = hist / (n1 * rho_pair * shell_volumes + 1e-10)
        cn_cumulative[(t1, t2)] = cn_cumulative[(t1, t2)] / n1

    return r, rdfs, cn_cumulative
