"""Structural analysis functions for multicomponent glass systems.

Author: Achraf Atila (achraf.atila@bam.de)

"""

import numpy as np

from pyiron_glass.neighbors import get_neighbors
from pyiron_glass.shared import count_distribution


def compute_coordination(
    ids: np.ndarray,
    types: np.ndarray,
    coords: np.ndarray,
    box_size: np.ndarray,
    target_type: int,
    cutoff: float,
    neighbor_types: list[int] | None = None,
) -> tuple[dict[int, int], dict[int, int]]:
    """Compute coordination number for atoms of a target type.

    Args:
        ids (np.ndarray): Atom IDs.
        types (np.ndarray): Atom types.
        coords (np.ndarray): Atom coordinates.
        box_size (np.ndarray): Simulation box dimensions.
        target_type (int): Atom type for which to compute coordination.
        cutoff (float): Cutoff radius.
        neighbor_types (List[int] or None): Valid neighbor types.

    Returns:
        Tuple containing:
            - coordination number distribution (dict): coordination number → count
            - per-atom coordination numbers (dict): atom ID → coordination number

    """
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


def compute_rdf(
    coords: np.ndarray,
    types: np.ndarray,
    box_size: np.ndarray,
    r_max: float = 10.0,
    n_bins: int = 500,
    type_pairs: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray]]:
    """Compute radial distribution functions (RDFs) and cumulative coordination numbers.

    Calculates the pair-wise radial distribution function g(r) for specified
    atom-type pairs under periodic boundary conditions, along with the
    cumulative coordination number n(r), i.e., the average number of
    neighbors within radius r.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        Cartesian coordinates of N particles (in Å).
    types : np.ndarray, shape (N,)
        Integer type identifier for each particle.
    box_size : np.ndarray, shape (3,)
        Simulation box dimensions along x, y, z (Å).
    r_max : float, optional
        Maximum distance to evaluate RDF (default is 10.0 Å).
    n_bins : int, optional
        Number of radial bins between 0 and r_max (default is 500).
    type_pairs : list of tuple(int, int), optional
        List of type index pairs to compute RDF for. If None, computes all
        combinations of present types.

    Returns
    -------
    r : np.ndarray, shape (n_bins,)
        Radial bin centers (Å).
    rdfs : dict[(int, int), np.ndarray]
        Normalized RDF values g(r) for each type pair.
    cn_cumulative : dict[(int, int), np.ndarray]
        Cumulative coordination number n(r), average count of neighbors
        within distance r for each reference type.

    Notes
    -----
    - Periodic boundaries handled via minimum image convention.
    - Normalization accounts for shell volume and pair densities.
    - Cumulative coordination is normalized per reference particle.
    - If t1 == t2, self-correlation is corrected via density and count.

    """
    # set bins
    bin_width = float(r_max / n_bins)
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    r = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    dr = bin_width

    # init storage and setup of pairs of unique types
    unique_types = np.unique(types)
    if type_pairs is None:
        type_pairs = [(t1, t2) for t1 in unique_types for t2 in unique_types]
    rdfs = {pair: np.zeros(n_bins) for pair in type_pairs}
    cn_cumulative = {pair: np.zeros(n_bins) for pair in type_pairs}

    # distance calculation
    rij = coords[:, None, :] - coords[None, :, :]
    rij -= box_size * np.round(rij / box_size)
    distances = np.linalg.norm(rij, axis=-1)

    # take only valid distances
    mask = (distances > 0) & (distances <= r_max)
    i, j = np.where(mask)
    type_i, type_j = types[i], types[j]

    # compute counts per type pair
    for t1, t2 in type_pairs:
        pair_mask = ((type_i == t1) & (type_j == t2)) | ((t1 == t2) & (type_i == t1) & (type_j == t2))
        hist, _ = np.histogram(distances[mask][pair_mask], bins=bin_edges)
        rdfs[(t1, t2)] = hist
        cn_cumulative[(t1, t2)] = np.cumsum(hist)

    # norm
    volume = np.prod(box_size)
    for (t1, t2), hist in rdfs.items():
        n1 = np.sum(types == t1)
        n2 = np.sum(types == t2) if t1 != t2 else n1

        # ideal pair density
        rho_pair = n2 / volume

        # shell volume and normalization
        shell_volumes = 4 * np.pi * (r**2) * dr
        rdfs[(t1, t2)] = hist / (n1 * rho_pair * shell_volumes)

        # coordination number normalization
        cn_cumulative[(t1, t2)] = cn_cumulative[(t1, t2)] / n1

    return r, rdfs, cn_cumulative
