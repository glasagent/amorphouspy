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
    coord_numbers = {
        ids[idx]: len(neighbors[idx])
        for idx, atom_type in enumerate(types)
        if atom_type == target_type
    }
    coord_numbers_distribution = count_distribution(coord_numbers)
    return dict(sorted(coord_numbers_distribution.items())), coord_numbers


def compute_rdf(
    types: np.ndarray,
    coords: np.ndarray,
    box_size: np.ndarray,
    type_i: int,
    type_j: int,
    r_max: float,
    bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute radial distribution function g(r) between atoms of type_i and type_j.

    Args:
        types (np.ndarray): Atom types.
        coords (np.ndarray): Atom coordinates.
        box_size (np.ndarray): Simulation box dimensions.
        type_i (int): First atom type.
        type_j (int): Second atom type.
        r_max (float): Maximum distance to consider.
        bins (int): Number of bins (default: 100).

    Returns:
        tuple: (r, g_r) where r are bin centers and g_r is the RDF values.
    """
    n_atoms = len(types)
    volume = np.prod(box_size)
    density_j = np.sum(types == type_j) / volume

    dr = r_max / bins
    hist = np.zeros(bins, dtype=float)

    indices_i = np.where(types == type_i)[0]
    indices_j = np.where(types == type_j)[0]

    for i_idx in indices_i:
        ri = coords[i_idx]
        for j_idx in indices_j:
            if i_idx == j_idx:
                continue
            rij = coords[j_idx] - ri
            # Minimum image convention
            rij -= box_size * np.round(rij / box_size)
            dist = np.linalg.norm(rij)
            if dist < r_max:
                bin_idx = int(dist / dr)
                hist[bin_idx] += 1

    norm = (4/3) * np.pi * (np.arange(1, bins+1)**3 - np.arange(bins)**3) * (dr**3)
    shell_volume = norm
    n_i = len(indices_i)
    g_r = hist / (n_i * density_j * shell_volume)

    r = (np.arange(bins) + 0.5) * dr
    return r, g_r
