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

