"""Structural analysis functions for multicomponent glass systems.

Author: Achraf Atila (achraf.atila@bam.de)

This module provides tools to compute local structural descriptors
in amorphous materials, focusing on bond angle distributions around
selected atoms. It supports type-specific neighbor searches
using periodic boundary conditions and customizable cutoffs.

Currently implemented:

- compute_angles: Calculates bond angle distributions for atom
  triplets of the form neighbor-center-neighbor, filtered by atomic
  types and distance cutoff.
"""

import numpy as np
from ase import Atoms

from pyiron_glass.io_utils import get_properties_for_structure_analysis
from pyiron_glass.neighbors import get_neighbors

MIN_NEIGHBORS_FOR_ANGLE = 2


def compute_angles(
    structure: Atoms,
    center_type: int,
    neighbor_type: int,
    cutoff: float,
    bins: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute bond angle distribution between triplets of neighbor_type-center-neighbor_type.

    Args:
        structure (Atoms): Atomic structure.
        center_type (int): Atom type at the angle center.
        neighbor_type (int): Atom type forming the angle with center.
        cutoff (float): Neighbor search cutoff.
        bins (int): Number of histogram bins (default: 180).

    Returns:
        tuple[np.ndarray, np.ndarray]: Bin centers (degrees), normalized angle histogram.

    """
    ids, types, coords, box_size = get_properties_for_structure_analysis(structure)

    neighbors = get_neighbors(
        coords,
        types,
        box_size,
        cutoff,
        center_type,
        [neighbor_type],
    )
    angles = []
    for i, atom_type in enumerate(types):
        if atom_type != center_type:
            continue
        neigh_ids = neighbors[i]
        if len(neigh_ids) < MIN_NEIGHBORS_FOR_ANGLE:
            continue
        for j, id_j in enumerate(neigh_ids):
            for k in range(j + 1, len(neigh_ids)):
                id_k = neigh_ids[k]
                v1 = coords[id_j] - coords[i]
                v2 = coords[id_k] - coords[i]
                v1 -= box_size * np.round(v1 / box_size)
                v2 -= box_size * np.round(v2 / box_size)
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                if norm_v1 == 0 or norm_v2 == 0:
                    continue
                cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
                angle = np.arccos(cos_theta) * 180 / np.pi
                angles.append(angle)
    angle_hist, bin_edges = np.histogram(
        angles,
        bins=bins,
        range=(0, 180),
        density=True,
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, angle_hist
