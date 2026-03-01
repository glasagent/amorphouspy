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

from amorphouspy.neighbors import get_neighbors

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
        structure: Atomic structure.
        center_type: Atom type at the angle center.
        neighbor_type: Atom type forming the angle with center.
        cutoff: Neighbor search cutoff.
        bins: Number of histogram bins. Defaults to 180.

    Returns:
        A tuple containing:
            - bin_centers: Bin centers (degrees).
            - angle_hist: Normalized angle histogram.

    Example:
        >>> bins, hist = compute_angles(structure, center_type=1, neighbor_type=2, cutoff=3.0)

    """
    # Wrap and extract positions/cell once — needed for minimum-image vectors
    structure_wrapped = structure.copy()
    structure_wrapped.wrap()
    coords = structure_wrapped.get_positions()
    cell = structure_wrapped.get_cell().array
    is_orthogonal = np.allclose(cell - np.diag(np.diag(cell)), 0.0, atol=1e-10)

    # Build ID → array index map to look up coordinates by real atom ID
    if "id" in structure_wrapped.arrays:
        raw_ids = structure_wrapped.arrays["id"]
    else:
        raw_ids = np.arange(1, len(structure_wrapped) + 1)
    id_to_idx = {int(aid): i for i, aid in enumerate(raw_ids)}

    # get_neighbors returns List[Tuple[central_id, List[neighbor_ids]]]
    neighbors = get_neighbors(
        structure,
        cutoff=cutoff,
        target_types=[center_type],
        neighbor_types=[neighbor_type],
    )

    angles = []

    for central_id, nn_ids in neighbors:
        if len(nn_ids) < MIN_NEIGHBORS_FOR_ANGLE:
            continue

        ci = id_to_idx[central_id]
        center_coord = coords[ci]

        # Resolve neighbor coordinates by real ID
        nn_indices = [id_to_idx[nid] for nid in nn_ids]

        for j_pos, j_idx in enumerate(nn_indices):
            for k_idx in nn_indices[j_pos + 1 :]:
                v1 = coords[j_idx] - center_coord
                v2 = coords[k_idx] - center_coord

                # Minimum-image correction
                if is_orthogonal:
                    box = np.diag(cell)
                    v1 -= box * np.round(v1 / box)
                    v2 -= box * np.round(v2 / box)
                else:
                    inv_cell = np.linalg.inv(cell)
                    df1 = inv_cell @ v1
                    df1 -= np.round(df1)
                    v1 = cell.T @ df1
                    df2 = inv_cell @ v2
                    df2 -= np.round(df2)
                    v2 = cell.T @ df2

                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                if norm_v1 == 0 or norm_v2 == 0:
                    continue

                cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cos_theta)))

    angle_hist, bin_edges = np.histogram(
        angles,
        bins=bins,
        range=(0, 180),
        density=True,
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, angle_hist
