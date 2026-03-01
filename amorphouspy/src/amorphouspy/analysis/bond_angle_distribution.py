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

        # Resolve neighbor coordinates by real ID — shape (k, 3)
        nn_indices = np.array([id_to_idx[nid] for nid in nn_ids], dtype=np.int32)
        vecs = coords[nn_indices] - center_coord  # (k, 3)

        # Minimum-image correction — vectorized for all neighbor vectors at once
        if is_orthogonal:
            box = np.diag(cell)
            vecs -= box * np.round(vecs / box)
        else:
            inv_cell = np.linalg.inv(cell)
            delta_frac = (inv_cell @ vecs.T).T
            delta_frac -= np.round(delta_frac)
            vecs = (cell.T @ delta_frac.T).T

        # Normalise all vectors at once
        norms = np.linalg.norm(vecs, axis=1)  # (k,)
        valid = norms > 0
        if valid.sum() < MIN_NEIGHBORS_FOR_ANGLE:
            continue
        vecs = vecs[valid]
        norms = norms[valid]
        unit_vecs = vecs / norms[:, np.newaxis]  # (k, 3)

        # Compute all unique pair cosines via matrix multiply: (k, k)
        # Upper triangle gives each unique i<j pair
        cos_mat = np.clip(unit_vecs @ unit_vecs.T, -1.0, 1.0)
        i_idx, j_idx = np.triu_indices(len(unit_vecs), k=1)
        cos_angles = cos_mat[i_idx, j_idx]
        angles.extend(np.degrees(np.arccos(cos_angles)).tolist())

    angle_hist, bin_edges = np.histogram(
        angles,
        bins=bins,
        range=(0, 180),
        density=True,
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, angle_hist
