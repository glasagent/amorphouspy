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

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from amorphouspy.neighbors import get_neighbors

if TYPE_CHECKING:
    from ase import Atoms

MIN_NEIGHBORS_FOR_ANGLE = 2


def _compute_angles_frame_average(
    structure: list[Atoms],
    center_type: int,
    neighbor_type: int,
    cutoff: float,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_frames = len(structure)
    per_frame_results = [compute_angles(s, center_type, neighbor_type, cutoff, bins) for s in structure]
    bin_centers = per_frame_results[0][0]
    angle_histogram_arrays = np.stack([frame_result[1] for frame_result in per_frame_results])
    angle_histogram_mean = angle_histogram_arrays.mean(axis=0)
    degrees_of_freedom = 1 if num_frames > 1 else 0
    angle_histogram_sem = angle_histogram_arrays.std(axis=0, ddof=degrees_of_freedom) / np.sqrt(num_frames)
    return bin_centers, angle_histogram_mean, angle_histogram_sem


def compute_angles(
    structure: Atoms | list[Atoms],
    center_type: int,
    neighbor_type: int,
    cutoff: float,
    bins: int = 180,
    *,
    frame_averaging: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bond angle distribution between triplets of neighbor_type-center-neighbor_type.

    Args:
        structure: Atomic structure, or a list of frames when frame_averaging=True.
        center_type: Atom type at the angle center.
        neighbor_type: Atom type forming the angle with center.
        cutoff: Neighbor search cutoff.
        bins: Number of histogram bins. Defaults to 180.
        frame_averaging: If True, average results over all frames in structure (list[Atoms]).

    Returns:
        A 3-tuple containing:
            bin_centers: Bin centers (degrees).
            angle_hist: Normalized angle histogram (or mean when frame_averaging=True).
            angle_hist_sem: SEM of angle histogram (zeros when frame_averaging=False).

    Example:
        >>> bins, hist, sem = compute_angles(structure, center_type=1, neighbor_type=2, cutoff=3.0)

    """
    if frame_averaging:
        if isinstance(structure, list) and len(structure) == 0:
            msg = "frame_averaging=True requires a non-empty list[Atoms]"
            raise ValueError(msg)
        frames = cast("list[Atoms]", structure if isinstance(structure, list) else [structure])
        return _compute_angles_frame_average(frames, center_type, neighbor_type, cutoff, bins)
    if isinstance(structure, list):
        structure = cast("Atoms", structure[0])
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
    hist_sem = np.zeros_like(angle_hist)
    return bin_centers, angle_hist, hist_sem
