"""Structural analysis functions to get the Qn and network connectivity of multicomponent glass systems.

Author: Achraf Atila (achraf.atila@bam.de)
This module provides functions to compute the Q^n distribution,
which characterizes the connectivity of tetrahedral network-forming
units based on the number of bridging oxygens (BOs) per former atom.
It also computes the average network connectivity based on the Q^n
distribution.
"""

from collections import defaultdict

import numpy as np

from pyiron_glass.analysis.radial_distribution_functions import compute_coordination
from pyiron_glass.neighbors import get_neighbors

MIN_COORDINATION_FOR_BRIDGING = 2


def compute_qn(
    ids: np.ndarray,
    types: np.ndarray,
    coords: np.ndarray,
    box_size: np.ndarray,
    cutoff: float,
    former_types: list[int],
    o_type: int,
) -> tuple[dict[int, int], dict[int, dict[int, int]]]:
    """Calculate Qn distribution.

    The Q^n distribution characterizes connectivity of tetrahedral
    network-forming units based on bridging oxygens (BOs) count.

    Args:
        ids (np.ndarray): Atom IDs.
        types (np.ndarray): Atom types.
        coords (np.ndarray): Atom coordinates.
        box_size (np.ndarray): Simulation box dimensions.
        cutoff (float): Cutoff radius for former-O neighbor search.
        former_types (list[int]): Atom types considered as formers.
        o_type (int): Atom type considered as oxygen.

    Returns:
        tuple[
            dict[int, int],             # Total Qn distribution
            dict[int, dict[int, int]]   # Partial Qn per former type
        ]

    """
    neighbors = dict(
        enumerate(
            get_neighbors(coords, types, box_size, cutoff, former_types, [o_type])
        )
    )

    _, coord_numbers_o = compute_coordination(
        ids,
        types,
        coords,
        box_size,
        o_type,
        cutoff,
        neighbor_types=former_types,
    )

    total_qn_counts = defaultdict(int)
    partial_qn_counts = {f_type: defaultdict(int) for f_type in former_types}

    for idx, atom_type in enumerate(types):
        if atom_type in former_types:
            bridging_count = 0
            for neigh_idx in neighbors.get(idx, []):
                if (
                    types[neigh_idx] == o_type
                    and coord_numbers_o.get(ids[neigh_idx], 0) >= MIN_COORDINATION_FOR_BRIDGING
                ):
                    bridging_count += 1
            total_qn_counts[bridging_count] += 1
            partial_qn_counts[atom_type][bridging_count] += 1

    total_qn_counts = {n: total_qn_counts.get(n, 0) for n in range(7)}
    for f_type in former_types:
        partial_qn_counts[f_type] = {n: partial_qn_counts[f_type].get(n, 0) for n in range(7)}

    return total_qn_counts, partial_qn_counts


def compute_network_connectivity(qn_dist: dict[int, int]) -> float:
    """Compute average network connectivity based on Qn distribution.

    Args:
        qn_dist (dict[int, int]): Qn distribution histogram.

    Returns:
        float: Average network connectivity.

    Raises:
        ValueError: If qn_dist is empty or sum of counts is zero.

    """
    total_formers = sum(qn_dist.values())
    if total_formers == 0:
        error_msg = "total_formers is zero, cannot compute network connectivity."
        raise ValueError(error_msg)

    return sum(n * (count / total_formers) for n, count in qn_dist.items())
