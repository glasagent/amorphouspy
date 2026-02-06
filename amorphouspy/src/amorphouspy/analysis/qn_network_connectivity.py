"""Structural analysis functions to get the Qn and network connectivity of multicomponent glass systems.

Author: Achraf Atila (achraf.atila@bam.de)
This module provides functions to compute the Q^n distribution,
which characterizes the connectivity of tetrahedral network-forming
units based on the number of bridging oxygens (BOs) per former atom.
It also computes the average network connectivity based on the Q^n
distribution.
"""

from collections import defaultdict

from ase import Atoms

from amorphouspy.analysis.radial_distribution_functions import compute_coordination
from amorphouspy.io_utils import get_properties_for_structure_analysis
from amorphouspy.neighbors import get_neighbors

MIN_COORDINATION_FOR_BRIDGING = 2


def compute_qn(
    structure: Atoms,
    cutoff: float,
    former_types: list[int],
    o_type: int,
) -> tuple[dict[int, int], dict[int, dict[int, int]]]:
    """Calculate Qn distribution.

    The Q^n distribution characterizes connectivity of tetrahedral
    network-forming units based on bridging oxygens (BOs) count.

    Args:
        structure: The atomic structure as ASE object.
        cutoff: Cutoff radius for former-O neighbor search.
        former_types: Atom types considered as formers.
        o_type: Atom type considered as oxygen.

    Returns:
        A tuple containing:
            - dict[int, int]: Total Qn distribution
            - dict[int, dict[int, int]]: Partial Qn per former type

    Example:
        >>> structure = read('glass.xyz')
        >>> total_qn, partial_qn = compute_qn(
        ...     structure, cutoff=2.0, former_types=[14], o_type=8
        ... )

    """
    ids, types, coords, box_size = get_properties_for_structure_analysis(structure)

    neighbors = dict(enumerate(get_neighbors(coords, types, box_size, cutoff, former_types, [o_type])))

    _, coord_numbers_o = compute_coordination(
        structure,
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
        qn_dist: Qn distribution histogram.

    Returns:
        Average network connectivity.

    Raises:
        ValueError: If qn_dist is empty or sum of counts is zero.

    Example:
        >>> qn_dist = {0: 10, 1: 50, 2: 100, 3: 200, 4: 50}
        >>> connectivity = compute_network_connectivity(qn_dist)

    """
    total_formers = sum(qn_dist.values())
    if total_formers == 0:
        error_msg = "total_formers is zero, cannot compute network connectivity."
        raise ValueError(error_msg)

    return sum(n * (count / total_formers) for n, count in qn_dist.items())
