"""
Author: Achraf Atila (achraf.atila@bam.de)

Structural analysis functions to get the Qn and network connectivity
 of multicomponent glass systems.



Note: For now, only LAMMPS dump files can be handled.
It reads a lammps dump file and uses a cell list algorithm for neighbor search
under periodic boundary conditions (PBC).
"""

from collections import defaultdict
import numpy as np
from pyiron_glass.neighbors import get_neighbors
from pyiron_glass.analysis.radial_distribution_functions import compute_coordination

# constnts

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
    """
    Calculate Qn distribution: number of bridging oxygens per former atom,
    and partial Qn distributions for each former type.

    The Q^n distribution characterizes the connectivity of tetrahedral
    network-forming units (e.g., SiO_4) based on the number of bridging
    oxygens (BOs) per tetrahedron. Each unit is classified as Q^n, where
    n is the number of BOs and (4 - n) is the number of non-bridging
    oxygens (NBOs).

    Q^0: 0 BOs - isolated tetrahedron
    Q^1: 1 BO  - end of a chain
    Q^2: 2 BOs - middle of a chain or ring
    Q^3: 3 BOs - branched structure
    Q^4: 4 BOs - fully connected 3D network (e.g., in pure silica)

    The Q^n distribution is sensitive to the glass composition; the
    addition of network modifiers (e.g., Na, Ca) increases the number
    of NBOs, shifting the distribution toward lower Q^n species. This
    distribution is commonly used to quantify the degree of network
    polymerization.


    Args:
        ids (np.ndarray): Atom IDs.
        types (np.ndarray): Atom types.
        coords (np.ndarray): Atom coordinates.
        box_size (np.ndarray): Simulation box dimensions.
        cutoff (float): Cutoff radius for former-O neighbor search.
        former_types (List[int]): Atom types considered as formers (e.g., Si, B, etc.).
        o_type (int): Atom type considered as oxygen.

    Returns:
        Tuple[
            Dict[int, int],              # Total Qn distribution
            Dict[int, Dict[int, int]]   # Partial Qn per former type
        ]

    """
    neighbors = dict(
        enumerate(
            get_neighbors(coords, types, box_size, cutoff, former_types, [o_type]),
        ),
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
            for j in neighbors.get(idx, []):
                if (
                    types[j] == o_type
                    and coord_numbers_o.get(ids[j], 0) >= MIN_COORDINATION_FOR_BRIDGING
                ):
                    bridging_count += 1
            total_qn_counts[bridging_count] += 1
            partial_qn_counts[atom_type][bridging_count] += 1

    # Normalize output
    total_qn_counts = {n: total_qn_counts.get(n, 0) for n in range(7)}
    for f_type in former_types:
        partial_qn_counts[f_type] = {
            n: partial_qn_counts[f_type].get(n, 0) for n in range(7)
        }

    return total_qn_counts, partial_qn_counts


def compute_network_connectivity(qn_dist: dict[int, int]) -> float:
    """
    Compute average network connectivity based on Qn distribution.

    Network connectivity quantifies the average number of bridging
    oxygens (BOs) per network-forming tetrahedral unit (e.g., SiO_4).
    It reflects the degree of polymerization of the glass network and
    is influenced by the presence of modifiers (e.g., Na, Ca), which
    create non-bridging oxygens (NBOs) and reduce connectivity.

    For silicate glasses, the theoretical maximum network connectivity
    at ambiant condictions is 4.0, corresponding to a fully polymerized
    structure (pure silica, all Q^4 units). The addition of modifiers
    breaks Si-O-Si bridges, reducing the number of BOs and thus
    decreasing the connectivity.

    Args:
        Qn_dist (Dict[int, int]): Qn distribution histogram.

    Returns:
        float: Average network connectivity.

    Raises:
        ValueError: If Qn_dist is empty or total_formers is zero.

    """
    total_formers = sum(qn_dist.values())

    if total_formers == 0:
        msg = "total_formers is zero, cannot compute network connectivity."
        raise ValueError(msg)

    return sum(n * (count / total_formers) for n, count in qn_dist.items())
