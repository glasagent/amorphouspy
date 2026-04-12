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
from ase import Atoms

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
            total_qn: Total Qn distribution (mapping from n to count).
            partial_qn: Partial Qn (mapping from former type to Qn distribution).

    Example:
        >>> structure = read('glass.xyz')
        >>> total_qn, partial_qn = compute_qn(
        ...     structure, cutoff=2.0, former_types=[14], o_type=8
        ... )

    """
    total_qn, partial_qn, _o_classes = compute_qn_and_classify(structure, cutoff, former_types, o_type)
    return total_qn, partial_qn


def compute_qn_and_classify(
    structure: Atoms,
    cutoff: float,
    former_types: list[int],
    o_type: int,
) -> tuple[dict[int, int], dict[int, dict[int, int]], dict[int, str]]:
    """Calculate Qn distribution and classify each oxygen as BO/NBO/free/tri.

    Performs a single neighbour search pass to compute both the Q^n
    distribution *and* the per-oxygen classification.

    Args:
        structure: The atomic structure as ASE object.
        cutoff: Cutoff radius for former-O neighbor search (Å).
        former_types: Atom types (atomic numbers) considered as formers.
        o_type: Atom type (atomic number) considered as oxygen.

    Returns:
        A tuple containing:
            total_qn: Total Qn distribution (mapping from n to count).
            partial_qn: Partial Qn (mapping from former type to Qn distribution).
            oxygen_classes: Mapping from real atom ID to oxygen class string
                (``"BO"``, ``"NBO"``, ``"free"``, or ``"tri"``).

    Example:
        >>> total_qn, partial_qn, o_classes = compute_qn_and_classify(
        ...     structure, cutoff=2.0, former_types=[14], o_type=8
        ... )

    """
    # Build real-ID -> atom type map
    types = structure.get_atomic_numbers()
    if "id" in structure.arrays:
        raw_ids = structure.arrays["id"].astype(np.int64)
    else:
        raw_ids = np.arange(1, len(structure) + 1, dtype=np.int64)
    id_to_type = {int(aid): int(t) for aid, t in zip(raw_ids, types, strict=False)}

    # --- Step 1: classify oxygens and identify bridging set -----------------
    oxygen_classes: dict[int, str] = {}
    bridging_o_ids: set[int] = set()

    for cid, nns in get_neighbors(
        structure,
        cutoff=cutoff,
        target_types=[o_type],
        neighbor_types=former_types,
    ):
        if id_to_type.get(cid) != o_type:
            continue
        n_formers = len(nns)
        if n_formers == 0:
            oxygen_classes[cid] = "free"
        elif n_formers == 1:
            oxygen_classes[cid] = "NBO"
        elif n_formers == MIN_COORDINATION_FOR_BRIDGING:
            oxygen_classes[cid] = "BO"
            bridging_o_ids.add(cid)
        else:
            oxygen_classes[cid] = "tri"
            bridging_o_ids.add(cid)

    # --- Step 2: count bridging O per former --------------------------------
    total_qn_counts: dict[int, int] = defaultdict(int)
    partial_qn_counts = {f_type: defaultdict(int) for f_type in former_types}

    for central_id, nn_ids in get_neighbors(
        structure,
        cutoff=cutoff,
        target_types=former_types,
        neighbor_types=[o_type],
    ):
        atom_type = id_to_type.get(central_id)
        if atom_type not in former_types:
            continue

        bridging_count = sum(1 for nid in nn_ids if nid in bridging_o_ids)
        total_qn_counts[bridging_count] += 1
        partial_qn_counts[atom_type][bridging_count] += 1

    # Normalise to Q0-Q6 keys
    total_qn_norm = {n: total_qn_counts.get(n, 0) for n in range(7)}
    partial_plain = {f_type: {n: partial_qn_counts[f_type].get(n, 0) for n in range(7)} for f_type in former_types}
    return total_qn_norm, partial_plain, oxygen_classes


def classify_oxygens(
    structure: Atoms,
    cutoff: float,
    former_types: list[int],
    o_type: int,
) -> dict[int, str]:
    """Classify each oxygen atom as bridging (BO), non-bridging (NBO), free, or triclustered.

    Convenience wrapper around :func:`compute_qn_and_classify` that returns
    only the oxygen classification.

    Args:
        structure: The atomic structure as ASE object.
        cutoff: Cutoff radius for former-O neighbor search (Å).
        former_types: Atom types (atomic numbers) considered as formers.
        o_type: Atom type (atomic number) considered as oxygen.

    Returns:
        A mapping from real atom ID to oxygen class string:
        ``"BO"`` (bridging, 2 former neighbours), ``"NBO"`` (non-bridging, 1),
        ``"free"`` (0), or ``"tri"`` (≥ 3 triclustered).

    Example:
        >>> o_classes = classify_oxygens(atoms, cutoff=2.0, former_types=[14], o_type=8)

    """
    _total_qn, _partial_qn, o_classes = compute_qn_and_classify(structure, cutoff, former_types, o_type)
    return o_classes


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
        msg = "total_formers is zero, cannot compute network connectivity."
        raise ValueError(msg)

    return sum(n * (count / total_formers) for n, count in qn_dist.items())
