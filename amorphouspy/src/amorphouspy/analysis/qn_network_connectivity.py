"""Structural analysis functions to get the Qn and network connectivity of multicomponent glass systems.

Author: Achraf Atila (achraf.atila@bam.de)
This module provides functions to compute the Q^n distribution,
which characterizes the connectivity of tetrahedral network-forming
units based on the number of bridging oxygens (BOs) per former atom.
It also computes the average network connectivity based on the Q^n
distribution.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, cast

import numpy as np

from amorphouspy.analysis.averaging import frame_average
from amorphouspy.neighbors import get_neighbors

if TYPE_CHECKING:
    from ase import Atoms

MIN_COORDINATION_FOR_BRIDGING = 2


def _classify_oxygens_raw(
    structure: Atoms,
    cutoff: float,
    former_types: list[int],
    o_type: int,
    id_to_type: dict[int, int],
) -> tuple[dict[int, str], set[int]]:
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
    return oxygen_classes, bridging_o_ids


def _count_bridging_per_former(
    structure: Atoms,
    cutoff: float,
    former_types: list[int],
    o_type: int,
    id_to_type: dict[int, int],
    bridging_o_ids: set[int],
) -> tuple[dict[int, int], dict[int, dict[int, int]]]:
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
    total_qn_norm = {n: total_qn_counts.get(n, 0) for n in range(7)}
    partial_plain = {f_type: {n: partial_qn_counts[f_type].get(n, 0) for n in range(7)} for f_type in former_types}
    return total_qn_norm, partial_plain


def compute_qn(
    structure: Atoms | list[Atoms],
    cutoff: float,
    former_types: list[int],
    o_type: int,
    *,
    frame_averaging: bool = False,
) -> tuple[dict[int, float], dict[int, dict[int, float]], dict[int, float], dict[int, dict[int, float]]]:
    """Calculate Qn distribution.

    The Q^n distribution characterizes connectivity of tetrahedral
    network-forming units based on bridging oxygens (BOs) count.

    Args:
        structure: The atomic structure as ASE object, or a list of frames when frame_averaging=True.
        cutoff: Cutoff radius for former-O neighbor search.
        former_types: Atom types considered as formers.
        o_type: Atom type considered as oxygen.
        frame_averaging: If True, average results over all frames in structure (list[Atoms]).

    Returns:
        A 4-tuple containing:
            total_qn: Total Qn distribution (or mean when frame_averaging=True).
            partial_qn: Partial Qn by former type (or mean when frame_averaging=True).
            total_qn_sem: SEM of total Qn (zeros when frame_averaging=False).
            partial_qn_sem: SEM of partial Qn (zeros when frame_averaging=False).

    Example:
        >>> structure = read('glass.xyz')
        >>> total_qn, partial_qn, total_sem, partial_sem = compute_qn(
        ...     structure, cutoff=2.0, former_types=[14], o_type=8
        ... )

    """
    if frame_averaging:
        if isinstance(structure, list) and len(structure) == 0:
            msg = "frame_averaging=True requires a non-empty list[Atoms]"
            raise ValueError(msg)
        frames = cast("list[Atoms]", structure if isinstance(structure, list) else [structure])
        means, sems = frame_average(
            lambda f: compute_qn(f, cutoff, former_types, o_type),
            frames,
            avg_indices=[0, 1],
        )
        return means[0], means[1], sems[0], sems[1]
    if isinstance(structure, list):
        structure = cast("Atoms", structure[0])
    total_qn, partial_qn, _o_classes, total_sem, partial_sem = compute_qn_and_classify(
        structure, cutoff, former_types, o_type
    )
    return total_qn, partial_qn, total_sem, partial_sem


def compute_qn_and_classify(
    structure: Atoms | list[Atoms],
    cutoff: float,
    former_types: list[int],
    o_type: int,
    *,
    frame_averaging: bool = False,
) -> tuple[
    dict[int, float], dict[int, dict[int, float]], dict[int, str], dict[int, float], dict[int, dict[int, float]]
]:
    """Calculate Qn distribution and classify each oxygen as BO/NBO/free/tri.

    Performs a single neighbour search pass to compute both the Q^n
    distribution *and* the per-oxygen classification.

    Args:
        structure: The atomic structure as ASE object, or a list of frames when frame_averaging=True.
        cutoff: Cutoff radius for former-O neighbor search (Å).
        former_types: Atom types (atomic numbers) considered as formers.
        o_type: Atom type (atomic number) considered as oxygen.
        frame_averaging: If True, average results over all frames in structure (list[Atoms]).

    Returns:
        A 5-tuple containing:
            total_qn: Total Qn distribution (or mean when frame_averaging=True).
            partial_qn: Partial Qn by former type (or mean when frame_averaging=True).
            oxygen_classes: Mapping from real atom ID to oxygen class string (last frame when frame_averaging=True).
            total_qn_sem: SEM of total Qn (zeros when frame_averaging=False).
            partial_qn_sem: SEM of partial Qn (zeros when frame_averaging=False).

    Example:
        >>> total_qn, partial_qn, o_classes = compute_qn_and_classify(
        ...     structure, cutoff=2.0, former_types=[14], o_type=8
        ... )

    """
    if frame_averaging:
        if isinstance(structure, list) and len(structure) == 0:
            msg = "frame_averaging=True requires a non-empty list[Atoms]"
            raise ValueError(msg)
        frames = cast("list[Atoms]", structure if isinstance(structure, list) else [structure])
        # avg_indices=[0, 1] — total_qn and partial_qn; oxygen_classes (index 2) passes through from last frame
        means, sems = frame_average(
            lambda f: compute_qn_and_classify(f, cutoff, former_types, o_type),
            frames,
            avg_indices=[0, 1],
        )
        return means[0], means[1], means[2], sems[0], sems[1]
    if isinstance(structure, list):
        structure = cast("Atoms", structure[0])
    types = structure.get_atomic_numbers()
    if "id" in structure.arrays:
        raw_ids = structure.arrays["id"].astype(np.int64)
    else:
        raw_ids = np.arange(1, len(structure) + 1, dtype=np.int64)
    id_to_type = {int(aid): int(t) for aid, t in zip(raw_ids, types, strict=False)}
    oxygen_classes, bridging_o_ids = _classify_oxygens_raw(structure, cutoff, former_types, o_type, id_to_type)
    total_qn_norm, partial_plain = _count_bridging_per_former(
        structure, cutoff, former_types, o_type, id_to_type, bridging_o_ids
    )
    total_qn_float: dict[int, float] = {n: float(v) for n, v in total_qn_norm.items()}
    partial_float: dict[int, dict[int, float]] = {
        f: {n: float(v) for n, v in qn.items()} for f, qn in partial_plain.items()
    }
    total_sem: dict[int, float] = dict.fromkeys(total_qn_norm, 0.0)
    partial_sem: dict[int, dict[int, float]] = {f: dict.fromkeys(qn, 0.0) for f, qn in partial_plain.items()}
    return total_qn_float, partial_float, oxygen_classes, total_sem, partial_sem


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
    _total_qn, _partial_qn, o_classes, _total_sem, _partial_sem = compute_qn_and_classify(
        structure, cutoff, former_types, o_type
    )
    return o_classes


def compute_network_connectivity(qn_dist: dict[int, int | float]) -> float:
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
