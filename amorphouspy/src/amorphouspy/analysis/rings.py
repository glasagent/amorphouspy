"""Guttman ring statistics for network glasses.

Author: Achraf Atila (achraf.atila@bam.de)

Implements Guttman's shortest-path (primitive) ring criterion for
multicomponent glass systems. A ring of size n is a closed alternating
path T₁-O-T₂-O-…-Tₙ-O-T₁ where each T is a network former (e.g. Si, Al)
and each O is a bridging oxygen bonded to at least two formers.

A ring is *primitive* (Guttman) when no shortcut exists through the
rest of the network: the graph shortest-path distance between any two
non-adjacent ring nodes equals their arc distance along the ring.

References:
    Guttman, L. Ring structure of the crystalline and amorphous forms of
    silicon dioxide. J. Non-Cryst. Solids 116, 145-147 (1990).
    https://doi.org/10.1016/0022-3093(90)90686-G
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from ase.data import atomic_numbers as ase_atomic_numbers

from amorphouspy.neighbors import get_neighbors
from amorphouspy.shared import type_to_dict

if TYPE_CHECKING:
    from ase import Atoms

_OXYGEN_ATOMIC_NUMBER: int = 8
_MIN_BRIDGING_COORDINATION: int = 2
_SMALLEST_ALLOWED_RING: int = 3


# ============================================================================
# Internal helpers
# ============================================================================


def _symbols_to_z_cutoffs(
    bond_lengths: dict[tuple[str, str], float],
) -> tuple[dict[tuple[int, int], float], list[int]]:
    """Convert element-symbol bond-length dict to atomic-number keyed cutoffs.

    Args:
        bond_lengths: Mapping from element-symbol pairs to bond cutoff in Å.
            Example: ``{('Si', 'O'): 1.8, ('Al', 'O'): 1.95}``.

    Returns:
        z_cutoffs: Symmetric per-pair cutoff dict keyed by atomic numbers.
        former_atomic_numbers: Sorted list of atomic numbers for all
            non-oxygen elements present in ``bond_lengths``.

    Examples:
        >>> z_cutoffs, formers = _symbols_to_z_cutoffs({('Si', 'O'): 1.8})
        >>> z_cutoffs[(14, 8)]
        1.8
        >>> formers
        [14]
    """
    z_cutoffs: dict[tuple[int, int], float] = {}
    all_atomic_numbers: set[int] = set()

    for (symbol_a, symbol_b), cutoff in bond_lengths.items():
        z_a = ase_atomic_numbers[symbol_a]
        z_b = ase_atomic_numbers[symbol_b]
        z_cutoffs[(z_a, z_b)] = cutoff
        z_cutoffs[(z_b, z_a)] = cutoff
        all_atomic_numbers.add(z_a)
        all_atomic_numbers.add(z_b)

    former_atomic_numbers = sorted(all_atomic_numbers - {_OXYGEN_ATOMIC_NUMBER})
    return z_cutoffs, former_atomic_numbers


def _build_former_graph(
    atoms: Atoms,
    z_cutoffs: dict[tuple[int, int], float],
    former_atomic_numbers: list[int],
    oxygen_atomic_number: int,
) -> nx.Graph:
    """Build an undirected T-T connectivity graph through bridging oxygens.

    Two network formers are connected by an edge when they share at least
    one bridging oxygen — an oxygen bonded to two or more formers within
    the specified cutoffs. An ID-to-type lookup dict is built once upfront
    to avoid repeated array searches per neighbor.

    Args:
        atoms: Atomic structure.
        z_cutoffs: Per-pair bond cutoffs keyed by atomic number pairs.
        former_atomic_numbers: Atomic numbers of network-former species.
        oxygen_atomic_number: Atomic number of oxygen (typically 8).

    Returns:
        Undirected graph whose nodes are former atom IDs and whose edges
        represent T-O-T linkages.

    Examples:
        >>> import numpy as np
        >>> from ase import Atoms
        >>> atoms = Atoms('SiOSi',
        ...     positions=[[0,0,0],[1.6,0,0],[3.2,0,0]],
        ...     cell=[10,10,10], pbc=True)
        >>> z_cuts = {(14,8): 2.0, (8,14): 2.0}
        >>> graph = _build_former_graph(atoms, z_cuts, [14], 8)
        >>> graph.number_of_edges()
        1
    """
    raw_ids = (
        atoms.arrays["id"].astype(np.int64) if "id" in atoms.arrays else np.arange(1, len(atoms) + 1, dtype=np.int64)
    )
    atomic_numbers = atoms.get_atomic_numbers()
    id_to_type: dict[int, int] = {
        int(atom_id): int(atom_type) for atom_id, atom_type in zip(raw_ids, atomic_numbers, strict=False)
    }
    former_set = set(former_atomic_numbers)
    former_graph = nx.Graph()

    neighbor_data = get_neighbors(
        atoms,
        cutoff=z_cutoffs,
        target_types=[oxygen_atomic_number],
        neighbor_types=former_atomic_numbers,
    )

    for _oxygen_atom_id, bonded_neighbor_ids in neighbor_data:
        bonded_former_ids = [
            neighbor_id for neighbor_id in bonded_neighbor_ids if id_to_type.get(neighbor_id) in former_set
        ]
        if len(bonded_former_ids) < _MIN_BRIDGING_COORDINATION:
            continue
        for index_a in range(len(bonded_former_ids)):
            for index_b in range(index_a + 1, len(bonded_former_ids)):
                former_graph.add_edge(bonded_former_ids[index_a], bonded_former_ids[index_b])

    return former_graph


def _canonical_ring(ring_nodes: list[int]) -> tuple[int, ...]:
    """Return a canonical hashable form for a ring to enable deduplication.

    Rotates the ring so the smallest node comes first, then chooses the
    lexicographically smaller traversal direction (forward vs. reversed).

    Args:
        ring_nodes: Ordered list of node IDs forming the ring.

    Returns:
        Canonical tuple suitable for use as a set/dict key.

    Examples:
        >>> _canonical_ring([3, 1, 2])
        (1, 2, 3)
        >>> _canonical_ring([3, 2, 1])
        (1, 2, 3)
    """
    smallest_node = min(ring_nodes)
    start_index = ring_nodes.index(smallest_node)
    rotated = ring_nodes[start_index:] + ring_nodes[:start_index]
    reversed_rotated = [rotated[0], *rotated[1:][::-1]]
    return tuple(rotated) if rotated <= reversed_rotated else tuple(reversed_rotated)


def _ring_is_primitive(ring_nodes: list[int], full_graph: nx.Graph) -> bool:
    """Return True if the ring satisfies the Guttman primitiveness criterion.

    A ring is primitive when the shortest path in the *full* graph between
    every pair of non-adjacent ring nodes equals their shorter arc distance
    along the ring itself — i.e. no shortcut exists through the network.
    Three-node rings are always primitive.

    Args:
        ring_nodes: Ordered list of node IDs forming the candidate ring.
            The edge between ring_nodes[0] and ring_nodes[-1] must be
            present in ``full_graph`` when this function is called.
        full_graph: The complete T-T connectivity graph (edge restored).

    Returns:
        True if the ring is primitive (Guttman), False otherwise.

    Examples:
        >>> import networkx as nx
        >>> triangle = nx.Graph([(0,1),(1,2),(2,0)])
        >>> _ring_is_primitive([0, 1, 2], triangle)
        True
    """
    ring_size = len(ring_nodes)
    if ring_size <= _SMALLEST_ALLOWED_RING:
        return True

    for position_i in range(ring_size):
        for position_j in range(position_i + 2, ring_size):
            # Skip the edge that closes the ring (first and last node)
            if position_j == ring_size - 1 and position_i == 0:
                continue
            node_i = ring_nodes[position_i]
            node_j = ring_nodes[position_j]
            arc_distance = min(
                position_j - position_i,
                ring_size - (position_j - position_i),
            )
            try:
                graph_distance = nx.shortest_path_length(full_graph, node_i, node_j)
            except nx.NetworkXNoPath:
                continue
            if graph_distance < arc_distance:
                return False
    return True


def _find_guttman_rings(
    former_graph: nx.Graph,
    max_ring_size: int,
) -> dict[int, int]:
    """Find all Guttman primitive rings up to ``max_ring_size``.

    For every undirected edge (u, v): temporarily removes it, finds all
    shortest paths from u back to v, restores the edge, then keeps each
    candidate ring that passes the Guttman primitiveness check. Canonical
    forms prevent counting rotations and reflections of the same ring twice.

    Args:
        former_graph: Undirected T-T connectivity graph from
            ``_build_former_graph_fast``.
        max_ring_size: Maximum number of former atoms in a ring.

    Returns:
        Mapping from ring size (number of T atoms) to ring count.

    Examples:
        >>> import networkx as nx
        >>> # Six-membered ring (benzene-like topology)
        >>> hexagonal_graph = nx.cycle_graph(6)
        >>> counts = _find_guttman_rings(hexagonal_graph, max_ring_size=8)
        >>> counts[6]
        1
    """
    found_canonical_rings: set[tuple[int, ...]] = set()
    ring_counts: defaultdict[int, int] = defaultdict(int)

    for node_u, node_v in list(former_graph.edges()):
        former_graph.remove_edge(node_u, node_v)

        if not nx.has_path(former_graph, node_u, node_v):
            former_graph.add_edge(node_u, node_v)
            continue

        shortest_path_length = nx.shortest_path_length(former_graph, node_u, node_v)
        ring_size = shortest_path_length + 1

        if not (_SMALLEST_ALLOWED_RING <= ring_size <= max_ring_size):
            former_graph.add_edge(node_u, node_v)
            continue

        all_shortest_paths = list(nx.all_shortest_paths(former_graph, node_u, node_v))

        # Restore edge before primitiveness check (full graph required)
        former_graph.add_edge(node_u, node_v)

        for path in all_shortest_paths:
            canonical_form = _canonical_ring(path)
            if canonical_form in found_canonical_rings:
                continue
            if _ring_is_primitive(path, former_graph):
                found_canonical_rings.add(canonical_form)
                ring_counts[ring_size] += 1

    return dict(ring_counts)


# ============================================================================
# Public API
# ============================================================================


def compute_guttmann_rings(
    structure: Atoms,
    bond_lengths: dict[tuple[str, str], float],
    max_size: int = 24,
    n_cpus: int = 1,  # noqa: ARG001
) -> tuple[dict[int, int], float]:
    """Compute the Guttman ring size distribution and mean ring size.

    Rings are detected using a native networkx-based BFS implementation of
    Guttman's shortest-path (primitive) ring criterion. Only closed primitive
    rings up to ``max_size`` former atoms are counted.

    Ring size is defined as the number of network-former atoms (T atoms) in
    the ring, following Guttman's original convention.

    Args:
        structure: ASE Atoms object containing atomic coordinates and types.
        bond_lengths: Maximum bond lengths for each element pair, e.g.
            ``{('Si', 'O'): 1.8, ('Al', 'O'): 1.95}``. All T-O pairs must
            be specified; T-T and O-O pairs are ignored.
        max_size: Maximum ring size (number of T atoms) to search for.
        n_cpus: Unused; kept for API compatibility.

    Returns:
        histogram: Mapping from ring size to ring count.
        mean_ring_size: Mean ring size weighted by count.

    Raises:
        ValueError: If ``bond_lengths`` contains no T-O pairs (i.e. all
            pairs involve only oxygen or only formers).

    Examples:
        >>> from ase.io import read
        >>> structure = read('glass.xyz')
        >>> histogram, mean_size = compute_guttmann_rings(
        ...     structure,
        ...     bond_lengths={('Si', 'O'): 1.8},
        ...     max_size=12,
        ... )
        >>> print(mean_size)
    """
    z_cutoffs, former_atomic_numbers = _symbols_to_z_cutoffs(bond_lengths)

    if not former_atomic_numbers:
        error_message = (
            "bond_lengths contains no network-former species. Provide at least one T-O pair such as ('Si', 'O')."
        )
        raise ValueError(error_message)

    wrapped_structure = structure.copy()
    wrapped_structure.wrap()

    former_graph = _build_former_graph(
        wrapped_structure,
        z_cutoffs,
        former_atomic_numbers,
        _OXYGEN_ATOMIC_NUMBER,
    )

    ring_counts = _find_guttman_rings(former_graph, max_size)

    if not ring_counts:
        return {}, 0.0

    total_rings = sum(ring_counts.values())
    mean_ring_size = sum(size * count for size, count in ring_counts.items()) / total_rings

    return ring_counts, float(mean_ring_size)


def generate_bond_length_dict(
    atoms: Atoms,
    specific_cutoffs: dict[tuple[str, str], float] | None = None,
    default_cutoff: float = -1.0,
) -> dict[tuple[str, str], float]:
    """Generate all symmetric element pairs and assign bond length cutoffs.

    Args:
        atoms: ASE Atoms object whose species determine the pair set.
        specific_cutoffs: Optional cutoff overrides for specific element
            pairs. Both orderings ``('A','B')`` and ``('B','A')`` are
            recognised.
        default_cutoff: Fallback bond length for pairs not in
            ``specific_cutoffs``.

    Returns:
        Dictionary mapping every symmetric element pair to its cutoff.

    Examples:
        >>> from ase.io import read
        >>> structure = read('glass.xyz')
        >>> bond_lengths = generate_bond_length_dict(
        ...     structure,
        ...     specific_cutoffs={('Si', 'O'): 1.8},
        ...     default_cutoff=2.0,
        ... )
    """
    if specific_cutoffs is None:
        specific_cutoffs = {}

    atomic_numbers = atoms.get_atomic_numbers()
    type_dict = type_to_dict(atomic_numbers)
    elements = list(type_dict.values())
    bond_dict: dict[tuple[str, str], float] = {}

    for element_a, element_b in combinations_with_replacement(elements, 2):
        cutoff = specific_cutoffs.get(
            (element_a, element_b),
            specific_cutoffs.get((element_b, element_a), default_cutoff),
        )
        bond_dict[(element_a, element_b)] = cutoff

    return bond_dict
