"""Guttman shortest-path ring statistics for network glasses.

Author: Achraf Atila (achraf.atila@bam.de)

Implements Guttman's shortest-path ring criterion for multicomponent glass
systems: the ring associated with a given bond is the shortest cycle that
contains it.

The search runs on the **bipartite T-O atom network**, network formers and
oxygens are both nodes, and the only edges are T-O bonds. Working at the atom
level rather than on a contracted T-T graph matters physically:

* an oxygen bonded to three or more formers (a tricluster) is a single node, so
  it cannot be traversed twice and does not masquerade as a small ring;
* two formers bridged by two distinct oxygens form a genuine four-node cycle,
  so edge-sharing polyhedra are detected instead of collapsing to one edge.

Ring size is reported as the number of network formers in the cycle, following
Guttman's convention: a cycle of ``2n`` atoms is an ``n``-membered ring. The
smallest reportable ring is therefore 2 (edge-sharing polyhedra).

Under periodic boundary conditions a candidate cycle is kept only when it
closes in real space, the minimum-image bond vectors around the cycle must sum
to zero. A path that leaves the cell and re-enters through the opposite face
returns to the same atom index but is a helix, not a ring. When every shortest
closure through a bond is rejected this way, the search deepens by two atoms at
a time until it finds the shortest closure that does close.

References:
    Guttman, L. Ring structure of the crystalline and amorphous forms of
    silicon dioxide. J. Non-Cryst. Solids 116, 145-147 (1990).
    https://doi.org/10.1016/0022-3093(90)90686-G
"""

from __future__ import annotations

import os
import warnings
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Iterable as _Iterable

try:
    from tqdm import tqdm as _tqdm
except ImportError:

    def _tqdm(iterable: _Iterable, **_: object) -> _Iterable:  # type: ignore[no-redef]
        """No-op fallback when tqdm is not installed."""
        return iterable


import numpy as np
from ase.data import atomic_numbers as ase_atomic_numbers

from amorphouspy.atoms.neighbors import get_neighbors
from amorphouspy.atoms.shared import type_to_dict

if TYPE_CHECKING:
    from ase import Atoms

_OXYGEN_ATOMIC_NUMBER: int = 8
_MIN_RING_COORDINATION: int = 2
_SMALLEST_ALLOWED_RING: int = 2

# A cycle that closes in real space sums its minimum-image bond vectors to zero
# up to ~1e-14 A of accumulated float64 error; one that only closes through the
# periodic image differs by a full lattice vector (several A). Any threshold
# between those two scales works — 1e-6 A sits eight orders clear of both.
_CLOSURE_TOLERANCE: float = 1e-6

# Upper bound on the depth-first states explored while deepening past a
# non-physical closure. Reached only by a bond whose neighbourhood wraps the
# cell without ever closing; such bonds are skipped and reported, never
# silently truncated.
_DEEPENING_STATE_BUDGET: int = 20_000

# Upper bound on the shortest closures enumerated for a single bond. A bond in a
# real network has a handful; a highly symmetric one could in principle have a
# combinatorial number, and truncating is preferable to exhausting memory. Bonds
# that hit this are counted and reported.
_MAX_CLOSURES_PER_BOND: int = 10_000

# Below this many bonds the search runs sequentially whatever ``n_cpus`` asks
# for, because spinning up worker processes costs more than it saves. Measured
# on a spawn-based platform: the search itself costs ~35 us per bond, while
# starting four workers — each re-importing the package — costs ~4 s, so four
# workers only break even past roughly 150 000 bonds. Fork-based platforms
# start workers far more cheaply, which makes this threshold conservative
# rather than wrong.
_PARALLEL_BOND_THRESHOLD: int = 100_000

# Bonds handed to a worker per task. Large chunks matter: on a 30 000-bond
# network, dropping from 256 to 2048 per chunk cut the map phase from 1.7 s to
# 0.37 s by amortising the per-task round trip.
_CHUNKS_PER_WORKER: int = 8

# Populated inside each worker process by _init_worker so that the adjacency and
# bond vectors cross the process boundary once instead of once per bond.
_WORKER_STATE: dict[str, Any] = {}


# ============================================================================
# Internal helpers — network construction
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


def _prune_ring_incapable(adjacency: list[list[int]]) -> None:
    """Iteratively strip nodes with fewer than two bonds, in place.

    A node of degree 0 or 1 cannot lie on any cycle, and removing it can drop a
    neighbour below the same threshold. Repeating to a fixed point deletes every
    dangling branch — non-bridging oxygens above all — before any search runs.

    Args:
        adjacency: Neighbour lists indexed by node, mutated in place. Pruned
            nodes end up with an empty list and are removed from every
            surviving neighbour list.

    Examples:
        >>> adjacency = [[1], [0, 2], [1, 3], [2]]
        >>> _prune_ring_incapable(adjacency)
        >>> adjacency
        [[], [], [], []]
    """
    degrees = [len(neighbors) for neighbors in adjacency]
    removed = [False] * len(adjacency)
    queue = deque(node for node, degree in enumerate(degrees) if degree < _MIN_RING_COORDINATION)

    while queue:
        node = queue.popleft()
        if removed[node]:
            continue
        removed[node] = True
        for neighbor in adjacency[node]:
            if removed[neighbor]:
                continue
            degrees[neighbor] -= 1
            if degrees[neighbor] < _MIN_RING_COORDINATION:
                queue.append(neighbor)

    for node, neighbors in enumerate(adjacency):
        adjacency[node] = [] if removed[node] else [n for n in neighbors if not removed[n]]


def _build_bipartite_network(
    atoms: Atoms,
    z_cutoffs: dict[tuple[int, int], float],
    former_atomic_numbers: list[int],
    oxygen_atomic_number: int,
) -> tuple[list[tuple[int, ...]], dict[tuple[int, int], tuple[float, float, float]]]:
    """Build the bipartite T-O network together with its minimum-image bond vectors.

    Only T-O bonds become edges, so the network is bipartite by construction —
    any T-T or O-O cutoff present in ``z_cutoffs`` is ignored here. Atom IDs are
    remapped to contiguous indices; only ring *sizes* are reported, so the
    original IDs are never needed again.

    Args:
        atoms: Atomic structure, already wrapped into the cell.
        z_cutoffs: Per-pair bond cutoffs keyed by atomic-number pairs.
        former_atomic_numbers: Atomic numbers of the network-former species.
        oxygen_atomic_number: Atomic number of oxygen (typically 8).

    Returns:
        adjacency: Neighbour indices per node, with ring-incapable nodes pruned.
        bond_vectors: Minimum-image displacement in Å for every directed bond,
            stored in both directions.

    Examples:
        >>> from ase import Atoms
        >>> atoms = Atoms('SiOSi',
        ...     positions=[[0, 0, 0], [1.6, 0, 0], [3.2, 0, 0]],
        ...     cell=[10, 10, 10], pbc=True)
        >>> adjacency, vectors = _build_bipartite_network(atoms, {(14, 8): 2.0, (8, 14): 2.0}, [14], 8)
        >>> sum(len(neighbors) for neighbors in adjacency)
        0
    """
    atomic_numbers = atoms.get_atomic_numbers()
    raw_ids = (
        atoms.arrays["id"].astype(np.int64) if "id" in atoms.arrays else np.arange(1, len(atoms) + 1, dtype=np.int64)
    )
    id_to_type: dict[int, int] = {
        int(atom_id): int(atom_type) for atom_id, atom_type in zip(raw_ids, atomic_numbers, strict=False)
    }
    former_set = set(former_atomic_numbers)

    neighbor_data = get_neighbors(
        atoms,
        cutoff=z_cutoffs,
        target_types=[oxygen_atomic_number],
        neighbor_types=former_atomic_numbers,
        return_vectors=True,
    )

    node_index: dict[int, int] = {}
    adjacency: list[list[int]] = []
    bond_vectors: dict[tuple[int, int], tuple[float, float, float]] = {}

    def _node(atom_id: int) -> int:
        index = node_index.get(atom_id)
        if index is None:
            index = len(adjacency)
            node_index[atom_id] = index
            adjacency.append([])
        return index

    for oxygen_id, bonded_ids, vectors in neighbor_data:
        if id_to_type.get(oxygen_id) != oxygen_atomic_number:
            continue
        for position, former_id in enumerate(bonded_ids):
            if id_to_type.get(former_id) not in former_set:
                continue
            oxygen_node = _node(oxygen_id)
            former_node = _node(former_id)
            adjacency[oxygen_node].append(former_node)
            adjacency[former_node].append(oxygen_node)
            vector = vectors[position]
            forward = (float(vector[0]), float(vector[1]), float(vector[2]))
            bond_vectors[(oxygen_node, former_node)] = forward
            bond_vectors[(former_node, oxygen_node)] = (-forward[0], -forward[1], -forward[2])

    _prune_ring_incapable(adjacency)
    return [tuple(neighbors) for neighbors in adjacency], bond_vectors


# ============================================================================
# Internal helpers — cycle bookkeeping
# ============================================================================


def _canonical_ring(ring_nodes: list[int]) -> tuple[int, ...]:
    """Return a canonical hashable form for a ring to enable deduplication.

    Rotates the ring so the smallest node comes first, then chooses the
    lexicographically smaller traversal direction (forward vs. reversed).

    The canonical form covers the full T-O cycle, oxygens included. Two rings
    spanning the same formers through different bridging oxygens are distinct
    rings and must not collapse onto one key.

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


def _cycle_is_physical(
    cycle: tuple[int, ...],
    bond_vectors: dict[tuple[int, int], tuple[float, float, float]],
) -> bool:
    """Return True when the cycle closes in real space rather than through an image.

    Walks the minimum-image bond vectors around the closed cycle and checks that
    they sum to zero. A cycle that only closes by crossing the periodic boundary
    accumulates a full lattice vector instead.

    Args:
        cycle: Ordered node indices; the closing bond runs from the last node
            back to the first.
        bond_vectors: Minimum-image displacement per directed bond, in Å.

    Returns:
        True if the summed displacement is below ``_CLOSURE_TOLERANCE``.

    Examples:
        >>> vectors = {(0, 1): (1.0, 0.0, 0.0), (1, 0): (-1.0, 0.0, 0.0)}
        >>> _cycle_is_physical((0, 1), vectors)
        True
    """
    sum_x = sum_y = sum_z = 0.0
    previous = cycle[-1]
    for node in cycle:
        vector = bond_vectors[(previous, node)]
        sum_x += vector[0]
        sum_y += vector[1]
        sum_z += vector[2]
        previous = node
    return sum_x * sum_x + sum_y * sum_y + sum_z * sum_z <= _CLOSURE_TOLERANCE * _CLOSURE_TOLERANCE


def _backtrack_paths(
    predecessors: dict[int, tuple[int, ...]],
    start: int,
    goal: int,
) -> tuple[list[tuple[int, ...]], bool]:
    """Expand a BFS predecessor DAG into every shortest path from start to goal.

    Each backtracking step moves to a predecessor exactly one BFS level closer
    to ``start``, so the depth strictly decreases and no node can repeat — every
    path produced is simple without needing a separate check.

    Args:
        predecessors: For each node, the neighbours one BFS level closer to
            ``start``.
        start: Source node of the BFS.
        goal: Node whose shortest paths are being reconstructed.

    Returns:
        paths: Shortest paths, each ordered from ``start`` to ``goal``.
        truncated: True if ``_MAX_CLOSURES_PER_BOND`` was reached and paths were
            left unexplored.

    Examples:
        >>> _backtrack_paths({0: (), 1: (0,), 2: (1,)}, 0, 2)
        ([(0, 1, 2)], False)
    """
    paths: list[tuple[int, ...]] = []
    stack: list[tuple[int, tuple[int, ...]]] = [(goal, (goal,))]
    while stack:
        if len(paths) >= _MAX_CLOSURES_PER_BOND:
            return paths, True
        node, path = stack.pop()
        if node == start:
            paths.append(path)
            continue
        stack.extend((predecessor, (predecessor, *path)) for predecessor in predecessors[node])
    return paths, False


# ============================================================================
# Internal helpers — the per-bond search
# ============================================================================


def _new_scratch(node_count: int) -> dict[str, list[int]]:
    """Allocate the breadth-first buffers reused across every bond of one search.

    Reallocating these per bond would dominate the runtime, so they are built
    once and recycled through a monotonically increasing visit stamp: a node
    counts as unvisited whenever its stamp predates the current sweep, which
    avoids clearing the arrays between bonds.

    Args:
        node_count: Number of nodes in the network.

    Returns:
        The buffers, keyed by name. ``visit`` is a one-element list so callees
        can advance the counter in place.

    Examples:
        >>> sorted(_new_scratch(3))
        ['backward_distance', 'backward_stamp', 'forward_distance', 'forward_stamp', 'visit']
    """
    return {
        "visit": [0],
        "forward_stamp": [0] * node_count,
        "forward_distance": [0] * node_count,
        "backward_stamp": [0] * node_count,
        "backward_distance": [0] * node_count,
    }


def _shortest_closures(
    start: int,
    goal: int,
    adjacency: list[tuple[int, ...]],
    max_path_length: int,
    scratch: dict[str, list[int]],
) -> tuple[int, list[tuple[int, ...]], bool]:
    """Enumerate every shortest path from start to goal with their direct bond suppressed.

    The breadth-first sweep stops as soon as the level containing ``goal`` is
    complete, so it touches only the nodes within the ring's own radius rather
    than the whole network.

    Args:
        start: One endpoint of the suppressed bond.
        goal: The other endpoint.
        adjacency: Neighbour indices per node.
        max_path_length: Longest path worth exploring, in edges.
        scratch: Buffers from ``_new_scratch``.

    Returns:
        length: Number of edges on a shortest path, or -1 if none exists within
            ``max_path_length``.
        paths: Every shortest path, ordered from ``start`` to ``goal``.
        truncated: True if the closure cap stopped the enumeration early.
    """
    stamp = scratch["forward_stamp"]
    distance = scratch["forward_distance"]
    scratch["visit"][0] += 1
    visit = scratch["visit"][0]

    stamp[start] = visit
    distance[start] = 0
    predecessors: dict[int, tuple[int, ...]] = {start: ()}
    frontier = [start]
    depth = 0

    while frontier and depth < max_path_length:
        depth += 1
        next_frontier: list[int] = []
        for node in frontier:
            for neighbor in adjacency[node]:
                if node == start and neighbor == goal:
                    continue
                if stamp[neighbor] != visit:
                    stamp[neighbor] = visit
                    distance[neighbor] = depth
                    predecessors[neighbor] = (node,)
                    next_frontier.append(neighbor)
                elif distance[neighbor] == depth:
                    predecessors[neighbor] += (node,)
        if stamp[goal] == visit:
            paths, truncated = _backtrack_paths(predecessors, start, goal)
            return distance[goal], paths, truncated
        frontier = next_frontier

    return -1, [], False


def _mark_backward_distances(
    start: int,
    goal: int,
    adjacency: list[tuple[int, ...]],
    max_path_length: int,
    scratch: dict[str, list[int]],
) -> int:
    """Record every node's distance to goal, with the start-goal bond suppressed.

    Used to prune the deepening search: a node further from ``goal`` than the
    remaining budget cannot reach it in time.

    Args:
        start: One endpoint of the suppressed bond.
        goal: The other endpoint, and the source of this sweep.
        adjacency: Neighbour indices per node.
        max_path_length: Longest path worth exploring, in edges.
        scratch: Buffers from ``_new_scratch``.

    Returns:
        The visit stamp identifying this sweep.
    """
    stamp = scratch["backward_stamp"]
    distance = scratch["backward_distance"]
    scratch["visit"][0] += 1
    visit = scratch["visit"][0]

    stamp[goal] = visit
    distance[goal] = 0
    frontier = [goal]
    depth = 0

    while frontier and depth < max_path_length:
        depth += 1
        next_frontier: list[int] = []
        for node in frontier:
            for neighbor in adjacency[node]:
                if node == goal and neighbor == start:
                    continue
                if stamp[neighbor] != visit:
                    stamp[neighbor] = visit
                    distance[neighbor] = depth
                    next_frontier.append(neighbor)
        frontier = next_frontier

    return visit


def _paths_of_length(
    start: int,
    goal: int,
    length: int,
    adjacency: list[tuple[int, ...]],
    visit: int,
    budget: int,
    scratch: dict[str, list[int]],
) -> tuple[list[tuple[int, ...]], int]:
    """Enumerate simple paths of exactly ``length`` edges from start to goal.

    Args:
        start: One endpoint of the suppressed bond.
        goal: The other endpoint.
        length: Exact number of edges the paths must have.
        adjacency: Neighbour indices per node.
        visit: Visit stamp from ``_mark_backward_distances``.
        budget: Remaining depth-first states this search may explore.
        scratch: Buffers from ``_new_scratch``.

    Returns:
        paths: Every simple path of exactly ``length`` edges.
        spent: Number of states explored; equals ``budget`` when exhausted.
    """
    stamp = scratch["backward_stamp"]
    distance = scratch["backward_distance"]
    paths: list[tuple[int, ...]] = []
    stack: list[tuple[int, tuple[int, ...], frozenset[int]]] = [(start, (start,), frozenset((start,)))]
    spent = 0

    while stack:
        spent += 1
        if spent >= budget:
            return [], budget
        node, path, visited = stack.pop()
        remaining = length - (len(path) - 1)
        if node == goal:
            if remaining == 0:
                paths.append(path)
            continue
        if remaining <= 0 or stamp[node] != visit or distance[node] > remaining:
            continue
        for neighbor in adjacency[node]:
            if node == start and neighbor == goal:
                continue
            if neighbor in visited:
                continue
            stack.append((neighbor, (*path, neighbor), visited | {neighbor}))

    return paths, spent


def _deepen(
    start: int,
    goal: int,
    shortest_length: int,
    adjacency: list[tuple[int, ...]],
    bond_vectors: dict[tuple[int, int], tuple[float, float, float]],
    max_path_length: int,
    scratch: dict[str, list[int]],
) -> tuple[list[tuple[int, ...]], bool]:
    """Search past a non-physical shortest closure for the shortest one that closes.

    The network is bipartite, so every path between the two endpoints of a bond
    has odd length; only lengths of matching parity are tried.

    Args:
        start: One endpoint of the suppressed bond.
        goal: The other endpoint.
        shortest_length: Length of the rejected shortest closure.
        adjacency: Neighbour indices per node.
        bond_vectors: Minimum-image displacement per directed bond, in Å.
        max_path_length: Longest path worth exploring, in edges.
        scratch: Buffers from ``_new_scratch``.

    Returns:
        cycles: Every physical closure at the first length that yields one, or
            an empty list if none exists within the search bounds.
        exhausted: True if the state budget ran out before a closure was found.
    """
    visit = _mark_backward_distances(start, goal, adjacency, max_path_length, scratch)
    budget = _DEEPENING_STATE_BUDGET

    for length in range(shortest_length + 2, max_path_length + 1, 2):
        candidates, spent = _paths_of_length(start, goal, length, adjacency, visit, budget, scratch)
        budget -= spent
        if budget <= 0:
            return [], True
        physical = [path for path in candidates if _cycle_is_physical(path, bond_vectors)]
        if physical:
            return physical, False

    return [], False


def _closures_through(
    start: int,
    goal: int,
    adjacency: list[tuple[int, ...]],
    bond_vectors: dict[tuple[int, int], tuple[float, float, float]],
    max_path_length: int,
    scratch: dict[str, list[int]],
) -> tuple[list[tuple[int, ...]], bool, bool]:
    """Return the shortest physical cycles running through the bond (start, goal).

    Args:
        start: One endpoint of the bond.
        goal: The other endpoint.
        adjacency: Neighbour indices per node.
        bond_vectors: Minimum-image displacement per directed bond, in Å.
        max_path_length: Longest path worth exploring, in edges.
        scratch: Buffers from ``_new_scratch``.

    Returns:
        cycles: Every shortest physical cycle through that bond, each ordered
            from ``start`` to ``goal`` with the bond itself closing the loop.
        exhausted: True if deepening gave up against its state budget.
        truncated: True if the closure enumeration hit its cap.
    """
    length, paths, truncated = _shortest_closures(start, goal, adjacency, max_path_length, scratch)
    if length < 0:
        return [], False, truncated
    physical = [path for path in paths if _cycle_is_physical(path, bond_vectors)]
    if physical:
        return physical, False, truncated
    cycles, exhausted = _deepen(start, goal, length, adjacency, bond_vectors, max_path_length, scratch)
    return cycles, exhausted, truncated


def _init_worker(
    adjacency: list[tuple[int, ...]],
    bond_vectors: dict[tuple[int, int], tuple[float, float, float]],
    max_ring_size: int,
) -> None:
    """Stash the network in each worker process so it crosses the boundary once.

    Args:
        adjacency: Neighbour indices per node.
        bond_vectors: Minimum-image displacement per directed bond, in Å.
        max_ring_size: Largest ring, in network formers, worth searching for.
    """
    _WORKER_STATE["adjacency"] = adjacency
    _WORKER_STATE["bond_vectors"] = bond_vectors
    _WORKER_STATE["max_path_length"] = 2 * max_ring_size - 1
    _WORKER_STATE["scratch"] = _new_scratch(len(adjacency))


def _process_bond(bond: tuple[int, int]) -> tuple[list[tuple[int, ...]], bool, bool]:
    """Find the shortest physical cycles through one bond inside a worker process.

    Args:
        bond: The two endpoint indices.

    Returns:
        cycles: Shortest physical cycles through the bond.
        exhausted: True if the deepening budget was hit for this bond.
        truncated: True if the closure enumeration was capped for this bond.
    """
    start, goal = bond
    return _closures_through(
        start,
        goal,
        _WORKER_STATE["adjacency"],
        _WORKER_STATE["bond_vectors"],
        _WORKER_STATE["max_path_length"],
        _WORKER_STATE["scratch"],
    )


def _tally_cycles(
    cycles: list[tuple[int, ...]],
    max_ring_size: int,
    seen: set[tuple[int, ...]],
    ring_counts: defaultdict[int, int],
) -> None:
    """Fold newly found cycles into the running histogram, skipping duplicates.

    Args:
        cycles: Candidate cycles as ordered node indices.
        max_ring_size: Largest ring size, in network formers, to retain.
        seen: Canonical forms already counted; updated in place.
        ring_counts: Ring size to count; updated in place.
    """
    for cycle in cycles:
        ring_size = len(cycle) // 2
        if not (_SMALLEST_ALLOWED_RING <= ring_size <= max_ring_size):
            continue
        canonical_form = _canonical_ring(list(cycle))
        if canonical_form in seen:
            continue
        seen.add(canonical_form)
        ring_counts[ring_size] += 1


def _find_guttman_rings(
    adjacency: list[tuple[int, ...]],
    bond_vectors: dict[tuple[int, int], tuple[float, float, float]],
    max_ring_size: int,
    n_cpus: int = 1,
) -> tuple[dict[int, int], int, int]:
    """Find the shortest physical cycle through every bond of the network.

    Args:
        adjacency: Neighbour indices per node, from ``_build_bipartite_network``.
        bond_vectors: Minimum-image displacement per directed bond, in Å.
        max_ring_size: Maximum number of network formers in a ring.
        n_cpus: Number of worker processes. Networks with fewer than
            ``_PARALLEL_BOND_THRESHOLD`` bonds run sequentially regardless,
            since starting workers would cost more than it saves.
            ``1``  — sequential (default, no process overhead).
            ``N``  — use N worker processes.
            ``-1`` — use all logical CPUs.

    Returns:
        ring_counts: Mapping from ring size (network formers) to ring count.
        exhausted_bonds: Number of bonds abandoned by the deepening budget.
        truncated_bonds: Number of bonds whose closure enumeration was capped.

    Examples:
        >>> # Two formers bridged by two oxygens: one edge-sharing 2-ring.
        >>> adjacency = [(1, 3), (0, 2), (1, 3), (2, 0)]
        >>> vectors = {}
        >>> for a, b, dx in ((0, 1, 1.0), (1, 2, 1.0), (2, 3, -1.0), (3, 0, -1.0)):
        ...     vectors[(a, b)] = (dx, 0.0, 0.0)
        ...     vectors[(b, a)] = (-dx, 0.0, 0.0)
        >>> counts, exhausted, truncated = _find_guttman_rings(adjacency, vectors, max_ring_size=4)
        >>> counts
        {2: 1}
    """
    bonds = [(node, neighbor) for node, neighbors in enumerate(adjacency) for neighbor in neighbors if neighbor > node]
    seen: set[tuple[int, ...]] = set()
    ring_counts: defaultdict[int, int] = defaultdict(int)
    exhausted_bonds = 0
    truncated_bonds = 0

    if n_cpus == 1 or len(bonds) < _PARALLEL_BOND_THRESHOLD:
        max_path_length = 2 * max_ring_size - 1
        scratch = _new_scratch(len(adjacency))
        for start, goal in _tqdm(bonds, desc="Finding rings", unit="bond"):
            cycles, exhausted, truncated = _closures_through(
                start, goal, adjacency, bond_vectors, max_path_length, scratch
            )
            _tally_cycles(cycles, max_ring_size, seen, ring_counts)
            exhausted_bonds += exhausted
            truncated_bonds += truncated
    else:
        workers = (os.cpu_count() or 1) if n_cpus == -1 else n_cpus
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(adjacency, bond_vectors, max_ring_size),
        ) as pool:
            results = _tqdm(
                pool.map(_process_bond, bonds, chunksize=max(1, len(bonds) // (workers * _CHUNKS_PER_WORKER))),
                total=len(bonds),
                desc="Finding rings",
                unit="bond",
            )
            for cycles, exhausted, truncated in results:
                _tally_cycles(cycles, max_ring_size, seen, ring_counts)
                exhausted_bonds += exhausted
                truncated_bonds += truncated

    return dict(ring_counts), exhausted_bonds, truncated_bonds


# ============================================================================
# Public API
# ============================================================================


def compute_guttmann_rings(
    structure: Atoms | list[Atoms],
    bond_lengths: dict[tuple[str, str], float],
    max_size: int = 24,
    n_cpus: int = 1,
) -> tuple[dict[int, float], float]:
    """Compute the Guttman ring size distribution and mean ring size.

    The ring associated with each T-O bond is the shortest cycle containing it.
    The search runs on the bipartite T-O atom network, so oxygen triclusters
    cannot masquerade as small rings and edge-sharing polyhedra are detected.
    Under periodic boundary conditions only cycles that close in real space are
    counted; see the module docstring for the full criterion.

    Ring size is the number of network-former atoms (T atoms) in the ring,
    following Guttman's original convention. The smallest reportable ring is 2,
    two formers bridged by two distinct oxygens.

    Args:
        structure: ASE Atoms object containing atomic coordinates and types.
            Pass a list to use the first frame.
        bond_lengths: Maximum bond lengths for each element pair, e.g.
            ``{('Si', 'O'): 1.8, ('Al', 'O'): 1.95}``. All T-O pairs must
            be specified; T-T and O-O pairs are ignored.
        max_size: Maximum ring size (number of T atoms) to search for.
        n_cpus: Number of worker processes for parallel ring search. Only
            very large networks (>100 000 T-O bonds, roughly a 10⁵-atom cell)
            benefit; below that the search runs sequentially whatever is asked
            for, because starting worker processes costs more than it saves.
            ``1``  — sequential execution (default).
            ``N``  — distribute the bond loop across N worker processes.
            ``-1`` — use all logical CPUs.

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
        >>> # Parallel on 4 cores
        >>> histogram, mean_size = compute_guttmann_rings(
        ...     structure,
        ...     bond_lengths={('Si', 'O'): 1.8},
        ...     n_cpus=4,
        ... )
    """
    if isinstance(structure, list):
        structure = cast("Atoms", structure[0])
    z_cutoffs, former_atomic_numbers = _symbols_to_z_cutoffs(bond_lengths)

    if not former_atomic_numbers:
        error_message = (
            "bond_lengths contains no network-former species. Provide at least one T-O pair such as ('Si', 'O')."
        )
        raise ValueError(error_message)

    wrapped_structure = structure.copy()
    wrapped_structure.wrap()

    adjacency, bond_vectors = _build_bipartite_network(
        wrapped_structure,
        z_cutoffs,
        former_atomic_numbers,
        _OXYGEN_ATOMIC_NUMBER,
    )

    ring_counts, exhausted_bonds, truncated_bonds = _find_guttman_rings(
        adjacency, bond_vectors, max_size, n_cpus=n_cpus
    )

    if exhausted_bonds:
        warnings.warn(
            f"Ring search abandoned {exhausted_bonds} bond(s) after exhausting the deepening budget of "
            f"{_DEEPENING_STATE_BUDGET} states; their rings are missing from the histogram. This happens when a "
            f"bond's neighbourhood wraps the periodic cell without closing — a larger cell or a smaller max_size "
            f"avoids it.",
            RuntimeWarning,
            stacklevel=2,
        )

    if truncated_bonds:
        warnings.warn(
            f"Ring search capped the closure enumeration at {_MAX_CLOSURES_PER_BOND} for {truncated_bonds} bond(s); "
            f"some rings through them are missing from the histogram. This means the network is far more degenerate "
            f"than a glass or an ordinary crystal — check the bond cutoffs.",
            RuntimeWarning,
            stacklevel=2,
        )

    if not ring_counts:
        return {}, 0.0

    total_rings = sum(ring_counts.values())
    mean_ring_size = sum(size * count for size, count in ring_counts.items()) / total_rings
    ring_counts_float: dict[int, float] = {k: float(v) for k, v in ring_counts.items()}
    return ring_counts_float, float(mean_ring_size)


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
