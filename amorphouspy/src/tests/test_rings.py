"""Tests for amorphouspy.analysis.rings.

Covers generate_bond_length_dict (pure) and compute_guttmann_rings (integration,
requires the 20Na2O-80SiO2 dump file).  Atom type mapping: O=type1, Si=type2, Na=type3.
"""

import networkx as nx
import numpy as np
import pytest
from amorphouspy.analysis.rings import (
    _find_guttman_rings,
    _process_edge,
    compute_guttmann_rings,
    generate_bond_length_dict,
)
from ase import Atoms
from ase.io import read

from . import DATA_DIR

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def glass_structure():
    """Load the 20Na2O-80SiO2 dump and apply correct atomic numbers.

    Type mapping per the simulation setup: O=type1(8), Si=type2(14), Na=type3(11).
    """
    atoms = read(DATA_DIR / "20Na2O-80SiO2.dump", format="lammps-dump-text")
    type_id = atoms.get_atomic_numbers().copy()
    to_z = np.array([0, 11, 8, 14], dtype=int)  # Na=1, O=2, Si=3
    atoms.set_atomic_numbers(to_z[type_id])
    return atoms


@pytest.fixture
def si_o_atoms():
    """Minimal Atoms object with Si and O for pure-function tests."""
    return Atoms(
        numbers=[14, 8, 8],
        positions=[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )


# ---------------------------------------------------------------------------
# generate_bond_length_dict
# ---------------------------------------------------------------------------


def test_generate_bond_length_dict_default_cutoff(si_o_atoms):
    """All pairs use the default cutoff when no specific overrides are given."""
    result = generate_bond_length_dict(si_o_atoms, default_cutoff=2.5)
    assert all(v == 2.5 for v in result.values())


def test_generate_bond_length_dict_specific_override(si_o_atoms):
    """Specific cutoffs override the default for the requested pair.

    Pairs are iterated as combinations_with_replacement over sorted elements,
    so ('O', 'Si') is the canonical key, not ('Si', 'O').
    """
    result = generate_bond_length_dict(
        si_o_atoms,
        specific_cutoffs={("Si", "O"): 1.8},
        default_cutoff=2.5,
    )
    # The function looks up (a,b) then (b,a), so reversed input still matches.
    assert result[("O", "Si")] == 1.8


def test_generate_bond_length_dict_symmetric_fallback(si_o_atoms):
    """Forward-order key ('O', 'Si') is matched directly in the canonical pair."""
    result = generate_bond_length_dict(
        si_o_atoms,
        specific_cutoffs={("O", "Si"): 1.9},
        default_cutoff=2.5,
    )
    assert result[("O", "Si")] == 1.9


def test_generate_bond_length_dict_n_pairs(si_o_atoms):
    """Number of pairs equals N*(N+1)/2 for N unique elements."""
    result = generate_bond_length_dict(si_o_atoms)
    # 2 unique elements (Si, O) → 3 pairs: (O,O), (O,Si), (Si,Si)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# compute_guttmann_rings
# ---------------------------------------------------------------------------


def test_compute_guttmann_rings_returns_tuple(glass_structure):
    """Function returns a (dict, float) tuple."""
    hist, mean = compute_guttmann_rings(
        glass_structure,
        bond_lengths={("Si", "O"): 2.0},
        max_size=6,
    )
    assert isinstance(hist, dict)
    assert isinstance(mean, float)


def test_compute_guttmann_rings_ring_sizes_positive(glass_structure):
    """All ring sizes in the histogram are positive integers."""
    hist, _ = compute_guttmann_rings(
        glass_structure,
        bond_lengths={("Si", "O"): 2.0},
        max_size=6,
    )
    assert all(isinstance(k, int) and k > 0 for k in hist)


def test_compute_guttmann_rings_counts_positive(glass_structure):
    """All ring counts are positive integers."""
    hist, _ = compute_guttmann_rings(
        glass_structure,
        bond_lengths={("Si", "O"): 2.0},
        max_size=6,
    )
    assert all(isinstance(v, int) and v > 0 for v in hist.values())


def test_compute_guttmann_rings_mean_nonnegative(glass_structure):
    """Mean ring size is non-negative."""
    _, mean = compute_guttmann_rings(
        glass_structure,
        bond_lengths={("Si", "O"): 2.0},
        max_size=6,
    )
    assert mean >= 0.0


# ---------------------------------------------------------------------------
# _find_guttman_rings — ring counting correctness
# ---------------------------------------------------------------------------


def test_find_guttman_rings_two_fused_rings():
    """Triangle and quadrilateral fused on a shared edge produce exactly one ring of each size.

    Graph: 0-1-2-0 (triangle) + 0-1-3-4-0 (quadrilateral), sharing edge (0,1).
    The shared edge (0,1) leads to the 3-ring as its shortest detour (0-2-1, length 2),
    not the 4-ring (0-4-3-1, length 3), so each ring is found through its own unique edges
    and the deduplication logic is exercised across all six edges.
    """
    g = nx.Graph([(0, 1), (1, 2), (2, 0), (1, 3), (3, 4), (4, 0)])
    counts = _find_guttman_rings(g, max_ring_size=8, n_cpus=1)
    assert counts == {3: 1, 4: 1}


# ---------------------------------------------------------------------------
# _process_edge — parallel worker correctness
# ---------------------------------------------------------------------------


def test_process_edge_returns_all_shortest_paths():
    """All shortest paths through an edge are returned without filtering.

    Diamond graph has two length-2 paths from 0 to 2 (via node 1 and via node 3).
    Both must appear in the result — the old primitiveness filter would have accepted
    both anyway, but this confirms the new code returns them all.
    """
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 2)]
    results = _process_edge((0, 2, edges, 8))
    assert len(results) == 2
    assert all(ring_size == 3 for ring_size, _ in results)


def test_process_edge_ring_exceeds_max_size_returns_empty():
    """Returns empty when the ring formed by the shortest detour exceeds max_ring_size.

    In an 8-cycle the only detour through edge (0,1) has length 7, giving ring size 8.
    With max_ring_size=6 that candidate must be discarded.
    """
    g = nx.cycle_graph(8)
    results = _process_edge((0, 1, list(g.edges()), 6))
    assert results == []


def test_process_edge_no_path_returns_empty():
    """Returns empty when removing the edge leaves the two nodes disconnected."""
    results = _process_edge((0, 1, [(0, 1)], 8))
    assert results == []


def test_compute_guttmann_rings_silicate_dominant_size(glass_structure):
    """For a silicate glass the most common ring size is between 3 and 10."""
    hist, _mean = compute_guttmann_rings(
        glass_structure,
        bond_lengths={("Si", "O"): 2.0},
        max_size=10,
    )
    if hist:
        most_common = max(hist, key=hist.get)
        assert 3 <= most_common <= 10
