"""Tests for amorphouspy.properties.structural.rings.

Covers generate_bond_length_dict (pure), the bipartite T-O ring search, and
compute_guttmann_rings (integration, requires the glass dump files).
Atom type mapping for 20Na2O-80SiO2: O=type1, Si=type2, Na=type3.
"""

import numpy as np
import pytest
from amorphouspy.properties.structural.averaging import average_over_frames
from amorphouspy.properties.structural.rings import (
    _closures_through,
    _new_scratch,
    compute_guttmann_rings,
    generate_bond_length_dict,
)
from ase import Atoms
from ase.build import bulk
from ase.io import read

from . import DATA_DIR

SI_O_BONDS = {("Si", "O"): 2.0}

# ---------------------------------------------------------------------------
# Fixtures and structure builders
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


@pytest.fixture(scope="module")
def borosilicate_structure():
    """Load the 30080-atom 20Na2O-10B2O3-70SiO2 dump.

    Type mapping per the simulation setup: O=type1(8), Si=type2(14), B=type3(5),
    Na=type4(11).
    """
    atoms = read(DATA_DIR / "20Na2O-10B2O3-70SiO2.dump", format="lammps-dump-text")
    type_id = atoms.get_atomic_numbers().copy()
    to_z = np.array([0, 8, 14, 5, 11], dtype=int)
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


def isolated_ring(n_formers, cell=40.0):
    """Build one isolated n-membered ring: n formers and n oxygens on a circle.

    Alternating T and O sit on a circle whose radius is chosen to give 1.6 A
    T-O bonds. The next-nearest T-O separation is far beyond any sane cutoff,
    so the connectivity is unambiguous.
    """
    radius = 0.8 / np.sin(np.pi / (2 * n_formers))
    positions, numbers = [], []
    for k in range(2 * n_formers):
        angle = np.pi * k / n_formers
        positions.append([cell / 2 + radius * np.cos(angle), cell / 2 + radius * np.sin(angle), cell / 2])
        numbers.append(14 if k % 2 == 0 else 8)
    return Atoms(numbers=numbers, positions=positions, cell=[cell] * 3, pbc=True)


def beta_cristobalite(a=7.16, repeat=2):
    """Build ideal beta-cristobalite: Si on the diamond net, O at every bond midpoint.

    Constructed from first principles rather than tabulated Wyckoff positions,
    so every property is analytically checkable: Si-O = a*sqrt(3)/8, Si is
    4-coordinate, O is 2-coordinate, and the stoichiometry is exactly SiO2.
    """
    silicon = bulk("Si", "diamond", a=a, cubic=True).repeat(repeat)
    positions = silicon.get_positions()
    cell = np.array(silicon.cell)
    bond_length = a * np.sqrt(3) / 4
    oxygens = []
    for i in range(len(silicon)):
        for j in range(i + 1, len(silicon)):
            fractional = np.linalg.solve(cell.T, positions[j] - positions[i])
            vector = cell.T @ (fractional - np.round(fractional))
            if abs(np.linalg.norm(vector) - bond_length) < 1e-6:
                oxygens.append(positions[i] + vector / 2)
    return silicon + Atoms("O" * len(oxygens), positions=oxygens, cell=silicon.cell, pbc=True)


def sizes(histogram):
    """Return the histogram with integer counts, for exact comparison."""
    return {size: int(count) for size, count in histogram.items()}


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


def test_generate_bond_length_dict_default_marks_pairs_unbonded(si_o_atoms):
    """The default cutoff is 0.0, which get_neighbors reads as "this pair never bonds".

    A negative default would be squared into a positive cutoff downstream, so the
    value must stay non-positive and 0.0 is the physically meaningful choice: a zero
    bonding radius.
    """
    result = generate_bond_length_dict(si_o_atoms, specific_cutoffs={("Si", "O"): 1.8})
    assert result[("O", "Si")] == 1.8
    assert result[("O", "O")] == 0.0
    assert result[("Si", "Si")] == 0.0


def test_rings_with_default_cutoff_excludes_unlisted_pairs(glass_structure):
    """Rings run end to end with the default, which sends 0.0 cutoffs through get_neighbors.

    Regression for the structure_characterization failure: the sentinel for "never
    bonded" reached the per-pair cutoff parser and was rejected as a non-positive
    distance.
    """
    bond_lengths = generate_bond_length_dict(glass_structure, specific_cutoffs={("Si", "O"): 2.0})
    assert bond_lengths[("O", "Na")] == 0.0  # modifier pairs excluded by the default

    histogram, mean_size = compute_guttmann_rings(glass_structure, bond_lengths=bond_lengths, max_size=10)
    explicit, explicit_mean = compute_guttmann_rings(glass_structure, bond_lengths=SI_O_BONDS, max_size=10)
    # Naming only the Si-O pair must match passing that pair alone.
    assert histogram == explicit
    assert mean_size == pytest.approx(explicit_mean)


# ---------------------------------------------------------------------------
# Ring search on hand-built topologies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_formers", [3, 4, 6])
def test_isolated_ring_is_found_at_its_own_size(n_formers):
    """A single closed T-O-T loop of n formers is reported exactly once, as size n."""
    histogram, mean_size = compute_guttmann_rings(isolated_ring(n_formers), bond_lengths=SI_O_BONDS, max_size=10)
    assert sizes(histogram) == {n_formers: 1}
    assert mean_size == pytest.approx(float(n_formers))


def test_ring_larger_than_max_size_is_not_searched():
    """A 6-ring is invisible when max_size caps the search below it.

    max_size bounds the breadth-first sweep itself, so the closure is never
    reached rather than being found and then discarded.
    """
    histogram, _ = compute_guttmann_rings(isolated_ring(6), bond_lengths=SI_O_BONDS, max_size=4)
    assert histogram == {}


def test_disconnected_pair_has_no_rings():
    """A lone T-O pair cannot close, so no ring is reported."""
    atoms = Atoms(numbers=[14, 8], positions=[[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]], cell=[40.0] * 3, pbc=True)
    histogram, mean_size = compute_guttmann_rings(atoms, bond_lengths=SI_O_BONDS, max_size=8)
    assert histogram == {}
    assert mean_size == 0.0


def test_all_shortest_closures_through_a_bond_are_kept():
    """Two formers bridged by three oxygens give three distinct 2-rings.

    Every bond has two equally short closures, through each of the other two
    oxygens. All of them are genuine rings and none may be dropped, but the
    three rings must be deduplicated down from the six (bond, closure) pairs
    that find them.
    """
    atoms = Atoms(
        numbers=[14, 14, 8, 8, 8],
        positions=[
            [-1.2, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.05, 0.0],
            [0.0, -0.525, 0.909],
            [0.0, -0.525, -0.909],
        ],
        cell=[40.0] * 3,
        pbc=True,
    )
    histogram, _ = compute_guttmann_rings(atoms, bond_lengths=SI_O_BONDS, max_size=8)
    assert sizes(histogram) == {2: 3}


# ---------------------------------------------------------------------------
# Regression tests for the three defects of the old T-T implementation
# ---------------------------------------------------------------------------


def test_oxygen_tricluster_is_not_a_ring():
    """Three formers sharing one oxygen form no ring.

    The old T-T graph drew an edge between every pair of formers on a shared
    oxygen, turning a tricluster into a triangle and reporting a phantom
    3-ring ({3: 1.0}). On the bipartite network the loop would have to pass
    through that single oxygen three times, so it is correctly absent.
    """
    atoms = Atoms(
        numbers=[8, 14, 14, 14],
        positions=[[20.0, 20.0, 20.0], [21.6, 20.0, 20.0], [19.2, 21.386, 20.0], [19.2, 18.614, 20.0]],
        cell=[40.0] * 3,
        pbc=True,
    )
    histogram, _ = compute_guttmann_rings(atoms, bond_lengths=SI_O_BONDS, max_size=10)
    assert histogram == {}


def test_edge_sharing_polyhedra_give_a_two_ring():
    """Two formers bridged by two distinct oxygens are a 2-membered ring.

    The old T-T graph collapsed both bridges onto one edge, so this ring was
    unreachable at any max_size (it returned {}).
    """
    atoms = Atoms(
        numbers=[14, 14, 8, 8],
        positions=[[18.8, 20.0, 20.0], [21.2, 20.0, 20.0], [20.0, 21.05, 20.0], [20.0, 18.95, 20.0]],
        cell=[40.0] * 3,
        pbc=True,
    )
    histogram, mean_size = compute_guttmann_rings(atoms, bond_lengths=SI_O_BONDS, max_size=10)
    assert sizes(histogram) == {2: 1}
    assert mean_size == pytest.approx(2.0)


def test_cycle_closing_only_through_pbc_is_rejected():
    """A chain that meets its own periodic image is a helix, not a ring.

    Eight atoms spaced 1.6 A along x exactly fill a 12.8 A cell, so the last
    oxygen bonds to the first silicon through the boundary. The old code had
    no closure check and reported {4: 1.0}; summing the minimum-image bond
    vectors around the loop gives a full lattice vector, not zero.
    """
    atoms = Atoms(
        numbers=[14, 8] * 4,
        positions=[[1.6 * k, 15.0, 15.0] for k in range(8)],
        cell=[12.8, 30.0, 30.0],
        pbc=True,
    )
    histogram, _ = compute_guttmann_rings(atoms, bond_lengths=SI_O_BONDS, max_size=10)
    assert histogram == {}


# ---------------------------------------------------------------------------
# Deepening past a non-physical shortest closure
# ---------------------------------------------------------------------------


def _theta_network():
    """Return (adjacency, bond_vectors) for a bond with a short and a long closure.

    Nodes 0 and 1 are joined directly, by a 3-edge path 0-3-2-1, and by a
    5-edge path 0-5-4-7-6-1. Bond vectors are exact coordinate differences, so
    every cycle closes to zero until one is deliberately broken.
    """
    positions = {
        0: (0.0, 0.0, 0.0),
        1: (1.0, 0.0, 0.0),
        2: (1.0, 1.0, 0.0),
        3: (0.0, 1.0, 0.0),
        4: (-1.0, -1.0, 0.0),
        5: (-1.0, 0.0, 0.0),
        6: (1.0, -1.0, 0.0),
        7: (0.0, -1.0, 0.0),
    }
    adjacency = [(1, 3, 5), (0, 2, 6), (1, 3), (0, 2), (5, 7), (0, 4), (1, 7), (4, 6)]
    bond_vectors = {
        (start, end): tuple(positions[end][axis] - positions[start][axis] for axis in range(3))
        for start, neighbors in enumerate(adjacency)
        for end in neighbors
    }
    return adjacency, bond_vectors


def test_closure_enumeration_cap_is_reported_not_hidden(monkeypatch):
    """Capping the closures for a bond is counted and surfaced as a warning.

    Two formers bridged by three oxygens give every bond two equally short
    closures. Lowering the cap to one forces truncation, which must show up as
    a RuntimeWarning rather than a silently short histogram.
    """
    atoms = Atoms(
        numbers=[14, 14, 8, 8, 8],
        positions=[
            [-1.2, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 1.05, 0.0],
            [0.0, -0.525, 0.909],
            [0.0, -0.525, -0.909],
        ],
        cell=[40.0] * 3,
        pbc=True,
    )
    monkeypatch.setattr("amorphouspy.properties.structural.rings._MAX_CLOSURES_PER_BOND", 1)

    with pytest.warns(RuntimeWarning, match="capped the closure enumeration"):
        histogram, _ = compute_guttmann_rings(atoms, bond_lengths=SI_O_BONDS, max_size=8)

    # Truncated, so fewer than the three rings the uncapped search finds.
    assert sum(histogram.values()) < 3


def _search_theta(adjacency, bond_vectors):
    """Run the per-bond search on bond (0, 1) with a path bound of 5 edges.

    Five edges is what a 3-former ring allows (2 * 3 - 1), matching the longest
    closure the theta network contains.
    """
    return _closures_through(0, 1, adjacency, bond_vectors, 5, _new_scratch(len(adjacency)))


def test_shortest_closure_is_used_when_it_closes():
    """With every cycle physical, the bond takes its shortest closure."""
    cycles, exhausted, truncated = _search_theta(*_theta_network())
    assert cycles == [(0, 3, 2, 1)]
    assert not exhausted
    assert not truncated


def test_deepening_finds_the_shortest_closure_that_actually_closes():
    """When the shortest closure does not close in real space, the search deepens.

    The 3-edge closure is given a 10 A offset on one bond, exactly as a cycle
    wrapping the cell would look. The 5-edge closure still sums to zero, so it
    becomes the ring for this bond.
    """
    adjacency, bond_vectors = _theta_network()
    bond_vectors[(0, 3)] = (bond_vectors[(0, 3)][0] - 10.0, 0.0, 0.0)
    bond_vectors[(3, 0)] = (bond_vectors[(3, 0)][0] + 10.0, 0.0, 0.0)

    cycles, exhausted, _truncated = _search_theta(adjacency, bond_vectors)
    assert cycles == [(0, 5, 4, 7, 6, 1)]
    assert not exhausted


# ---------------------------------------------------------------------------
# Limiting case: an ideal crystal with an analytically known ring spectrum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("repeat", [1, 2])
def test_beta_cristobalite_is_entirely_six_rings(repeat):
    """Ideal beta-cristobalite has exactly two 6-rings per silicon and nothing else.

    Its Si sublattice is the diamond net, in which every vertex lies on twelve
    6-rings; each ring covers six vertices, so the net carries 12/6 = 2 rings
    per vertex and no ring of any other size. Both the sizes and the absolute
    count are therefore fixed by topology, independent of this implementation.
    """
    atoms = beta_cristobalite(repeat=repeat)
    n_silicon = int((atoms.get_atomic_numbers() == 14).sum())

    histogram, mean_size = compute_guttmann_rings(atoms, bond_lengths=SI_O_BONDS, max_size=12)

    assert sizes(histogram) == {6: 2 * n_silicon}
    assert mean_size == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# compute_guttmann_rings on real glass structures
# ---------------------------------------------------------------------------


def test_compute_guttmann_rings_returns_tuple(glass_structure):
    """Function returns a (dict, float) tuple."""
    hist, mean = compute_guttmann_rings(glass_structure, bond_lengths=SI_O_BONDS, max_size=6)
    assert isinstance(hist, dict)
    assert isinstance(mean, float)


def test_compute_guttmann_rings_ring_sizes_positive(glass_structure):
    """All ring sizes in the histogram are positive integers."""
    hist, _ = compute_guttmann_rings(glass_structure, bond_lengths=SI_O_BONDS, max_size=6)
    assert all(isinstance(k, int) and k > 0 for k in hist)


def test_compute_guttmann_rings_counts_positive(glass_structure):
    """All ring counts are positive integers."""
    hist, _ = compute_guttmann_rings(glass_structure, bond_lengths=SI_O_BONDS, max_size=6)
    assert all(isinstance(v, (int, float)) and v > 0 for v in hist.values())


def test_compute_guttmann_rings_mean_nonnegative(glass_structure):
    """Mean ring size is non-negative."""
    _, mean = compute_guttmann_rings(glass_structure, bond_lengths=SI_O_BONDS, max_size=6)
    assert mean >= 0.0


def test_compute_guttmann_rings_silicate_dominant_size(glass_structure):
    """For a silicate glass the most common ring size is between 3 and 10."""
    hist, _mean = compute_guttmann_rings(glass_structure, bond_lengths=SI_O_BONDS, max_size=10)
    if hist:
        most_common = max(hist, key=hist.get)
        assert 3 <= most_common <= 10


def test_borosilicate_counts_match_independent_defect_census(borosilicate_structure):
    """Ring counts on the 30080-atom borosilicate match a direct count of the defects.

    Counting the neighbour lists directly, the structure holds 87 oxygens bonded
    to three or more formers and 32 former pairs bridged by two oxygens. The old
    T-T implementation reported 396 3-rings and no 2-rings; each tricluster
    contributed exactly one phantom 3-ring (396 - 87 = 309) and every
    edge-sharing pair was invisible.
    """
    hist, _ = compute_guttmann_rings(
        borosilicate_structure,
        bond_lengths={("Si", "O"): 2.0, ("B", "O"): 1.9},
        max_size=12,
    )
    assert hist[2] == 32
    assert hist[3] == 309


def test_parallel_search_matches_sequential(monkeypatch, glass_structure):
    """Worker processes reproduce the sequential histogram exactly.

    The threshold that normally keeps small networks sequential is lowered so
    the process pool actually runs on a structure small enough to test quickly.
    """
    sequential, _ = compute_guttmann_rings(glass_structure, bond_lengths=SI_O_BONDS, max_size=10)

    monkeypatch.setattr("amorphouspy.properties.structural.rings._PARALLEL_BOND_THRESHOLD", 0)
    parallel, _ = compute_guttmann_rings(glass_structure, bond_lengths=SI_O_BONDS, max_size=10, n_cpus=2)

    assert parallel == sequential


# ---------------------------------------------------------------------------
# average_over_frames — compute_guttmann_rings
# ---------------------------------------------------------------------------


def test_average_over_frames_rings_identical_frames(glass_structure):
    """Three identical frames: mean equals single-frame, SEM ≈ 0."""
    hist_s, mean_s = compute_guttmann_rings(glass_structure, bond_lengths=SI_O_BONDS, max_size=6)
    (hist_mean, mean_size_mean), (hist_sem, mean_size_sem) = average_over_frames(
        compute_guttmann_rings,
        [glass_structure, glass_structure, glass_structure],
        bond_lengths=SI_O_BONDS,
        max_size=6,
    )
    assert isinstance(hist_mean, dict)
    assert isinstance(mean_size_mean, float)
    assert mean_size_mean == pytest.approx(mean_s, abs=1e-10)
    assert mean_size_sem == pytest.approx(0.0, abs=1e-10)
    for size, count in hist_s.items():
        assert hist_mean[size] == pytest.approx(count, abs=1e-10)
        assert hist_sem[size] == pytest.approx(0.0, abs=1e-10)


def test_average_over_frames_empty_list_raises():
    """average_over_frames with empty list raises ValueError."""
    with pytest.raises(ValueError, match="requires a non-empty list"):
        average_over_frames(compute_guttmann_rings, [], bond_lengths=SI_O_BONDS)


def test_compute_guttmann_rings_list_uses_first_frame(glass_structure):
    """Passing a list without average_over_frames uses the first frame."""
    hist_s, mean_s = compute_guttmann_rings(glass_structure, bond_lengths=SI_O_BONDS, max_size=6)
    hist_a, mean_a = compute_guttmann_rings([glass_structure, glass_structure], bond_lengths=SI_O_BONDS, max_size=6)
    assert mean_a == pytest.approx(mean_s, abs=1e-10)
    for size in hist_s:
        assert hist_a[size] == pytest.approx(hist_s[size], abs=1e-10)
