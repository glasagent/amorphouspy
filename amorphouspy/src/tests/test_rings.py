"""Tests for amorphouspy.analysis.rings.

Covers generate_bond_length_dict (pure) and compute_guttmann_rings (integration,
requires the 20Na2O-80SiO2 dump file).  Atom type mapping: O=type1, Si=type2, Na=type3.
"""

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from amorphouspy.analysis.rings import compute_guttmann_rings, generate_bond_length_dict

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
    to_z = np.array([0, 8, 14, 11], dtype=int)  # O=1, Si=2, Na=3
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


def test_generate_bond_length_dict_returns_dict(si_o_atoms):
    """Return type is a dict with tuple keys."""
    result = generate_bond_length_dict(si_o_atoms)
    assert isinstance(result, dict)
    assert all(isinstance(k, tuple) and len(k) == 2 for k in result)


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
