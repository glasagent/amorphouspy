"""Tests for pure functions in amorphouspy.properties.vibrational."""

import itertools
from pathlib import Path

import numpy as np
import pytest
from amorphouspy.properties.structural.qn import compute_qn
from amorphouspy.properties.vibrational import classify_vibrational_groups, convert_frequency
from ase import Atoms
from ase.io import read

DATA_DIR = Path(__file__).parent / "data"
SI_O_BOND = 1.6

UNITS = ["THz", "cm-1", "meV", "rad/ps"]

# ---------------------------------------------------------------------------
# convert_frequency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("unit", "expected"), [("cm-1", 33.35641), ("meV", 4.135668), ("rad/ps", 6.283185)])
def test_convert_frequency_reference_values(unit, expected):
    """1 THz matches the reference value in each unit."""
    assert convert_frequency(1.0, "THz", unit) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(("from_unit", "to_unit"), list(itertools.product(UNITS, UNITS)))
def test_convert_frequency_round_trip(from_unit, to_unit):
    """Converting there and back returns the input for every unit pair."""
    values = np.array([-3.7, 0.0, 0.25, 12.5, 40.0])
    back = convert_frequency(convert_frequency(values, from_unit, to_unit), to_unit, from_unit)
    np.testing.assert_allclose(back, values, rtol=1e-12)
    scalar_back = convert_frequency(convert_frequency(12.5, from_unit, to_unit), to_unit, from_unit)
    assert scalar_back == pytest.approx(12.5, rel=1e-12)


def test_convert_frequency_array_keeps_shape_and_float64():
    """Array input returns a float64 array of the same shape."""
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = convert_frequency(values, "THz", "meV")
    assert result.shape == (2, 3)
    assert result.dtype == np.float64


def test_convert_frequency_integer_input():
    """Integer scalar returns float; integer array returns float64."""
    scalar = convert_frequency(2, "THz", "cm-1")
    assert type(scalar) is float
    array = convert_frequency(np.array([1, 2, 3]), "THz", "cm-1")
    assert array.dtype == np.float64


def test_convert_frequency_negative_sign_preserving():
    """Negative frequencies (imaginary modes) keep their sign."""
    assert convert_frequency(-1.0, "THz", "cm-1") == pytest.approx(-33.35641, rel=1e-6)
    result = convert_frequency(np.array([-2.0, 2.0]), "THz", "meV")
    assert result[0] == pytest.approx(-result[1])
    assert result[0] < 0


def test_convert_frequency_input_unchanged():
    """The input array is not modified in place."""
    values = np.array([-1.0, 1.0, 5.0])
    original = values.copy()
    convert_frequency(values, "THz", "rad/ps")
    np.testing.assert_array_equal(values, original)


def test_convert_frequency_same_unit_returns_copy():
    """from_unit == to_unit returns an equal array that is not the input object."""
    values = np.array([1.0, 2.0])
    result = convert_frequency(values, "meV", "meV")
    np.testing.assert_array_equal(result, values)
    assert result is not values
    assert not np.shares_memory(result, values)


@pytest.mark.parametrize(("from_unit", "to_unit"), [("Hz", "THz"), ("THz", "cm^-1")])
def test_convert_frequency_unknown_unit_raises(from_unit, to_unit):
    """An unknown unit raises ValueError naming the allowed units."""
    with pytest.raises(ValueError, match="allowed units"):
        convert_frequency(1.0, from_unit, to_unit)


# ---------------------------------------------------------------------------
# classify_vibrational_groups
# ---------------------------------------------------------------------------


def _si2o7_dimer() -> Atoms:
    """Two Si sharing one bridging O (position 2), each with three terminal O, in a large box."""
    si1 = np.array([10.0, 10.0, 10.0])
    si2 = si1 + np.array([2 * SI_O_BOND, 0.0, 0.0])
    bridging = si1 + np.array([SI_O_BOND, 0.0, 0.0])
    terminals_1 = [si1 + d for d in ([-SI_O_BOND, 0, 0], [0, SI_O_BOND, 0], [0, 0, SI_O_BOND])]
    terminals_2 = [si2 + d for d in ([SI_O_BOND, 0, 0], [0, SI_O_BOND, 0], [0, 0, SI_O_BOND])]
    positions = [si1, si2, bridging, *terminals_1, *terminals_2]
    return Atoms("Si2O7", positions=positions, cell=[30.0, 30.0, 30.0], pbc=True)


def _assert_partition(groups: dict[str, np.ndarray], expected: np.ndarray) -> None:
    """The groups are sorted int64 arrays that together hold every expected position exactly once."""
    for idx in groups.values():
        assert idx.dtype == np.int64
        assert len(idx) > 0
        np.testing.assert_array_equal(idx, np.sort(idx))
    combined = np.concatenate(list(groups.values())) if groups else np.array([], dtype=np.int64)
    np.testing.assert_array_equal(np.sort(combined), np.sort(expected))


def _assert_all_partitions(atoms: Atoms, groups: dict[str, dict[str, np.ndarray]], former_types: list[int]) -> None:
    numbers = atoms.get_atomic_numbers()
    _assert_partition(groups["element"], np.arange(len(atoms)))
    _assert_partition(groups["oxygen"], np.flatnonzero(numbers == 8))
    _assert_partition(groups["qn"], np.flatnonzero(np.isin(numbers, former_types)))


def test_classify_vibrational_groups_si2o7_dimer():
    """Isolated Si2O7: one BO, six NBO, both Si are Q1."""
    groups = classify_vibrational_groups(_si2o7_dimer(), cutoff=2.0, former_types=[14])
    assert set(groups["element"]) == {"Si", "O"}
    np.testing.assert_array_equal(groups["element"]["Si"], [0, 1])
    np.testing.assert_array_equal(groups["element"]["O"], np.arange(2, 9))
    assert set(groups["oxygen"]) == {"O_BO", "O_NBO"}
    np.testing.assert_array_equal(groups["oxygen"]["O_BO"], [2])
    np.testing.assert_array_equal(groups["oxygen"]["O_NBO"], [3, 4, 5, 6, 7, 8])
    assert set(groups["qn"]) == {"Si_Q1"}
    np.testing.assert_array_equal(groups["qn"]["Si_Q1"], [0, 1])
    _assert_all_partitions(_si2o7_dimer(), groups, [14])


def test_classify_vibrational_groups_sio2_glass_defaults_match_compute_qn():
    """Default cutoffs and formers partition the glass and reproduce compute_qn's totals."""
    atoms = read(DATA_DIR / "SiO2_glass_300_atoms.xyz")
    groups = classify_vibrational_groups(atoms)
    _assert_all_partitions(atoms, groups, [14])
    assert set(groups["element"]) == {"Si", "O"}

    # Same cutoff as the default derivation (first Si-O RDF minimum), checked for consistency below.
    explicit = classify_vibrational_groups(atoms, cutoff={(14, 8): 1.816}, former_types=[14])
    for key in ("oxygen", "qn"):
        assert explicit[key].keys() == groups[key].keys()
        for label in groups[key]:
            np.testing.assert_array_equal(explicit[key][label], groups[key][label])

    total_qn, _partial_qn = compute_qn(atoms, cutoff={(14, 8): 1.816}, former_types=[14], o_type=8)
    sizes = {n: len(groups["qn"].get(f"Si_Q{n}", [])) for n in total_qn}
    assert sizes == {n: int(count) for n, count in total_qn.items()}


def test_classify_vibrational_groups_shuffled_ids_give_positions():
    """A shuffled, non-contiguous "id" array still yields positions, not IDs."""
    atoms = _si2o7_dimer()
    reference = classify_vibrational_groups(atoms, cutoff=2.0, former_types=[14])
    rng = np.random.default_rng(42)
    atoms.arrays["id"] = rng.permutation(np.arange(1, len(atoms) + 1)) * 7 + 100
    groups = classify_vibrational_groups(atoms, cutoff=2.0, former_types=[14])
    for key in ("element", "oxygen", "qn"):
        assert groups[key].keys() == reference[key].keys()
        for label in reference[key]:
            np.testing.assert_array_equal(groups[key][label], reference[key][label])


def test_classify_vibrational_groups_duplicate_ids_raise():
    """Duplicate IDs cannot be mapped to positions."""
    atoms = _si2o7_dimer()
    atoms.arrays["id"] = np.ones(len(atoms), dtype=np.int64)
    with pytest.raises(ValueError, match="duplicate"):
        classify_vibrational_groups(atoms, cutoff=2.0, former_types=[14])


def test_classify_vibrational_groups_no_oxygen():
    """Without oxygen only the element grouping is filled."""
    atoms = Atoms("Si2Na", positions=[[1, 1, 1], [4, 4, 4], [7, 7, 7]], cell=[20, 20, 20], pbc=True)
    groups = classify_vibrational_groups(atoms)
    assert groups["oxygen"] == {}
    assert groups["qn"] == {}
    _assert_partition(groups["element"], np.arange(3))


def test_classify_vibrational_groups_no_formers():
    """With oxygen but no formers, qn is empty and every O is free."""
    atoms = Atoms("Na2O", positions=[[5, 5, 5], [9, 5, 5], [7, 5, 5]], cell=[20, 20, 20], pbc=True)
    groups = classify_vibrational_groups(atoms)
    assert groups["qn"] == {}
    assert set(groups["oxygen"]) == {"O_free"}
    np.testing.assert_array_equal(groups["oxygen"]["O_free"], [2])
    _assert_partition(groups["element"], np.arange(3))
