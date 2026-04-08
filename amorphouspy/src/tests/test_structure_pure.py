"""Tests for pure helper functions in amorphouspy.structure."""

import amorphouspy.structure as ps
import amorphouspy.structure.planner as ps_planner
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# parse_formula
# ---------------------------------------------------------------------------


def test_parse_formula_simple():
    """SiO2 yields one Si and two O."""
    assert ps.parse_formula("SiO2") == {"Si": 1, "O": 2}


def test_parse_formula_multidigit():
    """Al2O3 yields two Al and three O."""
    assert ps.parse_formula("Al2O3") == {"Al": 2, "O": 3}


def test_parse_formula_no_count():
    """Single-atom formula without a count defaults to 1."""
    assert ps.parse_formula("O") == {"O": 1}


def test_parse_formula_multi_element():
    """Na2O yields two Na and one O."""
    assert ps.parse_formula("Na2O") == {"Na": 2, "O": 1}


# ---------------------------------------------------------------------------
# formula_mass_g_per_mol
# ---------------------------------------------------------------------------


def test_formula_mass_sio2():
    """Molar mass of SiO2 is approximately 60.09 g/mol."""
    mass = ps.formula_mass_g_per_mol("SiO2")
    assert pytest.approx(mass, abs=0.1) == 60.09


def test_formula_mass_al2o3():
    """Molar mass of Al2O3 is approximately 101.96 g/mol."""
    mass = ps.formula_mass_g_per_mol("Al2O3")
    assert pytest.approx(mass, abs=0.1) == 101.96


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def test_normalize_basic():
    """Values sum to 1 after normalization."""
    result = ps.normalize({"A": 1.0, "B": 3.0})
    assert pytest.approx(result["A"]) == 0.25
    assert pytest.approx(result["B"]) == 0.75


def test_normalize_already_normalized():
    """Already-normalized dict is returned unchanged."""
    d = {"X": 0.4, "Y": 0.6}
    result = ps.normalize(d)
    assert pytest.approx(sum(result.values())) == 1.0


def test_normalize_raises_on_zero_sum():
    """Raises ValueError when all values are zero."""
    with pytest.raises(ValueError, match="non-positive"):
        ps.normalize({"A": 0.0, "B": 0.0})


# ---------------------------------------------------------------------------
# weight_percent_to_mol_fraction
# ---------------------------------------------------------------------------


def test_weight_percent_to_mol_fraction_sum_to_one():
    """Molar fractions from weight percents sum to 1."""
    comp = {"SiO2": 60.0, "Na2O": 40.0}
    result = ps.weight_percent_to_mol_fraction(comp)
    assert pytest.approx(sum(result.values())) == 1.0


def test_weight_percent_to_mol_fraction_heavier_oxide_lower_mol():
    """Heavier oxide has a lower molar fraction than a lighter one at equal weight."""
    comp = {"SiO2": 50.0, "Na2O": 50.0}
    result = ps.weight_percent_to_mol_fraction(comp)
    # Na2O (M≈62) is heavier than SiO2 (M≈60), so mol fraction of Na2O < SiO2
    assert result["SiO2"] > result["Na2O"]


# ---------------------------------------------------------------------------
# get_composition
# ---------------------------------------------------------------------------


def test_get_composition_molar_normalizes():
    """Molar mode returns fractions summing to 1."""
    comp = {"SiO2": 0.75, "Na2O": 0.25}
    result = ps.get_composition(comp, mode="molar")
    assert pytest.approx(sum(result.values())) == 1.0
    assert pytest.approx(result["SiO2"]) == 0.75


def test_get_composition_weight_mode():
    """Weight mode converts and returns fractions summing to 1."""
    comp = {"SiO2": 60.0, "Na2O": 40.0}
    result = ps.get_composition(comp, mode="weight")
    assert pytest.approx(sum(result.values())) == 1.0


def test_get_composition_invalid_mode():
    """Unknown mode raises ValueError."""
    with pytest.raises(ValueError, match="Invalid mode"):
        ps.get_composition({"SiO2": 1.0}, mode="atomic")


# ---------------------------------------------------------------------------
# _atoms_per_fu_map
# ---------------------------------------------------------------------------


def test_atoms_per_fu_map():
    """SiO2 has 3 and Na2O has 3 atoms per formula unit."""
    result = ps_planner._atoms_per_fu_map({"SiO2": 0.7, "Na2O": 0.3})  # noqa: SLF001
    assert result["SiO2"] == 3
    assert result["Na2O"] == 3


def test_atoms_per_fu_map_al2o3():
    """Al2O3 has 5 atoms per formula unit."""
    result = ps_planner._atoms_per_fu_map({"Al2O3": 1.0})  # noqa: SLF001
    assert result["Al2O3"] == 5


# ---------------------------------------------------------------------------
# _integer_fu_from_total
# ---------------------------------------------------------------------------


def test_integer_fu_from_total_sums_to_target():
    """Allocated formula units sum exactly to Nfu_target."""
    mol_frac = {"SiO2": 0.75, "Na2O": 0.25}
    result = ps_planner._integer_fu_from_total(100, mol_frac)  # noqa: SLF001
    assert sum(result.values()) == 100


def test_integer_fu_from_total_proportional():
    """Allocation is proportional to molar fractions."""
    mol_frac = {"SiO2": 0.75, "Na2O": 0.25}
    result = ps_planner._integer_fu_from_total(100, mol_frac)  # noqa: SLF001
    assert result["SiO2"] == 75
    assert result["Na2O"] == 25


# ---------------------------------------------------------------------------
# allocate_formula_units_to_target_atoms
# ---------------------------------------------------------------------------


def test_allocate_formula_units_returns_close_to_target():
    """Total atoms in allocation is close to target."""
    mol_frac = {"SiO2": 0.75, "Na2O": 0.25}
    _ni, n_atoms = ps.allocate_formula_units_to_target_atoms(mol_frac, target_atoms=300)
    assert abs(n_atoms - 300) <= 5


def test_allocate_formula_units_returns_dict_and_int():
    """Return type is (dict, int)."""
    mol_frac = {"SiO2": 1.0}
    ni, n_atoms = ps.allocate_formula_units_to_target_atoms(mol_frac, target_atoms=30)
    assert isinstance(ni, dict)
    assert isinstance(n_atoms, int)


# ---------------------------------------------------------------------------
# element_counts_from_formula_units
# ---------------------------------------------------------------------------


def test_element_counts_from_formula_units_basic():
    """Two SiO2 formula units yield 2 Si and 4 O."""
    result = ps.element_counts_from_formula_units({"SiO2": 2})
    assert result == {"Si": 2, "O": 4}


def test_element_counts_from_formula_units_mixed():
    """Mixed oxides accumulate O correctly."""
    result = ps.element_counts_from_formula_units({"SiO2": 1, "Na2O": 1})
    assert result["Si"] == 1
    assert result["Na"] == 2
    assert result["O"] == 3


# ---------------------------------------------------------------------------
# plan_system
# ---------------------------------------------------------------------------


def test_plan_system_atoms_target():
    """plan_system with target_type='atoms' returns a total close to target."""
    plan = ps.plan_system({"SiO2": 0.75, "Na2O": 0.25}, target=300, target_type="atoms")
    assert abs(plan["total_atoms"] - 300) <= 5
    assert "mol_fraction" in plan
    assert "element_counts" in plan


def test_plan_system_molecules_target():
    """plan_system with target_type='molecules' returns a result."""
    plan = ps.plan_system({"SiO2": 0.75, "Na2O": 0.25}, target=100, target_type="molecules")
    assert plan["total_atoms"] > 0


def test_plan_system_invalid_target_type():
    """Unknown target_type raises ValueError."""
    with pytest.raises(ValueError, match="Invalid target_type"):
        ps.plan_system({"SiO2": 1.0}, target=100, target_type="particles")


# ---------------------------------------------------------------------------
# validate_target_mode
# ---------------------------------------------------------------------------


def test_validate_target_mode_raises_if_both_none():
    """Raises if both n_molecules and target_atoms are None."""
    with pytest.raises(ValueError, match="Either"):
        ps.validate_target_mode(None, None)


def test_validate_target_mode_raises_if_both_set():
    """Raises if both n_molecules and target_atoms are specified."""
    with pytest.raises(ValueError, match="Only one"):
        ps.validate_target_mode(10, 100)


def test_validate_target_mode_passes_with_one():
    """Does not raise if exactly one is specified."""
    ps.validate_target_mode(10, None)
    ps.validate_target_mode(None, 100)


# ---------------------------------------------------------------------------
# extract_composition
# ---------------------------------------------------------------------------


def test_extract_composition_fractions_unchanged():
    """Fractions already summing to 1 pass through unchanged."""
    result = ps.extract_composition({"SiO2": 0.75, "Na2O": 0.25})
    assert pytest.approx(result["SiO2"]) == 0.75


def test_extract_composition_percent_converted():
    """Values summing to 100 are divided by 100."""
    result = ps.extract_composition({"SiO2": 75.0, "Na2O": 25.0})
    assert pytest.approx(result["SiO2"]) == 0.75


def test_extract_composition_empty_raises():
    """Empty composition raises ValueError."""
    with pytest.raises(ValueError, match="Empty"):
        ps.extract_composition({})


def test_extract_composition_negative_raises():
    """Negative fraction raises ValueError."""
    with pytest.raises(ValueError, match="Negative"):
        ps.extract_composition({"SiO2": -0.1, "Na2O": 1.1})


def test_extract_composition_invalid_sum_raises():
    """Sum too far from 100 or 1.0 raises ValueError."""
    with pytest.raises(ValueError, match="Invalid composition sum"):
        ps.extract_composition({"SiO2": 50.0, "Na2O": 10.0})


def test_extract_composition_within_tolerance_rescales():
    """Sum within default tolerance of 100 is accepted and rescaled to 1.0."""
    result = ps.extract_composition({"SiO2": 74.95, "Na2O": 25.0})
    total = sum(result.values())
    assert pytest.approx(total) == 1.0
    assert pytest.approx(result["SiO2"]) == 74.95 / 99.95


def test_extract_composition_exceeds_tolerance_raises():
    """Sum more than default tolerance away from 100 raises ValueError."""
    with pytest.raises(ValueError, match="Invalid composition sum"):
        ps.extract_composition({"SiO2": 74.0, "Na2O": 25.0})


def test_extract_composition_custom_tolerance():
    """Custom tolerance of 0.03 accepts sums within 3% of 100."""
    result = ps.extract_composition({"SiO2": 74.0, "Na2O": 25.0}, tolerance=0.03)
    total = sum(result.values())
    assert pytest.approx(total) == 1.0
    assert pytest.approx(result["SiO2"]) == 74.0 / 99.0


def test_extract_composition_custom_tolerance_exceeded():
    """Sum exceeding custom tolerance raises ValueError."""
    with pytest.raises(ValueError, match="Total exceeds"):
        ps.extract_composition({"SiO2": 80.0, "Na2O": 25.0}, tolerance=0.03)


# ---------------------------------------------------------------------------
# minimum_image_distance
# ---------------------------------------------------------------------------


def test_minimum_image_distance_no_pbc():
    """Standard Euclidean distance when both points are well inside the box."""
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([4.0, 0.0, 0.0])
    assert pytest.approx(ps.minimum_image_distance(p1, p2, box_length=10.0)) == 3.0


def test_minimum_image_distance_with_pbc():
    """Periodic image is closer than the direct distance."""
    p1 = np.array([0.5, 0.0, 0.0])
    p2 = np.array([9.5, 0.0, 0.0])
    # Direct distance = 9.0, PBC distance = 1.0
    assert pytest.approx(ps.minimum_image_distance(p1, p2, box_length=10.0)) == 1.0


def test_minimum_image_distance_zero():
    """Distance between identical points is zero."""
    p = np.array([3.0, 3.0, 3.0])
    assert pytest.approx(ps.minimum_image_distance(p, p, box_length=10.0)) == 0.0
