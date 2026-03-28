"""Tests for the remaining pure functions in amorphouspy.structure.

Covers: check_neutral_oxide, extract_stoichiometry, get_glass_density_from_model,
get_box_from_density, create_random_atoms, get_ase_structure, get_structure_dict.
"""

from collections import Counter

import numpy as np
import pytest
from ase.atoms import Atoms as AseAtoms

import amorphouspy.structure as ps

# ---------------------------------------------------------------------------
# check_neutral_oxide
# ---------------------------------------------------------------------------


def test_check_neutral_oxide_sio2_does_not_raise():
    """SiO2 is charge-neutral and must not raise."""
    ps.check_neutral_oxide("SiO2")


def test_check_neutral_oxide_na2o_does_not_raise():
    """Na2O is charge-neutral and must not raise."""
    ps.check_neutral_oxide("Na2O")


def test_check_neutral_oxide_al2o3_does_not_raise():
    """Al2O3 is charge-neutral and must not raise."""
    ps.check_neutral_oxide("Al2O3")


def test_check_neutral_oxide_invalid_element_raises():
    """A formula containing a non-existent element raises ValueError."""
    with pytest.raises(ValueError, match=r"not a valid Element|Invalid oxide formula"):
        ps.check_neutral_oxide("ZzO2")


def test_check_neutral_oxide_non_neutral_raises():
    """A stoichiometry that cannot be charge-neutral raises ValueError."""
    with pytest.raises(ValueError, match=r"not charge neutral|Cannot determine"):
        ps.check_neutral_oxide("SiO3")  # Si4+ + 3*O2- = -2 net charge


# ---------------------------------------------------------------------------
# extract_stoichiometry
# ---------------------------------------------------------------------------


def test_extract_stoichiometry_returns_nested_dict():
    """Returns oxide → element-count dict for each component."""
    result = ps.extract_stoichiometry({"SiO2": 0.8, "Na2O": 0.2})
    assert result["SiO2"] == {"Si": 1, "O": 2}
    assert result["Na2O"] == {"Na": 2, "O": 1}


def test_extract_stoichiometry_percent_input():
    """Percentage inputs (summing to 100) are accepted and normalised internally."""
    result = ps.extract_stoichiometry({"SiO2": 80.0, "Na2O": 20.0})
    assert result["SiO2"] == {"Si": 1, "O": 2}


def test_extract_stoichiometry_all_oxides_present():
    """Every input oxide appears as a key in the output."""
    comp = {"SiO2": 0.6, "Na2O": 0.2, "Al2O3": 0.2}
    result = ps.extract_stoichiometry(comp)
    assert set(result.keys()) == set(comp.keys())


# ---------------------------------------------------------------------------
# get_glass_density_from_model
# ---------------------------------------------------------------------------


def test_get_glass_density_sio2_near_baseline():
    """Pure SiO2 density should be close to the model intercept (~2.12 g/cm³)."""
    density = ps.get_glass_density_from_model({"SiO2": 1.0})
    assert pytest.approx(density, abs=0.02) == 2.12


def test_get_glass_density_returns_positive():
    """Density for any valid composition is positive."""
    density = ps.get_glass_density_from_model({"SiO2": 0.8, "Na2O": 0.2})
    assert density > 0


def test_get_glass_density_na2o_increases_density():
    """Adding Na2O to SiO2 increases density relative to pure SiO2."""
    d_pure = ps.get_glass_density_from_model({"SiO2": 1.0})
    d_mixed = ps.get_glass_density_from_model({"SiO2": 0.8, "Na2O": 0.2})
    assert d_mixed > d_pure


def test_get_glass_density_unsupported_component_raises():
    """A component absent from the density model's coefficients raises ValueError."""
    with pytest.raises(ValueError, match="not in density model"):
        ps.get_glass_density_from_model({"SiO2": 0.8, "GeO2": 0.2})


# ---------------------------------------------------------------------------
# get_box_from_density
# ---------------------------------------------------------------------------


def test_get_box_from_density_positive_n_molecules():
    """Box length is positive when using n_molecules mode."""
    length = ps.get_box_from_density(
        {"SiO2": 0.8, "Na2O": 0.2},
        n_molecules=100,
        target_atoms=None,
    )
    assert length > 0


def test_get_box_from_density_positive_target_atoms():
    """Box length is positive when using target_atoms mode with explicit density."""
    length = ps.get_box_from_density(
        {"SiO2": 1.0},
        n_molecules=None,
        target_atoms=30,
        density=2.2,
    )
    assert length > 0


def test_get_box_from_density_larger_system_larger_box():
    """A larger target_atoms yields a proportionally larger box."""
    comp = {"SiO2": 1.0}
    l_small = ps.get_box_from_density(comp, n_molecules=None, target_atoms=30, density=2.2)
    l_large = ps.get_box_from_density(comp, n_molecules=None, target_atoms=300, density=2.2)
    assert l_large > l_small


# ---------------------------------------------------------------------------
# create_random_atoms
# ---------------------------------------------------------------------------


def test_create_random_atoms_return_types():
    """Return value is a (list, dict) tuple."""
    atoms_list, counts = ps.create_random_atoms({"SiO2": 1.0}, target_atoms=9, box_length=15.0)
    assert isinstance(atoms_list, list)
    assert isinstance(counts, dict)


def test_create_random_atoms_element_counts_consistent():
    """Element counts in the list match the returned counts dict."""
    atoms_list, counts = ps.create_random_atoms({"SiO2": 1.0}, target_atoms=9, box_length=15.0)
    list_counts = Counter(a["element"] for a in atoms_list)
    assert list_counts["Si"] == counts["Si"]
    assert list_counts["O"] == counts["O"]


def test_create_random_atoms_min_distance_respected():
    """All atom pairs satisfy the minimum-distance constraint."""
    atoms_list, _ = ps.create_random_atoms({"SiO2": 1.0}, target_atoms=9, box_length=15.0, min_distance=1.5)
    positions = np.array([a["position"] for a in atoms_list])
    box = 15.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = ps.minimum_image_distance(positions[i], positions[j], box)
            assert dist >= 1.5 - 1e-9, f"atoms {i} and {j} too close: {dist:.4f} Å"


def test_create_random_atoms_n_molecules_mode():
    """n_molecules mode places the correct number of formula units."""
    _atoms_list, counts = ps.create_random_atoms({"SiO2": 1.0}, n_molecules=3, box_length=15.0)
    assert counts.get("Si", 0) == 3
    assert counts.get("O", 0) == 6


# ---------------------------------------------------------------------------
# get_ase_structure
# ---------------------------------------------------------------------------


def test_get_ase_structure_returns_atoms_object():
    """Result is an ASE Atoms object."""
    atoms_list, _ = ps.create_random_atoms({"SiO2": 1.0}, target_atoms=9, box_length=15.0)
    result = ps.get_ase_structure({"atoms": atoms_list, "box": 15.0})
    assert isinstance(result, AseAtoms)


def test_get_ase_structure_correct_atom_count():
    """Atom count in result equals the number of atoms in the input dict."""
    atoms_list, _ = ps.create_random_atoms({"SiO2": 1.0}, target_atoms=9, box_length=15.0)
    result = ps.get_ase_structure({"atoms": atoms_list, "box": 15.0})
    assert len(result) == len(atoms_list)


def test_get_ase_structure_replicate_doubles_atoms():
    """Replicating (2, 1, 1) doubles the total atom count."""
    atoms_list, _ = ps.create_random_atoms({"SiO2": 1.0}, target_atoms=9, box_length=15.0)
    atoms_dict = {"atoms": atoms_list, "box": 15.0}
    result = ps.get_ase_structure(atoms_dict, replicate=(2, 1, 1))
    assert len(result) == 2 * len(atoms_list)


# ---------------------------------------------------------------------------
# get_structure_dict
# ---------------------------------------------------------------------------


def test_get_structure_dict_required_keys():
    """All required keys are present in the result."""
    result = ps.get_structure_dict({"SiO2": 1.0}, target_atoms=9)
    for key in ("atoms", "box", "formula_units", "total_atoms", "element_counts", "mol_fraction"):
        assert key in result, f"Missing key: {key}"


def test_get_structure_dict_total_atoms_close_to_target():
    """total_atoms is within a small tolerance of the target."""
    result = ps.get_structure_dict({"SiO2": 1.0}, target_atoms=30)
    assert abs(result["total_atoms"] - 30) <= 5


def test_get_structure_dict_box_positive():
    """Box length is a positive number."""
    result = ps.get_structure_dict({"SiO2": 1.0}, target_atoms=9)
    assert result["box"] > 0


def test_get_structure_dict_n_molecules_mode():
    """n_molecules mode also produces a valid structure dict."""
    result = ps.get_structure_dict({"SiO2": 1.0}, n_molecules=3)
    assert result["total_atoms"] == 9  # 3 * (1 Si + 2 O) = 9
    assert result["box"] > 0
