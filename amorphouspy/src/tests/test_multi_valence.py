"""Tests for multi-valence oxide → BMP element label threading."""

from amorphouspy.fabrication.planner import element_counts_from_formula_units, plan_system


def test_mixed_iron_oxides_produce_both_labels() -> None:
    """FeO + Fe2O3 must yield both "Fe" and "Fe3" in the correct atom counts."""
    # 2 FU of FeO → 2 Fe + 2 O
    # 3 FU of Fe2O3 → 6 Fe3 + 9 O
    ni = {"FeO": 2, "Fe2O3": 3}
    counts = element_counts_from_formula_units(ni)

    assert counts["Fe"] == 2
    assert counts["Fe3"] == 6
    assert counts["O"] == 11
    assert "Fe" in counts
    assert "Fe3" in counts
    assert len([k for k in counts if k not in ("Fe", "Fe3", "O")]) == 0


def test_standard_oxides_unaffected() -> None:
    """SiO2 and Na2O should produce bare element symbols, not BMP-style labels."""
    ni = {"SiO2": 4, "Na2O": 2}
    counts = element_counts_from_formula_units(ni)

    assert counts["Si"] == 4
    assert counts["Na"] == 4
    assert counts["O"] == 10
    assert "Si2" not in counts
    assert "Na2" not in counts


def test_cerium_both_valences() -> None:
    """Ce2O3 + CeO2 should yield both "Ce3" and "Ce4" in the correct atom counts."""
    ni = {"Ce2O3": 2, "CeO2": 3}
    counts = element_counts_from_formula_units(ni)

    assert counts["Ce3"] == 4
    assert counts["Ce4"] == 3
    assert counts["O"] == 12  # 6 + 6


def test_manganese_valences() -> None:
    """MnO + Mn2O3 + MnO2 should yield "Mn" and "Mn3" in the correct atom counts."""
    # Mn (Mn²⁺, q=1.2) and Mn3 (Mn³⁺, q=1.8) have distinct parameters per Bertani et al. 2021.
    ni = {"MnO": 1, "Mn2O3": 2}
    counts = element_counts_from_formula_units(ni)

    assert counts["Mn"] == 1
    assert counts["Mn3"] == 4
    assert counts["O"] == 7  # 1 + 6


def test_vanadium_valences() -> None:
    """V2O4 + V2O5 should yield "V4" and "V5" in the correct atom counts."""
    ni = {"V2O4": 1, "V2O5": 1}
    counts = element_counts_from_formula_units(ni)

    assert counts["V4"] == 2
    assert counts["V5"] == 2
    assert counts["O"] == 9  # 4 + 5


def test_copper_valences() -> None:
    """Cu2O + CuO should yield "Cu" and "Cu2" in the correct atom counts."""
    ni = {"Cu2O": 1, "CuO": 2}
    counts = element_counts_from_formula_units(ni)

    assert counts["Cu"] == 2  # from Cu2O
    assert counts["Cu2"] == 2  # from CuO
    assert counts["O"] == 3


# ---------------------------------------------------------------------------
# plan_system integration
# ---------------------------------------------------------------------------


def test_plan_system_mixed_iron_element_counts() -> None:
    """plan_system with FeO + Fe2O3 returns element_counts with "Fe" and "Fe3"."""
    comp = {"FeO": 0.5, "Fe2O3": 0.5}
    plan = plan_system(comp, target=200, mode="molar", target_type="atoms")

    ec = plan["element_counts"]
    assert "Fe" in ec
    assert "Fe3" in ec
    assert ec["Fe"] > 0
    assert ec["Fe3"] > 0
    assert "O" in ec


def test_plan_system_standard_composition_unaffected() -> None:
    """SiO2 + Na2O composition produces bare element symbols in element_counts."""
    comp = {"SiO2": 0.8, "Na2O": 0.2}
    plan = plan_system(comp, target=200, mode="molar", target_type="atoms")

    ec = plan["element_counts"]
    assert "Si" in ec
    assert "Na" in ec
    assert "O" in ec
    assert "Si2" not in ec
    assert "Na2" not in ec


def test_plan_system_total_atoms_consistent() -> None:
    """Total atom count equals sum of all element_counts."""
    comp = {"FeO": 0.4, "Fe2O3": 0.3, "SiO2": 0.3}
    plan = plan_system(comp, target=300, mode="molar", target_type="atoms")

    ec = plan["element_counts"]
    assert sum(ec.values()) == plan["total_atoms"]
