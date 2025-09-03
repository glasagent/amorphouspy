"""Testing structure generation."""

import re

import pytest

import pyiron_glass.structure as ps


def test_reproducibility() -> None:
    """Test that the random structure generation is reproducible when using a fixed seed."""
    seed = 42
    comp = "0.25CaO-0.25Al2O3-0.5SiO2"
    n_molecules = 10
    box_length = 10.0  # Angstrom
    min_distance = 1.6  # Angstrom
    max_attempts_per_atom = 100000  # Max attempts to place an atom

    kwargs = {
        "n_molecules": n_molecules,
        "box_length": box_length,
        "min_distance": min_distance,
        "max_attempts_per_atom": max_attempts_per_atom,
        "stoichiometry": ps.extract_stoichiometry(comp),
        "seed": seed,
    }

    atoms1 = ps.create_random_atoms(comp, **kwargs)
    atoms2 = ps.create_random_atoms(comp, **kwargs)

    assert atoms1 == atoms2, "Random structures should be identical with the same seed."

    # Test with a different seed
    seed = 12345
    kwargs["seed"] = seed
    atoms3 = ps.create_random_atoms(comp, **kwargs)

    assert atoms1 != atoms3, "Random structures should differ with different seeds."


def mol_to_weight(composition: str) -> dict[str, float]:
    """Convert a molar composition string into weight percent composition.

    Example:
        "0.25CaO-0.25Al2O3-0.50SiO2" → {"CaO": 21.92, "Al2O3": 30.89, "SiO2": 47.19}

    Args:
        composition (str): Molar fraction composition string.

    Returns:
        dict[str, float]: Weight percent composition normalized to 100.

    """
    # Atomic weights (g/mol)
    atomic_weights = {
        "O": 15.999,
        "Si": 28.085,
        "Al": 26.982,
        "Ca": 40.078,
    }

    # Parse composition
    components = composition.split("-")
    mole_fractions = []
    formulas = []

    for comp in components:
        match = re.match(r"([0-9.]+)?([A-Za-z0-9]+)", comp)
        if not match:
            error_msg = f"Invalid component: {comp}"
            raise ValueError(error_msg)
        frac_str, formula = match.groups()
        frac = float(frac_str) if frac_str else 1.0
        mole_fractions.append(frac)
        formulas.append(formula)

    # Compute molar mass of a formula
    def molar_mass(formula: str) -> float:
        mass = 0.0
        for elem, i in re.findall(r"([A-Z][a-z]*)(\d*)", formula):
            n = int(i) if i else 1
            if elem not in atomic_weights:
                error_msg = f"Unknown element: {elem}"
                raise KeyError(error_msg)
            mass += atomic_weights[elem] * n
        return mass

    # Compute weight fractions
    masses = [x * molar_mass(f) for x, f in zip(mole_fractions, formulas, strict=False)]
    total_mass = sum(masses)

    return {f: (100 * m / total_mass) for f, m in zip(formulas, masses, strict=False)}


# Constants for tests
EXPECTED_COUNTS = {"Ca": 25, "Al": 50, "Si": 50, "O": 200}


@pytest.mark.parametrize(
    ("mode", "n_molecules", "target_atoms"),
    [
        ("molar", 100, None),
        ("molar", None, 325),
        ("weight", 100, None),
        ("weight", None, 325),
    ],
)
def test_structure_atom_counts(mode: str, n_molecules: int | None, target_atoms: int | None) -> None:
    """Verify that atom counts are preserved for all input modes.

    Expected counts:
        Ca = 25
        Al = 50
        Si = 50
        O  = 200
    """
    composition = "0.25CaO-0.25Al2O3-0.50SiO2"
    box_length = 100  # any value, irrelevant for counting
    min_distance = 2  # any value, irrelevant for counting
    seed = 42
    max_attempts_per_atom = 10000  # any value, irrelevant for counting

    stoichiometry = ps.extract_stoichiometry(composition)

    if mode == "weight":
        comp = mol_to_weight(composition)
        composition = f"{comp['CaO']}CaO-{comp['Al2O3']}Al2O3-{comp['SiO2']}SiO2"

    atoms, atom_counts = ps.create_random_atoms(
        composition=composition,
        n_molecules=n_molecules,
        target_atoms=target_atoms,
        mode=mode,
        stoichiometry=stoichiometry,
        box_length=box_length,
        min_distance=min_distance,
        seed=seed,
        max_attempts_per_atom=max_attempts_per_atom,
    )

    # Verify counts directly from returned dictionary
    for elem, expected in EXPECTED_COUNTS.items():
        assert atom_counts[elem] == expected, f"{elem} atoms should be {expected}."
