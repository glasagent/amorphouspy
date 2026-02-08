"""Testing structure generation."""

import amorphouspy.structure as ps


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


# Constants for tests
EXPECTED_COUNTS = {"Ca": 25, "Al": 50, "Si": 50, "O": 200}


def test_structure_atom_counts_molar() -> None:
    """Verify that atom counts are preserved for molar input mode."""
    composition = "0.25CaO-0.25Al2O3-0.50SiO2"
    box_length = 100
    min_distance = 2
    seed = 42
    n_molecules = 100
    target_atoms = 325
    max_attempts_per_atom = 10000
    stoichiometry = ps.extract_stoichiometry(composition)

    # Test with n_molecules
    atoms, atom_counts = ps.create_random_atoms(
        composition=composition,
        n_molecules=n_molecules,
        target_atoms=None,
        mode="molar",
        stoichiometry=stoichiometry,
        box_length=box_length,
        min_distance=min_distance,
        seed=seed,
        max_attempts_per_atom=max_attempts_per_atom,
    )

    for elem, expected in EXPECTED_COUNTS.items():
        assert atom_counts[elem] == expected, f"{elem} atoms should be {expected} for {n_molecules} mode."

    # Test with target_atoms
    _atoms, atom_counts = ps.create_random_atoms(
        composition=composition,
        n_molecules=None,
        target_atoms=target_atoms,
        mode="molar",
        stoichiometry=stoichiometry,
        box_length=box_length,
        min_distance=min_distance,
        seed=seed,
        max_attempts_per_atom=max_attempts_per_atom,
    )

    for elem, expected in EXPECTED_COUNTS.items():
        assert atom_counts[elem] == expected, f"{elem} atoms should be {expected} for {target_atoms} mode."


def test_structure_atom_counts_weight() -> None:
    """Verify that atom counts are preserved for weight input mode."""
    weight_composition = "20.2CaO-36.6Al2O3-43.2SiO2"
    box_length = 100
    min_distance = 2
    seed = 42
    max_attempts_per_atom = 10000
    n_molecules = 100
    target_atoms = 325
    stoichiometry = ps.extract_stoichiometry(weight_composition)

    # Test with n_molecules
    atoms, atom_counts = ps.create_random_atoms(
        composition=weight_composition,
        n_molecules=n_molecules,
        target_atoms=None,
        mode="weight",
        stoichiometry=stoichiometry,
        box_length=box_length,
        min_distance=min_distance,
        seed=seed,
        max_attempts_per_atom=max_attempts_per_atom,
    )

    for elem, expected in EXPECTED_COUNTS.items():
        assert atom_counts[elem] == expected, f"{elem} atoms should be {expected} for {n_molecules} mode."

    # Test with target_atoms
    _atoms, atom_counts = ps.create_random_atoms(
        composition=weight_composition,
        n_molecules=None,
        target_atoms=target_atoms,
        mode="weight",
        stoichiometry=stoichiometry,
        box_length=box_length,
        min_distance=min_distance,
        seed=seed,
        max_attempts_per_atom=max_attempts_per_atom,
    )

    for elem, expected in EXPECTED_COUNTS.items():
        assert atom_counts[elem] == expected, f"{elem} atoms should be {expected} for {target_atoms} mode."
