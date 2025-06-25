"""Testing structure generation."""

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
