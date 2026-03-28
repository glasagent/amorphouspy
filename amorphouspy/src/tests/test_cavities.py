"""Tests for amorphouspy.analysis.cavities.

Covers compute_cavities (integration test using the 20Na2O-80SiO2 dump file).
Atom type mapping: O=type1, Si=type2, Na=type3.
"""

import numpy as np
import pytest
from ase.io import read

from amorphouspy.analysis.cavities import compute_cavities

from . import DATA_DIR

# ---------------------------------------------------------------------------
# Fixture
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


# ---------------------------------------------------------------------------
# compute_cavities
# ---------------------------------------------------------------------------


def test_compute_cavities_returns_dict(glass_structure):
    """Return type is a dict."""
    result = compute_cavities(glass_structure, resolution=16)
    assert isinstance(result, dict)


def test_compute_cavities_required_keys(glass_structure):
    """All expected property keys are present in the result."""
    result = compute_cavities(glass_structure, resolution=16)
    expected_keys = {"volumes", "surface_areas", "asphericities", "acylindricities", "anisotropies"}
    assert expected_keys.issubset(result.keys())


def test_compute_cavities_volumes_nonnegative(glass_structure):
    """All cavity volumes are non-negative."""
    result = compute_cavities(glass_structure, resolution=16)
    volumes = result["volumes"]
    assert np.all(volumes >= 0)


def test_compute_cavities_arrays_same_length(glass_structure):
    """All property arrays have the same length (one entry per cavity)."""
    result = compute_cavities(glass_structure, resolution=16)
    lengths = {k: len(v) for k, v in result.items()}
    assert len(set(lengths.values())) == 1, f"Mismatched lengths: {lengths}"


def test_compute_cavities_custom_cutoff_radii(glass_structure):
    """Passing explicit cutoff_radii does not raise and returns a valid dict."""
    result = compute_cavities(
        glass_structure,
        resolution=16,
        cutoff_radii={"O": 1.52, "Si": 1.10, "Na": 1.86},
    )
    assert "volumes" in result
