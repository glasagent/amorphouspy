"""Tests for structure factor analysis functions related to glassy systems.

Author: Achraf Atila (achraf.atila@bam.de)
"""

import numpy as np
import numpy.typing as npt
import pytest
from ase import Atoms

# Assuming the original script is named structure_factor.py
from amorphouspy.analysis.structure_factor import (
    _neutron_scattering_length,
    _sine_transform_rdf,
    _xray_form_factor,
    compute_structure_factor,
)


def test_neutron_scattering_length() -> None:
    """Test the retrieval of NIST neutron scattering lengths.

    Verifies that correct values are returned for common elements and that
    a KeyError is raised for unsupported atomic numbers.
    """
    # Test known values
    assert _neutron_scattering_length(14) == 4.1491  # Si
    assert _neutron_scattering_length(8) == 5.803  # O

    # Test error handling
    with pytest.raises(KeyError):
        _neutron_scattering_length(999)


def test_xray_form_factor() -> None:
    """Test the Doyle-Turner X-ray form factor calculation.

    Ensures the form factor equals the atomic number at q=0 and
    returns the correct array shape.
    """
    q: npt.NDArray[np.float64] = np.array([0.0, 1.0, 5.0])
    # At q=0, f(q) should equal Z (atomic number)
    f_si: npt.NDArray[np.float64] = _xray_form_factor(14, q)
    assert np.isclose(f_si[0], 14.0)
    assert len(f_si) == 3
    assert all(f_si > 0)


def test_sine_transform_rdf() -> None:
    """Test the Faber-Ziman sine transform math.

    Uses an ideal gas baseline (g(r) = 1) where the resulting
    structure factor S(q) must be exactly 1.0.
    """
    # Create a dummy RDF: g(r) = 1 (ideal gas / no correlation)
    r: npt.NDArray[np.float64] = np.linspace(0.1, 10.0, 100)
    gr: npt.NDArray[np.float64] = np.ones_like(r)
    q_values: npt.NDArray[np.float64] = np.linspace(0.5, 5.0, 10)
    number_density: float = 0.04  # typical for solids

    sq: npt.NDArray[np.float64] = _sine_transform_rdf(r, gr, q_values, number_density, lorch_damping=False)

    # For gr=1, the integral of r*(1-1) is 0, so S(q) should be exactly 1.0
    np.testing.assert_allclose(sq, 1.0, atol=1e-7)


def test_compute_structure_factor_integration() -> None:
    """Integration test for the total structure factor computation.

    Tests the full pipeline from an Atoms object to S(q) using
    neutron radiation settings.
    """
    # Setup a simple Silicon FCC cell
    a: float = 5.43
    lattice: Atoms = Atoms("Si2", scaled_positions=[(0, 0, 0), (0.25, 0.25, 0.25)], cell=(a, a, a), pbc=True)

    try:
        results: tuple[
            npt.NDArray[np.float64], npt.NDArray[np.float64], dict[tuple[int, int], npt.NDArray[np.float64]]
        ] = compute_structure_factor(lattice, q_min=1.0, q_max=10.0, n_q=50, r_max=5.0, radiation="neutron")

        q, sq, partials = results

        assert len(q) == 50
        assert len(sq) == 50
        assert (14, 14) in partials
        assert not np.isnan(sq).any()

    except ImportError:
        pytest.skip("amorphouspy or dependencies not available for integration test")


def test_radiation_value_error() -> None:
    """Verify that unsupported radiation types raise a ValueError."""
    lattice: Atoms = Atoms("Si", cell=(5, 5, 5), pbc=True)
    with pytest.raises(ValueError, match="radiation must be 'neutron' or 'xray'"):
        compute_structure_factor(lattice, radiation="electron")


@pytest.mark.parametrize("rad", ["neutron", "xray"])
def test_structure_factor_output_shapes(rad: str) -> None:
    """Test output array consistency for different radiation types.

    Args:
        rad: The radiation type to test ('neutron' or 'xray').
    """
    # Minimal system for shape checking
    structure: Atoms = Atoms("SiO2", positions=[(0, 0, 0), (1, 1, 1), (2, 2, 2)], cell=(5, 5, 5), pbc=True)

    n_q: int = 20
    q, sq, partials = compute_structure_factor(structure, n_q=n_q, radiation=rad, r_max=4.0)

    assert q.shape == (n_q,)
    assert sq.shape == (n_q,)
    # SiO2 should have Si-Si, O-O, and Si-O partials
    assert len(partials) == 3
