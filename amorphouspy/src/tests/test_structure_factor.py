"""Tests for structure factor analysis functions related to glassy systems.

Author: Achraf Atila (achraf.atila@bam.de)
"""

import numpy as np
import numpy.typing as npt
import pytest
from amorphouspy.structure_characterization.averaging import average_over_frames
from amorphouspy.structure_characterization.structure_factor import (
    _neutron_scattering_length,
    _sine_transform_rdf,
    _xray_form_factor,
    compute_structure_factor,
)
from ase import Atoms


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
        results = compute_structure_factor(lattice, q_min=1.0, q_max=10.0, n_q=50, r_max=5.0, radiation="neutron")

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


# ---------------------------------------------------------------------------
# average_over_frames — compute_structure_factor
# ---------------------------------------------------------------------------


def test_average_over_frames_structure_factor_identical_frames() -> None:
    """Three identical frames: mean equals single-frame result, SEM ≈ 0."""
    structure = Atoms("SiO2", positions=[(0, 0, 0), (1, 1, 1), (2, 2, 2)], cell=(5, 5, 5), pbc=True)
    n_q = 20
    q_s, sq_s, partials_s = compute_structure_factor(structure, n_q=n_q, r_max=4.0)
    (q_a, sq_mean, partials_mean), (_, sq_sem, partials_sem) = average_over_frames(
        compute_structure_factor, [structure, structure, structure], n_q=n_q, r_max=4.0
    )
    assert np.allclose(q_a, q_s)
    assert np.allclose(sq_mean, sq_s)
    assert np.allclose(sq_sem, 0.0, atol=1e-10)
    for k in partials_s:
        assert np.allclose(partials_mean[k], partials_s[k])
        assert np.allclose(partials_sem[k], 0.0, atol=1e-10)


def test_compute_structure_factor_list_uses_first_frame() -> None:
    """Passing a list without average_over_frames uses the first frame."""
    structure = Atoms("SiO2", positions=[(0, 0, 0), (1, 1, 1), (2, 2, 2)], cell=(5, 5, 5), pbc=True)
    q_s, sq_s, _ = compute_structure_factor(structure, n_q=20, r_max=4.0)
    q_a, sq_a, _ = compute_structure_factor([structure, structure], n_q=20, r_max=4.0)
    assert np.allclose(q_a, q_s)
    assert np.allclose(sq_a, sq_s)


def test_compute_structure_factor_unknown_xray_element_raises() -> None:
    """X-ray form factor for an element not in ATOMIC_SCATTERING_PARAMS raises KeyError."""
    # Oganesson (Z=118) is not in the Doyle-Turner table
    structure = Atoms([118], positions=[(0, 0, 0)], cell=(5, 5, 5), pbc=True)
    with pytest.raises(KeyError):
        compute_structure_factor(structure, radiation="xray", n_q=10, r_max=4.0)
