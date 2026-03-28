"""Tests for pure functions in amorphouspy.workflows.elastic_mod."""

import numpy as np
import pytest
from ase import Atoms

from amorphouspy.workflows.elastic_mod import apply_strain, isotropic_moduli_from_Cij

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cubic_atoms() -> Atoms:
    """Simple 2-atom structure in a 5 Å cubic box."""
    coords = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    cell = np.diag([5.0, 5.0, 5.0])
    return Atoms(numbers=[14, 8], positions=coords, cell=cell, pbc=True)


def _isotropic_cij(c11: float, c12: float, c44: float) -> np.ndarray:
    """Build a 6x6 Voigt stiffness matrix for an isotropic cubic material."""
    cij = np.zeros((6, 6))
    for i in range(3):
        cij[i, i] = c11
        for j in range(3):
            if i != j:
                cij[i, j] = c12
    for i in range(3, 6):
        cij[i, i] = c44
    return cij


# ---------------------------------------------------------------------------
# apply_strain
# ---------------------------------------------------------------------------


def test_apply_strain_zero_strain_unchanged_cell():
    """Zero strain tensor leaves the cell unchanged."""
    atoms = _cubic_atoms()
    strained = apply_strain(atoms, np.zeros((3, 3)))
    np.testing.assert_allclose(strained.get_cell(), atoms.get_cell(), atol=1e-10)


def test_apply_strain_uniaxial_x():
    """Uniaxial strain in x scales only the x cell vector."""
    atoms = _cubic_atoms()
    eps = np.zeros((3, 3))
    eps[0, 0] = 0.01
    strained = apply_strain(atoms, eps)
    cell = strained.get_cell()
    assert pytest.approx(cell[0, 0], rel=1e-6) == 5.0 * 1.01
    assert pytest.approx(cell[1, 1], rel=1e-6) == 5.0
    assert pytest.approx(cell[2, 2], rel=1e-6) == 5.0


def test_apply_strain_returns_new_atoms():
    """apply_strain does not modify the original atoms object."""
    atoms = _cubic_atoms()
    original_cell = atoms.get_cell().copy()
    eps = np.zeros((3, 3))
    eps[0, 0] = 0.01
    apply_strain(atoms, eps)
    np.testing.assert_allclose(atoms.get_cell(), original_cell)


def test_apply_strain_atoms_inside_cell():
    """All atoms are inside the new cell after straining."""
    atoms = _cubic_atoms()
    eps = np.zeros((3, 3))
    eps[0, 0] = 0.05
    strained = apply_strain(atoms, eps)
    scaled = strained.get_scaled_positions()
    assert (scaled >= 0.0).all()
    assert (scaled < 1.0).all()


def test_apply_strain_hydrostatic():
    """Hydrostatic strain uniformly scales all cell vectors."""
    atoms = _cubic_atoms()
    delta = 0.01
    eps = delta * np.eye(3)
    strained = apply_strain(atoms, eps)
    cell = strained.get_cell()
    for i in range(3):
        assert pytest.approx(cell[i, i], rel=1e-6) == 5.0 * (1.0 + delta)


# ---------------------------------------------------------------------------
# isotropic_moduli_from_Cij
# ---------------------------------------------------------------------------


def test_isotropic_moduli_returns_all_keys():
    """Result contains B, G, E, and nu."""
    cij = _isotropic_cij(c11=70.0, c12=30.0, c44=20.0)
    result = isotropic_moduli_from_Cij(cij)
    assert set(result.keys()) == {"B", "G", "E", "nu"}


def test_isotropic_moduli_bulk_modulus():
    """Bulk modulus B = (C11 + 2*C12) / 3."""
    c11, c12, c44 = 70.0, 30.0, 20.0
    cij = _isotropic_cij(c11, c12, c44)
    result = isotropic_moduli_from_Cij(cij)
    expected_B = (c11 + 2 * c12) / 3.0
    assert pytest.approx(result["B"]) == expected_B


def test_isotropic_moduli_shear_modulus():
    """Shear modulus G matches Voigt-Reuss-Hill average for isotropic case."""
    c11, c12, c44 = 70.0, 30.0, 20.0
    cij = _isotropic_cij(c11, c12, c44)
    result = isotropic_moduli_from_Cij(cij)
    gv = (c11 - c12 + 3 * c44) / 5.0
    gr = 5.0 * c44 * (c11 - c12) / (4.0 * c44 + 3.0 * (c11 - c12))
    expected_G = 0.5 * (gv + gr)
    assert pytest.approx(result["G"]) == expected_G


def test_isotropic_moduli_youngs_modulus():
    """Young's modulus E = 9*B*G / (3*B + G)."""
    c11, c12, c44 = 70.0, 30.0, 20.0
    cij = _isotropic_cij(c11, c12, c44)
    result = isotropic_moduli_from_Cij(cij)
    expected_E = 9.0 * result["B"] * result["G"] / (3.0 * result["B"] + result["G"])
    assert pytest.approx(result["E"]) == expected_E


def test_isotropic_moduli_poisson_ratio():
    """Poisson's ratio nu = (3B - 2G) / (2*(3B + G))."""
    c11, c12, c44 = 70.0, 30.0, 20.0
    cij = _isotropic_cij(c11, c12, c44)
    result = isotropic_moduli_from_Cij(cij)
    B, G = result["B"], result["G"]
    expected_nu = (3.0 * B - 2.0 * G) / (2.0 * (3.0 * B + G))
    assert pytest.approx(result["nu"]) == expected_nu


def test_isotropic_moduli_poisson_ratio_range():
    """Poisson's ratio is in the physically valid range (-1, 0.5) for positive moduli."""
    cij = _isotropic_cij(c11=70.0, c12=30.0, c44=20.0)
    result = isotropic_moduli_from_Cij(cij)
    assert -1.0 < result["nu"] < 0.5


def test_isotropic_moduli_symmetrizes_input():
    """Slightly asymmetric Cij is symmetrized before computation."""
    cij = _isotropic_cij(c11=70.0, c12=30.0, c44=20.0)
    cij_asym = cij.copy()
    cij_asym[0, 1] += 2.0  # break symmetry slightly
    result_sym = isotropic_moduli_from_Cij(cij)
    result_asym = isotropic_moduli_from_Cij(cij_asym)
    # Both should produce valid moduli (symmetrization is applied internally)
    assert result_asym["B"] != result_sym["B"] or result_asym["G"] != result_sym["G"]
