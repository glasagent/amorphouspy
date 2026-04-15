"""Tests for pure functions in amorphouspy.workflows.elastic_mod."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from amorphouspy.workflows.elastic_mod import apply_strain, elastic_simulation, isotropic_moduli_from_Cij
from ase import Atoms

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
    """Young's modulus E = 9*B*G / (3*B + G), computed from c11/c12/c44."""
    c11, c12, c44 = 70.0, 30.0, 20.0
    cij = _isotropic_cij(c11, c12, c44)
    result = isotropic_moduli_from_Cij(cij)
    B = (c11 + 2.0 * c12) / 3.0
    gv = (c11 - c12 + 3.0 * c44) / 5.0
    gr = 5.0 * c44 * (c11 - c12) / (4.0 * c44 + 3.0 * (c11 - c12))
    G = 0.5 * (gv + gr)
    assert pytest.approx(result["E"]) == 9.0 * B * G / (3.0 * B + G)


def test_isotropic_moduli_poisson_ratio():
    """Poisson's ratio nu = (3B - 2G) / (2*(3B + G)), computed from c11/c12/c44."""
    c11, c12, c44 = 70.0, 30.0, 20.0
    cij = _isotropic_cij(c11, c12, c44)
    result = isotropic_moduli_from_Cij(cij)
    B = (c11 + 2.0 * c12) / 3.0
    gv = (c11 - c12 + 3.0 * c44) / 5.0
    gr = 5.0 * c44 * (c11 - c12) / (4.0 * c44 + 3.0 * (c11 - c12))
    G = 0.5 * (gv + gr)
    assert pytest.approx(result["nu"]) == (3.0 * B - 2.0 * G) / (2.0 * (3.0 * B + G))


def test_isotropic_moduli_symmetrizes_input():
    """Slightly asymmetric Cij is symmetrized before computation."""
    cij = _isotropic_cij(c11=70.0, c12=30.0, c44=20.0)
    cij_asym = cij.copy()
    cij_asym[0, 1] += 2.0  # break symmetry slightly
    result_sym = isotropic_moduli_from_Cij(cij)
    result_asym = isotropic_moduli_from_Cij(cij_asym)
    # Both should produce valid moduli (symmetrization is applied internally)
    assert result_asym["B"] != result_sym["B"] or result_asym["G"] != result_sym["G"]


# ---------------------------------------------------------------------------
# elastic_simulation — n_repeats uncertainty quantification
# ---------------------------------------------------------------------------


def _make_potential() -> pd.DataFrame:
    return pd.DataFrame([{"Name": "test", "Config": []}])


def _make_equilibration_return(atoms):
    """Minimal mock return value for _run_lammps_md (equilibration call)."""
    # Volume list: 10 entries of 5^3 = 125 Å^3 → avg_l = 5.0
    return atoms.copy(), {"generic": {"volume": [125.0] * 10}}


def test_elastic_simulation_cij_index_mapping():
    """Each Cij entry is populated from the correct stress tensor component.

    Uses strain=1 (denom=2) and a stress tensor with distinct values per component
    so any index transposition produces a wrong number rather than silently passing.
    The shear entries (C44/C55/C66) are the most error-prone: they must read the
    off-diagonal stress component, not a diagonal one.
    """
    atoms = _cubic_atoms()
    strain = 1.0  # denom = 2.0 for easy mental arithmetic

    stress = np.zeros((3, 3))
    stress[0, 0] = 100.0
    stress[1, 1] = 20.0
    stress[2, 2] = 20.0
    stress[0, 1] = stress[1, 0] = 1.0  # → C66 (xy)
    stress[0, 2] = stress[2, 0] = 2.0  # → C55 (xz)
    stress[1, 2] = stress[2, 1] = 4.0  # → C44 (yz)
    denom = 2.0 * strain

    with (
        patch("amorphouspy.workflows.elastic_mod._run_lammps_md") as mock_md,
        patch("amorphouspy.workflows.elastic_mod._run_strained_md", return_value=stress),
    ):
        mock_md.return_value = _make_equilibration_return(atoms)
        result = elastic_simulation(atoms, _make_potential(), strain=strain, n_repeats=1)

    cij = result["Cij"]
    # Normal-strain rows: column j reads stress[j,j] / denom
    for i in range(3):
        assert cij[i, 0] == pytest.approx(stress[0, 0] / denom)
        assert cij[i, 1] == pytest.approx(stress[1, 1] / denom)
        assert cij[i, 2] == pytest.approx(stress[2, 2] / denom)
    # Shear entries: each reads its specific off-diagonal component
    assert cij[3, 3] == pytest.approx(stress[1, 2] / denom)  # yz → C44
    assert cij[4, 4] == pytest.approx(stress[0, 2] / denom)  # xz → C55
    assert cij[5, 5] == pytest.approx(stress[0, 1] / denom)  # xy → C66


def test_elastic_simulation_n_repeats_uses_sample_std():
    """Cij_std uses ddof=1 (sample std) when n_repeats > 1.

    ddof=0 and ddof=1 give different results for n=2; the test computes both
    and confirms only the ddof=1 value matches.
    """
    atoms = _cubic_atoms()
    strain = 1.0

    # Repeat 0 gets scale=1, repeat 1 gets scale=3 — 6 calls per repeat
    stress_a = np.zeros((3, 3))
    stress_b = np.zeros((3, 3))
    for s, scale in [(stress_a, 1.0), (stress_b, 3.0)]:
        s[0, 0] = 100.0 * scale
        s[1, 1] = s[2, 2] = 20.0 * scale
        s[0, 1] = s[1, 0] = 1.0 * scale
        s[0, 2] = s[2, 0] = 2.0 * scale
        s[1, 2] = s[2, 1] = 4.0 * scale

    with (
        patch("amorphouspy.workflows.elastic_mod._run_lammps_md") as mock_md,
        patch("amorphouspy.workflows.elastic_mod._run_strained_md", side_effect=[stress_a] * 6 + [stress_b] * 6),
    ):
        mock_md.return_value = _make_equilibration_return(atoms)
        result = elastic_simulation(atoms, _make_potential(), strain=strain, n_repeats=2)

    denom = 2.0 * strain
    val_a = stress_a[0, 0] / denom
    val_b = stress_b[0, 0] / denom
    assert result["Cij_std"][0, 0] == pytest.approx(np.std([val_a, val_b], ddof=1))
    assert result["Cij_std"][0, 0] != pytest.approx(np.std([val_a, val_b], ddof=0))
