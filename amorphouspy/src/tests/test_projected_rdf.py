"""Tests for the projected RDF (Y₂₀ uniaxial + Y₂₁/Y₂₂ shear) analysis."""

from __future__ import annotations

import numpy as np
import pytest
from amorphouspy.structure_characterization.projected_rdf import compute_projected_rdf
from amorphouspy.structure_characterization.radial_distribution_functions import compute_rdf
from ase import Atoms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_cubic_sc(n: int = 4, a: float = 3.0) -> Atoms:
    """Simple-cubic supercell — fully isotropic, single species."""
    base = Atoms("Si", positions=[(0, 0, 0)], cell=[(a, 0, 0), (0, a, 0), (0, 0, a)], pbc=True)
    return base.repeat(n)


def _strained_z(atoms: Atoms, strain: float = 0.05) -> Atoms:
    """Apply uniaxial strain along z by scaling the cell and positions."""
    strained = atoms.copy()
    cell = strained.get_cell().array.copy()
    cell[2] *= 1.0 + strain
    strained.set_cell(cell, scale_atoms=True)
    return strained


def _sheared_xy(atoms: Atoms, gamma: float = 0.05) -> Atoms:
    """Apply xy shear by tilting the x lattice vector."""
    sheared = atoms.copy()
    cell = sheared.get_cell().array.copy()
    cell[0, 1] += gamma * cell[1, 1]
    sheared.set_cell(cell, scale_atoms=True)
    return sheared


# ---------------------------------------------------------------------------
# Output shape / basic contract
# ---------------------------------------------------------------------------


def test_output_shapes() -> None:
    """Output shapes and None-routing are correct for all three calling modes."""
    atoms = _simple_cubic_sc(n=3)
    n_bins = 50

    r, uniaxial, shear = compute_projected_rdf(atoms, deformation_axis="z", n_bins=n_bins, r_max=4.0)
    assert r.shape == (n_bins,)
    assert uniaxial is not None
    assert shear is None
    assert (14, 14) in uniaxial
    assert uniaxial[(14, 14)].shape == (n_bins,)

    r, uniaxial, shear = compute_projected_rdf(atoms, shear_plane="xy", n_bins=n_bins, r_max=4.0)
    assert r.shape == (n_bins,)
    assert uniaxial is None
    assert shear is not None
    assert (14, 14) in shear

    _, uniaxial, shear = compute_projected_rdf(atoms, deformation_axis="z", shear_plane="xy", n_bins=n_bins, r_max=4.0)
    assert uniaxial is not None
    assert shear is not None


def test_neither_axis_nor_plane_raises() -> None:
    """ValueError raised when neither deformation_axis nor shear_plane is set."""
    atoms = _simple_cubic_sc(n=3)
    with pytest.raises(ValueError, match="At least one"):
        compute_projected_rdf(atoms, r_max=4.0)


# ---------------------------------------------------------------------------
# Isotropy: signals near zero for undeformed cubic cell
# ---------------------------------------------------------------------------


def test_uniaxial_near_zero_for_isotropic_cell() -> None:
    """An undeformed SC lattice has no uniaxial anisotropy."""
    atoms = _simple_cubic_sc(n=5)
    _, uniaxial, _ = compute_projected_rdf(atoms, deformation_axis="z", n_bins=200, r_max=5.0)
    assert uniaxial is not None
    # The mean absolute value over the full range should be small relative to 1
    assert np.mean(np.abs(uniaxial[(14, 14)])) < 0.05


def test_shear_near_zero_for_isotropic_cell() -> None:
    """An undeformed SC lattice has no shear anisotropy."""
    atoms = _simple_cubic_sc(n=5)
    _, _, shear = compute_projected_rdf(atoms, shear_plane="xy", n_bins=200, r_max=5.0)
    assert shear is not None
    assert np.mean(np.abs(shear[(14, 14)])) < 0.05


# ---------------------------------------------------------------------------
# Deformation signal
# ---------------------------------------------------------------------------


def test_uniaxial_signal_nonzero_after_z_strain() -> None:
    """Z-strained cell should produce a measurable uniaxial signal."""
    atoms = _simple_cubic_sc(n=5)
    strained = _strained_z(atoms, strain=0.10)
    _, uniaxial_ref, _ = compute_projected_rdf(atoms, deformation_axis="z", n_bins=200, r_max=5.0)
    _, uniaxial_def, _ = compute_projected_rdf(strained, deformation_axis="z", n_bins=200, r_max=5.0)
    assert uniaxial_ref is not None
    assert uniaxial_def is not None
    delta = np.max(np.abs(uniaxial_def[(14, 14)] - uniaxial_ref[(14, 14)]))
    assert delta > 1e-4


def test_shear_signal_nonzero_after_xy_shear() -> None:
    """XY-sheared cell should produce a measurable shear signal."""
    atoms = _simple_cubic_sc(n=5)
    sheared = _sheared_xy(atoms, gamma=0.10)
    _, _, shear_ref = compute_projected_rdf(atoms, shear_plane="xy", n_bins=200, r_max=5.0)
    _, _, shear_def = compute_projected_rdf(sheared, shear_plane="xy", n_bins=200, r_max=5.0)
    assert shear_ref is not None
    assert shear_def is not None
    delta = np.max(np.abs(shear_def[(14, 14)] - shear_ref[(14, 14)]))
    assert delta > 1e-4


# ---------------------------------------------------------------------------
# Axis remapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_all_deformation_axes_run(axis: str) -> None:
    """All three deformation axes run without error."""
    atoms = _simple_cubic_sc(n=3)
    _, uniaxial, _ = compute_projected_rdf(atoms, deformation_axis=axis, n_bins=50, r_max=4.0)
    assert uniaxial is not None
    assert not np.isnan(uniaxial[(14, 14)]).any()


@pytest.mark.parametrize("plane", ["xy", "xz", "yz"])
def test_all_shear_planes_run(plane: str) -> None:
    """All three shear planes run without error."""
    atoms = _simple_cubic_sc(n=3)
    _, _, shear = compute_projected_rdf(atoms, shear_plane=plane, n_bins=50, r_max=4.0)
    assert shear is not None
    assert not np.isnan(shear[(14, 14)]).any()


# ---------------------------------------------------------------------------
# Multi-frame averaging
# ---------------------------------------------------------------------------


def test_identical_frames_averaging_stable() -> None:
    """Average of identical frames equals single-frame result; SEM should be zero."""
    atoms = _simple_cubic_sc(n=3)
    r_s, uni_s, shear_s = compute_projected_rdf(atoms, deformation_axis="z", shear_plane="xy", n_bins=50, r_max=4.0)
    r_m, uni_m, shear_m = compute_projected_rdf(
        [atoms, atoms, atoms], deformation_axis="z", shear_plane="xy", n_bins=50, r_max=4.0
    )
    assert np.allclose(r_s, r_m)
    assert uni_s is not None
    assert uni_m is not None
    assert shear_s is not None
    assert shear_m is not None
    assert np.allclose(uni_s[(14, 14)], uni_m[(14, 14)])
    assert np.allclose(shear_s[(14, 14)], shear_m[(14, 14)])


# ---------------------------------------------------------------------------
# Triclinic cell
# ---------------------------------------------------------------------------


def test_triclinic_cell_handled() -> None:
    """Triclinic cell should not raise and should produce finite values."""
    cell = np.array([[6.0, 0.5, 0.0], [0.0, 6.0, 0.3], [0.0, 0.0, 6.0]])
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
    types = [14, 14, 14, 14]
    atoms = Atoms(numbers=types, positions=positions, cell=cell, pbc=True)
    _, uniaxial, _ = compute_projected_rdf(atoms, deformation_axis="z", n_bins=30, r_max=4.0)
    assert uniaxial is not None
    assert not np.isnan(uniaxial[(14, 14)]).any()


# ---------------------------------------------------------------------------
# r_max clamping warning
# ---------------------------------------------------------------------------


def test_rmax_clamping_warning() -> None:
    """r_max exceeding half the cell height should trigger a UserWarning."""
    atoms = Atoms("Si", positions=[(0, 0, 0)], cell=(4.0, 4.0, 4.0), pbc=True)
    with pytest.warns(UserWarning, match="r_max"):
        _, _, _ = compute_projected_rdf(atoms, deformation_axis="z", r_max=10.0, n_bins=20)


def test_rmax_raises_on_tiny_box() -> None:
    """ValueError raised when box is too small for any valid integer r_max."""
    tiny = Atoms("Si", positions=[(0, 0, 0)], cell=(1.0, 1.0, 1.0), pbc=True)
    with pytest.raises(ValueError, match="no valid integer cutoff"):
        compute_projected_rdf(tiny, deformation_axis="z", r_max=1.0, n_bins=10)


# ---------------------------------------------------------------------------
# Multi-species: explicit type_pairs + cross-species normalisation
# ---------------------------------------------------------------------------


def test_explicit_type_pairs() -> None:
    """Explicit type_pairs is canonicalized; cross-species normalisation branch is exercised."""
    rng = np.random.default_rng(42)
    n_si, n_o = 8, 16
    positions = rng.uniform(0, 10, size=(n_si + n_o, 3))
    numbers = np.array([14] * n_si + [8] * n_o)
    atoms = Atoms(numbers=numbers, positions=positions, cell=(10, 10, 10), pbc=True)

    _, uniaxial, _ = compute_projected_rdf(
        atoms,
        deformation_axis="z",
        n_bins=50,
        r_max=4.0,
        type_pairs=[(14, 8), (8, 14)],
    )
    assert uniaxial is not None
    assert list(uniaxial.keys()) == [(8, 14)]
    assert not np.isnan(uniaxial[(8, 14)]).any()


def test_cross_species_auto_pairs() -> None:
    """Auto-detected pairs for a binary system include cross-species pairs with correct normalisation."""
    rng = np.random.default_rng(0)
    n_si, n_o = 8, 16
    positions = rng.uniform(0, 10, size=(n_si + n_o, 3))
    numbers = np.array([14] * n_si + [8] * n_o)
    atoms = Atoms(numbers=numbers, positions=positions, cell=(10, 10, 10), pbc=True)

    _, uniaxial, _ = compute_projected_rdf(atoms, deformation_axis="z", n_bins=50, r_max=4.0)
    assert uniaxial is not None
    assert (8, 8) in uniaxial
    assert (14, 14) in uniaxial
    assert (8, 14) in uniaxial
    assert not np.isnan(uniaxial[(8, 14)]).any()


# ---------------------------------------------------------------------------
# Y₀₀ normalization consistency: g₂₀(z) = sqrt(5)*g(r) when all bonds are along z
# ---------------------------------------------------------------------------


def _z_aligned_dimer_crystal(n: int = 4, a: float = 4.0, dz: float = 1.5) -> Atoms:
    """Elongated cell with Si-Si dimers aligned strictly along z.

    Each unit cell has two Si at (0,0,0) and (0,0,dz) in a box of (a, a, a).
    With integer repeat n, the supercell has 2*n³ atoms. All nearest-neighbor
    bonds are along z, so Y₂₀(ẑ) = N₂₀*(1.5-0.5) = N₂₀ = sqrt(5/4π) for
    every bond in the first peak.  After sqrt(4π) scale: g₂₀ = sqrt(5)*g(r).
    """
    base = Atoms(
        "Si2",
        positions=[(0.0, 0.0, 0.0), (0.0, 0.0, dz)],
        cell=[(a, 0, 0), (0, a, 0), (0, 0, a)],
        pbc=True,
    )
    return base.repeat(n)


def test_g20_scale_z_aligned_bonds() -> None:
    """g₂₀(z-axis) equals sqrt(5)*g(r) for a crystal with all bonds along z.

    Derivation: each bond r̂ = ẑ gives Y₂₀(ẑ) = sqrt(5/4π).
    After normalisation by sqrt(4pi)/(N * p * Omega): g20 = sqrt(5/4pi)*sqrt(4pi)*g(r) = sqrt(5)*g(r).
    This confirms the Y₀₀ convention: sqrt(4π)*Y₀₀ = 1, so g₀₀ would recover g(r).
    """
    atoms = _z_aligned_dimer_crystal(n=3, a=4.0, dz=1.5)
    r_max, n_bins = 3.0, 300

    r_proj, uniaxial, _ = compute_projected_rdf(atoms, deformation_axis="z", r_max=r_max, n_bins=n_bins)
    _r_rdf, rdfs, _ = compute_rdf(atoms, r_max=r_max, n_bins=n_bins)

    g20 = uniaxial[(14, 14)]
    g_r = rdfs[(14, 14)]

    # Find bins in the first peak (dz=1.5 Å ± 0.3 Å)
    peak_mask = (r_proj > 1.2) & (r_proj < 1.8)
    assert peak_mask.any(), "No bins found in first-peak window"

    g20_peak = g20[peak_mask]
    gr_peak = g_r[peak_mask]

    # All peak bins should satisfy g₂₀ ≈ sqrt(5) * g(r)
    expected = np.sqrt(5.0) * gr_peak
    np.testing.assert_allclose(g20_peak, expected, rtol=0.01, err_msg="g₂₀(z-axis) != sqrt(5)*g(r) for z-aligned bonds")
