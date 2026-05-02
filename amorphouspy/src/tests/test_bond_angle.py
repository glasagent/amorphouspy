"""Tests for amorphouspy.analysis.bond_angle_distribution."""

import numpy as np
import pytest
from amorphouspy.analysis.bond_angle_distribution import compute_angles
from ase import Atoms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _atoms_with_known_angle(angle_deg: float) -> Atoms:
    """Return a structure with one Si center and two O neighbors at `angle_deg`."""
    r = 1.6  # bond length in Å
    angle_rad = np.deg2rad(angle_deg)
    # Place O atoms symmetrically about the x-axis in the xy-plane
    o1 = np.array([r * np.cos(angle_rad / 2), r * np.sin(angle_rad / 2), 0.0])
    o2 = np.array([r * np.cos(angle_rad / 2), -r * np.sin(angle_rad / 2), 0.0])
    center = np.array([5.0, 5.0, 5.0])
    coords = np.array([center, center + o1, center + o2])
    types = np.array([14, 8, 8])
    cell = np.diag([10.0, 10.0, 10.0])
    return Atoms(numbers=types, positions=coords, cell=cell, pbc=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compute_angles_90deg_peak() -> None:
    """90° geometry → histogram peak in the 85-95° range."""
    atoms = _atoms_with_known_angle(90.0)
    bin_centers, hist, _sem = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=2.0)
    peak_angle = bin_centers[np.argmax(hist)]
    assert 85.0 <= peak_angle <= 95.0, f"Peak at {peak_angle:.1f}°, expected ~90°"


def test_compute_angles_180deg_peak() -> None:
    """180° (linear) geometry → histogram peak in the 175-180° range."""
    # Si at center, two O along ±x axis
    center = np.array([5.0, 5.0, 5.0])
    r = 1.6
    coords = np.array([center, center + [r, 0, 0], center + [-r, 0, 0]])  # noqa: RUF005
    types = np.array([14, 8, 8])
    cell = np.diag([10.0, 10.0, 10.0])
    atoms = Atoms(numbers=types, positions=coords, cell=cell, pbc=True)
    bin_centers, hist, _sem = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=2.0)
    peak_angle = bin_centers[np.argmax(hist)]
    assert 170.0 <= peak_angle <= 180.0, f"Peak at {peak_angle:.1f}°, expected ~180°"


def test_compute_angles_output_shape_default_bins() -> None:
    """Default bins=180 → bin_centers has 180 elements."""
    atoms = _atoms_with_known_angle(109.47)  # tetrahedral-ish
    bin_centers, hist, _sem = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=2.0)
    assert bin_centers.shape == (180,)
    assert hist.shape == (180,)


def test_compute_angles_custom_bins() -> None:
    """Custom bins parameter changes the output array length."""
    atoms = _atoms_with_known_angle(90.0)
    bin_centers, hist, _sem = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=2.0, bins=90)
    assert bin_centers.shape == (90,)
    assert hist.shape == (90,)


def test_compute_angles_normalized() -> None:
    """Histogram is density-normalized (integrates to ~1 over [0, 180])."""
    atoms = _atoms_with_known_angle(90.0)
    bin_centers, hist, _sem = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=2.0)
    dr = bin_centers[1] - bin_centers[0]
    integral = np.sum(hist) * dr
    assert integral == pytest.approx(1.0, abs=0.02)


def test_compute_angles_no_neighbors_within_cutoff() -> None:
    """Cutoff too small → no neighbors found → histogram has no finite non-zero values."""
    atoms = _atoms_with_known_angle(90.0)
    _bin_centers, hist, _sem = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=0.5)
    # density=True on empty data produces NaN; treat NaN and 0 both as "no signal"
    assert np.nansum(hist) == pytest.approx(0.0)


def test_compute_angles_single_neighbor_skipped() -> None:
    """Atoms with only one neighbor do not contribute angles."""
    # Only one O near the Si center
    center = np.array([5.0, 5.0, 5.0])
    coords = np.array([center, center + [1.6, 0, 0]])  # noqa: RUF005
    types = np.array([14, 8])
    cell = np.diag([10.0, 10.0, 10.0])
    atoms = Atoms(numbers=types, positions=coords, cell=cell, pbc=True)
    _, hist, _sem = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=2.0)
    assert np.nansum(hist) == pytest.approx(0.0)


def test_compute_angles_wrong_center_type() -> None:
    """No atoms of the requested center type → histogram has no finite non-zero values."""
    atoms = _atoms_with_known_angle(90.0)
    _, hist, _sem = compute_angles(atoms, center_type=11, neighbor_type=8, cutoff=2.0)  # Na not present
    assert np.nansum(hist) == pytest.approx(0.0)


def test_compute_angles_bin_centers_range() -> None:
    """Bin centers span (0, 180) degrees."""
    atoms = _atoms_with_known_angle(90.0)
    bin_centers, _, _sem = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=2.0)
    assert bin_centers[0] > 0.0
    assert bin_centers[-1] < 180.0


# ---------------------------------------------------------------------------
# frame_averaging
# ---------------------------------------------------------------------------


def test_compute_angles_frame_averaging_identical_frames() -> None:
    """Three identical frames: mean equals single-frame result, SEM ≈ 0."""
    atoms = _atoms_with_known_angle(90.0)
    bins_s, hist_s, _sem_s = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=2.0)
    bins_a, hist_mean, hist_sem = compute_angles(
        [atoms, atoms, atoms], center_type=14, neighbor_type=8, cutoff=2.0, frame_averaging=True
    )
    assert np.allclose(bins_a, bins_s)
    assert np.allclose(hist_mean, hist_s)
    assert np.allclose(hist_sem, 0.0, atol=1e-10)


def test_compute_angles_frame_averaging_empty_list_raises() -> None:
    """frame_averaging=True with empty list raises ValueError."""
    with pytest.raises(ValueError, match="frame_averaging=True requires"):
        compute_angles([], 14, 8, 2.0, frame_averaging=True)


def test_compute_angles_frame_averaging_single_atoms_fallback() -> None:
    """frame_averaging=True with single Atoms falls back to single-frame result with SEM=0."""
    atoms = _atoms_with_known_angle(90.0)
    bins_s, hist_s, _sem_s = compute_angles(atoms, 14, 8, 2.0)
    bins_a, hist_mean, hist_sem = compute_angles(atoms, 14, 8, 2.0, frame_averaging=True)
    assert np.allclose(bins_a, bins_s)
    assert np.allclose(hist_mean, hist_s)
    assert np.allclose(hist_sem, 0.0, atol=1e-10)


def test_compute_angles_list_uses_first_frame() -> None:
    """Passing a list without frame_averaging=True uses the first frame."""
    atoms = _atoms_with_known_angle(90.0)
    bins_s, hist_s, _sem_s = compute_angles(atoms, 14, 8, 2.0)
    bins_a, hist_a, _sem_a = compute_angles([atoms, atoms], 14, 8, 2.0)
    assert np.allclose(bins_a, bins_s)
    assert np.allclose(hist_a, hist_s)


def test_compute_angles_triclinic_cell() -> None:
    """Triclinic (sheared) cell produces same peak as orthogonal cell — covers fractional minimum-image branch."""
    atoms_ortho = _atoms_with_known_angle(90.0)
    cell_shear = atoms_ortho.get_cell().array.copy()
    cell_shear[1, 0] = 1.0  # shear b-vector along a
    atoms_tri = Atoms(
        numbers=atoms_ortho.get_atomic_numbers(),
        positions=atoms_ortho.get_positions(),
        cell=cell_shear,
        pbc=True,
    )
    bins_ortho, hist_ortho, _sem_ortho = compute_angles(atoms_ortho, center_type=14, neighbor_type=8, cutoff=2.0)
    bins_tri, hist_tri, _sem_tri = compute_angles(atoms_tri, center_type=14, neighbor_type=8, cutoff=2.0)
    peak_ortho = bins_ortho[np.argmax(hist_ortho)]
    peak_tri = bins_tri[np.argmax(hist_tri)]
    assert abs(peak_ortho - peak_tri) < 2.0


def test_compute_angles_fewer_than_two_neighbors() -> None:
    """Atom with only 1 neighbor after filtering is skipped — histogram has no signal."""
    # Single Si with only one O neighbor within cutoff
    coords = np.array([[5.0, 5.0, 5.0], [6.6, 5.0, 5.0]])
    types = np.array([14, 8])
    atoms = Atoms(numbers=types, positions=coords, cell=np.diag([10.0, 10.0, 10.0]), pbc=True)
    bins, hist, _sem = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=2.0)
    assert bins.shape == hist.shape
    assert np.nansum(hist) == pytest.approx(0.0)
