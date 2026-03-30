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
    bin_centers, hist = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=2.0)
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
    bin_centers, hist = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=2.0)
    peak_angle = bin_centers[np.argmax(hist)]
    assert 170.0 <= peak_angle <= 180.0, f"Peak at {peak_angle:.1f}°, expected ~180°"


def test_compute_angles_output_shape_default_bins() -> None:
    """Default bins=180 → bin_centers has 180 elements."""
    atoms = _atoms_with_known_angle(109.47)  # tetrahedral-ish
    bin_centers, hist = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=2.0)
    assert bin_centers.shape == (180,)
    assert hist.shape == (180,)


def test_compute_angles_custom_bins() -> None:
    """Custom bins parameter changes the output array length."""
    atoms = _atoms_with_known_angle(90.0)
    bin_centers, hist = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=2.0, bins=90)
    assert bin_centers.shape == (90,)
    assert hist.shape == (90,)


def test_compute_angles_normalized() -> None:
    """Histogram is density-normalized (integrates to ~1 over [0, 180])."""
    atoms = _atoms_with_known_angle(90.0)
    bin_centers, hist = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=2.0)
    dr = bin_centers[1] - bin_centers[0]
    integral = np.sum(hist) * dr
    assert integral == pytest.approx(1.0, abs=0.02)


def test_compute_angles_no_neighbors_within_cutoff() -> None:
    """Cutoff too small → no neighbors found → histogram has no finite non-zero values."""
    atoms = _atoms_with_known_angle(90.0)
    _bin_centers, hist = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=0.5)
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
    _, hist = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=2.0)
    assert np.nansum(hist) == pytest.approx(0.0)


def test_compute_angles_wrong_center_type() -> None:
    """No atoms of the requested center type → histogram has no finite non-zero values."""
    atoms = _atoms_with_known_angle(90.0)
    _, hist = compute_angles(atoms, center_type=11, neighbor_type=8, cutoff=2.0)  # Na not present
    assert np.nansum(hist) == pytest.approx(0.0)


def test_compute_angles_bin_centers_range() -> None:
    """Bin centers span (0, 180) degrees."""
    atoms = _atoms_with_known_angle(90.0)
    bin_centers, _ = compute_angles(atoms, center_type=14, neighbor_type=8, cutoff=2.0)
    assert bin_centers[0] > 0.0
    assert bin_centers[-1] < 180.0
