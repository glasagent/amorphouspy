"""Tests for amorphouspy.analysis.radial_distribution_functions."""

import numpy as np
import pytest
from amorphouspy.analysis.radial_distribution_functions import (
    _compute_cn_cumulative,
    _compute_distances,
    _compute_rdf_histograms,
    compute_rdf,
)
from ase import Atoms
from ase.io import read

from . import DATA_DIR

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_sio2_atoms() -> Atoms:
    """Minimal SiO2 unit: Si at centre, two O at 1.6 Å along x/y in a 10 Å box."""
    coords = np.array([[5.0, 5.0, 5.0], [6.6, 5.0, 5.0], [5.0, 6.6, 5.0]])
    types = np.array([14, 8, 8])
    cell = np.diag([10.0, 10.0, 10.0])
    return Atoms(numbers=types, positions=coords, cell=cell, pbc=True)


# ---------------------------------------------------------------------------
# _compute_distances
# ---------------------------------------------------------------------------


def test_compute_distances_orthogonal_finds_pair() -> None:
    """Two atoms 1.6 Å apart are returned with correct distance."""
    atoms = Atoms("SiO", positions=[[4.0, 5.0, 5.0], [5.6, 5.0, 5.0]], cell=[10, 10, 10], pbc=True)
    distances, _i_idx, _j_idx = _compute_distances(atoms, r_max=3.0)
    assert len(distances) >= 1
    assert distances[0] == pytest.approx(1.6, abs=1e-5)


def test_compute_distances_pbc_wrapping() -> None:
    """Atoms near opposite faces of the box are found via periodic images."""
    # Place atoms near x=0 and x≈10 so the minimum-image distance is ~1 Å
    atoms = Atoms("SiO", positions=[[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]], cell=[10, 10, 10], pbc=True)
    distances, _, _ = _compute_distances(atoms, r_max=2.0)
    assert len(distances) == 1
    assert distances[0] == pytest.approx(1.0, abs=1e-5)


def test_compute_distances_triclinic() -> None:
    """_compute_distances works for a triclinic cell and finds close pairs."""
    cell = np.array([[6.0, 0.0, 0.0], [1.0, 6.0, 0.0], [0.0, 0.0, 6.0]])
    coords = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]])
    atoms = Atoms(numbers=[14, 8], positions=coords, cell=cell, pbc=True)
    distances, _, _ = _compute_distances(atoms, r_max=3.0)
    assert len(distances) >= 1
    # Both atoms are within 3 Å of each other regardless of shear
    assert distances[0] <= 3.0


# ---------------------------------------------------------------------------
# _compute_rdf_histograms
# ---------------------------------------------------------------------------


def _make_histogram_inputs(same_type: bool) -> dict:  # noqa: FBT001
    """Build minimal inputs for _compute_rdf_histograms."""
    bin_edges = np.linspace(0, 5, 51)
    r = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    dr = bin_edges[1] - bin_edges[0]
    shell_volumes = 4.0 * np.pi * r**2 * dr
    volume = 1000.0

    if same_type:
        # 10 distances between type-8 atoms at ~1.5 Å
        distances = np.full(10, 1.5)
        type_i = np.full(10, 8, dtype=int)
        type_j = np.full(10, 8, dtype=int)
        type_counts = {8: 5}
        unordered_pairs = [(8, 8)]
    else:
        # 10 distances between type-8 (O) and type-14 (Si)
        distances = np.full(10, 1.6)
        type_i = np.full(10, 8, dtype=int)
        type_j = np.full(10, 14, dtype=int)
        type_counts = {8: 10, 14: 5}
        unordered_pairs = [(8, 14)]

    return {
        "unordered_pairs": unordered_pairs,
        "type_i": type_i,
        "type_j": type_j,
        "distances": distances,
        "type_counts": type_counts,
        "volume": volume,
        "bin_edges": bin_edges,
        "shell_volumes": shell_volumes,
    }


def test_compute_rdf_histograms_same_type_key() -> None:
    """Same-type pair is stored under (t, t) canonical key."""
    kwargs = _make_histogram_inputs(same_type=True)
    rdfs, _hist_dir = _compute_rdf_histograms(**kwargs)
    assert (8, 8) in rdfs
    assert rdfs[(8, 8)].shape[0] == 50


def test_compute_rdf_histograms_same_type_factor2() -> None:
    """Same-type histogram applies factor of 2 to the raw counts."""
    kwargs = _make_histogram_inputs(same_type=True)
    _rdfs, hist_dir = _compute_rdf_histograms(**kwargs)
    # The directed histogram is raw (not doubled)
    raw_hist = hist_dir[(8, 8)]
    # g(r) numerator = hist*2, so g(r)*denominator ≈ hist*2
    assert raw_hist.sum() == 10  # 10 pairs inserted


def test_compute_rdf_histograms_cross_type_symmetric() -> None:
    """Cross-type histogram sums forward and reverse directions."""
    kwargs = _make_histogram_inputs(same_type=False)
    rdfs, hist_dir = _compute_rdf_histograms(**kwargs)
    assert (8, 14) in rdfs
    # Only forward distances were inserted → hist_dir sum == 10
    assert hist_dir[(8, 14)].sum() == 10


# ---------------------------------------------------------------------------
# _compute_cn_cumulative
# ---------------------------------------------------------------------------


def test_compute_cn_cumulative_same_type_factor2() -> None:
    """Same-type CN cumulative applies factor=2."""
    hist = np.array([0, 5, 3, 0, 0], dtype=float)
    hist_directed = {(8, 8): hist}
    type_counts = {8: 4}
    result = _compute_cn_cumulative([(8, 8)], hist_directed, type_counts)
    # factor=2 → cumsum([0,10,6,0,0]) / 4 = [0, 2.5, 4.0, 4.0, 4.0]
    expected = np.cumsum(hist * 2) / 4
    np.testing.assert_allclose(result[(8, 8)], expected)


def test_compute_cn_cumulative_cross_type_factor1() -> None:
    """Cross-type CN cumulative applies factor=1."""
    hist = np.array([0, 4, 4, 0, 0], dtype=float)
    hist_directed = {(8, 14): hist}
    type_counts = {8: 2, 14: 2}
    result = _compute_cn_cumulative([(8, 14)], hist_directed, type_counts)
    expected = np.cumsum(hist * 1) / 2
    np.testing.assert_allclose(result[(8, 14)], expected)


def test_compute_cn_cumulative_ordered_pair() -> None:
    """Ordered pair (14, 8) uses type_counts[14] as the reference count."""
    hist = np.array([0, 4, 4, 0, 0], dtype=float)
    hist_directed = {(8, 14): hist}
    type_counts = {8: 10, 14: 5}
    result = _compute_cn_cumulative([(14, 8)], hist_directed, type_counts)
    # t1=14 → n_ref = type_counts[14] = 5, factor=1
    expected = np.cumsum(hist) / 5
    np.testing.assert_allclose(result[(14, 8)], expected)


# ---------------------------------------------------------------------------
# compute_rdf — integration
# ---------------------------------------------------------------------------


def test_compute_rdf_returns_correct_shapes() -> None:
    """compute_rdf returns r, rdfs, cn with consistent shapes."""
    atoms = _simple_sio2_atoms()
    r, rdfs, cn = compute_rdf(atoms, r_max=4.0, n_bins=100)
    assert r.shape == (100,)
    for arr in rdfs.values():
        assert arr.shape == (100,)
    for arr in cn.values():
        assert arr.shape == (100,)


def test_compute_rdf_rmax_clamping() -> None:
    """r_max larger than half the box height is clamped with a UserWarning."""
    atoms = _simple_sio2_atoms()  # 10 Å box → max allowed = 5 Å
    with pytest.warns(UserWarning, match="r_max=8.0000"):
        r, _rdfs, _cn = compute_rdf(atoms, r_max=8.0, n_bins=50)
    # r should be <= adjusted r_max (floor of 5.0 = 5)
    assert r.max() <= 5.0 + 1e-6


def test_compute_rdf_rmax_too_small_raises() -> None:
    """r_max is adjusted to 0 for a tiny box → ValueError."""
    coords = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    cell = np.diag([1.5, 1.5, 1.5])
    atoms = Atoms(numbers=[8, 14], positions=coords, cell=cell, pbc=True)
    with pytest.raises(ValueError, match="r_max_allowed"):
        compute_rdf(atoms, r_max=10.0, n_bins=10)


def test_compute_rdf_type_pairs_filter() -> None:
    """Explicit type_pairs returns only the requested keys."""
    atoms = _simple_sio2_atoms()
    _r, rdfs, _cn = compute_rdf(atoms, r_max=4.0, n_bins=50, type_pairs=[(8, 14)])
    assert set(rdfs.keys()) == {(8, 14)}


def test_compute_rdf_with_dump_si_o_peak() -> None:
    """On a real glass dump, the Si-O first peak should be near 1.6 Å."""
    filename = DATA_DIR / "20Na2O-80SiO2.dump"
    atoms = read(filename, format="lammps-dump-text")
    type_id = atoms.get_atomic_numbers().copy()
    to_z = np.array([0, 11, 8, 14], dtype=int)
    atoms.set_atomic_numbers(to_z[type_id])

    r, rdfs, _ = compute_rdf(atoms, r_max=6.0, n_bins=300, type_pairs=[(8, 14)])
    g_si_o = rdfs[(8, 14)]
    # First peak of Si-O RDF is ~1.6 Å
    peak_r = r[np.argmax(g_si_o)]
    assert 1.4 <= peak_r <= 1.9, f"Si-O peak at unexpected position: {peak_r:.3f} Å"
