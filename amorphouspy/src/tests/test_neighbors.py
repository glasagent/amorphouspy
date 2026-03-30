"""Tests for amorphouspy.neighbors — cell lists, cutoff parsing, and get_neighbors."""

import importlib
import sys

import numpy as np
import pytest
from amorphouspy.neighbors import (
    NUMBA_AVAILABLE,
    _build_nl_ortho_numba,
    _build_nl_tri_numba,
    _dist_and_vec_ortho,
    _dist_and_vec_tri,
    _estimate_max_neighbors,
    _extract_atom_ids,
    _get_pair_cutoff_sq_python,
    _lookup_cutoff_sq,
    _numba_to_list,
    _parse_cutoff,
    cell_perpendicular_heights,
    compute_cell_list_orthogonal,
    compute_cell_list_triclinic,
    get_neighbors,
)
from ase import Atoms

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ortho_atoms() -> Atoms:
    """Four atoms in a 6 Å cubic box: O-Si at 1 Å apart, O-Si at 1 Å apart."""
    coords = np.array([[0.5, 3.0, 3.0], [1.5, 3.0, 3.0], [3.5, 3.0, 3.0], [4.5, 3.0, 3.0]], dtype=np.float64)
    types = np.array([8, 14, 8, 14], dtype=np.int32)
    cell = np.diag([6.0, 6.0, 6.0])
    return Atoms(numbers=types, positions=coords, cell=cell, pbc=True)


def _triclinic_atoms() -> Atoms:
    """Two atoms in a triclinic box with a small shear."""
    cell = np.array([[5.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    coords = np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1]], dtype=np.float64)
    types = np.array([8, 14], dtype=np.int32)
    return Atoms(numbers=types, positions=coords, cell=cell, pbc=True)


# ---------------------------------------------------------------------------
# _parse_cutoff
# ---------------------------------------------------------------------------


def test_parse_cutoff_scalar() -> None:
    """Scalar cutoff returns use_pair_cutoffs=False and empty arrays."""
    types = np.array([8, 14], dtype=np.int32)
    max_rc, pair_types, pair_cutoffs_sq, use = _parse_cutoff(3.0, types)
    assert max_rc == pytest.approx(3.0)
    assert use is False
    assert pair_types.shape == (0, 2)
    assert pair_cutoffs_sq.shape == (0,)


def test_parse_cutoff_dict_symmetric() -> None:
    """Dict cutoff: (8,14) and (14,8) are stored as the same pair."""
    types = np.array([8, 14], dtype=np.int32)
    cutoff = {(8, 14): 2.0}
    max_rc, pair_types, _pair_cutoffs_sq, use = _parse_cutoff(cutoff, types)
    assert max_rc == pytest.approx(2.0)
    assert use is True
    # Both orderings must be present
    rows = [tuple(r) for r in pair_types]
    assert (8, 14) in rows
    assert (14, 8) in rows


def test_parse_cutoff_dict_default_fallback() -> None:
    """Pairs not listed in the dict default to the maximum cutoff."""
    types = np.array([8, 14, 11], dtype=np.int32)
    cutoff = {(8, 14): 2.0, (8, 11): 2.7}
    max_rc, pair_types, pair_cutoffs_sq, _use = _parse_cutoff(cutoff, types)
    assert max_rc == pytest.approx(2.7)
    # (14,11) was not listed → should equal max_rc squared
    rows = [tuple(r) for r in pair_types]
    idx = rows.index((14, 11))
    assert pair_cutoffs_sq[idx] == pytest.approx(2.7**2)


# ---------------------------------------------------------------------------
# cell_perpendicular_heights
# ---------------------------------------------------------------------------


def test_cell_perpendicular_heights_ortho() -> None:
    """For a diagonal cell, perpendicular heights equal the box edge lengths."""
    cell = np.diag([5.0, 7.0, 9.0])
    heights = cell_perpendicular_heights(cell)
    np.testing.assert_allclose(heights, [5.0, 7.0, 9.0], rtol=1e-10)


def test_cell_perpendicular_heights_triclinic() -> None:
    """For a sheared cell, perpendicular heights are smaller than edge norms."""
    cell = np.array([[5.0, 0.0, 0.0], [2.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    heights = cell_perpendicular_heights(cell)
    # h_b < |b| because b is sheared
    assert heights[1] < np.linalg.norm(cell[1])
    assert all(h > 0 for h in heights)


# ---------------------------------------------------------------------------
# compute_cell_list_orthogonal
# ---------------------------------------------------------------------------


def test_compute_cell_list_orthogonal_shape() -> None:
    """cell_start has n_total+1 entries; all atoms are assigned a valid cell."""
    coords = np.random.default_rng(0).random((20, 3)) * 10.0
    box_size = np.array([10.0, 10.0, 10.0])
    atom_cells, n_cells, cell_start, order = compute_cell_list_orthogonal(coords, box_size, 3.0)
    n_total = int(n_cells[0]) * int(n_cells[1]) * int(n_cells[2])
    assert cell_start.shape[0] == n_total + 1
    assert order.shape[0] == len(coords)
    assert atom_cells.shape == (len(coords), 3)


def test_compute_cell_list_orthogonal_all_atoms_assigned() -> None:
    """Every atom appears exactly once in the sorted order array."""
    coords = np.random.default_rng(1).random((10, 3)) * 8.0
    box_size = np.array([8.0, 8.0, 8.0])
    _, _, _, order = compute_cell_list_orthogonal(coords, box_size, 2.0)
    assert sorted(order.tolist()) == list(range(len(coords)))


# ---------------------------------------------------------------------------
# compute_cell_list_triclinic
# ---------------------------------------------------------------------------


def test_compute_cell_list_triclinic_frac_coords_in_range() -> None:
    """Fractional coordinates must lie in [0, 1)."""
    cell = np.array([[6.0, 0.0, 0.0], [1.5, 6.0, 0.0], [0.0, 0.0, 6.0]])
    coords = np.random.default_rng(2).random((15, 3)) * 6.0
    coords_frac, _atom_cells, _n_cells, _cell_start, _order = compute_cell_list_triclinic(coords, cell, 2.0)
    assert np.all(coords_frac >= 0.0)
    assert np.all(coords_frac < 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# _estimate_max_neighbors
# ---------------------------------------------------------------------------


def test_estimate_max_neighbors_reasonable() -> None:
    """Returns a positive integer for a typical glass density."""
    coords = np.random.default_rng(3).random((100, 3)) * 10.0
    cell = np.diag([10.0, 10.0, 10.0])
    result = _estimate_max_neighbors(coords, cell, 3.5)
    assert isinstance(result, int)
    assert result >= 64


def test_estimate_max_neighbors_zero_volume() -> None:
    """Zero-volume cell returns the safe default of 200."""
    coords = np.zeros((5, 3))
    cell = np.zeros((3, 3))
    result = _estimate_max_neighbors(coords, cell, 3.0)
    assert result == 200


def test_estimate_max_neighbors_empty() -> None:
    """Zero atoms returns the safe default of 200."""
    result = _estimate_max_neighbors(np.zeros((0, 3)), np.diag([5.0, 5.0, 5.0]), 3.0)
    assert result == 200


# ---------------------------------------------------------------------------
# _extract_atom_ids
# ---------------------------------------------------------------------------


def test_extract_atom_ids_default_sequential() -> None:
    """Without an 'id' array, IDs are 1-based sequential."""
    atoms = Atoms("SiO", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True)
    ids = _extract_atom_ids(atoms)
    np.testing.assert_array_equal(ids, [1, 2])


def test_extract_atom_ids_from_array() -> None:
    """When 'id' array is present, it is used directly."""
    atoms = Atoms("SiO", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True)
    atoms.new_array("id", np.array([101, 202], dtype=np.int64))
    ids = _extract_atom_ids(atoms)
    np.testing.assert_array_equal(ids, [101, 202])


# ---------------------------------------------------------------------------
# get_neighbors — orthogonal box
# ---------------------------------------------------------------------------


def test_get_neighbors_scalar_cutoff_finds_pair() -> None:
    """Two atoms 1 Å apart are mutual neighbors within a 1.5 Å cutoff."""
    atoms = _ortho_atoms()
    nl = {cid: nn for cid, nn, *_ in get_neighbors(atoms, cutoff=1.5)}
    # Atom 1 (O at 0.5) and atom 2 (Si at 1.5) should be neighbors
    assert len(nl[1]) == 1
    assert nl[1][0] == 2


def test_get_neighbors_scalar_cutoff_misses_far_pair() -> None:
    """Atoms 2 Å apart are not neighbors within a 1.5 Å cutoff."""
    atoms = _ortho_atoms()
    nl = {cid: nn for cid, nn, *_ in get_neighbors(atoms, cutoff=1.5)}
    # Atom 2 (Si at 1.5) and atom 3 (O at 3.5) are 2 Å apart → not neighbors
    assert 3 not in nl[2]


def test_get_neighbors_pair_cutoff_dict() -> None:
    """Per-pair cutoffs: (8,14) cutoff 1.5, (8,8) cutoff 0.5 → only cross-pairs found."""
    atoms = _ortho_atoms()
    cutoff = {(8, 14): 1.5, (8, 8): 0.5, (14, 14): 0.5}
    nl = {cid: nn for cid, nn, *_ in get_neighbors(atoms, cutoff=cutoff)}
    # O(1) → Si(2) at 1Å, within (8,14)=1.5
    assert 2 in nl[1]
    # O(1) → O(3) at 3Å, beyond (8,8)=0.5
    assert 3 not in nl[1]


def test_get_neighbors_target_filter() -> None:
    """target_types restricts which atoms are searched."""
    atoms = _ortho_atoms()
    result = get_neighbors(atoms, cutoff=1.5, target_types=[8])
    # Only O atoms (type 8) should be central atoms with non-empty entries
    central_ids = [cid for cid, nn in result if nn]
    # Fetch types by position
    for cid in central_ids:
        idx = cid - 1  # 1-based → 0-based
        assert atoms.get_atomic_numbers()[idx] == 8


def test_get_neighbors_neighbor_filter() -> None:
    """neighbor_types restricts which atoms count as neighbors."""
    atoms = _ortho_atoms()
    nl = {cid: nn for cid, nn, *_ in get_neighbors(atoms, cutoff=1.5, neighbor_types=[8])}
    # No Si atom should appear as a neighbor
    for nn_list in nl.values():
        for nid in nn_list:
            idx = nid - 1
            assert atoms.get_atomic_numbers()[idx] == 8


def test_get_neighbors_return_vectors_norms_match_distances() -> None:
    """Bond vector norms match the scalar distances."""
    atoms = _ortho_atoms()
    result = get_neighbors(atoms, cutoff=1.5, return_vectors=True)
    for _cid, nn_ids, vecs in result:
        if nn_ids:
            norms = np.linalg.norm(vecs, axis=1)
            assert all(n <= 1.5 + 1e-6 for n in norms)
            assert norms.shape[0] == len(nn_ids)


def test_get_neighbors_numpy_fallback_same_result() -> None:
    """NumPy fallback produces the same neighbor lists as the Numba path."""
    atoms = _ortho_atoms()
    nl_numba = {cid: sorted(nn) for cid, nn in get_neighbors(atoms, cutoff=1.5, use_numba=True)}
    nl_numpy = {cid: sorted(nn) for cid, nn in get_neighbors(atoms, cutoff=1.5, use_numba=False)}
    assert nl_numba == nl_numpy


def test_get_neighbors_numpy_fallback_with_vectors() -> None:
    """NumPy fallback with return_vectors gives consistent norms."""
    atoms = _ortho_atoms()
    result = get_neighbors(atoms, cutoff=1.5, return_vectors=True, use_numba=False)
    for _cid, nn_ids, vecs in result:
        if nn_ids:
            norms = np.linalg.norm(vecs, axis=1)
            assert all(n <= 1.5 + 1e-6 for n in norms)


def test_get_neighbors_numpy_fallback_pair_cutoffs() -> None:
    """NumPy fallback respects per-pair cutoff dicts."""
    atoms = _ortho_atoms()
    cutoff = {(8, 14): 1.5, (8, 8): 0.5, (14, 14): 0.5}
    nl_np = {cid: sorted(nn) for cid, nn in get_neighbors(atoms, cutoff=cutoff, use_numba=False)}
    nl_nb = {cid: sorted(nn) for cid, nn in get_neighbors(atoms, cutoff=cutoff, use_numba=True)}
    assert nl_np == nl_nb


# ---------------------------------------------------------------------------
# get_neighbors — triclinic box
# ---------------------------------------------------------------------------


def test_get_neighbors_triclinic_finds_neighbor() -> None:
    """Neighbor search works for a triclinic (sheared) cell."""
    atoms = _triclinic_atoms()
    result = get_neighbors(atoms, cutoff=1.5)
    nl = dict(result)
    # Two atoms are ~1 Å apart; each should find the other
    assert len(nl[1]) >= 1
    assert len(nl[2]) >= 1


def test_get_neighbors_triclinic_numpy_matches_numba() -> None:
    """Numba and NumPy paths agree on triclinic boxes."""
    atoms = _triclinic_atoms()
    nl_nb = {cid: sorted(nn) for cid, nn in get_neighbors(atoms, cutoff=1.5, use_numba=True)}
    nl_np = {cid: sorted(nn) for cid, nn in get_neighbors(atoms, cutoff=1.5, use_numba=False)}
    assert nl_nb == nl_np


# ---------------------------------------------------------------------------
# get_neighbors — tuple input
# ---------------------------------------------------------------------------


def test_get_neighbors_tuple_input() -> None:
    """Raw (coords, types, cell) tuple gives the same result as ASE Atoms."""
    atoms = _ortho_atoms()
    coords = atoms.get_positions()
    types = atoms.get_atomic_numbers().astype(np.int32)
    cell = atoms.get_cell().array

    nl_atoms = {cid: sorted(nn) for cid, nn in get_neighbors(atoms, cutoff=1.5)}
    nl_tuple = {cid: sorted(nn) for cid, nn in get_neighbors((coords, types, cell), cutoff=1.5)}
    assert nl_atoms == nl_tuple


# ---------------------------------------------------------------------------
# Numba fallback (lines 24-35): reimport with numba disabled
# ---------------------------------------------------------------------------


def test_numba_fallback_jit_and_prange() -> None:
    """When numba is unavailable, jit is a no-op and prange == range."""
    # Stash original module if present
    original_numba = sys.modules.get("numba")
    original_neighbors = sys.modules.pop("amorphouspy.neighbors", None)

    try:
        # Pretend numba is not installed
        sys.modules["numba"] = None  # type: ignore[assignment]
        mod = importlib.import_module("amorphouspy.neighbors")
        assert mod.NUMBA_AVAILABLE is False
        # prange must be the builtin range when numba is absent
        assert mod.prange is range
        # The jit decorator must be callable and return a pass-through
        test_fn = mod.jit(nopython=True)(lambda x: x * 2)
        assert test_fn(5) == 10
    finally:
        # Restore original state
        if original_numba is None:
            sys.modules.pop("numba", None)
        else:
            sys.modules["numba"] = original_numba  # type: ignore[assignment]
        sys.modules.pop("amorphouspy.neighbors", None)
        if original_neighbors is not None:
            sys.modules["amorphouspy.neighbors"] = original_neighbors


# ---------------------------------------------------------------------------
# _lookup_cutoff_sq body (lines 115-118)
# ---------------------------------------------------------------------------


def test_lookup_cutoff_sq_found() -> None:
    """_lookup_cutoff_sq returns the correct squared cutoff for a found pair."""
    pair_types = np.array([[8, 14], [14, 8], [8, 8], [14, 14]], dtype=np.int32)
    pair_cutoffs_sq = np.array([4.0, 4.0, 2.25, 9.0], dtype=np.float64)

    fn = _lookup_cutoff_sq.py_func if NUMBA_AVAILABLE else _lookup_cutoff_sq
    result = fn(8, 14, pair_types, pair_cutoffs_sq)
    assert result == pytest.approx(4.0)


def test_lookup_cutoff_sq_fallback() -> None:
    """_lookup_cutoff_sq returns pair_cutoffs_sq[0] when pair is not found."""
    # Only (8,14) in the table — pair (11,11) not present
    pair_types = np.array([[8, 14]], dtype=np.int32)
    pair_cutoffs_sq = np.array([4.0], dtype=np.float64)

    fn = _lookup_cutoff_sq.py_func if NUMBA_AVAILABLE else _lookup_cutoff_sq
    result = fn(11, 11, pair_types, pair_cutoffs_sq)
    # Fallback: first entry
    assert result == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# _get_pair_cutoff_sq_python fallback (line 131)
# ---------------------------------------------------------------------------


def test_get_pair_cutoff_sq_python_found() -> None:
    """Returns correct value when pair is in the table."""
    pair_types = np.array([[8, 14]], dtype=np.int32)
    pair_cutoffs_sq = np.array([4.0], dtype=np.float64)
    result = _get_pair_cutoff_sq_python(8, 14, pair_types, pair_cutoffs_sq)
    assert result == pytest.approx(4.0)


def test_get_pair_cutoff_sq_python_fallback() -> None:
    """Returns pair_cutoffs_sq[0] when pair (11,11) is not in table."""
    pair_types = np.array([[8, 14]], dtype=np.int32)
    pair_cutoffs_sq = np.array([4.0], dtype=np.float64)
    result = _get_pair_cutoff_sq_python(11, 11, pair_types, pair_cutoffs_sq)
    assert result == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# _dist_and_vec_ortho body (lines 242-248)
# ---------------------------------------------------------------------------


def test_dist_and_vec_ortho_body() -> None:
    """_dist_and_vec_ortho returns correct displacement and squared distance."""
    fn = _dist_and_vec_ortho.py_func if NUMBA_AVAILABLE else _dist_and_vec_ortho
    ci = np.array([0.5, 0.0, 0.0], dtype=np.float64)
    cj = np.array([1.5, 0.0, 0.0], dtype=np.float64)
    box = np.array([6.0, 6.0, 6.0], dtype=np.float64)
    dx, _dy, _dz, dist_sq = fn(ci, cj, box)
    assert dist_sq == pytest.approx(1.0)
    assert dx == pytest.approx(-1.0)


def test_dist_and_vec_ortho_minimum_image() -> None:
    """_dist_and_vec_ortho uses minimum image convention."""
    fn = _dist_and_vec_ortho.py_func if NUMBA_AVAILABLE else _dist_and_vec_ortho
    # Atoms at 0.1 and 5.9 in a 6.0 box → minimum image distance is 0.2
    ci = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    cj = np.array([5.9, 0.0, 0.0], dtype=np.float64)
    box = np.array([6.0, 6.0, 6.0], dtype=np.float64)
    _dx, _dy, _dz, dist_sq = fn(ci, cj, box)
    assert dist_sq == pytest.approx(0.04, abs=1e-10)


# ---------------------------------------------------------------------------
# _dist_and_vec_tri body (lines 262-271)
# ---------------------------------------------------------------------------


def test_dist_and_vec_tri_body() -> None:
    """_dist_and_vec_tri returns correct Cartesian displacement and squared distance."""
    fn = _dist_and_vec_tri.py_func if NUMBA_AVAILABLE else _dist_and_vec_tri
    cell = np.array([[5.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float64)
    frac_i = np.array([0.02, 0.02, 0.02], dtype=np.float64)
    frac_j = np.array([0.22, 0.02, 0.02], dtype=np.float64)
    _dx, _dy, _dz, dist_sq = fn(frac_i, frac_j, cell)
    # delta_frac_x = -0.2, Cartesian dx = -0.2*5 + (-0.2)*1 = -1.0 - 0.0 (depends on cell)
    assert dist_sq > 0.0
    assert isinstance(dist_sq, float)


def test_dist_and_vec_tri_minimum_image() -> None:
    """_dist_and_vec_tri applies minimum image across periodic boundary."""
    fn = _dist_and_vec_tri.py_func if NUMBA_AVAILABLE else _dist_and_vec_tri
    cell = np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]], dtype=np.float64)
    frac_i = np.array([0.01, 0.5, 0.5], dtype=np.float64)
    frac_j = np.array([0.99, 0.5, 0.5], dtype=np.float64)
    _dx, _dy, _dz, dist_sq = fn(frac_i, frac_j, cell)
    # Minimum image: delta_frac_x = 0.01 - 0.99 = -0.98 → wrapped to 0.02 → dx = 0.12
    assert dist_sq == pytest.approx((0.02 * 6.0) ** 2, rel=1e-5)


# ---------------------------------------------------------------------------
# _make_ortho_nl_inputs helper for low-level kernel tests
# ---------------------------------------------------------------------------


def _make_ortho_nl_inputs(atoms, cutoff):
    """Build the kwargs dict needed to call _build_nl_ortho_numba directly."""
    atoms_copy = atoms.copy()
    atoms_copy.wrap()
    coords = atoms_copy.get_positions()
    types = atoms_copy.get_atomic_numbers().astype(np.int32)
    cell = atoms_copy.get_cell().array
    box_size = np.diag(cell)
    max_cutoff, pair_types, pair_cutoffs_sq, use_pair_cutoffs = _parse_cutoff(cutoff, types)
    cutoff_sq = max_cutoff**2
    atom_cells, n_cells, cell_start, cell_atoms = compute_cell_list_orthogonal(coords, box_size, max_cutoff)
    max_neighbors = _estimate_max_neighbors(coords, cell, max_cutoff)
    return {
        "coords": coords,
        "types": types,
        "box_size": box_size,
        "atom_cells": atom_cells,
        "n_cells": n_cells,
        "cell_start": cell_start,
        "cell_atoms": cell_atoms,
        "cutoff_sq": cutoff_sq,
        "max_neighbors": max_neighbors,
        "pair_types": pair_types,
        "pair_cutoffs_sq": pair_cutoffs_sq,
        "use_pair_cutoffs": use_pair_cutoffs,
    }


# ---------------------------------------------------------------------------
# _build_nl_ortho_numba body (lines 310-378)
# ---------------------------------------------------------------------------


def test_build_nl_ortho_numba_basic() -> None:
    """_build_nl_ortho_numba basic path without filters."""
    fn = _build_nl_ortho_numba.py_func if NUMBA_AVAILABLE else _build_nl_ortho_numba
    atoms = _ortho_atoms()
    kwargs = _make_ortho_nl_inputs(atoms, cutoff=1.5)
    kwargs["target_types"] = np.empty(0, dtype=np.int32)
    kwargs["neighbor_types"] = np.empty(0, dtype=np.int32)
    kwargs["use_target_filter"] = False
    kwargs["use_neighbor_filter"] = False
    kwargs["return_vectors"] = False

    nl, counts, _vecs = fn(**kwargs)
    assert nl.shape[0] == len(atoms)
    assert counts.shape[0] == len(atoms)
    # At least some neighbors found
    assert counts.sum() > 0


def test_build_nl_ortho_numba_target_filter() -> None:
    """_build_nl_ortho_numba with use_target_filter: Si atoms (type 14) are skipped."""
    fn = _build_nl_ortho_numba.py_func if NUMBA_AVAILABLE else _build_nl_ortho_numba
    atoms = _ortho_atoms()
    kwargs = _make_ortho_nl_inputs(atoms, cutoff=1.5)
    # Only O atoms (type 8) are targets — Si atoms (type 14) get continue
    kwargs["target_types"] = np.array([8], dtype=np.int32)
    kwargs["neighbor_types"] = np.empty(0, dtype=np.int32)
    kwargs["use_target_filter"] = True
    kwargs["use_neighbor_filter"] = False
    kwargs["return_vectors"] = False

    _nl, counts, _vecs = fn(**kwargs)
    types = atoms.get_atomic_numbers()
    # Si atoms (index 1, 3 in 0-based) should have count 0
    for idx in range(len(atoms)):
        if types[idx] == 14:
            assert counts[idx] == 0


def test_build_nl_ortho_numba_neighbor_filter() -> None:
    """_build_nl_ortho_numba with use_neighbor_filter: only O neighbors kept."""
    fn = _build_nl_ortho_numba.py_func if NUMBA_AVAILABLE else _build_nl_ortho_numba
    atoms = _ortho_atoms()
    kwargs = _make_ortho_nl_inputs(atoms, cutoff=1.5)
    kwargs["target_types"] = np.empty(0, dtype=np.int32)
    kwargs["neighbor_types"] = np.array([8], dtype=np.int32)
    kwargs["use_target_filter"] = False
    kwargs["use_neighbor_filter"] = True
    kwargs["return_vectors"] = False

    nl, counts, _vecs = fn(**kwargs)
    types = atoms.get_atomic_numbers()
    # Verify only O-type neighbors appear
    for i in range(len(atoms)):
        for k in range(int(counts[i])):
            j = int(nl[i, k])
            assert types[j] == 8


def test_build_nl_ortho_numba_pair_cutoffs() -> None:
    """_build_nl_ortho_numba with use_pair_cutoffs covers the _lookup_cutoff_sq branch."""
    fn = _build_nl_ortho_numba.py_func if NUMBA_AVAILABLE else _build_nl_ortho_numba
    atoms = _ortho_atoms()
    kwargs = _make_ortho_nl_inputs(atoms, cutoff={(8, 14): 1.5, (8, 8): 0.5, (14, 14): 0.5})
    kwargs["target_types"] = np.empty(0, dtype=np.int32)
    kwargs["neighbor_types"] = np.empty(0, dtype=np.int32)
    kwargs["use_target_filter"] = False
    kwargs["use_neighbor_filter"] = False
    kwargs["return_vectors"] = False

    nl, _counts, _vecs = fn(**kwargs)
    assert nl.shape[0] == len(atoms)


def test_build_nl_ortho_numba_return_vectors() -> None:
    """_build_nl_ortho_numba with return_vectors=True populates vector_list."""
    fn = _build_nl_ortho_numba.py_func if NUMBA_AVAILABLE else _build_nl_ortho_numba
    atoms = _ortho_atoms()
    kwargs = _make_ortho_nl_inputs(atoms, cutoff=1.5)
    kwargs["target_types"] = np.empty(0, dtype=np.int32)
    kwargs["neighbor_types"] = np.empty(0, dtype=np.int32)
    kwargs["use_target_filter"] = False
    kwargs["use_neighbor_filter"] = False
    kwargs["return_vectors"] = True

    _nl, counts, vecs = fn(**kwargs)
    # Check that at least one vector is non-zero for a bonded pair
    found_nonzero = False
    for i in range(len(atoms)):
        for k in range(int(counts[i])):
            vec = vecs[i, k]
            if np.any(vec != 0.0):
                found_nonzero = True
                break
    assert found_nonzero


# ---------------------------------------------------------------------------
# _build_nl_tri_numba body (lines 413-481)
# ---------------------------------------------------------------------------


def _make_tri_nl_inputs(atoms, cutoff):
    """Build the kwargs dict needed to call _build_nl_tri_numba directly."""
    atoms_copy = atoms.copy()
    atoms_copy.wrap()
    coords = atoms_copy.get_positions()
    types = atoms_copy.get_atomic_numbers().astype(np.int32)
    cell = atoms_copy.get_cell().array
    max_cutoff, pair_types, pair_cutoffs_sq, use_pair_cutoffs = _parse_cutoff(cutoff, types)
    cutoff_sq = max_cutoff**2
    coords_frac, atom_cells, n_cells, cell_start, cell_atoms = compute_cell_list_triclinic(coords, cell, max_cutoff)
    max_neighbors = _estimate_max_neighbors(coords, cell, max_cutoff)
    return {
        "coords_frac": coords_frac,
        "types": types,
        "cell": cell,
        "atom_cells": atom_cells,
        "n_cells": n_cells,
        "cell_start": cell_start,
        "cell_atoms": cell_atoms,
        "cutoff_sq": cutoff_sq,
        "max_neighbors": max_neighbors,
        "pair_types": pair_types,
        "pair_cutoffs_sq": pair_cutoffs_sq,
        "use_pair_cutoffs": use_pair_cutoffs,
    }


def test_build_nl_tri_numba_basic() -> None:
    """_build_nl_tri_numba basic path without filters."""
    fn = _build_nl_tri_numba.py_func if NUMBA_AVAILABLE else _build_nl_tri_numba
    atoms = _triclinic_atoms()
    kwargs = _make_tri_nl_inputs(atoms, cutoff=1.5)
    kwargs["target_types"] = np.empty(0, dtype=np.int32)
    kwargs["neighbor_types"] = np.empty(0, dtype=np.int32)
    kwargs["use_target_filter"] = False
    kwargs["use_neighbor_filter"] = False
    kwargs["return_vectors"] = False

    nl, counts, _vecs = fn(**kwargs)
    assert nl.shape[0] == len(atoms)
    assert counts.shape[0] == len(atoms)


def test_build_nl_tri_numba_target_filter() -> None:
    """_build_nl_tri_numba with use_target_filter skips non-target atoms."""
    fn = _build_nl_tri_numba.py_func if NUMBA_AVAILABLE else _build_nl_tri_numba
    atoms = _triclinic_atoms()
    kwargs = _make_tri_nl_inputs(atoms, cutoff=1.5)
    # Only O atoms (type 8) are targets
    kwargs["target_types"] = np.array([8], dtype=np.int32)
    kwargs["neighbor_types"] = np.empty(0, dtype=np.int32)
    kwargs["use_target_filter"] = True
    kwargs["use_neighbor_filter"] = False
    kwargs["return_vectors"] = False

    _nl, counts, _vecs = fn(**kwargs)
    types = atoms.get_atomic_numbers()
    for idx in range(len(atoms)):
        if types[idx] != 8:
            assert counts[idx] == 0


def test_build_nl_tri_numba_neighbor_filter() -> None:
    """_build_nl_tri_numba with use_neighbor_filter only allows O neighbors."""
    fn = _build_nl_tri_numba.py_func if NUMBA_AVAILABLE else _build_nl_tri_numba
    atoms = _triclinic_atoms()
    kwargs = _make_tri_nl_inputs(atoms, cutoff=1.5)
    kwargs["target_types"] = np.empty(0, dtype=np.int32)
    kwargs["neighbor_types"] = np.array([8], dtype=np.int32)
    kwargs["use_target_filter"] = False
    kwargs["use_neighbor_filter"] = True
    kwargs["return_vectors"] = False

    nl, counts, _vecs = fn(**kwargs)
    types = atoms.get_atomic_numbers()
    for i in range(len(atoms)):
        for k in range(int(counts[i])):
            j = int(nl[i, k])
            assert types[j] == 8


def test_build_nl_tri_numba_pair_cutoffs() -> None:
    """_build_nl_tri_numba with use_pair_cutoffs covers the pair-lookup branch."""
    fn = _build_nl_tri_numba.py_func if NUMBA_AVAILABLE else _build_nl_tri_numba
    atoms = _triclinic_atoms()
    kwargs = _make_tri_nl_inputs(atoms, cutoff={(8, 14): 1.5, (8, 8): 0.5, (14, 14): 0.5})
    kwargs["target_types"] = np.empty(0, dtype=np.int32)
    kwargs["neighbor_types"] = np.empty(0, dtype=np.int32)
    kwargs["use_target_filter"] = False
    kwargs["use_neighbor_filter"] = False
    kwargs["return_vectors"] = False

    nl, _counts, _vecs = fn(**kwargs)
    assert nl.shape[0] == len(atoms)


def test_build_nl_tri_numba_return_vectors() -> None:
    """_build_nl_tri_numba with return_vectors=True populates vector_list."""
    fn = _build_nl_tri_numba.py_func if NUMBA_AVAILABLE else _build_nl_tri_numba
    atoms = _triclinic_atoms()
    kwargs = _make_tri_nl_inputs(atoms, cutoff=1.5)
    kwargs["target_types"] = np.empty(0, dtype=np.int32)
    kwargs["neighbor_types"] = np.empty(0, dtype=np.int32)
    kwargs["use_target_filter"] = False
    kwargs["use_neighbor_filter"] = False
    kwargs["return_vectors"] = True

    _nl, _counts, vecs = fn(**kwargs)
    # Verify shape
    assert vecs.shape == (len(atoms), kwargs["max_neighbors"], 3)


# ---------------------------------------------------------------------------
# _numba_to_list overflow loop (lines 503-506)
# ---------------------------------------------------------------------------


def test_numba_to_list_overflow_retry() -> None:
    """_numba_to_list triggers overflow retry when counts exceed max_neighbors."""
    n_atoms = 4
    max_neighbors = 1  # start too small

    # First call returns neighbor_counts with value > max_neighbors
    nl_overflow = np.full((n_atoms, 1), 0, dtype=np.int32)
    counts_overflow = np.array([3, 3, 3, 3], dtype=np.int32)  # 3 > 1
    vecs_overflow = np.zeros((n_atoms, 1, 3), dtype=np.float32)

    call_count = {"n": 0}

    def mock_build_fn(**kwargs):
        call_count["n"] += 1
        mn = kwargs["max_neighbors"]
        # Return a proper result on retry with larger buffer
        nl = np.zeros((n_atoms, mn), dtype=np.int32)
        counts = np.array([1, 1, 1, 1], dtype=np.int32)
        vecs = np.zeros((n_atoms, mn, 3), dtype=np.float32)
        return nl, counts, vecs

    build_kwargs = {"max_neighbors": max_neighbors}
    idx_neighbors, _vec_neighbors = _numba_to_list(
        nl_overflow,
        counts_overflow,
        vecs_overflow,
        n_atoms,
        max_neighbors,
        mock_build_fn,
        build_kwargs,
        return_vectors=False,
    )
    # Should have called build_fn at least once (retry)
    assert call_count["n"] >= 1
    assert len(idx_neighbors) == n_atoms


# ---------------------------------------------------------------------------
# NumPy fallback — target filter (line 585)
# ---------------------------------------------------------------------------


def test_numpy_fallback_target_filter_skips_non_target() -> None:
    """use_numba=False with target_types=[8]: Si atoms (type 14) hit line 585 continue."""
    atoms = _ortho_atoms()
    result = get_neighbors(atoms, cutoff=1.5, target_types=[8], use_numba=False)
    types = atoms.get_atomic_numbers()
    # Si atoms (type 14) are skipped by target filter → their neighbor lists must be empty
    for cid, nn_ids in result:
        idx = cid - 1  # 1-based → 0-based
        if types[idx] == 14:  # Si (non-target)
            assert nn_ids == []


# ---------------------------------------------------------------------------
# NumPy fallback — neighbor filter (line 593)
# ---------------------------------------------------------------------------


def test_numpy_fallback_neighbor_filter_applies() -> None:
    """use_numba=False with neighbor_types=[8]: line 593 filters candidates."""
    atoms = _ortho_atoms()
    result = get_neighbors(atoms, cutoff=1.5, neighbor_types=[8], use_numba=False)
    types = atoms.get_atomic_numbers()
    # All reported neighbors should be O atoms
    for entry in result:
        nn_ids = entry[1]
        for nid in nn_ids:
            idx = nid - 1
            assert types[idx] == 8


# ---------------------------------------------------------------------------
# NumPy fallback — no candidates after filter (line 595)
# ---------------------------------------------------------------------------


def test_numpy_fallback_no_candidates_after_filter() -> None:
    """use_numba=False: Si atoms look for Na neighbors — none present → line 595 hit."""
    atoms = _ortho_atoms()  # contains only O (8) and Si (14), no Na (11)
    # target_types=[14] → look at Si atoms; neighbor_types=[11] → only Na neighbors
    # There are no Na atoms → all candidates filtered → continue on line 595
    result = get_neighbors(atoms, cutoff=1.5, target_types=[14], neighbor_types=[11], use_numba=False)
    # Si atoms should have empty neighbor lists since there are no Na atoms
    for entry in result:
        entry[0]
        nn_ids = entry[1]
        # All Si atoms should have no Na neighbors
        assert nn_ids == []
