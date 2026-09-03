"""Tests for amorphouspy.atoms.neighbors — cell lists, cutoff parsing, and get_neighbors."""

import importlib
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from amorphouspy.atoms.neighbors import (
    NUMBA_AVAILABLE,
    _build_nl_ortho_numba,
    _build_nl_tri_numba,
    _cell_offsets,
    _clamp_cell_count,
    _dist_and_vec_ortho,
    _dist_and_vec_tri,
    _dist_and_vec_tri_exact,
    _estimate_max_neighbors,
    _extract_atom_ids,
    _flatten_distance_buffers,
    _get_pair_cutoff_sq_python,
    _grow_until_fits,
    _half_min_height,
    _image_search_bound,
    _lookup_cutoff_sq,
    _min_periodic_lattice_vector,
    _normalize_type_filter,
    _pad_nonperiodic,
    _parse_cutoff,
    _stencil_grid,
    build_distances,
    cell_perpendicular_heights,
    compute_cell_list_orthogonal,
    compute_cell_list_triclinic,
    get_neighbors,
)
from ase import Atoms
from ase.build import molecule
from ase.geometry import find_mic
from ase.io import read
from ase.neighborlist import neighbor_list

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


def _triclinic_atoms_4() -> Atoms:
    """Four atoms in a triclinic box: two O-Si pairs ~1 Å apart, well-separated from each other.

    Cell has a shear component so the orthogonal path is not taken.
    Atoms: O(8) at 2.0 along x, Si(14) at 3.0 along x (pair 1);
           O(8) at 5.5 along x, Si(14) at 6.5 along x (pair 2).
    All y/z are the same Cartesian coordinate so pairs are clearly separated.
    """
    cell = np.array([[8.0, 0.0, 0.0], [2.0, 8.0, 0.0], [0.0, 0.0, 8.0]])
    coords = np.array(
        [[2.0, 4.0, 4.0], [3.0, 4.0, 4.0], [5.5, 4.0, 4.0], [6.5, 4.0, 4.0]],
        dtype=np.float64,
    )
    types = np.array([8, 14, 8, 14], dtype=np.int32)
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
    original_neighbors = sys.modules.pop("amorphouspy.atoms.neighbors", None)

    try:
        # Pretend numba is not installed
        sys.modules["numba"] = None  # type: ignore[assignment]
        mod = importlib.import_module("amorphouspy.atoms.neighbors")
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
        sys.modules.pop("amorphouspy.atoms.neighbors", None)
        if original_neighbors is not None:
            sys.modules["amorphouspy.atoms.neighbors"] = original_neighbors


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
    stencil_start, stencil_count = _cell_offsets(n_cells)
    max_neighbors = _estimate_max_neighbors(coords, cell, max_cutoff)
    return {
        "coords": coords,
        "types": types,
        "box_size": box_size,
        "atom_cells": atom_cells,
        "n_cells": n_cells,
        "cell_start": cell_start,
        "cell_atoms": cell_atoms,
        "stencil_start": stencil_start,
        "stencil_count": stencil_count,
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
    stencil_start, stencil_count = _cell_offsets(n_cells)
    max_neighbors = _estimate_max_neighbors(coords, cell, max_cutoff)
    return {
        "coords_frac": coords_frac,
        "types": types,
        "cell": cell,
        "atom_cells": atom_cells,
        "n_cells": n_cells,
        "cell_start": cell_start,
        "cell_atoms": cell_atoms,
        "stencil_start": stencil_start,
        "stencil_count": stencil_count,
        "cutoff_sq": cutoff_sq,
        "max_neighbors": max_neighbors,
        "pair_types": pair_types,
        "pair_cutoffs_sq": pair_cutoffs_sq,
        "use_pair_cutoffs": use_pair_cutoffs,
        "half_height_sq": _half_min_height(cell) ** 2,
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
# _grow_until_fits overflow loop
# ---------------------------------------------------------------------------


def test_grow_until_fits_retries_with_a_larger_buffer() -> None:
    """When counts exceed max_neighbors the kernel is re-run with max_neighbors = int(1.2 * overflow) + 1."""
    n_atoms = 4
    calls: list[int] = []

    def mock_build_fn(**kwargs):
        max_neighbors = kwargs["max_neighbors"]
        calls.append(max_neighbors)
        # The kernels report the true count whatever the buffer size.
        counts = np.full(n_atoms, 3, dtype=np.int32)
        nl = np.zeros((n_atoms, max_neighbors), dtype=np.int32)
        vecs = np.zeros((n_atoms, max_neighbors, 3), dtype=np.float32)
        return nl, counts, vecs

    build_kwargs = {"max_neighbors": 1}
    neighbor_list, counts, _vecs = _grow_until_fits(mock_build_fn, build_kwargs)

    assert calls == [1, 4]
    assert neighbor_list.shape == (n_atoms, 4)
    assert build_kwargs["max_neighbors"] == 4
    np.testing.assert_array_equal(counts, 3)


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


# ---------------------------------------------------------------------------
# _flatten_distance_buffers
# ---------------------------------------------------------------------------


def test_flatten_distance_buffers_basic() -> None:
    """_flatten_distance_buffers unpacks (N, max_pairs) buffers into flat arrays."""
    dist_buf = np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float64)
    j_buf = np.array([[5, 7, 0], [9, 0, 0]], dtype=np.int32)
    counts = np.array([2, 1], dtype=np.int32)

    dist_out, i_out, j_out = _flatten_distance_buffers(dist_buf, j_buf, counts)

    assert len(dist_out) == 3
    assert len(i_out) == 3
    assert len(j_out) == 3
    np.testing.assert_array_equal(i_out, [0, 0, 1])
    np.testing.assert_array_equal(j_out, [5, 7, 9])
    np.testing.assert_allclose(dist_out, [1.0, 2.0, 3.0])


def test_flatten_distance_buffers_empty() -> None:
    """_flatten_distance_buffers returns empty arrays when counts are all zero."""
    dist_buf = np.zeros((3, 5), dtype=np.float64)
    j_buf = np.zeros((3, 5), dtype=np.int32)
    counts = np.zeros(3, dtype=np.int32)

    dist_out, i_out, j_out = _flatten_distance_buffers(dist_buf, j_buf, counts)

    assert len(dist_out) == 0
    assert len(i_out) == 0
    assert len(j_out) == 0


# ---------------------------------------------------------------------------
# build_distances — orthogonal box
# ---------------------------------------------------------------------------


def test_build_distances_ortho_all_pairs() -> None:
    """build_distances returns exactly the expected half-pairs within r_max."""
    atoms = _ortho_atoms()
    atoms.wrap()
    # At r_max=1.5: only the two O-Si pairs at 1 Å are within range.
    # Pairs at 2 Å and 3 Å must be absent.
    dists, i_idx, j_idx = build_distances(atoms, r_max=1.5)

    assert dists.dtype == np.float64
    assert i_idx.dtype == np.int32
    assert j_idx.dtype == np.int32
    assert np.all(j_idx > i_idx)
    assert len(dists) == 2
    assert set(zip(i_idx.tolist(), j_idx.tolist(), strict=False)) == {(0, 1), (2, 3)}
    assert np.allclose(sorted(dists), [1.0, 1.0], atol=1e-10)


def test_build_distances_ortho_type_filter() -> None:
    """build_distances type filter suppresses same-species pairs that are in range."""
    atoms = _ortho_atoms()
    atoms.wrap()
    types = atoms.get_atomic_numbers()
    # r_max=3.5 puts O-O (3 Å) and Si-Si (3 Å) within range; the filter must drop them.
    # Expected O-Si pairs within 3.5 Å: (0,1)@1Å, (0,3)@2Å, (1,2)@2Å, (2,3)@1Å — 4 pairs.
    dists, i_idx, j_idx = build_distances(atoms, r_max=3.5, types=types, unordered_pairs=[(8, 14)])

    assert len(dists) == 4
    for i, j in zip(i_idx, j_idx, strict=False):
        pair = tuple(sorted([types[i], types[j]]))
        assert pair == (8, 14)


def test_build_distances_ortho_no_duplicate_pairs_small_box() -> None:
    """No (i, j) pair appears more than once even when the box is smaller than 2*r_max (n_cells=1)."""
    atoms = _ortho_atoms()
    atoms.wrap()
    # r_max=4.0 on a 6 Å box → n_cells=1, so all 27 cell-neighbor offsets wrap to the same cell.
    _dists, i_idx, j_idx = build_distances(atoms, r_max=4.0)

    pairs = list(zip(i_idx.tolist(), j_idx.tolist(), strict=False))
    assert len(pairs) == len(set(pairs)), f"duplicate pairs: {pairs}"


def test_build_distances_ortho_no_pairs_in_range() -> None:
    """build_distances with a very small r_max returns empty arrays."""
    atoms = _ortho_atoms()
    atoms.wrap()
    dists, _i_idx, _j_idx = build_distances(atoms, r_max=0.1)

    assert len(dists) == 0


def test_build_distances_ortho_matches_brute_force() -> None:
    """build_distances distances match direct pairwise computation."""
    atoms = _ortho_atoms()
    atoms.wrap()
    r_max = 1.5
    dists, i_idx, j_idx = build_distances(atoms, r_max=r_max)

    coords = atoms.get_positions()
    cell = atoms.get_cell().array
    box = np.diag(cell)
    for d, i, j in zip(dists, i_idx, j_idx, strict=False):
        diff = coords[i] - coords[j]
        diff -= box * np.round(diff / box)
        expected = float(np.linalg.norm(diff))
        assert d == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# build_distances — triclinic box
# ---------------------------------------------------------------------------


def test_build_distances_triclinic_matches_brute_force() -> None:
    """build_distances distances match direct pairwise computation for a triclinic cell."""
    atoms = _triclinic_atoms_4()
    atoms.wrap()
    r_max = 1.5
    dists, i_idx, j_idx = build_distances(atoms, r_max=r_max)

    assert len(dists) >= 2
    assert np.all(j_idx > i_idx)

    coords = atoms.get_positions()
    cell = atoms.get_cell().array
    # Ground truth from ASE, which is independent of this module's conventions
    for d, i, j in zip(dists, i_idx, j_idx, strict=False):
        expected = float(find_mic(coords[i] - coords[j], cell)[1])
        assert d == pytest.approx(expected, abs=1e-10)


def test_build_distances_triclinic_type_filter_excludes_same_species() -> None:
    """build_distances type filter for triclinic cells excludes O-O and Si-Si pairs."""
    atoms = _triclinic_atoms_4()
    atoms.wrap()
    types = atoms.get_atomic_numbers()
    # With a wide enough cutoff, O-O pairs at ~4 Å would appear without the filter
    dists, i_idx, j_idx = build_distances(atoms, r_max=5.0, types=types, unordered_pairs=[(8, 14)])

    assert len(dists) > 0
    for i, j in zip(i_idx, j_idx, strict=False):
        pair = tuple(sorted([types[i], types[j]]))
        assert pair == (8, 14)


def test_build_distances_triclinic_no_duplicate_pairs_small_box() -> None:
    """No (i, j) pair appears more than once for a triclinic box smaller than 2*r_max (n_cells=1)."""
    atoms = _triclinic_atoms_4()
    atoms.wrap()
    # r_max=5.0 on an 8 Å box → n_cells=1 in every dimension, all 27 offsets map to same cell.
    _dists, i_idx, j_idx = build_distances(atoms, r_max=5.0)

    pairs = list(zip(i_idx.tolist(), j_idx.tolist(), strict=False))
    assert len(pairs) == len(set(pairs)), f"duplicate pairs found: {pairs}"


# ---------------------------------------------------------------------------
# Conditional vector_list allocation in _build_nl_ortho_numba / _build_nl_tri_numba
# ---------------------------------------------------------------------------


def test_build_nl_ortho_numba_no_vectors_shape() -> None:
    """With return_vectors=False the kernel returns a (0,0,3) vector_list, not a full array."""
    fn = _build_nl_ortho_numba.py_func if NUMBA_AVAILABLE else _build_nl_ortho_numba
    atoms = _ortho_atoms()
    kwargs = _make_ortho_nl_inputs(atoms, cutoff=1.5)
    kwargs["target_types"] = np.empty(0, dtype=np.int32)
    kwargs["neighbor_types"] = np.empty(0, dtype=np.int32)
    kwargs["use_target_filter"] = False
    kwargs["use_neighbor_filter"] = False
    kwargs["return_vectors"] = False

    _nl, _counts, vecs = fn(**kwargs)
    assert vecs.shape == (0, 0, 3), f"expected (0,0,3), got {vecs.shape}"


def test_build_nl_tri_numba_no_vectors_shape() -> None:
    """With return_vectors=False the triclinic kernel returns a (0,0,3) vector_list."""
    fn = _build_nl_tri_numba.py_func if NUMBA_AVAILABLE else _build_nl_tri_numba
    atoms = _triclinic_atoms()
    kwargs = _make_tri_nl_inputs(atoms, cutoff=1.5)
    kwargs["target_types"] = np.empty(0, dtype=np.int32)
    kwargs["neighbor_types"] = np.empty(0, dtype=np.int32)
    kwargs["use_target_filter"] = False
    kwargs["use_neighbor_filter"] = False
    kwargs["return_vectors"] = False

    _nl, _counts, vecs = fn(**kwargs)
    assert vecs.shape == (0, 0, 3), f"expected (0,0,3), got {vecs.shape}"


# ---------------------------------------------------------------------------
# _numpy_fallback cutoff_matrix — vectorized per-pair lookup matches Numba path
# ---------------------------------------------------------------------------


def test_numpy_fallback_pair_cutoffs_match_numba() -> None:
    """NumPy fallback with per-pair cutoffs returns the same neighbor sets as the Numba path."""
    atoms = _ortho_atoms()
    cutoff = {(8, 14): 1.5, (8, 8): 0.5, (14, 14): 0.5}

    result_numba = get_neighbors(atoms, cutoff=cutoff, use_numba=True)
    result_numpy = get_neighbors(atoms, cutoff=cutoff, use_numba=False)

    assert len(result_numba) == len(result_numpy)
    for (cid_n, nn_n), (cid_p, nn_p) in zip(result_numba, result_numpy, strict=False):
        assert cid_n == cid_p
        assert sorted(nn_n) == sorted(nn_p), f"atom {cid_n}: numba={sorted(nn_n)}, numpy={sorted(nn_p)}"


def test_numpy_fallback_pair_cutoffs_tighter_cutoff_excludes_pairs() -> None:
    """A tight per-pair cutoff in the NumPy path correctly excludes pairs beyond that distance.

    O-Si pairs are at 1 Å and 2 Å in _ortho_atoms. Setting cutoff (8,14)=1.2 should keep
    only the 1 Å pairs and drop the 2 Å ones, while a wider global cutoff would include both.
    This verifies the cutoff_matrix lookup actually applies per-pair values, not the global max.
    """
    atoms = _ortho_atoms()
    # global max would be 2.0; per-pair (8,14) is tighter at 1.2
    cutoff = {(8, 14): 1.2, (8, 8): 2.0, (14, 14): 2.0}

    result = get_neighbors(atoms, cutoff=cutoff, use_numba=False)
    types = atoms.get_atomic_numbers()

    for cid, nn_ids in result:
        idx = cid - 1
        for nid in nn_ids:
            jdx = nid - 1
            if types[idx] in (8, 14) and types[jdx] in (8, 14) and types[idx] != types[jdx]:
                # O-Si pair must be within 1.2 Å — the 2 Å pairs should be absent
                coords = atoms.get_positions()
                box = np.diag(atoms.get_cell().array)
                diff = coords[idx] - coords[jdx]
                diff -= box * np.round(diff / box)
                assert np.linalg.norm(diff) <= 1.2 + 1e-9


# ---------------------------------------------------------------------------
# Regression: duplicate neighbours when a box edge spans fewer than 3 cells
# ---------------------------------------------------------------------------


def _glass_300() -> Atoms:
    """The 300-atom SiO2 glass fixture, wrapped, as an orthogonal 16.4876 Å cube."""
    return read(Path(__file__).parent / "data" / "SiO2_glass_300_atoms.xyz")


@pytest.mark.parametrize("cutoff", [2.0, 5.0, 5.5, 6.0, 9.0])
@pytest.mark.parametrize("use_numba", [True, False])
def test_get_neighbors_no_duplicates_across_cell_counts(cutoff: float, use_numba: bool) -> None:  # noqa: FBT001
    """No atom is reported twice, including when n_cells drops to 2 or 1.

    n_cells = max(1, floor(height / cutoff)), so a 16.4876 Å box gives n_cells of
    8, 3, 2, 2 and 1 for these cutoffs. Below 3 several of the 27 offsets wrap
    onto the same cell, which used to append every atom in it again.
    """
    atoms = _glass_300()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = get_neighbors(atoms, cutoff=cutoff, use_numba=use_numba)
    for central_id, nn_ids in result:
        assert len(nn_ids) == len(set(nn_ids)), f"atom {central_id} has duplicate neighbours at cutoff {cutoff}"


def test_stencil_grid_visits_each_cell_once() -> None:
    """_stencil_grid yields each wrapped cell once, not 27 raw offsets."""
    n_cells = np.array([1, 2, 3], dtype=np.int32)
    cells = (np.array([0, 0, 0], dtype=np.int32) + _stencil_grid(n_cells)) % n_cells
    assert len(cells) == len({tuple(c) for c in cells})
    # 1 * 2 * 3 distinct cells exist in total, and all are within one offset of the origin
    assert len(cells) == 6


def _wrapped_cells_with_dedup(cell_idx: tuple[int, ...], n_cells: tuple[int, ...]) -> set[tuple[int, ...]]:
    """Reference stencil: the 27 raw offsets, wrapped, then deduplicated."""
    return {
        tuple(((np.array(cell_idx) + np.array(offset)) % np.array(n_cells)).tolist())
        for offset in itertools.product((-1, 0, 1), repeat=3)
    }


def test_cell_offsets_visit_each_cell_exactly_once() -> None:
    """The restricted per-dimension ranges enumerate exactly the deduplicated 27-offset stencil, never twice."""
    for n_cells in itertools.product(range(1, 6), repeat=3):
        stencil = _stencil_grid(np.array(n_cells, dtype=np.int32))
        for cell_idx in itertools.product(*(range(n) for n in n_cells)):
            visited = [tuple(c) for c in ((np.array(cell_idx) + stencil) % np.array(n_cells)).tolist()]
            assert len(visited) == len(set(visited)), (n_cells, cell_idx)
            assert set(visited) == _wrapped_cells_with_dedup(cell_idx, n_cells), (n_cells, cell_idx)


def test_clamp_cell_count_caps_total_and_keeps_each_axis_positive() -> None:
    """A grid past the cap is halved until it fits; small grids pass through unchanged."""
    clamped = _clamp_cell_count(np.array([2000, 2000, 2000]))
    assert int(np.prod(clamped, dtype=np.int64)) <= 1_000_000
    assert np.all(clamped >= 1)
    np.testing.assert_array_equal(_clamp_cell_count(np.array([8, 3, 2])), [8, 3, 2])


# ---------------------------------------------------------------------------
# Regression: fractional-coordinate handedness (ASE stores lattice vectors as rows)
# ---------------------------------------------------------------------------


def test_compute_cell_list_triclinic_frac_roundtrip() -> None:
    """coords_frac @ cell must reproduce the Cartesian coordinates.

    Guards against computing fractional coordinates as inv(cell) @ r, which is
    the column-vector convention and silently distorts every non-orthogonal cell.
    """
    cell = np.array([[8.0, 0.0, 0.0], [3.0, 8.0, 0.0], [2.0, 1.5, 8.0]])
    rng = np.random.default_rng(11)
    coords = rng.random((25, 3)) @ cell  # inside the cell by construction
    coords_frac, _atom_cells, _n_cells, _cell_start, _order = compute_cell_list_triclinic(coords, cell, 2.0)
    assert np.allclose(coords_frac @ cell, coords, atol=1e-10)


def test_compute_cell_list_triclinic_frac_matches_ase() -> None:
    """Fractional coordinates agree with ASE's own scaled positions."""
    cell = np.array([[8.0, 0.0, 0.0], [3.0, 8.0, 0.0], [2.0, 1.5, 8.0]])
    rng = np.random.default_rng(12)
    coords = rng.random((25, 3)) @ cell
    atoms = Atoms(numbers=np.full(25, 14), positions=coords, cell=cell, pbc=True)
    coords_frac, *_ = compute_cell_list_triclinic(coords, cell, 2.0)
    assert np.allclose(coords_frac, atoms.get_scaled_positions() % 1.0, atol=1e-10)


@pytest.mark.parametrize("shear", [0.0, 0.03, 0.12, 0.30])
@pytest.mark.parametrize("cutoff", [2.0, 3.5])
@pytest.mark.parametrize("use_numba", [True, False])
def test_get_neighbors_matches_ase_under_shear(shear: float, cutoff: float, use_numba: bool) -> None:  # noqa: FBT001
    """Neighbour pairs match ase.neighborlist exactly for sheared (triclinic) cells."""
    atoms = _glass_300()
    box_length = atoms.get_cell().array[0, 0]
    cell = atoms.get_cell().array.copy()
    cell[1, 0] = shear * box_length
    cell[2, 0] = 0.5 * shear * box_length
    atoms.set_cell(cell, scale_atoms=False)
    atoms.set_pbc(True)
    atoms.wrap()

    i_ase, j_ase = neighbor_list("ij", atoms, cutoff)
    expected = set(zip(i_ase.tolist(), j_ase.tolist(), strict=False))

    # get_neighbors returns 1-based IDs for a file without an explicit id column
    obtained = {
        (central_id - 1, neighbor_id - 1)
        for central_id, nn_ids in get_neighbors(atoms, cutoff=cutoff, use_numba=use_numba)
        for neighbor_id in nn_ids
    }
    assert obtained == expected


def test_build_distances_matches_ase_under_shear() -> None:
    """build_distances half-pairs match ase.neighborlist for a sheared cell."""
    atoms = _glass_300()
    cell = atoms.get_cell().array.copy()
    cell[1, 0] = 2.0
    atoms.set_cell(cell, scale_atoms=False)
    atoms.set_pbc(True)
    atoms.wrap()

    dists, i_idx, j_idx = build_distances(atoms, r_max=3.5)
    obtained = {(int(i), int(j)) for i, j in zip(i_idx, j_idx, strict=False)}

    i_ase, j_ase = neighbor_list("ij", atoms, 3.5)
    expected = {(int(i), int(j)) for i, j in zip(i_ase, j_ase, strict=False) if j > i}
    assert obtained == expected

    coords = atoms.get_positions()
    for d, i, j in zip(dists, i_idx, j_idx, strict=False):
        assert d == pytest.approx(float(find_mic(coords[i] - coords[j], cell)[1]), abs=1e-9)


# ---------------------------------------------------------------------------
# _dist_and_vec_tri_exact — guarded minimum-image search
# ---------------------------------------------------------------------------


def test_dist_and_vec_tri_exact_fast_path_matches_rounding() -> None:
    """Below half the smallest perpendicular height the rounded image is returned unchanged."""
    fast = _dist_and_vec_tri.py_func if NUMBA_AVAILABLE else _dist_and_vec_tri
    exact = _dist_and_vec_tri_exact.py_func if NUMBA_AVAILABLE else _dist_and_vec_tri_exact
    cell = np.array([[10.0, 0.0, 0.0], [1.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    half_height_sq = _half_min_height(cell) ** 2
    frac_i = np.array([0.10, 0.10, 0.10])
    frac_j = np.array([0.15, 0.10, 0.10])
    assert exact(frac_i, frac_j, cell, half_height_sq) == fast(frac_i, frac_j, cell)


def test_dist_and_vec_tri_exact_beats_rounding_on_skewed_cell() -> None:
    """On a strongly skewed cell the search finds a closer image than per-component rounding."""
    fast = _dist_and_vec_tri.py_func if NUMBA_AVAILABLE else _dist_and_vec_tri
    exact = _dist_and_vec_tri_exact.py_func if NUMBA_AVAILABLE else _dist_and_vec_tri_exact
    cell = np.array([[10.0, 0.0, 0.0], [7.0, 10.0, 0.0], [7.0, 7.0, 10.0]])
    half_height_sq = _half_min_height(cell) ** 2

    rng = np.random.default_rng(0)
    n_improved = 0
    for _ in range(500):
        frac_i, frac_j = rng.random(3), rng.random(3)
        *_, dist_sq_fast = fast(frac_i, frac_j, cell)
        *_, dist_sq_exact = exact(frac_i, frac_j, cell, half_height_sq)
        assert dist_sq_exact <= dist_sq_fast + 1e-12
        # ASE's find_mic is the independent ground truth
        reference = float(find_mic((frac_i - frac_j) @ cell, cell)[1])
        assert np.sqrt(dist_sq_exact) == pytest.approx(reference, abs=1e-9)
        if dist_sq_exact < dist_sq_fast - 1e-9:
            n_improved += 1
    assert n_improved > 0, "skewed cell should expose cases where rounding is not the nearest image"


# ---------------------------------------------------------------------------
# Input validation at the public boundary
# ---------------------------------------------------------------------------


def test_parse_cutoff_empty_dict_raises() -> None:
    """An empty cutoff dict names the argument instead of failing inside max()."""
    types = np.array([8, 14], dtype=np.int32)
    with pytest.raises(ValueError, match="empty dict"):
        _parse_cutoff({}, types)


@pytest.mark.parametrize("bad_cutoff", [0.0, -1.0])
def test_parse_cutoff_non_positive_scalar_raises(bad_cutoff: float) -> None:
    """A non-positive scalar cutoff is rejected."""
    types = np.array([8, 14], dtype=np.int32)
    with pytest.raises(ValueError, match="positive distance"):
        _parse_cutoff(bad_cutoff, types)


def test_parse_cutoff_non_positive_pair_marks_pair_excluded() -> None:
    """A non-positive per-pair cutoff means "never bonded", the convention generate_bond_length_dict uses.

    Squaring -1.0 would turn it into a 1.0 A cutoff, which only looks like "no bond"
    because no real pair sits that close, so the encoded value must stay negative.
    """
    types = np.array([8, 14], dtype=np.int32)
    max_cutoff, pair_types, pair_cutoffs_sq, use_pair_cutoffs = _parse_cutoff({(14, 8): 1.8, (8, 8): -1.0}, types)

    assert use_pair_cutoffs is True
    assert max_cutoff == pytest.approx(1.8)
    rows = [tuple(row) for row in pair_types]
    assert pair_cutoffs_sq[rows.index((8, 8))] < 0.0
    assert pair_cutoffs_sq[rows.index((14, 8))] == pytest.approx(1.8**2)


def test_parse_cutoff_all_pairs_excluded_raises() -> None:
    """A dict with no positive cutoff can never produce a neighbour, so it is rejected."""
    types = np.array([8, 14], dtype=np.int32)
    with pytest.raises(ValueError, match="no pair in the cutoff dict has a positive distance"):
        _parse_cutoff({(14, 8): -1.0, (8, 8): -1.0}, types)


@pytest.mark.parametrize("use_numba", [True, False])
def test_excluded_pair_never_bonds_even_when_atoms_overlap(use_numba: bool) -> None:  # noqa: FBT001
    """An excluded pair stays unbonded at any separation, including below the 1 A the old squaring implied."""
    atoms = Atoms(
        numbers=[14, 8, 8],
        positions=[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [5.5, 5.0, 5.0]],
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=True,
    )
    cutoff = {(14, 8): 1.8, (8, 8): -1.0, (14, 14): -1.0}
    neighbours = {central: sorted(nn) for central, nn in get_neighbors(atoms, cutoff, use_numba=use_numba)}
    assert neighbours == {1: [], 2: [], 3: []}


def test_normalize_type_filter_accepts_scalar() -> None:
    """A bare atomic number is wrapped into a one-element list."""
    assert _normalize_type_filter(14, "target_types") == [14]
    assert _normalize_type_filter(np.int32(8), "target_types") == [8]
    assert _normalize_type_filter([8, 14], "target_types") == [8, 14]
    assert _normalize_type_filter(None, "target_types") is None


def test_normalize_type_filter_rejects_string() -> None:
    """A non-integer type filter raises a TypeError naming the argument."""
    with pytest.raises(TypeError, match="neighbor_types"):
        _normalize_type_filter("Si", "neighbor_types")


def test_get_neighbors_accepts_scalar_type_filters() -> None:
    """target_types=14 behaves exactly like target_types=[14]."""
    atoms = _ortho_atoms()
    scalar = {cid: sorted(nn) for cid, nn in get_neighbors(atoms, cutoff=1.5, target_types=14, neighbor_types=8)}
    listed = {cid: sorted(nn) for cid, nn in get_neighbors(atoms, cutoff=1.5, target_types=[14], neighbor_types=[8])}
    assert scalar == listed


def test_get_neighbors_warns_when_cutoff_exceeds_minimum_image() -> None:
    """A cutoff beyond half the shortest lattice vector warns instead of silently truncating."""
    atoms = _ortho_atoms()  # 6 Å cube → half the shortest lattice vector is 3 Å
    with pytest.warns(RuntimeWarning, match="minimum-image"):
        get_neighbors(atoms, cutoff=4.0)


def test_get_neighbors_does_not_warn_within_minimum_image() -> None:
    """No warning while the cutoff stays within half the shortest lattice vector."""
    atoms = _ortho_atoms()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        get_neighbors(atoms, cutoff=2.5)


# ---------------------------------------------------------------------------
# _image_search_bound — when the exact-MIC guard is compiled in at all
# ---------------------------------------------------------------------------


def test_image_search_bound_none_for_fine_grid() -> None:
    """A grid fine enough that no candidate pair can beat rounding needs no guard."""
    cell = np.diag([49.5, 49.5, 49.5])
    assert _image_search_bound(cell, np.array([14, 14, 14])) is None


def test_image_search_bound_set_for_coarse_grid() -> None:
    """With few cells per edge a candidate can exceed the bound, so the guard is required."""
    cell = np.diag([49.5, 49.5, 49.5])
    bound = _image_search_bound(cell, np.array([2, 2, 2]))
    assert bound is not None
    assert bound == pytest.approx(_half_min_height(cell) ** 2)


def test_image_search_bound_set_for_skewed_cell() -> None:
    """A strongly skewed cell keeps the guard even on a moderately fine grid."""
    cell = np.array([[10.0, 0.0, 0.0], [7.0, 10.0, 0.0], [7.0, 7.0, 10.0]])
    assert _image_search_bound(cell, np.array([4, 4, 4])) is not None


@pytest.mark.parametrize("shear", [0.05, 0.30, 0.70])
@pytest.mark.parametrize("cutoff", [2.0, 3.5, 6.0])
def test_image_search_pruning_does_not_change_results(shear: float, cutoff: float) -> None:
    """Pruning the guard must never change a neighbour list.

    Runs the triclinic kernel twice — once with whatever _image_search_bound
    decides, once with the guard forced on — and requires identical output.
    """
    atoms = _glass_300()
    cell = atoms.get_cell().array.copy()
    cell[1, 0] = shear * cell[0, 0]
    cell[2, 1] = 0.5 * shear * cell[1, 1]
    atoms.set_cell(cell, scale_atoms=False)
    atoms.set_pbc(True)
    atoms.wrap()

    coords = atoms.get_positions()
    types = atoms.get_atomic_numbers().astype(np.int32)
    coords_frac, atom_cells, n_cells, cell_start, cell_atoms = compute_cell_list_triclinic(coords, cell, cutoff)
    stencil_start, stencil_count = _cell_offsets(n_cells)

    fn = _build_nl_tri_numba
    common = {
        "coords_frac": coords_frac,
        "types": types,
        "cell": cell,
        "atom_cells": atom_cells,
        "n_cells": n_cells,
        "cell_start": cell_start,
        "cell_atoms": cell_atoms,
        "stencil_start": stencil_start,
        "stencil_count": stencil_count,
        "cutoff_sq": cutoff**2,
        "target_types": np.empty(0, dtype=np.int32),
        "neighbor_types": np.empty(0, dtype=np.int32),
        "use_target_filter": False,
        "use_neighbor_filter": False,
        "max_neighbors": _estimate_max_neighbors(coords, cell, cutoff),
        "pair_types": np.empty((0, 2), dtype=np.int32),
        "pair_cutoffs_sq": np.empty(0, dtype=np.float64),
        "use_pair_cutoffs": False,
        "return_vectors": True,
    }
    nl_auto, counts_auto, vecs_auto = fn(**common, half_height_sq=_image_search_bound(cell, n_cells))
    nl_forced, counts_forced, vecs_forced = fn(**common, half_height_sq=_half_min_height(cell) ** 2)

    np.testing.assert_array_equal(counts_auto, counts_forced)
    # The kernel buffers are uninitialised past each atom's count; compare only the filled slots.
    filled = np.arange(nl_auto.shape[1])[np.newaxis, :] < counts_auto[:, np.newaxis]
    np.testing.assert_array_equal(nl_auto[filled], nl_forced[filled])
    np.testing.assert_allclose(vecs_auto[filled], vecs_forced[filled], atol=1e-12)


# ---------------------------------------------------------------------------
# pbc handling — non-periodic directions via box padding, ASE as ground truth
# ---------------------------------------------------------------------------

_PBC_CELLS = {
    "orthogonal": np.diag([16.0, 16.0, 16.0]),
    "shear_b": np.array([[16.0, 0.0, 0.0], [4.0, 16.0, 0.0], [0.0, 0.0, 16.0]]),
    "shear_c": np.array([[16.0, 0.0, 0.0], [0.0, 16.0, 0.0], [5.0, 4.0, 16.0]]),
    "skewed": np.array([[16.0, 0.0, 0.0], [9.0, 14.0, 0.0], [8.0, 7.0, 13.0]]),
}
_ALL_PBC = list(itertools.product([True, False], repeat=3))


def _pbc_id(pbc: tuple[bool, ...]) -> str:
    return "".join("T" if flag else "F" for flag in pbc)


def _random_atoms(cell: np.ndarray, pbc, n_atoms: int = 200, seed: int = 0) -> Atoms:
    """Random Si/O atoms filling the cell, with the requested pbc flags."""
    rng = np.random.default_rng(seed)
    positions = rng.random((n_atoms, 3)) @ cell
    numbers = np.resize([14, 8, 8], n_atoms)
    return Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)


def _pairs_ase(atoms: Atoms, cutoff: float) -> set[tuple[int, int]]:
    """Ordered (i, j) pairs within cutoff from ASE, which honours pbc."""
    i_ase, j_ase = neighbor_list("ij", atoms, cutoff)
    return set(zip(i_ase.tolist(), j_ase.tolist(), strict=True))


def _pairs_ours(atoms: Atoms, cutoff, use_numba: bool = True) -> set[tuple[int, int]]:  # noqa: FBT001
    """Ordered (i, j) pairs from get_neighbors, converted from 1-based ids to 0-based indices."""
    return {
        (central_id - 1, neighbor_id - 1)
        for central_id, nn_ids in get_neighbors(atoms, cutoff=cutoff, use_numba=use_numba)
        for neighbor_id in nn_ids
    }


@pytest.mark.parametrize("pbc", _ALL_PBC, ids=_pbc_id)
@pytest.mark.parametrize("cell_name", list(_PBC_CELLS))
@pytest.mark.parametrize("use_numba", [True, False])
def test_get_neighbors_honours_pbc(pbc: tuple[bool, ...], cell_name: str, use_numba: bool) -> None:  # noqa: FBT001
    """Every pbc combination matches ASE exactly; the flags index lattice vectors, not Cartesian axes."""
    atoms = _random_atoms(_PBC_CELLS[cell_name], pbc)
    assert _pairs_ours(atoms, 3.2, use_numba) == _pairs_ase(atoms, 3.2)


def test_slab_in_registry_has_no_cross_vacuum_bonds() -> None:
    """Padding must be 2 * cutoff: with 1 * cutoff the surface layers of this slab bond across the vacuum.

    Two 4 x 4 layers exactly 7 Å apart in a cell exactly 7 Å tall. With
    pbc=[T, T, F] the layers must not see each other, yet the wrapped image of
    the top layer sits exactly one cutoff (3.5 Å) below the bottom layer unless
    the padding leaves 2 * cutoff of clearance — and the kernel accepts
    dist <= cutoff.
    """
    grid = np.arange(4) * 2.5
    thickness = 7.0
    positions = np.array([[x, y, z] for x in grid for y in grid for z in (0.0, thickness)])
    atoms = Atoms("H32", positions=positions, cell=np.diag([10.0, 10.0, thickness]), pbc=[True, True, False])
    pairs = _pairs_ours(atoms, 3.5)
    assert pairs == _pairs_ase(atoms, 3.5)
    assert all(positions[i, 2] == positions[j, 2] for i, j in pairs), "a pair crosses the vacuum"


def test_slab_with_zero_length_nonperiodic_vector() -> None:
    """A zero-length c vector (ASE's usual slab setup) is completed instead of crashing inside numba."""
    grid = np.arange(4) * 2.5
    positions = np.array([[x, y, 0.0] for x in grid for y in grid])
    cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.0]])
    atoms = Atoms("H16", positions=positions, cell=cell, pbc=[True, True, False])
    expected = _pairs_ase(atoms, 3.0)
    assert _pairs_ours(atoms, 3.0) == expected
    _dists, i_idx, j_idx = build_distances(atoms, r_max=3.0)
    assert {(int(i), int(j)) for i, j in zip(i_idx, j_idx, strict=True)} == {(i, j) for i, j in expected if j > i}


def test_isolated_molecule_without_cell() -> None:
    """A bare molecule (zero cell, pbc=False) finds its bonds instead of crashing or returning nothing."""
    benzene = molecule("C6H6")
    pairs = _pairs_ours(benzene, 1.6)
    assert pairs == _pairs_ase(benzene, 1.6)
    assert len(pairs) == 24  # 6 C-C + 6 C-H bonds, both directions


def test_fully_periodic_degenerate_cell_raises() -> None:
    """A zero-volume cell that claims to be periodic is rejected with a ValueError, not a numba SystemError."""
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], cell=np.diag([10.0, 10.0, 0.0]), pbc=True)
    with pytest.raises(ValueError, match="degenerate"):
        get_neighbors(atoms, cutoff=2.0)
    with pytest.raises(ValueError, match="degenerate"):
        build_distances(atoms, r_max=2.0)


def test_nan_coordinates_raise() -> None:
    """A NaN coordinate is rejected up front; under fastmath it would silently vanish from the kernels."""
    positions = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    atoms = Atoms("H2", positions=positions, cell=np.diag([10.0, 10.0, 10.0]), pbc=True)
    with pytest.raises(ValueError, match="NaN"):
        get_neighbors(atoms, cutoff=2.0)
    with pytest.raises(ValueError, match="NaN"):
        build_distances(atoms, r_max=2.0)


@pytest.mark.parametrize("cell_name", ["orthogonal", "shear_c"])
@pytest.mark.parametrize("use_numba", [True, False])
def test_atoms_outside_box_along_nonperiodic_axis(cell_name: str, use_numba: bool) -> None:  # noqa: FBT001
    """Unwrapped coordinates far outside the cell along a non-periodic vector are handled exactly."""
    cell = _PBC_CELLS[cell_name]
    atoms = _random_atoms(cell, [True, True, False], seed=3)
    positions = atoms.get_positions()
    positions[::3] -= 2.5 * cell[2]
    positions[1::3] += 2.5 * cell[2]
    atoms.set_positions(positions)
    assert _pairs_ours(atoms, 3.2, use_numba) == _pairs_ase(atoms, 3.2)


def test_monolayer_thinner_than_cutoff() -> None:
    """A single layer in a cell far thinner than the cutoff is padded out, not wrapped onto itself."""
    grid = np.arange(6) * 2.0
    positions = np.array([[x, y, 0.2] for x in grid for y in grid])
    atoms = Atoms("H36", positions=positions, cell=np.diag([12.0, 12.0, 0.5]), pbc=[True, True, False])
    assert _pairs_ours(atoms, 3.0) == _pairs_ase(atoms, 3.0)


def test_internal_vacuum_two_slabs() -> None:
    """Two slabs separated by more than the cutoff along a non-periodic axis never bond across the gap."""
    grid = np.arange(4) * 2.5
    lower = np.array([[x, y, z] for x in grid for y in grid for z in (0.0, 2.0)])
    upper = lower + np.array([0.0, 0.0, 12.0])
    positions = np.vstack([lower, upper])
    cell = np.diag([10.0, 10.0, 14.0])
    atoms = Atoms(f"H{len(positions)}", positions=positions, cell=cell, pbc=[True, True, False])
    pairs = _pairs_ours(atoms, 3.5)
    assert pairs == _pairs_ase(atoms, 3.5)
    assert all((positions[i, 2] < 5.0) == (positions[j, 2] < 5.0) for i, j in pairs), "a pair crosses the gap"


@pytest.mark.parametrize("pbc", [pbc for pbc in _ALL_PBC if not all(pbc)], ids=_pbc_id)
def test_pad_nonperiodic_invariants(pbc: tuple[bool, ...]) -> None:
    """Padded heights are exactly E + 2 * cutoff, periodic heights are untouched, distances are preserved."""
    rng = np.random.default_rng(5)
    cell = np.array([[15.0, 0.0, 0.0], [4.0, 14.0, 0.0], [3.0, 5.0, 13.0]])
    coords = (rng.random((150, 3)) @ cell) * 1.4 - 2.0  # spills outside the cell on purpose
    pbc_arr = np.array(pbc)
    cutoff = 3.0
    heights_before = cell_perpendicular_heights(cell)
    frac_before = coords @ np.linalg.inv(cell)
    extent = (frac_before.max(axis=0) - frac_before.min(axis=0)) * heights_before

    new_cell, new_coords = _pad_nonperiodic(cell, coords, pbc_arr, cutoff)
    heights_after = cell_perpendicular_heights(new_cell)
    frac_after = new_coords @ np.linalg.inv(new_cell)

    np.testing.assert_allclose(heights_after[pbc_arr], heights_before[pbc_arr], rtol=1e-10)
    np.testing.assert_allclose(heights_after[~pbc_arr] - extent[~pbc_arr], 2.0 * cutoff, rtol=1e-10)
    assert np.all(frac_after[:, ~pbc_arr] >= -1e-12)
    assert np.all(frac_after[:, ~pbc_arr] < 1.0)
    np.testing.assert_allclose(
        np.linalg.norm(new_coords[1:] - new_coords[0], axis=1),
        np.linalg.norm(coords[1:] - coords[0], axis=1),
        rtol=1e-10,
    )


def test_pad_nonperiodic_is_identity_when_fully_periodic() -> None:
    """The fully periodic hot path returns the very same objects, untouched."""
    cell = np.diag([10.0, 10.0, 10.0])
    coords = np.random.default_rng(0).random((20, 3)) * 10.0
    out_cell, out_coords = _pad_nonperiodic(cell, coords, np.ones(3, dtype=bool), 3.0)
    assert out_cell is cell
    assert out_coords is coords


@pytest.mark.parametrize("n_atoms", [0, 1, 2])
@pytest.mark.parametrize("pbc", [True, False])
def test_tiny_systems(n_atoms: int, pbc: bool) -> None:  # noqa: FBT001
    """0, 1 and 2 atoms run through both entry points, periodic or not, and agree with ASE."""
    positions = np.array([[0.6 * k, 5.0, 5.0] for k in range(n_atoms)]).reshape(n_atoms, 3)
    atoms = Atoms("H" * n_atoms, positions=positions, cell=np.diag([10.0, 10.0, 10.0]), pbc=pbc)
    expected = _pairs_ase(atoms, 2.0) if n_atoms else set()

    result = get_neighbors(atoms, cutoff=2.0)
    assert len(result) == n_atoms
    assert _pairs_ours(atoms, 2.0) == expected

    dists, _i_idx, _j_idx = build_distances(atoms, r_max=2.0)
    assert len(dists) == len(expected) // 2


def test_pair_cutoffs_and_filters_on_slab() -> None:
    """Per-pair cutoffs and type filters on a slab: numba and NumPy agree with an in-plane brute force."""
    atoms = _random_atoms(np.diag([12.0, 12.0, 6.0]), [True, True, False], n_atoms=120, seed=7)
    cutoff = {(8, 14): 2.2, (8, 8): 3.0, (14, 14): 3.4}
    kwargs = {"cutoff": cutoff, "target_types": [14], "neighbor_types": [8, 14]}
    numba_result = {cid: sorted(nn) for cid, nn in get_neighbors(atoms, use_numba=True, **kwargs)}
    numpy_result = {cid: sorted(nn) for cid, nn in get_neighbors(atoms, use_numba=False, **kwargs)}
    assert numba_result == numpy_result

    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    box = np.diag(atoms.get_cell().array)
    for cid, nn_ids in numba_result.items():
        i = cid - 1
        if numbers[i] != 14:
            assert nn_ids == []
            continue
        expected = []
        for j in range(len(atoms)):
            if j == i:
                continue
            delta = positions[i] - positions[j]
            delta[:2] -= box[:2] * np.round(delta[:2] / box[:2])  # in-plane minimum image only
            pair = tuple(sorted((int(numbers[i]), int(numbers[j]))))
            if np.linalg.norm(delta) <= cutoff[pair]:
                expected.append(j + 1)
        assert nn_ids == expected


def test_build_distances_honours_pbc() -> None:
    """build_distances half-pairs and distances match ASE on a cell that is non-periodic along b."""
    atoms = _random_atoms(_PBC_CELLS["shear_c"], [True, False, True], seed=11)
    dists, i_idx, j_idx = build_distances(atoms, r_max=3.2)
    obtained = {(int(i), int(j)) for i, j in zip(i_idx, j_idx, strict=True)}
    assert obtained == {(i, j) for i, j in _pairs_ase(atoms, 3.2) if j > i}

    i_ase, j_ase, d_ase = neighbor_list("ijd", atoms, 3.2)
    reference = {(int(i), int(j)): float(d) for i, j, d in zip(i_ase, j_ase, d_ase, strict=True)}
    for d, i, j in zip(dists, i_idx, j_idx, strict=True):
        assert d == pytest.approx(reference[(int(i), int(j))], abs=1e-9)


@pytest.mark.parametrize("use_numba", [True, False])
def test_return_vectors_match_ase_with_opposite_sign(use_numba: bool) -> None:  # noqa: FBT001
    """Bond vectors are r_i - r_j, i.e. exactly -D from ASE, on a slab."""
    atoms = _random_atoms(_PBC_CELLS["shear_b"], [True, True, False], seed=13)
    i_ase, j_ase, d_ase = neighbor_list("ijD", atoms, 3.2)
    reference = {(int(i), int(j)): vec for i, j, vec in zip(i_ase, j_ase, d_ase, strict=True)}
    for central_id, nn_ids, vecs in get_neighbors(atoms, cutoff=3.2, return_vectors=True, use_numba=use_numba):
        for neighbor_id, vec in zip(nn_ids, vecs, strict=True):
            np.testing.assert_allclose(vec, -reference[(central_id - 1, neighbor_id - 1)], atol=1e-5)


def test_get_neighbors_output_types_are_plain_python() -> None:
    """Central ids are int and neighbour lists are list[int]: callers use them as dict keys and serialise them."""
    result = get_neighbors(_glass_300(), cutoff=2.0, return_vectors=True)
    for central_id, nn_ids, vecs in result:
        assert type(central_id) is int
        assert type(nn_ids) is list
        assert all(type(neighbor_id) is int for neighbor_id in nn_ids)
        assert vecs.dtype == np.float64
        assert vecs.shape == (len(nn_ids), 3)


def test_build_distances_numpy_branch_matches_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    """With Numba disabled build_distances takes the shared NumPy fallback and agrees on a small box and a slab."""
    small = _ortho_atoms()  # 6 Å box, r_max=4 → n_cells=1, the duplicate-pair regime
    slab = _random_atoms(_PBC_CELLS["shear_c"], [True, True, False], seed=17)
    types = slab.get_atomic_numbers()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        numba_small = build_distances(small, r_max=4.0)
        numba_slab = build_distances(slab, r_max=3.2, types=types, unordered_pairs=[(8, 14)])
        monkeypatch.setattr("amorphouspy.atoms.neighbors.NUMBA_AVAILABLE", False)
        numpy_small = build_distances(small, r_max=4.0)
        numpy_slab = build_distances(slab, r_max=3.2, types=types, unordered_pairs=[(8, 14)])

    for (d_nb, i_nb, j_nb), (d_np, i_np, j_np) in ((numba_small, numpy_small), (numba_slab, numpy_slab)):
        pairs_nb = {(int(i), int(j)): float(d) for d, i, j in zip(d_nb, i_nb, j_nb, strict=True)}
        pairs_np = {(int(i), int(j)): float(d) for d, i, j in zip(d_np, i_np, j_np, strict=True)}
        assert len(pairs_np) == len(d_np), "duplicate pairs in the NumPy branch"
        assert pairs_nb.keys() == pairs_np.keys()
        for key, d in pairs_nb.items():
            assert pairs_np[key] == pytest.approx(d, abs=1e-9)


# ---------------------------------------------------------------------------
# Minimum-image warning criterion — shortest periodic lattice vector
# ---------------------------------------------------------------------------


def test_min_periodic_lattice_vector_uses_shortest_vector_not_height() -> None:
    """For a=(10,0,0), c=(9,0,10) the perpendicular height is 7.43 Å but the shortest lattice vector is 10 Å."""
    cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [9.0, 0.0, 10.0]])
    assert cell_perpendicular_heights(cell).min() == pytest.approx(7.4329, abs=1e-3)
    assert _min_periodic_lattice_vector(cell, np.array([True, True, True])) == pytest.approx(10.0)
    assert _min_periodic_lattice_vector(cell, np.array([True, True, False])) == pytest.approx(10.0)
    assert _min_periodic_lattice_vector(cell, np.array([False, False, True])) == pytest.approx(np.sqrt(181.0))
    assert _min_periodic_lattice_vector(cell, np.array([False, False, False])) == np.inf


def test_no_spurious_warning_for_skewed_cell_within_shortest_vector() -> None:
    """A 4.5 Å cutoff is below half the shortest lattice vector (5 Å) though above half the height (3.7 Å)."""
    cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [9.0, 0.0, 10.0]])
    atoms = _random_atoms(cell, pbc=True, n_atoms=50)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        get_neighbors(atoms, cutoff=4.5)


def test_slab_does_not_warn_about_nonperiodic_axis() -> None:
    """A thin slab is judged on its in-plane lattice only; the padded axis never triggers the warning."""
    atoms = _random_atoms(np.diag([20.0, 20.0, 4.0]), [True, True, False], n_atoms=60)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        get_neighbors(atoms, cutoff=3.5)
