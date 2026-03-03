"""Tests for amorphouspy.neighbors — cell lists, cutoff parsing, and get_neighbors."""

import numpy as np
import pytest
from ase import Atoms

from amorphouspy.neighbors import (
    _estimate_max_neighbors,
    _extract_atom_ids,
    _parse_cutoff,
    cell_perpendicular_heights,
    compute_cell_list_orthogonal,
    compute_cell_list_triclinic,
    get_neighbors,
)


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
    max_rc, pair_types, pair_cutoffs_sq, use = _parse_cutoff(cutoff, types)
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
    max_rc, pair_types, pair_cutoffs_sq, use = _parse_cutoff(cutoff, types)
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
    coords_frac, atom_cells, n_cells, cell_start, order = compute_cell_list_triclinic(coords, cell, 2.0)
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
    for cid, nn_ids, vecs in result:
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
    for cid, nn_ids, vecs in result:
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
    nl = {cid: nn for cid, nn in result}
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
