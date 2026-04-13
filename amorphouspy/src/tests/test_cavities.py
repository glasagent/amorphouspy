"""Tests for amorphouspy.analysis.cavities.

Covers compute_cavities (integration test using the 20Na2O-80SiO2 dump file).
Atom type mapping: O=type1, Si=type2, Na=type3.
"""

import builtins

import numpy as np
import plotly.graph_objects as go
import pytest
from amorphouspy.analysis.cavities import (
    _build_occupied_grid,
    _compute_grid_shape,
    _compute_gyration_tensor_descriptors,
    _compute_triangle_mesh_area,
    _compute_voxel_step,
    _label_voids_with_pbc,
    _merge_pbc_labels,
    compute_cavities,
    visualize_cavities,
)
from ase import Atoms
from ase.io import read

from . import DATA_DIR

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def glass_structure():
    """Load the 20Na2O-80SiO2 dump and apply correct atomic numbers.

    Type mapping per the simulation setup: O=type1(8), Si=type2(14), Na=type3(11).
    """
    atoms = read(DATA_DIR / "20Na2O-80SiO2.dump", format="lammps-dump-text")
    type_id = atoms.get_atomic_numbers().copy()
    to_z = np.array([0, 8, 14, 11], dtype=int)  # O=1, Si=2, Na=3
    atoms.set_atomic_numbers(to_z[type_id])
    return atoms


# ---------------------------------------------------------------------------
# compute_cavities
# ---------------------------------------------------------------------------


def test_compute_cavities_returns_dict(glass_structure):
    """Return type is a dict."""
    result = compute_cavities(glass_structure, resolution=16)
    assert isinstance(result, dict)


def test_compute_cavities_required_keys(glass_structure):
    """All expected property keys are present in the result."""
    result = compute_cavities(glass_structure, resolution=16)
    expected_keys = {"volumes", "surface_areas", "asphericities", "acylindricities", "anisotropies"}
    assert expected_keys.issubset(result.keys())


def test_compute_cavities_volumes_nonnegative(glass_structure):
    """All cavity volumes are non-negative."""
    result = compute_cavities(glass_structure, resolution=16)
    volumes = result["volumes"]
    assert np.all(volumes >= 0)


def test_compute_cavities_arrays_same_length(glass_structure):
    """All property arrays have the same length (one entry per cavity)."""
    result = compute_cavities(glass_structure, resolution=16)
    lengths = {k: len(v) for k, v in result.items()}
    assert len(set(lengths.values())) == 1, f"Mismatched lengths: {lengths}"


def test_compute_cavities_custom_cutoff_radii(glass_structure):
    """Passing explicit cutoff_radii does not raise and returns a valid dict."""
    result = compute_cavities(
        glass_structure,
        resolution=16,
        cutoff_radii={"O": 1.52, "Si": 1.10, "Na": 1.86},
    )
    assert "volumes" in result


# ---------------------------------------------------------------------------
# Triclinic cell tests
# ---------------------------------------------------------------------------


def _triclinic_glass():
    """Sparse Si-O structure in a triclinic box (2 Å shear on b-vector)."""
    cell = np.array([[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    positions = [[5.0, 5.0, 5.0], [2.0, 2.0, 2.0], [8.0, 8.0, 8.0]]
    return Atoms("SiOO", positions=positions, cell=cell, pbc=True)


def test_compute_cavities_triclinic_returns_required_keys():
    """compute_cavities runs without error on a triclinic cell."""
    result = compute_cavities(_triclinic_glass(), resolution=16)
    expected_keys = {"volumes", "surface_areas", "asphericities", "acylindricities", "anisotropies"}
    assert expected_keys.issubset(result.keys())


def test_compute_cavities_triclinic_volumes_nonnegative():
    """All cavity volumes are non-negative for a triclinic structure."""
    result = compute_cavities(_triclinic_glass(), resolution=16)
    assert np.all(result["volumes"] >= 0)


def test_compute_cavities_triclinic_total_volume_bounded():
    """Total void volume must be less than the full cell volume."""
    atoms = _triclinic_glass()
    cell_volume = float(abs(np.linalg.det(atoms.get_cell().array)))
    result = compute_cavities(atoms, resolution=16)
    assert float(result["volumes"].sum()) < cell_volume


def test_compute_cavities_triclinic_arrays_same_length():
    """All property arrays have the same length for a triclinic structure."""
    result = compute_cavities(_triclinic_glass(), resolution=16)
    lengths = {k: len(v) for k, v in result.items()}
    assert len(set(lengths.values())) == 1, f"Mismatched lengths: {lengths}"


# ---------------------------------------------------------------------------
# Internal helpers — unit tests
# ---------------------------------------------------------------------------


def test_compute_voxel_step():
    """Voxel step equals max box length divided by (resolution - padding voxels)."""
    lengths = np.array([20.0, 20.0, 20.0])
    step = _compute_voxel_step(lengths, resolution=64)
    assert step == pytest.approx(20.0 / 60.0)


def test_compute_grid_shape():
    """Grid shape adds padding voxels to the number of physical voxels per axis."""
    lengths = np.array([10.0, 10.0, 10.0])
    shape = _compute_grid_shape(lengths, voxel_step=1.0)
    assert shape == (14, 14, 14)


def test_compute_triangle_mesh_area_known():
    """Single right-isoceles triangle with legs 1: area = 0.5."""
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    assert _compute_triangle_mesh_area(vertices, faces) == pytest.approx(0.5)


def test_compute_gyration_tensor_descriptors_sphere():
    """Uniformly sampled sphere voxels → asphericity and acylindricity near 0."""
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((500, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * rng.uniform(0, 1, (500, 1))
    eta, c, _kappa = _compute_gyration_tensor_descriptors(pts)
    assert abs(eta) < 0.05
    assert abs(c) < 0.05


def test_compute_gyration_tensor_descriptors_single_voxel():
    """Single voxel → degenerate gyration tensor → all descriptors are 0."""
    pts = np.array([[1.0, 2.0, 3.0]])
    eta, c, kappa = _compute_gyration_tensor_descriptors(pts)
    assert eta == 0.0
    assert c == 0.0
    assert kappa == 0.0


def test_merge_pbc_labels_no_spanning():
    """Labels that don't touch opposite faces are unchanged after merge."""
    labels = np.zeros((10, 10, 10), dtype=int)
    labels[4, 4, 4] = 1  # interior — never touches a border face
    merged, n = _merge_pbc_labels(labels, 1)
    assert n == 1
    assert merged[4, 4, 4] == 1


def test_label_voids_with_pbc_fully_occupied():
    """A fully occupied grid returns zero voids."""
    grid = np.ones((20, 20, 20), dtype=bool)
    void_labels, n = _label_voids_with_pbc(grid)
    assert n == 0
    assert void_labels.max() == 0


def test_build_occupied_grid_dict_missing_element():
    """Element absent from dict falls back to covalent radius without error."""
    cell = np.diag([10.0, 10.0, 10.0])
    positions = np.array([[0.5, 0.5, 0.5]])  # fractional
    atomic_numbers = np.array([14])  # Si
    grid_shape = (20, 20, 20)
    # Provide a dict that has no entry for Si → covalent fallback (line 145)
    result = _build_occupied_grid(
        positions, atomic_numbers, grid_shape, voxel_step=0.5, radii_by_symbol={"O": 1.5}, cell=cell
    )
    assert result.any()


def test_build_occupied_grid_uniform_radius():
    """Float cutoff_radii marks voxels for a single atom."""
    cell = np.diag([10.0, 10.0, 10.0])
    positions = np.array([[0.5, 0.5, 0.5]])
    atomic_numbers = np.array([14])
    grid_shape = (20, 20, 20)
    result = _build_occupied_grid(positions, atomic_numbers, grid_shape, voxel_step=0.5, radii_by_symbol=1.5, cell=cell)
    assert result.any()


# ---------------------------------------------------------------------------
# compute_cavities — edge cases
# ---------------------------------------------------------------------------


def _dense_structure():
    """Atoms packed tightly enough (large uniform radius) to leave no cavities.

    At resolution=16 the voxel step is 0.5 Å, giving discrete_radius=7 for a
    3.5 Å cutoff. The farthest point from all atoms (box centre) is at distance
    4*sqrt(3) ≈ 6.93 voxels, which is ≤ 7, so the entire grid is occupied.
    """
    cell = np.diag([6.0, 6.0, 6.0])
    positions = [[x, y, z] for x in [1.5, 4.5] for y in [1.5, 4.5] for z in [1.5, 4.5]]
    return Atoms("Si8", positions=positions, cell=cell, pbc=True)


def test_compute_cavities_empty_when_fully_occupied():
    """Returns all-empty arrays when no cavity voxels exist."""
    result = compute_cavities(_dense_structure(), resolution=16, cutoff_radii=3.5)
    for arr in result.values():
        assert len(arr) == 0


def test_compute_cavities_float_cutoff_radii(glass_structure):
    """Uniform float cutoff_radii is accepted and returns a valid dict."""
    result = compute_cavities(glass_structure, resolution=16, cutoff_radii=2.8)
    assert "volumes" in result
    assert isinstance(result["volumes"], np.ndarray)


def test_compute_cavities_percolating_warning(glass_structure):
    """A UserWarning is raised when a percolating cavity is detected."""
    with pytest.warns(UserWarning, match="percolating"):
        compute_cavities(glass_structure, resolution=16)


def test_compute_cavities_shape_descriptors_bounded(glass_structure):
    """Asphericity, acylindricity, and anisotropy are in [0, 1]."""
    result = compute_cavities(glass_structure, resolution=16, cutoff_radii=2.8)
    for key in ("asphericities", "acylindricities", "anisotropies"):
        vals = result[key]
        if len(vals):
            assert np.all(vals >= -1e-9), f"{key} contains negative values"
            assert np.all(vals <= 1.0 + 1e-9), f"{key} exceeds 1"


# ---------------------------------------------------------------------------
# visualize_cavities
# ---------------------------------------------------------------------------


def test_visualize_cavities_returns_figure(glass_structure):
    """visualize_cavities returns a plotly Figure."""
    fig = visualize_cavities(glass_structure, resolution=16, cutoff_radii=2.8)
    assert isinstance(fig, go.Figure)


def test_visualize_cavities_no_atoms(glass_structure):
    """show_atoms=False returns a Figure without atom scatter traces."""
    fig = visualize_cavities(glass_structure, resolution=16, cutoff_radii=2.8, show_atoms=False)
    assert isinstance(fig, go.Figure)
    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter3d)]
    assert len(scatter_traces) == 0


def test_visualize_cavities_excluded_threshold(glass_structure):
    """excluded_cavities filters out large cavities from the figure."""
    fig_all = visualize_cavities(glass_structure, resolution=16, cutoff_radii=2.8, show_atoms=False)
    fig_filtered = visualize_cavities(
        glass_structure, resolution=16, cutoff_radii=2.8, show_atoms=False, excluded_cavities=1.0
    )
    mesh_all = [t for t in fig_all.data if isinstance(t, go.Mesh3d)]
    mesh_filtered = [t for t in fig_filtered.data if isinstance(t, go.Mesh3d)]
    assert len(mesh_filtered) <= len(mesh_all)


def test_visualize_cavities_import_error(monkeypatch):
    """ImportError is raised when plotly is not available."""
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "plotly.graph_objects":
            msg = "mocked"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.raises(ImportError, match="plotly is required"):
        visualize_cavities(_triclinic_glass(), resolution=16)
