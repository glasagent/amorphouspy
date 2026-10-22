"""Tests for amorphouspy.analysis.cavities.

Covers compute_cavities (integration test using the 20Na2O-80SiO2 dump file).
Atom type mapping: O=type1, Si=type2, Na=type3.
"""

import builtins

import numpy as np
import plotly.graph_objects as go
import pytest
from amorphouspy.properties.structural.cavities import (
    _atom_radii,
    _build_cavity_grid,
    _build_distance_field,
    _cavity_level_field,
    _compute_grid_shape,
    _compute_gyration_tensor_descriptors,
    _compute_triangle_mesh_area,
    _compute_voxel_step,
    _display_cavity_mesh,
    _exposed_surface_points,
    _iter_cavity_voxels,
    _label_voids_with_pbc,
    _local_cavity_mesh,
    _merge_pbc_labels,
    compute_cavities,
    visualize_cavities,
)
from ase import Atoms
from ase.io import read
from scipy import ndimage

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
    """Voxel step equals max box length divided by resolution (no padding)."""
    lengths = np.array([20.0, 20.0, 20.0])
    step = _compute_voxel_step(lengths, resolution=64)
    assert step == pytest.approx(20.0 / 64.0)


def test_compute_grid_shape():
    """Grid spans exactly one cell: one voxel per voxel_step along each axis."""
    lengths = np.array([10.0, 10.0, 10.0])
    shape = _compute_grid_shape(lengths, voxel_step=1.0)
    assert shape == (10, 10, 10)


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
    merged, percolating, offsets = _merge_pbc_labels(labels, 1)
    assert merged.max() == 1
    assert merged[4, 4, 4] == 1
    assert percolating == set()
    assert np.all(offsets == 0)


def test_merge_pbc_labels_across_corner():
    """Components touching only diagonally across a cell corner are merged."""
    labels = np.zeros((6, 6, 6), dtype=int)
    labels[0, 0, 0] = 1
    labels[5, 5, 5] = 2
    merged, percolating, offsets = _merge_pbc_labels(labels, 2)
    assert merged.max() == 1
    assert percolating == set()
    # Component 1 is the periodic image at +1 cell along every axis relative to component 2.
    assert np.all(offsets[1] - offsets[2] == 1)


def test_label_voids_with_pbc_fully_occupied():
    """A fully occupied grid returns zero voids."""
    grid = np.ones((20, 20, 20), dtype=bool)
    void_labels, percolating, _raw, _offsets = _label_voids_with_pbc(grid)
    assert void_labels.max() == 0
    assert percolating == set()


def test_label_voids_with_pbc_face_crossing_is_not_percolating():
    """A void cut by the cell face is one cavity, not a percolating one."""
    grid = np.ones((10, 10, 10), dtype=bool)
    grid[0:2, 4:6, 4:6] = False
    grid[8:10, 4:6, 4:6] = False
    void_labels, percolating, _raw, _offsets = _label_voids_with_pbc(grid)
    assert void_labels.max() == 1
    assert percolating == set()


def test_label_voids_with_pbc_channel_is_percolating():
    """A channel connecting to its own periodic image is percolating."""
    grid = np.ones((10, 10, 10), dtype=bool)
    grid[:, 4, 4] = False
    void_labels, percolating, _raw, _offsets = _label_voids_with_pbc(grid)
    assert void_labels.max() == 1
    assert percolating == {1}


def test_build_distance_field_dict_missing_element():
    """Element absent from dict falls back to covalent radius without error."""
    cell = np.diag([10.0, 10.0, 10.0])
    positions = np.array([[0.5, 0.5, 0.5]])  # fractional
    atomic_numbers = np.array([14])  # Si
    grid_shape = (20, 20, 20)
    # Provide a dict that has no entry for Si → covalent fallback
    result = _build_distance_field(positions, atomic_numbers, grid_shape, radii_by_symbol={"O": 1.5}, cell=cell)
    assert (result <= 0).any()


def test_build_distance_field_uniform_radius():
    """Float cutoff_radii marks voxels for a single atom."""
    cell = np.diag([10.0, 10.0, 10.0])
    positions = np.array([[0.5, 0.5, 0.5]])
    atomic_numbers = np.array([14])
    grid_shape = (20, 20, 20)
    result = _build_distance_field(positions, atomic_numbers, grid_shape, radii_by_symbol=1.5, cell=cell)
    assert (result <= 0).any()


def test_build_distance_field_unbiased_sphere_volume():
    """Averaged over sub-voxel atom positions, the occupied voxel count matches the sphere volume.

    With 1 Å voxels and a 1.4 Å radius, rounding the radius to whole voxels (the pyMolDyn/sovapy
    scheme) would always give 7 voxels, 39 % below the sphere volume of 11.49 Å³.
    """
    cell = np.diag([10.0, 10.0, 10.0])
    sub_voxel_shifts = (np.arange(6) + 0.5) / 60.0  # fractional shifts spanning one 1 Å voxel
    shifted_positions = [
        np.array([[0.5 + dx, 0.5 + dy, 0.5 + dz]])
        for dx in sub_voxel_shifts
        for dy in sub_voxel_shifts
        for dz in sub_voxel_shifts
    ]
    occupied_counts = [
        (_build_distance_field(position, np.array([8]), (10, 10, 10), 1.4, cell) <= 0).sum()
        for position in shifted_positions
    ]
    assert np.mean(occupied_counts) == pytest.approx(4.0 / 3.0 * np.pi * 1.4**3, rel=0.02)


def test_compute_cavities_sphere_surface_area():
    """The cavity surface around a single isolated atom is the full sphere area 4*pi*R^2.

    Every surface point is exposed and assigned, so the area is exact up to rounding. Marching cubes gave
    +8.35 % on the binary voxel mask and -0.34 % on the distance field here.
    """
    atoms = Atoms("O", positions=[[10.37, 9.81, 10.13]], cell=[20.0] * 3, pbc=True)
    result = compute_cavities(atoms, resolution=64, cutoff_radii=3.0)
    assert len(result["surface_areas"]) == 1
    assert result["surface_areas"][0] == pytest.approx(4.0 * np.pi * 3.0**2, rel=1e-9)


def test_exposed_surface_points_two_sphere_lens():
    """Two overlapping spheres of different radii: exposed area = both spheres minus the two caps inside the other."""
    radii = np.array([1.5, 2.0])
    separation = 2.5
    cap_heights = radii - np.array([0.9, 1.6])  # cap bases at (d^2 + r1^2 - r2^2) / 2d = 0.9 and d - 0.9
    exact = float(np.sum(4.0 * np.pi * radii**2 - 2.0 * np.pi * radii * cap_heights))
    positions = np.array([[5.0, 5.0, 5.0], [5.0 + separation, 5.0, 5.0]])
    _points, weights = _exposed_surface_points(positions, np.array([8, 14]), radii, np.eye(3) * 12.0)
    assert weights.sum() == pytest.approx(exact, rel=0.005)


def test_exposed_surface_points_several_images_in_small_cell():
    """An atom of radius 2.5 in a 4 A cubic cell overlaps six images of itself: exposed area 4 pi r^2 - 6 caps = 10 pi.

    The cutoff (5 A) exceeds half the cell, so the neighbour search must run on a replicated cell.
    """
    atoms = Atoms("O", positions=[[1.1, 0.7, 2.3]], cell=[4.0] * 3, pbc=True)
    _points, weights = _exposed_surface_points(atoms.positions, atoms.numbers, np.array([2.5]), atoms.cell.array)
    assert weights.sum() == pytest.approx(10.0 * np.pi, rel=0.005)
    result = compute_cavities(atoms, resolution=32, cutoff_radii=2.5)
    assert result["percolating"].all()
    assert result["surface_areas"].sum() == pytest.approx(10.0 * np.pi, rel=0.005)


def _simple_cubic_lattice(cell):
    """125 O atoms on a simple-cubic lattice with 3 A spacing, placed in a cell spanning the same 15 A lattice."""
    grid_1d = np.arange(0.0, 15.0, 3.0)
    lattice_sites = np.array(np.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij")).reshape(3, -1).T
    positions = lattice_sites + np.array([0.37, 0.21, 0.13])
    atoms = Atoms("O125", positions=positions, cell=cell, pbc=True)
    atoms.wrap()
    return atoms


@pytest.mark.parametrize(
    "cell",
    [np.eye(3) * 15.0, np.array([[15.0, 0.0, 0.0], [15.0, 15.0, 0.0], [0.0, 0.0, 15.0]])],
    ids=["cubic", "triclinic"],
)
def test_compute_cavities_simple_cubic_lattice_analytic(cell):
    """Spheres of radius 1.8 on a 3 A simple-cubic lattice, in the cubic cell and in the lattice's (a, a+b, c) cell.

    Each atom loses six caps of height 0.3 A to its neighbours, so the exposed area is
    N (4 pi r^2 - 12 pi r h) and the free fraction 1 - N (4/3 pi r^3 - 6 pi h^2 (3r - h) / 3) / V. The free space is
    one percolating network.
    """
    radius, cap_height = 1.8, 0.3
    atoms = _simple_cubic_lattice(cell)
    exact_area = 125 * (4.0 * np.pi * radius**2 - 12.0 * np.pi * radius * cap_height)
    atom_volume = 4.0 / 3.0 * np.pi * radius**3 - 6.0 * np.pi * cap_height**2 * (3.0 * radius - cap_height) / 3.0
    exact_free_fraction = 1.0 - 125 * atom_volume / 15.0**3
    result = compute_cavities(atoms, resolution=32, cutoff_radii=radius)
    assert result["percolating"].tolist() == [True]
    assert result["surface_areas"].sum() == pytest.approx(exact_area, rel=0.002)
    assert result["volumes"].sum() / atoms.get_volume() == pytest.approx(exact_free_fraction, rel=0.005)


def test_local_cavity_mesh_spherical_cavity_area():
    """A closed spherical cavity of radius 6 voxels is triangulated within 1.5 % of its area."""
    grid_points = np.indices((24, 24, 24)).reshape(3, -1).T
    distance_to_wall = 6.0 - np.linalg.norm(grid_points - np.array([12.3, 11.8, 12.1]), axis=1)
    field = distance_to_wall.reshape(24, 24, 24).astype(np.float32)
    vertices, faces = _local_cavity_mesh(np.argwhere(field > 0), field)
    assert _compute_triangle_mesh_area(vertices, faces) == pytest.approx(4.0 * np.pi * 6.0**2, rel=0.015)


@pytest.mark.parametrize(
    "cell",
    [np.eye(3) * 4.5, np.array([[5.0, 0.0, 0.0], [1.5, 4.5, 0.0], [0.7, 0.9, 4.8]])],
    ids=["orthogonal", "triclinic"],
)
def test_build_distance_field_matches_brute_force(cell):
    """The field equals min over atoms and periodic images of (|x - x_j| - r_j), capped at one voxel diagonal.

    The 2.5 A radius exceeds half the cell, so several images of an atom reach the same voxel.
    """
    fractional = np.array([[0.13, 0.27, 0.61], [0.72, 0.55, 0.08], [0.41, 0.93, 0.36]])
    atomic_numbers = np.array([14, 8, 8])
    grid_shape = (10, 9, 11)
    field = _build_distance_field(fractional, atomic_numbers, grid_shape, {"Si": 2.5, "O": 1.4}, cell)

    voxel_fractional = np.indices(grid_shape).reshape(3, -1).T / np.array(grid_shape)
    images = np.array(np.meshgrid(*[np.arange(-2, 3)] * 3, indexing="ij")).reshape(3, -1).T
    exact = np.full(len(voxel_fractional), np.inf)
    for atom_fractional, radius in zip(fractional, (2.5, 1.4, 1.4), strict=True):
        separations = (voxel_fractional[:, None, :] - atom_fractional - images[None, :, :]) @ cell
        exact = np.minimum(exact, np.linalg.norm(separations, axis=2).min(axis=1) - radius)
    diagonal_signs = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]])
    voxel_diagonal = np.max(np.linalg.norm((diagonal_signs / np.array(grid_shape)) @ cell, axis=1))
    np.testing.assert_allclose(field.ravel(), np.minimum(exact, voxel_diagonal), atol=1e-5)
    np.testing.assert_array_equal(field.ravel() <= 0, exact <= 0)


def test_cavity_level_field_no_foreign_empty_voxel_next_to_cavity(glass_structure):
    """In each closed cavity's unwrapped box, every 26-neighbour of a cavity voxel is a cavity voxel or occupied.

    So the -1 fill of foreign empty voxels can never create a surface next to the cavity.
    """
    void_labels, _occupied, _cell, percolating, raw_labels, image_offsets, field = _build_cavity_grid(
        glass_structure, 32, None
    )
    grid_shape = np.array(void_labels.shape)
    for void_index, voxel_indices, image_shifts in _iter_cavity_voxels(void_labels, raw_labels, image_offsets):
        if void_index in percolating:
            continue
        unwrapped = voxel_indices + image_shifts * grid_shape
        lower = unwrapped.min(axis=0) - 1
        local_shape = unwrapped.max(axis=0) - lower + 2
        axes = [(np.arange(size) + start) % n for size, start, n in zip(local_shape, lower, grid_shape, strict=True)]
        member = np.zeros(tuple(local_shape), dtype=bool)
        member[tuple((unwrapped - lower).T)] = True
        level = _cavity_level_field(field[np.ix_(*axes)], member)
        shell = ndimage.binary_dilation(member, structure=np.ones((3, 3, 3), dtype=bool)) & ~member
        assert np.all(level[shell] <= 0)
        assert np.all(level[member] > 0)


def _channel_field(size=12, radius=2.5):
    """Level field of a straight periodic channel of the given radius (in voxels) along x."""
    y, z = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    section = radius - np.hypot(y - (size - 1) / 2, z - (size - 1) / 2)
    return np.broadcast_to(section, (size, size, size)).astype(np.float32).copy()


@pytest.mark.parametrize(
    "mc_to_cart",
    [np.eye(3), np.array([[1.0, 0.0, 0.0], [0.25, 1.0, 0.0], [0.1, 0.2, 1.0]])],
    ids=["orthogonal", "triclinic"],
)
def test_display_cavity_mesh_percolating_channel_has_no_seam(mc_to_cart):
    """The percolating display mesh covers exactly one cell wherever the cell boundary cuts the channel.

    Dropped or duplicated marching-cubes cells at the periodic seam would change the area under a periodic shift.
    Before the fix the unshifted channel lost 9 % of its area and the shifted one 35 %.
    """
    field = _channel_field()
    areas = []
    for shift in ((0, 0, 0), (3, 5, 7), (6, 0, 11)):
        shifted = np.roll(field, shift, axis=(0, 1, 2))
        vertices, faces = _display_cavity_mesh(np.argwhere(shifted > 0), None, shifted)
        areas.append(_compute_triangle_mesh_area(vertices @ mc_to_cart, faces))
    assert areas == pytest.approx([areas[0]] * 3, rel=1e-9)
    if np.allclose(mc_to_cart, np.eye(3)):
        assert areas[0] == pytest.approx(2.0 * np.pi * 2.5 * 12, rel=0.015)


def test_compute_cavities_same_for_cubic_and_tilted_cell(glass_structure):
    """A negligible tilt of the same cell does not change the free volume."""
    tilted = glass_structure.copy()
    tilted_cell = tilted.get_cell().array.copy()
    tilted_cell[1, 0] = 1e-6
    tilted.set_cell(tilted_cell)
    free_cubic = compute_cavities(glass_structure, resolution=32)["volumes"].sum() / glass_structure.get_volume()
    free_tilted = compute_cavities(tilted, resolution=32)["volumes"].sum() / tilted.get_volume()
    assert free_tilted == pytest.approx(free_cubic, abs=1e-4)


# ---------------------------------------------------------------------------
# compute_cavities — edge cases
# ---------------------------------------------------------------------------


def _dense_structure():
    """Atoms packed tightly enough (large uniform radius) to leave no cavities.

    The atoms form a simple-cubic lattice with 3 Å spacing, so no point of the
    cell is farther than sqrt(3) * 1.5 ≈ 2.60 Å from an atom. With a 3.5 Å
    cutoff the entire grid is occupied at any resolution.
    """
    cell = np.diag([6.0, 6.0, 6.0])
    positions = [[x, y, z] for x in [1.5, 4.5] for y in [1.5, 4.5] for z in [1.5, 4.5]]
    return Atoms("Si8", positions=positions, cell=cell, pbc=True)


def test_compute_cavities_empty_when_fully_occupied():
    """Returns all-empty arrays when no cavity voxels exist."""
    result = compute_cavities(_dense_structure(), resolution=16, cutoff_radii=3.5)
    for arr in result.values():
        assert len(arr) == 0


def test_compute_cavities_integer_cutoff_radii(glass_structure):
    """An integer or numpy uniform radius gives the same result as the float (ints used to fall back to VdW radii)."""
    reference = compute_cavities(glass_structure, resolution=16, cutoff_radii=2.0)
    for radius in (2, np.int64(2), np.float32(2.0)):
        result = compute_cavities(glass_structure, resolution=16, cutoff_radii=radius)
        for key in ("volumes", "surface_areas", "percolating"):
            np.testing.assert_array_equal(result[key], reference[key])
    vdw = compute_cavities(glass_structure, resolution=16)
    assert not np.array_equal(vdw["volumes"], reference["volumes"])


def test_compute_cavities_bool_cutoff_radii_rejected(glass_structure):
    """A bool is not read as a 1 A radius."""
    with pytest.raises(TypeError, match="cutoff_radii"):
        compute_cavities(glass_structure, resolution=16, cutoff_radii=True)


def test_compute_cavities_float_cutoff_radii(glass_structure):
    """Uniform float cutoff_radii is accepted and returns a valid dict."""
    result = compute_cavities(glass_structure, resolution=16, cutoff_radii=2.8)
    assert "volumes" in result
    assert isinstance(result["volumes"], np.ndarray)


def test_compute_cavities_percolating_flag(glass_structure):
    """Percolating cavities are kept and flagged, with NaN shape descriptors.

    With 1.0 Å radii the empty space of the 20Na2O-80SiO2 glass is one connected
    network that wraps around the cell.
    """
    result = compute_cavities(glass_structure, resolution=16, cutoff_radii=1.0)
    percolating = result["percolating"]
    assert percolating.dtype == bool
    assert percolating.any()
    assert np.all(result["volumes"][percolating] > 0)
    assert np.all(np.isnan(result["asphericities"][percolating]))
    assert not np.any(np.isnan(result["asphericities"][~percolating]))


def test_compute_cavities_no_percolation_with_vdw_radii(glass_structure):
    """With Van der Waals radii the voids of this small test glass are closed cavities, none percolating."""
    result = compute_cavities(glass_structure, resolution=16)
    assert len(result["volumes"]) > 0
    assert not result["percolating"].any()


def test_compute_cavities_shape_descriptors_bounded(glass_structure):
    """Asphericity, acylindricity, and anisotropy are in [0, 1] for non-percolating cavities."""
    result = compute_cavities(glass_structure, resolution=16, cutoff_radii=2.8)
    for key in ("asphericities", "acylindricities", "anisotropies"):
        vals = result[key][~result["percolating"]]
        if len(vals):
            assert np.all(vals >= -1e-9), f"{key} contains negative values"
            assert np.all(vals <= 1.0 + 1e-9), f"{key} exceeds 1"


# ---------------------------------------------------------------------------
# compute_cavities — periodic boundary handling
# ---------------------------------------------------------------------------


def _lattice_with_hole(hole_centre, box_length=20.0, spacing=2.0, hole_radius=3.0):
    """Simple-cubic O lattice with one spherical hole, positioned anywhere in the periodic cell."""
    grid_1d = np.arange(0.0, box_length, spacing)
    positions = np.array(np.meshgrid(grid_1d, grid_1d, grid_1d, indexing="ij")).reshape(3, -1).T
    delta = positions - np.array(hole_centre)
    delta -= box_length * np.round(delta / box_length)
    positions = positions[np.linalg.norm(delta, axis=1) > hole_radius]
    return Atoms("O" * len(positions), positions=positions, cell=[box_length] * 3, pbc=True)


def _largest_cavity(result):
    index = int(np.argmax(result["volumes"]))
    return {key: values[index] for key, values in result.items()}


def test_compute_cavities_hole_independent_of_cell_boundary():
    """The same hole at the box centre, on a face, and at a corner gives the same cavity.

    Volumes and shapes are identical. Surface areas agree to point-sampling noise only: other atoms are removed
    for each hole position, so the per-atom point rotations differ.
    """
    centred_hole = _lattice_with_hole([10.0, 10.0, 10.0])
    reference = _largest_cavity(compute_cavities(centred_hole, resolution=64, cutoff_radii=1.6))
    assert reference["volumes"] > 50.0
    for centre in ([0.0, 10.0, 10.0], [0.0, 0.0, 10.0], [0.0, 0.0, 0.0]):
        cavity = _largest_cavity(compute_cavities(_lattice_with_hole(centre), resolution=64, cutoff_radii=1.6))
        assert not cavity["percolating"]
        assert cavity["volumes"] == pytest.approx(reference["volumes"])
        assert cavity["surface_areas"] == pytest.approx(reference["surface_areas"], rel=0.01)
        assert cavity["asphericities"] == pytest.approx(reference["asphericities"], abs=1e-9)


def test_compute_cavities_volume_conservation(glass_structure):
    """Every empty voxel belongs to exactly one cavity: summed volume = unoccupied volume."""
    result = compute_cavities(glass_structure, resolution=32)
    _labels, occupied_grid, cell_array, *_ = _build_cavity_grid(glass_structure, 32, None)
    unoccupied_volume = (~occupied_grid).sum() * abs(np.linalg.det(cell_array)) / occupied_grid.size
    assert result["volumes"].sum() == pytest.approx(unoccupied_volume)


def test_compute_cavities_channel_percolating():
    """A straight empty channel through the cell is flagged percolating with finite volume per cell."""
    atoms = _lattice_with_hole([10.0, 10.0, 10.0], hole_radius=0.0)
    on_channel = (np.abs(atoms.positions[:, 1] - 10.0) < 0.1) & (np.abs(atoms.positions[:, 2] - 10.0) < 0.1)
    del atoms[np.where(on_channel)[0]]
    result = compute_cavities(atoms, resolution=64, cutoff_radii=1.6)
    channel = _largest_cavity(result)
    assert channel["percolating"]
    assert 0.0 < channel["volumes"] < atoms.get_volume()
    assert channel["surface_areas"] > 0.0
    assert np.isnan(channel["asphericities"])


def test_compute_cavities_whole_voxel_translation(glass_structure):
    """Translating all atoms by whole voxels gives the same cavities (relabelled)."""
    resolution = 32
    translated = glass_structure.copy()
    translated.translate((np.array([3, 5, 7]) / resolution) @ glass_structure.cell.array)

    def sorted_cavities(result):
        order = np.lexsort((result["surface_areas"], result["volumes"]))
        return {key: values[order] for key, values in result.items()}

    reference = sorted_cavities(compute_cavities(glass_structure, resolution=resolution))
    result = sorted_cavities(compute_cavities(translated, resolution=resolution))
    np.testing.assert_allclose(result["volumes"], reference["volumes"], rtol=1e-12)
    np.testing.assert_allclose(result["surface_areas"], reference["surface_areas"], rtol=1e-9)
    np.testing.assert_allclose(result["asphericities"], reference["asphericities"], atol=1e-9)
    np.testing.assert_array_equal(result["percolating"], reference["percolating"])


def _lattice_with_channel(shift):
    """2 A simple-cubic O lattice in a 20 A cell with one line of atoms removed, shifted by ``shift``."""
    atoms = _lattice_with_hole([10.0, 10.0, 10.0], hole_radius=0.0)
    on_channel = (np.abs(atoms.positions[:, 1] - 10.0) < 0.1) & (np.abs(atoms.positions[:, 2] - 10.0) < 0.1)
    del atoms[np.where(on_channel)[0]]
    atoms.translate(shift)
    return atoms


def test_compute_cavities_channel_percolation_depends_on_resolution():
    """A channel is found percolating only when its narrowest neck is resolved by the grid.

    The channel's neck radius is 2 - r: 0.4 A for r = 1.6 (wider than the 0.27 A half voxel diagonal at
    resolution 64, so it percolates for any sub-voxel shift) and 0.1 A for r = 1.9 (narrower, so whether
    voxel centres fall inside the neck decides).
    """

    def percolating_count(shift, radius):
        return int(
            compute_cavities(_lattice_with_channel(shift), resolution=64, cutoff_radii=radius)["percolating"].sum()
        )

    centred, off_centre = [0.0, 0.0, 0.0], [0.0, 0.15625, 0.15625]
    for shift in (centred, off_centre, [0.11, 0.07, 0.29]):
        assert percolating_count(shift, 1.6) == 1
    assert percolating_count(centred, 1.9) == 1
    assert percolating_count(off_centre, 1.9) == 0


# ---------------------------------------------------------------------------
# compute_cavities — probe accessibility
# ---------------------------------------------------------------------------


def test_compute_cavities_probe_single_atom_analytic():
    """One atom of radius 2 with a 1.2 A probe in a 20 A box.

    Probe centres fit outside radius 3.2, the probe-centre surface is a 3.2 A sphere, and the largest included
    sphere sits half a cell diagonal from the atom.
    """
    atoms = Atoms("O", positions=[[10.37, 9.81, 10.13]], cell=[20.0] * 3, pbc=True)
    result = compute_cavities(atoms, resolution=64, cutoff_radii=2.0, probe_radius=1.2)
    cell_volume = 20.0**3
    assert result["percolating"].tolist() == [True]
    assert result["volumes"][0] == pytest.approx(cell_volume - 4.0 / 3.0 * np.pi * 3.2**3, rel=1e-3)
    assert result["surface_areas"][0] == pytest.approx(4.0 * np.pi * 3.2**2, rel=1e-9)
    half_voxel_diagonal = np.sqrt(3.0) * 20.0 / 64 / 2
    exact_inscribed = np.sqrt(3.0) * 10.0 - 2.0
    assert exact_inscribed - half_voxel_diagonal <= result["max_inscribed_radii"][0] <= exact_inscribed


@pytest.mark.parametrize(
    "cell",
    [np.eye(3) * 15.0, np.array([[15.0, 0.0, 0.0], [15.0, 15.0, 0.0], [0.0, 0.0, 15.0]])],
    ids=["cubic", "triclinic"],
)
def test_compute_cavities_probe_simple_cubic_lattice_analytic(cell):
    """Non-overlapping spheres (r = 1, a = 3) with a 0.3 A probe, in the cubic and the (a, a+b, c) cell.

    Inflated spheres (1.3 A) still do not touch: accessible = V - N 4/3 pi 1.3^3, one percolating region.
    """
    atoms = _simple_cubic_lattice(cell)
    result = compute_cavities(atoms, resolution=64, cutoff_radii=1.0, probe_radius=0.3)
    assert result["percolating"].tolist() == [True]
    assert result["volumes"].sum() == pytest.approx(15.0**3 - 125 * 4.0 / 3.0 * np.pi * 1.3**3, rel=0.01)


def test_compute_cavities_probe_fits_hole_only_below_inscribed_radius():
    """A hole whose largest included sphere is 2*sqrt(3) - 1.6 A holds a 1.5 A probe but not a 1.9 A probe."""
    atoms = _lattice_with_hole([10.0, 10.0, 10.0])
    exact_inscribed = 2.0 * np.sqrt(3.0) - 1.6  # centre to the nearest remaining lattice site, minus the radius
    without_probe = compute_cavities(atoms, resolution=64, cutoff_radii=1.6)
    assert without_probe["max_inscribed_radii"].max() == pytest.approx(exact_inscribed, abs=1e-6)
    fits = compute_cavities(atoms, resolution=64, cutoff_radii=1.6, probe_radius=1.5)
    assert len(fits["volumes"]) == 1
    assert fits["max_inscribed_radii"][0] == pytest.approx(exact_inscribed, abs=1e-6)
    assert len(compute_cavities(atoms, resolution=64, cutoff_radii=1.6, probe_radius=1.9)["volumes"]) == 0


def test_compute_cavities_probe_channel_percolation():
    """A probe passes the channel only if it fits through the 0.4 A neck; wider pockets remain for larger probes.

    The pockets between necks have an included-sphere radius of sqrt(5) - 1.6 A.
    """
    atoms = _lattice_with_channel([0.0, 0.0, 0.0])
    assert compute_cavities(atoms, resolution=128, cutoff_radii=1.6, probe_radius=0.3)["percolating"].sum() == 1
    blocked = compute_cavities(atoms, resolution=128, cutoff_radii=1.6, probe_radius=0.5)
    assert len(blocked["volumes"]) > 0
    assert not blocked["percolating"].any()
    assert blocked["max_inscribed_radii"].max() == pytest.approx(np.sqrt(5.0) - 1.6, abs=20.0 / 128 * np.sqrt(3) / 2)


def test_compute_cavities_inscribed_radius_inside_cavity(glass_structure):
    """Every cavity voxel is empty, so every cavity holds a sphere of positive radius."""
    result = compute_cavities(glass_structure, resolution=16)
    assert np.all(result["max_inscribed_radii"] > 0.0)


def test_compute_cavities_negative_probe_rejected(glass_structure):
    """A negative probe radius raises."""
    with pytest.raises(ValueError, match="probe_radius"):
        compute_cavities(glass_structure, resolution=16, probe_radius=-0.5)


def test_atom_radii_ionic():
    """'ionic' gives Shannon radii and rejects elements without a defined state and unknown strings."""
    assert _atom_radii(np.array([8, 14, 11, 13]), "ionic").tolist() == [1.35, 0.26, 1.02, 0.39]
    with pytest.raises(ValueError, match="Xe"):
        _atom_radii(np.array([8, 54]), "ionic")
    with pytest.raises(ValueError, match="ionic"):
        _atom_radii(np.array([8]), "covalent")


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
