"""Structural analysis functions for multicomponent glass systems.

Author: Achraf Atila (achraf.atila@bam.de)

Identifies and characterises cavities in atomic structures. The simulation
box is discretised into a 3D voxel grid; voxels whose centre lies within the
atomic radius (ASE Van der Waals radii by default) of any atom are marked as
occupied. Connected regions of empty voxels, detected with full
26-connectivity and periodic-boundary-condition awareness, constitute
cavities.

For each cavity the following properties are computed:

- Volume: number of empty voxels x voxel volume (A^3).
- Surface area: area of the exposed atom-sphere patches that bound the
  cavity (A^2), sampled with Shrake-Rupley points.
- Display meshes (``visualize_cavities``): marching cubes on the
  distance to the nearest atom surface.
- Shape descriptors: asphericity (eta), acylindricity (c), and
  relative shape anisotropy (kappa) derived from the volume-weighted
  gyration tensor over all cavity voxels (pyMolDyn definition).

References:
    Meyer, I. et al. pyMolDyn: Identification, structure, and properties
    of cavities/vacancies in condensed matter.
    J. Comput. Chem. 38, 389-394 (2017).
    https://doi.org/10.1002/jcc.24697
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

import mcubes
import numpy as np
from ase.data import atomic_numbers as ase_atomic_numbers
from ase.data import covalent_radii as ase_covalent_radii
from ase.data import vdw_radii as ase_vdw_radii
from pymatgen.core.periodic_table import Species
from scipy import ndimage
from scipy.spatial.transform import Rotation

from amorphouspy.atoms.neighbors import cell_perpendicular_heights, get_neighbors

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ase import Atoms

# ============================================================================
# Internal helpers
# ============================================================================

# Surface points per A^2 of atom sphere, and the minimum per atom
_SURFACE_POINTS_PER_A2 = 20.0
_MIN_SURFACE_POINTS = 50
# Voxel offsets searched around a surface point for the nearest empty voxel
_ASSIGNMENT_OFFSETS = np.arange(-3, 5)
# (oxidation state, coordination) for Shannon (1976) ionic radii: tetrahedral network formers,
# octahedral modifiers and two-coordinated oxygen
_IONIC_STATES = {
    "O": (-2, "II"),
    "Si": (4, "IV"),
    "B": (3, "IV"),
    "Al": (3, "IV"),
    "P": (5, "IV"),
    "Ge": (4, "IV"),
    "Ti": (4, "VI"),
    "Zn": (2, "IV"),
    "Li": (1, "IV"),
    "Na": (1, "VI"),
    "K": (1, "VI"),
    "Mg": (2, "VI"),
    "Ca": (2, "VI"),
    "Sr": (2, "VI"),
    "Ba": (2, "VI"),
}


def _compute_voxel_step(box_lengths: np.ndarray, resolution: int) -> float:
    """Compute the voxel edge length for a periodic grid spanning exactly one cell.

    Args:
        box_lengths: Side lengths of the simulation box in A, shape (3,).
        resolution: Grid resolution (number of voxels along the longest side).

    Returns:
        Voxel edge length in A.

    Examples:
        >>> import numpy as np
        >>> _compute_voxel_step(np.array([20.0, 20.0, 20.0]), resolution=64)
        0.3125
    """
    return float(np.max(box_lengths)) / resolution


def _compute_grid_shape(box_lengths: np.ndarray, voxel_step: float) -> tuple[int, ...]:
    """Compute the 3D grid shape for the given box and voxel size.

    The grid covers exactly one periodic cell (no padding), so voxel ``i`` and
    voxel ``n - 1`` along an axis are periodic neighbours.

    Args:
        box_lengths: Side lengths of the simulation box in A, shape (3,).
        voxel_step: Voxel edge length in A.

    Returns:
        Integer grid dimensions (nx, ny, nz).

    Examples:
        >>> import numpy as np
        >>> lengths = np.array([10.0, 10.0, 10.0])
        >>> _compute_grid_shape(lengths, voxel_step=1.0)
        (10, 10, 10)
    """
    return tuple(max(1, int(np.round(length / voxel_step))) for length in box_lengths)


def _atom_radii(atomic_numbers: np.ndarray, radii_by_symbol: dict[str, float] | float | str | None) -> np.ndarray:
    """Radius of every atom from the radius specification.

    Args:
        atomic_numbers: Atomic numbers for each atom, shape (n_atoms,).
        radii_by_symbol: None → element-specific ASE VdW radii (covalent
            fallback). A real number (int or float) → uniform radius for all
            atoms. A dict → per-element overrides (missing elements fall back
            to ASE covalent radii). ``"ionic"`` → Shannon (1976) ionic radii
            for the oxidation state and coordination in ``_IONIC_STATES``.

    Returns:
        Radii in A, shape (n_atoms,).

    Raises:
        TypeError: If ``radii_by_symbol`` is a bool.
        ValueError: If ``radii_by_symbol`` is a string other than ``"ionic"``,
            or ``"ionic"`` is used with an element not in ``_IONIC_STATES``.

    Examples:
        >>> import numpy as np
        >>> _atom_radii(np.array([14, 8]), 3).tolist()
        [3.0, 3.0]
        >>> _atom_radii(np.array([14, 8]), {"O": 1.5}).tolist()  # Si falls back to its covalent radius
        [1.11, 1.5]
        >>> _atom_radii(np.array([14, 8, 11]), "ionic").tolist()
        [0.26, 1.35, 1.02]
    """
    if isinstance(radii_by_symbol, bool):
        msg = f"cutoff_radii must be None, a number, a dict or 'ionic', got {radii_by_symbol!r}"
        raise TypeError(msg)
    z_to_symbol = {z: symbol for symbol, z in ase_atomic_numbers.items()}
    if isinstance(radii_by_symbol, str):
        if radii_by_symbol != "ionic":
            msg = f"cutoff_radii string must be 'ionic', got {radii_by_symbol!r}"
            raise ValueError(msg)
        symbols = {z_to_symbol.get(z, "X") for z in np.asarray(atomic_numbers).tolist()}
        missing = sorted(symbols - set(_IONIC_STATES))
        if missing:
            msg = f"no ionic radius defined for {missing}; pass a dict of radii instead"
            raise ValueError(msg)
        radii_by_symbol = {
            symbol: float(Species(symbol, _IONIC_STATES[symbol][0]).get_shannon_radius(_IONIC_STATES[symbol][1]))
            for symbol in symbols
        }
    radii = np.empty(len(atomic_numbers), dtype=np.float64)
    for atom_index, z in enumerate(np.asarray(atomic_numbers).tolist()):
        if isinstance(radii_by_symbol, numbers.Real):
            radii[atom_index] = float(radii_by_symbol)
        elif isinstance(radii_by_symbol, dict):
            radii[atom_index] = radii_by_symbol.get(z_to_symbol.get(z, "X"), float(ase_covalent_radii[z]))
        else:
            vdw = ase_vdw_radii[z]
            radii[atom_index] = float(vdw) if not np.isnan(vdw) else float(ase_covalent_radii[z])
    return radii


def _voxel_diagonal(grid_shape: tuple[int, ...], cell: np.ndarray) -> float:
    """Length of the longest body diagonal of one voxel, in A.

    Args:
        grid_shape: Integer grid dimensions (nx, ny, nz).
        cell: (3, 3) cell matrix with lattice vectors as rows.

    Returns:
        Longest voxel body diagonal in A.

    Examples:
        >>> import numpy as np
        >>> round(_voxel_diagonal((10, 10, 10), np.eye(3) * 10.0), 4)
        1.7321
    """
    diagonal_signs = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]])
    return float(np.max(np.linalg.norm((diagonal_signs / np.array(grid_shape)) @ cell, axis=1)))


def _build_distance_field(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    grid_shape: tuple[int, ...],
    radii_by_symbol: dict[str, float] | float | str | None,
    cell: np.ndarray,
    probe_radius: float = 0.0,
    cap: float | None = None,
) -> np.ndarray:
    """Distance from each voxel centre to the nearest atom surface.

    Voxel ``i`` sits at fractional coordinate ``i / grid_shape``. Its value is
    ``min_j(|x - x_j| - r_j)`` over atoms ``j``, from Cartesian distances to the
    true atom positions; the same rule applies to orthogonal and triclinic
    cells. A voxel is occupied when its value is at most 0. Atoms are not
    snapped to the grid and radii are not rounded to whole voxels (unlike
    pyMolDyn and sovapy), so the number of occupied voxels is an unbiased
    estimate of the occupied volume at any resolution.

    Values are capped at one voxel diagonal by default: marching cubes only
    needs exact values next to atom surfaces to place cavity surfaces between
    voxel centres. Values below the cap are exact.

    Periodic boundary conditions are handled by wrapping voxel indices
    modulo ``grid_shape``.

    When ``radii_by_symbol`` is None, the ASE Van der Waals radius for each
    element is used, falling back to the ASE covalent radius for elements
    without VdW data. When a single number is given, that radius is applied
    uniformly to all atoms.

    Args:
        positions: Fractional coordinates of atoms, shape (n_atoms, 3).
            Values are in [0, 1).
        atomic_numbers: Atomic numbers for each atom, shape (n_atoms,).
        grid_shape: Integer grid dimensions (nx, ny, nz).
        radii_by_symbol: Radius specification (see ``_atom_radii``).
        cell: (3, 3) cell matrix with lattice vectors as rows.
        probe_radius: Added to every atom radius (A), so the field is the
            distance to the surface traced by a probe sphere's centre.
        cap: Largest stored value in A; None → one voxel diagonal.

    Returns:
        Float32 array of shape ``grid_shape``: distance in A to the nearest
        atom surface, negative inside atoms; occupied voxels are ``<= 0``.

    Examples:
        >>> import numpy as np
        >>> field = _build_distance_field(
        ...     np.array([[0.5, 0.5, 0.5]]), np.array([14]), (10, 10, 10), 1.5, np.diag([10.0, 10.0, 10.0])
        ... )
        >>> int((field <= 0).sum())  # 1 A voxels within 1.5 A: centre, 6 face and 12 edge neighbours
        19
        >>> float(field[5, 5, 5]), float(field[6, 5, 5])
        (-1.5, -0.5)
    """
    grid_shape_array = np.array(grid_shape)
    if cap is None:
        cap = _voxel_diagonal(grid_shape, cell)
    distance_field = np.full(grid_shape, cap, dtype=np.float32)
    perp_heights = cell_perpendicular_heights(cell)
    offsets_by_radius: dict[float, np.ndarray] = {}

    for atom_index, atom_radius in enumerate((_atom_radii(atomic_numbers, radii_by_symbol) + probe_radius).tolist()):
        reach_radius = atom_radius + cap
        if atom_radius not in offsets_by_radius:
            # Voxel extent of the reach along each axis, plus one voxel for the sub-voxel atom position
            reach = int(np.ceil(float(np.max(reach_radius / perp_heights * grid_shape_array)))) + 1
            offset_range = np.arange(-reach, reach + 1)
            offset_x, offset_y, offset_z = np.meshgrid(offset_range, offset_range, offset_range, indexing="ij")
            offsets_by_radius[atom_radius] = np.stack([offset_x.ravel(), offset_y.ravel(), offset_z.ravel()], axis=1)

        candidate_indices = (
            np.floor(positions[atom_index] * grid_shape_array).astype(int) + offsets_by_radius[atom_radius]
        )
        distances = np.linalg.norm((candidate_indices / grid_shape_array - positions[atom_index]) @ cell, axis=1)
        within_reach = distances <= reach_radius
        field_indices = candidate_indices[within_reach] % grid_shape_array

        # minimum.at keeps the nearest image when wrapped offsets repeat a voxel in small cells
        np.minimum.at(
            distance_field,
            (field_indices[:, 0], field_indices[:, 1], field_indices[:, 2]),
            (distances[within_reach] - atom_radius).astype(np.float32),
        )

    return distance_field


def _periodic_face_edges(raw_labels: np.ndarray) -> np.ndarray:
    """Collect the label pairs that touch across the periodic cell faces.

    For every axis, each voxel on the high face (index ``n - 1``) is paired with
    its 9 periodic neighbours on the low face (index 0), with the in-face
    offsets wrapped along the other two axes. Each pair carries the integer
    image shift (in cell units) that maps the low-face voxel next to the
    high-face voxel.

    Args:
        raw_labels: Integer label array from ``scipy.ndimage.label``
            (0 = occupied).

    Returns:
        Unique rows ``(label_high, label_low, shift_x, shift_y, shift_z)``,
        shape (n_edges, 5).

    Examples:
        >>> import numpy as np
        >>> labels = np.zeros((4, 4, 4), dtype=int)
        >>> labels[0, 1, 1] = 1
        >>> labels[3, 1, 1] = 2
        >>> _periodic_face_edges(labels).tolist()
        [[2, 1, 1, 0, 0]]
    """
    edge_blocks = []
    for axis in range(3):
        other_axes = [a for a in range(3) if a != axis]
        n_b, n_c = (raw_labels.shape[a] for a in other_axes)
        high_face = np.take(raw_labels, raw_labels.shape[axis] - 1, axis=axis)
        low_face = np.take(raw_labels, 0, axis=axis)
        index_b = np.arange(n_b)[:, None]
        index_c = np.arange(n_c)[None, :]
        for delta_b in (-1, 0, 1):
            for delta_c in (-1, 0, 1):
                neighbour = np.roll(low_face, shift=(-delta_b, -delta_c), axis=(0, 1))
                touching = (high_face > 0) & (neighbour > 0)
                if not touching.any():
                    continue
                shift_b = np.broadcast_to((index_b + delta_b >= n_b).astype(int) - (index_b + delta_b < 0), (n_b, n_c))
                shift_c = np.broadcast_to((index_c + delta_c >= n_c).astype(int) - (index_c + delta_c < 0), (n_b, n_c))
                block = np.zeros((int(touching.sum()), 5), dtype=np.int64)
                block[:, 0] = high_face[touching]
                block[:, 1] = neighbour[touching]
                block[:, 2 + axis] = 1
                block[:, 2 + other_axes[0]] = shift_b[touching]
                block[:, 2 + other_axes[1]] = shift_c[touching]
                edge_blocks.append(block)
    if not edge_blocks:
        return np.zeros((0, 5), dtype=np.int64)
    return np.unique(np.concatenate(edge_blocks), axis=0)


def _merge_pbc_labels(raw_labels: np.ndarray, raw_label_count: int) -> tuple[np.ndarray, set[int], np.ndarray]:
    """Merge cavity labels across periodic faces, tracking periodic images.

    A weighted union-find stores, for every raw component, the image shift (in
    cell units) that places it next to the root component of its cavity. A face
    contact between two components that already belong to the same cavity but
    imply a different image shift means the cavity connects to its own periodic
    image: it wraps around the cell and is percolating (infinite under PBC).

    Args:
        raw_labels: Integer label array from ``scipy.ndimage.label``
            (0 = occupied).
        raw_label_count: Number of distinct labels (excluding background 0).

    Returns:
        void_labels: Relabelled array with PBC-merged cavities (1..n).
        percolating_labels: Labels (in ``void_labels``) of percolating cavities.
        image_offsets: Integer image shift of each raw component relative to
            its cavity, shape (raw_label_count + 1, 3). Unwrapped voxel index
            = voxel index + image_offsets[raw_label] * grid_shape.

    Examples:
        >>> import numpy as np
        >>> labels = np.zeros((4, 4, 4), dtype=int)
        >>> labels[0, 1, 1] = 1
        >>> labels[3, 1, 1] = 2
        >>> void_labels, percolating, offsets = _merge_pbc_labels(labels, 2)
        >>> int(void_labels.max()), percolating
        (1, set())
    """
    parent = list(range(raw_label_count + 1))
    offset: list[tuple[int, int, int]] = [(0, 0, 0)] * (raw_label_count + 1)
    percolating_roots: set[int] = set()

    def find_root(label: int) -> tuple[int, tuple[int, int, int]]:
        path = []
        while parent[label] != label:
            path.append(label)
            label = parent[label]
        root = label
        accumulated = (0, 0, 0)
        for node in reversed(path):
            accumulated = tuple(a + b for a, b in zip(accumulated, offset[node], strict=True))
            offset[node] = accumulated
            parent[node] = root
        return root, offset[path[0]] if path else (0, 0, 0)

    for label_high, label_low, *shift in _periodic_face_edges(raw_labels).tolist():
        root_high, offset_high = find_root(label_high)
        root_low, offset_low = find_root(label_low)
        # The low-face component must sit at image offset_high + shift relative to root_high.
        required = tuple(h + s for h, s in zip(offset_high, shift, strict=True))
        if root_high == root_low:
            if required != offset_low:
                percolating_roots.add(root_high)
            continue
        parent[root_low] = root_high
        offset[root_low] = tuple(r - lo for r, lo in zip(required, offset_low, strict=True))
        if root_low in percolating_roots:
            percolating_roots.discard(root_low)
            percolating_roots.add(root_high)

    roots = np.zeros(raw_label_count + 1, dtype=np.int64)
    image_offsets = np.zeros((raw_label_count + 1, 3), dtype=np.int64)
    for label in range(1, raw_label_count + 1):
        root, label_offset = find_root(label)
        roots[label] = root
        image_offsets[label] = label_offset

    unique_roots, new_ids = np.unique(roots[1:], return_inverse=True)
    lookup = np.zeros(raw_label_count + 1, dtype=raw_labels.dtype)
    lookup[1:] = new_ids + 1
    root_to_new = {int(r): i + 1 for i, r in enumerate(unique_roots)}
    percolating_labels = {root_to_new[r] for r in percolating_roots}
    return lookup[raw_labels], percolating_labels, image_offsets


def _label_voids_with_pbc(occupied_grid: np.ndarray) -> tuple[np.ndarray, set[int], np.ndarray, np.ndarray]:
    """Find connected empty-voxel components with periodic boundary conditions.

    The grid spans exactly one periodic cell. Empty voxels are labelled with
    26-connectivity, then components touching across opposite faces (including
    edges and corners) are merged with ``_merge_pbc_labels``.

    Args:
        occupied_grid: Boolean array; True = occupied voxel.

    Returns:
        void_labels: Integer label array (0 = occupied; 1..n = void cavities).
        percolating_labels: Labels of cavities that wrap around the cell.
        raw_labels: Labels before the periodic merge.
        image_offsets: Image shift of each raw component, shape (n_raw + 1, 3).

    Examples:
        >>> import numpy as np
        >>> grid = np.ones((6, 6, 6), dtype=bool)
        >>> grid[:, 2, 2] = False  # channel along x wraps around the cell
        >>> void_labels, percolating, _, _ = _label_voids_with_pbc(grid)
        >>> int(void_labels.max()), percolating
        (1, {1})
    """
    connectivity_structure = np.ones((3, 3, 3), dtype=int)
    raw_labels, raw_label_count = ndimage.label(~occupied_grid, structure=connectivity_structure)

    if raw_label_count == 0:
        return raw_labels, set(), raw_labels, np.zeros((1, 3), dtype=np.int64)

    void_labels, percolating_labels, image_offsets = _merge_pbc_labels(raw_labels, raw_label_count)
    return void_labels, percolating_labels, raw_labels, image_offsets


def _compute_triangle_mesh_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute the surface area of a triangle mesh.

    Args:
        vertices: Vertex coordinates, shape (n_vertices, 3).
        faces: Triangle face indices, shape (n_faces, 3).

    Returns:
        Total surface area as sum of individual triangle areas.
    """
    edge_a = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge_b = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    cross_products = np.cross(edge_a, edge_b)
    return float(0.5 * np.sum(np.linalg.norm(cross_products, axis=1)))


def _cavity_level_field(distance_field: np.ndarray, member: np.ndarray) -> np.ndarray:
    """Level-set field whose zero contour is the surface of one cavity.

    Keeps the distance to the nearest atom surface on the cavity's voxels and
    on occupied voxels, and sets every other empty voxel to -1. Those belong to
    other cavities or to other periodic images of this one; they are never
    26-adjacent to the cavity, so they cannot create surfaces.

    Args:
        distance_field: Distance to the nearest atom surface (see
            ``_build_distance_field``).
        member: Boolean mask of the cavity's voxels, same shape.

    Returns:
        Float32 array for marching cubes at level 0.

    Examples:
        >>> import numpy as np
        >>> field = np.array([[[0.5, -0.25, 0.75]]], dtype=np.float32)
        >>> _cavity_level_field(field, np.array([[[True, False, False]]])).tolist()
        [[[0.5, -0.25, -1.0]]]
    """
    return np.where(member | (distance_field <= 0), distance_field, np.float32(-1.0))


def _local_cavity_mesh(unwrapped_indices: np.ndarray, distance_field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate one cavity from its unwrapped (contiguous) voxel indices.

    Marching cubes runs at level 0 on the distance to the nearest atom surface,
    so the surface lies between voxel centres where the distance changes sign.

    Args:
        unwrapped_indices: Integer voxel indices of the cavity after removing
            periodic wrapping, shape (n_voxels, 3).
        distance_field: Distance to the nearest atom surface on the periodic
            grid (see ``_build_distance_field``).

    Returns:
        vertices: Mesh vertices in unwrapped voxel-index coordinates.
        faces: Triangle face indices.

    Examples:
        >>> import numpy as np
        >>> field = np.full((10, 10, 10), -1.0, dtype=np.float32)
        >>> field[5, 5, 5] = 0.5
        >>> vertices, faces = _local_cavity_mesh(np.array([[5, 5, 5]]), field)
        >>> len(faces) > 0
        True
    """
    lower = unwrapped_indices.min(axis=0) - 1
    local_shape = unwrapped_indices.max(axis=0) - lower + 2
    wrapped_axes = [
        (np.arange(size) + start) % n for size, start, n in zip(local_shape, lower, distance_field.shape, strict=True)
    ]
    local_field = distance_field[np.ix_(*wrapped_axes)]
    member = np.zeros(tuple(local_shape), dtype=bool)
    local_indices = unwrapped_indices - lower
    member[local_indices[:, 0], local_indices[:, 1], local_indices[:, 2]] = True
    vertices, faces = mcubes.marching_cubes(_cavity_level_field(local_field, member), 0.0)
    return vertices + lower, faces


def _display_cavity_mesh(
    voxel_indices: np.ndarray,
    image_shifts: np.ndarray | None,
    distance_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate one cavity for display, in voxel-index coordinates.

    A percolating cavity is drawn over exactly one cell: its level field is
    padded with one periodically wrapped layer at the high end of each axis,
    so the marching-cubes cells joining voxel ``n - 1`` to voxel 0 are included
    once and no cell is duplicated.

    Args:
        voxel_indices: In-cell voxel indices of the cavity, shape (n_voxels, 3).
        image_shifts: Per-voxel image shifts from the periodic merge, or None
            for a percolating cavity (drawn as its in-cell fragments).
        distance_field: Distance to the nearest atom surface on the periodic
            grid (see ``_build_distance_field``).

    Returns:
        vertices: Mesh vertices in voxel-index coordinates.
        faces: Triangle face indices.

    Examples:
        >>> import numpy as np
        >>> field = np.full((6, 6, 6), -1.0, dtype=np.float32)
        >>> field[0, 2, 2] = field[5, 2, 2] = 0.5  # neighbours across the x face
        >>> indices = np.array([[0, 2, 2], [5, 2, 2]])
        >>> shifts = np.array([[1, 0, 0], [0, 0, 0]])
        >>> vertices, faces = _display_cavity_mesh(indices, shifts, field)
        >>> bool(vertices[:, 0].max() <= 7.0)
        True
    """
    if image_shifts is None:
        void_mask = np.zeros(distance_field.shape, dtype=bool)
        void_mask[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True
        padded_field = np.pad(_cavity_level_field(distance_field, void_mask), ((0, 1),) * 3, mode="wrap")
        return mcubes.marching_cubes(padded_field, 0.0)
    # One contiguous surface, shifted so its centre lies inside the cell.
    grid_shape = np.array(distance_field.shape, dtype=np.int64)
    unwrapped_indices = voxel_indices + image_shifts * grid_shape
    unwrapped_indices -= np.floor(unwrapped_indices.mean(axis=0) / grid_shape).astype(np.int64) * grid_shape
    return _local_cavity_mesh(unwrapped_indices, distance_field)


def _fibonacci_sphere(point_count: int) -> np.ndarray:
    """Nearly uniform unit vectors on a sphere (Fibonacci lattice).

    Args:
        point_count: Number of points.

    Returns:
        Unit vectors, shape (point_count, 3).

    Examples:
        >>> import numpy as np
        >>> directions = _fibonacci_sphere(100)
        >>> directions.shape, bool(np.allclose(np.linalg.norm(directions, axis=1), 1.0))
        ((100, 3), True)
    """
    index = np.arange(point_count) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / point_count)
    azimuth = np.pi * (1.0 + 5.0**0.5) * index
    return np.stack([np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)], axis=1)


def _exposed_surface_points(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    radii: np.ndarray,
    cell: np.ndarray,
    points_per_area: float = _SURFACE_POINTS_PER_A2,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the parts of the atom spheres not covered by other atoms (Shrake-Rupley).

    Each atom carries ``max(50, round(points_per_area * 4 pi r^2))`` Fibonacci
    points, each standing for ``4 pi r^2 / point_count`` of area. Every atom's
    point set is turned by its own random rotation (fixed seed, so results are
    reproducible): a common orientation aliases with regular atom arrangements
    (1.7 % area error on a simple-cubic lattice at 10 points per A^2). A point
    is exposed when it lies outside every other atom sphere, periodic images
    included. Neighbours come from ``get_neighbors`` with pair cutoffs
    ``r_i + r_j``; when the cutoff exceeds half a perpendicular cell height the
    search runs on a replicated cell, so atoms overlapping several images of
    a neighbour are handled.

    Args:
        positions: Cartesian atom positions in A, shape (n_atoms, 3).
        atomic_numbers: Atomic numbers, shape (n_atoms,).
        radii: Atom radii in A, shape (n_atoms,); equal for atoms of one element.
        cell: (3, 3) cell matrix with lattice vectors as rows.
        points_per_area: Points per A^2 of sphere area.

    Returns:
        points: Cartesian coordinates of the exposed points, shape (n_points, 3).
        weights: Area in A^2 represented by each point, shape (n_points,).

    Examples:
        >>> import numpy as np
        >>> points, weights = _exposed_surface_points(
        ...     np.array([[5.0, 5.0, 5.0], [7.0, 5.0, 5.0]]), np.array([8, 8]), np.array([1.5, 1.5]), np.eye(3) * 10.0
        ... )
        >>> round(float(weights.sum()), 1)  # exact: 2 * 2 pi r (r + d / 2) = 47.12; the rest is sampling noise
        47.3
    """
    atom_count = len(positions)
    if atom_count == 0 or float(radii.max()) <= 0.0:
        return np.zeros((0, 3)), np.zeros(0)

    radius_by_number = dict(zip(np.asarray(atomic_numbers).tolist(), radii.tolist(), strict=True))
    pair_cutoffs = {
        (number_i, number_j): radius_i + radius_j
        for number_i, radius_i in radius_by_number.items()
        for number_j, radius_j in radius_by_number.items()
    }
    # Replicate the cell until the largest cutoff is at most half of every perpendicular height
    repeats = np.maximum(1, np.ceil(4.0 * float(radii.max()) / cell_perpendicular_heights(cell))).astype(int)
    shifts = np.array(np.meshgrid(*(np.arange(count) for count in repeats), indexing="ij")).reshape(3, -1).T
    search_positions = (positions[None, :, :] + (shifts @ cell)[:, None, :]).reshape(-1, 3)
    search_numbers = np.tile(np.asarray(atomic_numbers), len(shifts))
    neighbor_lists = get_neighbors(
        (search_positions, search_numbers, repeats[:, None] * cell), pair_cutoffs, return_vectors=True
    )

    rotations = Rotation.random(atom_count, np.random.default_rng(0)).as_matrix()
    unit_spheres: dict[int, np.ndarray] = {}
    point_blocks = []
    weight_blocks = []
    for central_id, neighbor_ids, vectors in neighbor_lists:
        atom_index = central_id - 1
        if atom_index >= atom_count:  # replica of an atom in the original cell
            continue
        sphere_area = 4.0 * np.pi * radii[atom_index] ** 2
        point_count = max(_MIN_SURFACE_POINTS, round(points_per_area * sphere_area))
        if point_count not in unit_spheres:
            unit_spheres[point_count] = _fibonacci_sphere(point_count)
        offsets = radii[atom_index] * unit_spheres[point_count] @ rotations[atom_index].T
        neighbor_radii = radii[(np.asarray(neighbor_ids, dtype=np.int64) - 1) % atom_count]
        # vectors are r_i - r_j, so offset + vector is the point relative to neighbour j. The nearest
        # neighbours bury most points, so only the survivors are tested against the others.
        nearest_first = np.argsort(np.einsum("ij,ij->i", vectors, vectors), kind="stable")
        exposed = np.ones(point_count, dtype=bool)
        for block in (nearest_first[:8], nearest_first[8:]):
            candidates = np.flatnonzero(exposed)
            if len(block) == 0 or len(candidates) == 0:
                continue
            separations = offsets[candidates][None, :, :] + vectors[block][:, None, :]
            exposed[candidates] = np.all(np.linalg.norm(separations, axis=2) >= neighbor_radii[block][:, None], axis=0)
        point_blocks.append(positions[atom_index] + offsets[exposed])
        weight_blocks.append(np.full(int(exposed.sum()), sphere_area / point_count))
    return np.concatenate(point_blocks), np.concatenate(weight_blocks)


def _assign_points_to_cavities(
    points: np.ndarray,
    weights: np.ndarray,
    void_labels: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    """Sum the area of exposed surface points per cavity.

    Each point goes to the cavity of the nearest empty voxel centre among the
    8 x 8 x 8 voxels around it (up to 3 voxels away along each axis). A point
    with no empty voxel there borders a crevice narrower than the grid
    resolves, and its area is left unassigned.

    Args:
        points: Cartesian coordinates of the points, shape (n_points, 3).
        weights: Area in A^2 represented by each point, shape (n_points,).
        void_labels: Integer label array (0 = occupied; 1..n = cavities).
        cell: (3, 3) cell matrix with lattice vectors as rows.

    Returns:
        Area in A^2 per label, shape (n_cavities + 1,); entry 0 is the
        unassigned area.

    Examples:
        >>> import numpy as np
        >>> labels = np.zeros((4, 4, 4), dtype=int)
        >>> labels[3, 0, 0] = 1
        >>> _assign_points_to_cavities(np.array([[0.4, 0.4, 0.4]]), np.array([2.0]), labels, np.eye(3) * 4.0).tolist()
        [0.0, 2.0]
    """
    grid_shape = np.array(void_labels.shape)
    fractional = points @ np.linalg.inv(cell)
    base_indices = np.floor(fractional * grid_shape).astype(np.int64)
    base_vectors = (base_indices / grid_shape - fractional) @ cell
    offsets = np.array(np.meshgrid(*(_ASSIGNMENT_OFFSETS,) * 3, indexing="ij")).reshape(3, -1).T
    inner = np.all((offsets >= -1) & (offsets <= 2), axis=1)  # noqa: PLR2004
    # Outer-shell voxels are at least two voxel heights from the point along some axis
    outer_bound_squared = (2.0 * float(np.min(cell_perpendicular_heights(cell) / grid_shape))) ** 2
    nearest_squared = np.full(len(points), np.inf)
    point_labels = np.zeros(len(points), dtype=np.int64)

    selection = np.arange(len(points))
    for shell in (inner, ~inner):
        for offset in offsets[shell]:
            candidate = (base_indices[selection] + offset) % grid_shape
            candidate_labels = void_labels[candidate[:, 0], candidate[:, 1], candidate[:, 2]]
            squared = np.sum((base_vectors[selection] + (offset / grid_shape) @ cell) ** 2, axis=1)
            closer = (candidate_labels > 0) & (squared < nearest_squared[selection])
            nearest_squared[selection[closer]] = squared[closer]
            point_labels[selection[closer]] = candidate_labels[closer]
        selection = np.flatnonzero(nearest_squared > outer_bound_squared)
    return np.bincount(point_labels, weights=weights, minlength=int(void_labels.max()) + 1)


def _compute_gyration_tensor_descriptors(
    cavity_voxel_positions: np.ndarray,
) -> tuple[float, float, float]:
    """Compute shape descriptors from the volume-weighted gyration tensor.

    Implements the pyMolDyn definition (Meyer et al., J. Comput. Chem. 38,
    389-394, 2017): the gyration tensor is computed over all voxels inside
    the cavity, each weighted equally (uniform voxel volume). This matches
    Eq. R = (1/V_C) sum_j v_j * r_j * r_j^T from the paper.

    Shape descriptors are derived from the ordered eigenvalues
    lambda1 >= lambda2 >= lambda3 of R:

        Rg2 = lambda1 + lambda2 + lambda3
        eta = (lambda1 - 0.5*(lambda2 + lambda3)) / Rg2
        c   = (lambda2 - lambda3) / Rg2
        kappa = sqrt(eta^2 + 0.75*c^2)     [Theodorou & Suter, 1985]

    Args:
        cavity_voxel_positions: Continuous coordinates of all voxels inside
            the cavity, shape (n_voxels, 3), in A.

    Returns:
        asphericity: eta in [0, 1]; zero for spherically symmetric cavities.
        acylindricity: c in [0, 1]; zero for cylindrically symmetric cavities.
        anisotropy: kappa in [0, 1]; zero for spherical, one for collinear.
    """
    centred_positions = cavity_voxel_positions - cavity_voxel_positions.mean(axis=0)
    gyration_tensor = (centred_positions.T @ centred_positions) / len(centred_positions)
    eigenvalues = sorted(np.linalg.eigvalsh(gyration_tensor).tolist(), reverse=True)

    squared_gyration_radius = sum(eigenvalues)
    if squared_gyration_radius == 0.0:
        return 0.0, 0.0, 0.0

    asphericity = (eigenvalues[0] - 0.5 * (eigenvalues[1] + eigenvalues[2])) / squared_gyration_radius
    acylindricity = (eigenvalues[1] - eigenvalues[2]) / squared_gyration_radius
    anisotropy = float((asphericity**2 + 0.75 * acylindricity**2) ** 0.5)

    return float(asphericity), float(acylindricity), anisotropy


# ============================================================================
# Internal grid builder (shared by compute_cavities and visualize_cavities)
# ============================================================================


def _build_cavity_grid(
    structure: Atoms,
    resolution: int,
    cutoff_radii: dict[str, float] | float | str | None,
    probe_radius: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[int], np.ndarray, np.ndarray, np.ndarray]:
    """Discretize the structure and label void cavities.

    Args:
        structure: ASE Atoms object (will be wrapped internally).
        resolution: Number of voxels along the longest box dimension.
        cutoff_radii: Radius specification passed through to ``_build_distance_field``.
        probe_radius: Probe radius in A added to every atom radius; cavities are
            then the regions where the probe centre fits.

    Raises:
        ValueError: If ``probe_radius`` is negative or NaN.

    Returns:
        void_labels: Integer label array (0 = occupied; 1..n = void cavities).
        occupied_grid: Boolean array; True = occupied voxel.
        cell_array: (3, 3) cell matrix with lattice vectors as rows.
        percolating_labels: Labels of cavities that wrap around the cell.
        raw_labels: Labels before the periodic merge.
        image_offsets: Image shift of each raw component, shape (n_raw + 1, 3).
        distance_field: Distance to the nearest atom surface (see ``_build_distance_field``).
    """
    if not probe_radius >= 0.0:
        msg = f"probe_radius must be a non-negative distance in A, got {probe_radius}"
        raise ValueError(msg)
    wrapped_structure = structure.copy()
    wrapped_structure.wrap()

    cell_array = wrapped_structure.get_cell().array
    box_lengths = cell_perpendicular_heights(cell_array)

    voxel_step = _compute_voxel_step(box_lengths, resolution)
    grid_shape = _compute_grid_shape(box_lengths, voxel_step)

    fractional_positions = wrapped_structure.get_scaled_positions()
    atomic_numbers = wrapped_structure.get_atomic_numbers()

    distance_field = _build_distance_field(
        fractional_positions,
        atomic_numbers,
        grid_shape,
        cutoff_radii,
        cell_array,
        probe_radius,
    )
    occupied_grid = distance_field <= 0

    void_labels, percolating_labels, raw_labels, image_offsets = _label_voids_with_pbc(occupied_grid)
    return void_labels, occupied_grid, cell_array, percolating_labels, raw_labels, image_offsets, distance_field


def _iter_cavity_voxels(
    void_labels: np.ndarray,
    raw_labels: np.ndarray,
    image_offsets: np.ndarray,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield the voxels of each cavity, grouped with a single sort.

    Args:
        void_labels: Integer label array (0 = occupied; 1..n = void cavities).
        raw_labels: Labels before the periodic merge.
        image_offsets: Image shift of each raw component, shape (n_raw + 1, 3).

    Yields:
        Tuples ``(void_index, voxel_indices, image_shifts)``: the cavity label,
        its in-cell voxel indices (n_voxels, 3) and the per-voxel image shifts
        (n_voxels, 3). Unwrapped indices = voxel_indices + image_shifts * grid_shape.

    Examples:
        >>> import numpy as np
        >>> labels = np.array([[[0, 1], [2, 1]]])
        >>> [(i, v.tolist()) for i, v, _ in _iter_cavity_voxels(labels, labels, np.zeros((3, 3), dtype=int))]
        [(1, [[0, 0, 1], [0, 1, 1]]), (2, [[0, 1, 0]])]
    """
    flat_labels = void_labels.ravel()
    flat_raw_labels = raw_labels.ravel()
    sort_order = np.argsort(flat_labels, kind="stable")
    boundaries = np.searchsorted(flat_labels[sort_order], np.arange(1, int(void_labels.max()) + 2))
    for void_index in range(1, int(void_labels.max()) + 1):
        flat_indices = sort_order[boundaries[void_index - 1] : boundaries[void_index]]
        voxel_indices = np.column_stack(np.unravel_index(flat_indices, void_labels.shape))
        yield void_index, voxel_indices, image_offsets[flat_raw_labels[flat_indices]]


# ============================================================================
# Public API
# ============================================================================


def compute_cavities(
    structure: Atoms,
    resolution: int = 128,
    cutoff_radii: dict[str, float] | float | str | None = None,
    probe_radius: float = 0.0,
) -> dict[str, np.ndarray]:
    """Compute cavity distribution and shape properties for a glass structure.

    Domain-based cavity detection following pyMolDyn (Meyer et al. 2017). The
    simulation box is discretised into a 3D voxel grid; voxels whose centre
    lies within the atomic radius of any atom are marked as occupied.
    Connected regions of empty voxels -- detected with full 26-connectivity
    and periodic-boundary-condition awareness -- constitute cavities (domains).

    Unlike pyMolDyn and sovapy, atoms are not snapped to the nearest voxel and
    radii are not rounded to whole voxels: every voxel is tested against the
    true atom position and radius, with the same rule for orthogonal and
    triclinic cells. The empty-voxel volume is therefore an unbiased estimate
    of the free volume at any resolution, and values differ from pyMolDyn and
    sovapy at the same resolution.

    The grid spans exactly one periodic cell. Cavities that cross a cell face
    are joined to their continuation on the opposite face (including edges and
    corners) with a union-find merge that tracks periodic images, so every
    cavity is found regardless of where the cell boundary cuts it.

    A cavity that connects to its own periodic image wraps around the cell and
    is infinite under PBC (percolating). It is kept, with its volume and
    surface area per cell, flagged in ``'percolating'``, and its shape
    descriptors set to NaN. Every empty voxel belongs to exactly one cavity, so
    ``volumes.sum()`` is the total unoccupied volume of the cell.

    The boundary of the empty space consists of the parts of the atom spheres
    not covered by other atoms. Surface areas are the areas of these exposed
    patches, sampled with 20 Shrake-Rupley points per A^2 (each atom's point
    set randomly rotated with a fixed seed), and each point is assigned to
    the cavity of the nearest empty voxel within 3 voxels along each axis.
    The resolution decides which cavity a patch belongs to and affects the
    total only through the dropped area: points with no empty voxel that
    close border crevices the grid does not resolve, and their area is
    dropped (0.07 % of the total for a 10,020-atom sodium silicate glass at
    resolution 128; 0.002 % at resolution 64, where the 3-voxel search
    reaches farther).

    The total free volume converges quickly with resolution, but cavity
    counts, percolation and the split of volume and area between cavities
    converge more slowly: channels and necks narrower than about one voxel
    are not resolved. Check them at two resolutions.

    Only domains are computed. pyMolDyn additionally offers center-based and
    surface-based cavities, which extend each domain into the atom spheres
    (to points nearer to a domain centre or domain surface point than to the
    nearest atom); those are not implemented here.

    With ``probe_radius`` > 0 (e.g. 1.20 A, the Van der Waals radius of a
    neutral H atom), every atom radius is increased by the probe radius, so
    the cavities are the regions where the probe centre fits: each one holds
    at least one probe, and a percolating one lets the probe cross the cell.
    ``'volumes'`` and ``'surface_areas'`` then describe the space of the probe
    centre (accessible volume and surface). The accessible volume converges
    with resolution, but in dense glasses most sites are pockets smaller than
    a voxel, so the number of sites keeps growing with resolution (180, 236
    and 301 at resolutions 128, 256 and 384 for H in a sodium silicate glass
    with ionic radii); the space the probe body can fill is not computed.
    The result depends strongly on the atom radii: with Van der Waals radii
    almost no site in a dense silicate glass fits an H atom.

    Args:
        structure: ASE Atoms object with atomic coordinates and cell.
        resolution: Number of voxels along the longest box dimension
            (default 128). Higher values resolve narrower channels and
            crevices between cavities and finer cavity shapes, at the cost of
            memory and runtime.
        cutoff_radii: Atomic radius specification. Four forms are accepted:

            - ``None`` (default): element-specific ASE Van der Waals radii,
              with ASE covalent radii as fallback.
            - a number (int or float, not bool): uniform radius in A applied
              to every atom (e.g. ``cutoff_radii=2.8``, sovapy's default
              radius).
            - ``dict[str, float]``: per-element overrides in A
              (e.g. ``{'Si': 2.1, 'O': 1.52}``); elements not in the dict
              fall back to ASE covalent radii.
            - ``"ionic"``: Shannon (1976) ionic radii via pymatgen for
              two-coordinated O2-, tetrahedral network formers (Si, B, Al,
              P, Ge, Zn), octahedral Ti and modifiers (Na, K, Mg, Ca, Sr, Ba)
              and tetrahedral Li; other elements raise ValueError.

        probe_radius: Radius in A of a probe sphere (default 0, no probe).

    Returns:
        Dictionary with the following keys, each mapping to a 1D array with one
        entry per detected cavity:

        - ``'volumes'``: cavity volumes in A^3 (per cell for percolating ones).
        - ``'surface_areas'``: cavity surface areas in A^2 (per cell for
          percolating ones): exposed atom-sphere area bounding the cavity.
          Totals are within 0.1 % of analytic values for overlapping-sphere
          test geometries; small cavities carry point-sampling noise
          (about 7 % at the 90th percentile for 2-10 A^2 cavities).
        - ``'asphericities'``: asphericity eta in [0, 1] (NaN if percolating).
        - ``'acylindricities'``: acylindricity c in [0, 1] (NaN if percolating).
        - ``'anisotropies'``: relative shape anisotropy kappa in [0, 1]
          (NaN if percolating).
        - ``'percolating'``: bool, True for cavities that wrap around the cell.
        - ``'max_inscribed_radii'``: radius in A of the largest sphere centred
          on a cavity voxel that touches no atom (largest included sphere),
          sampled at voxel centres, so low by up to half a voxel diagonal.

    Raises:
        TypeError: If ``cutoff_radii`` is a bool.
        ValueError: If ``probe_radius`` is negative, or ``cutoff_radii`` is an
            unknown string or ``"ionic"`` with an element without ionic radius.

    Examples:
        >>> from ase.io import read
        >>> structure = read('glass.xyz')
        >>> cavities = compute_cavities(structure, resolution=128)
        >>> free_volume_fraction = cavities['volumes'].sum() / structure.get_volume()
    """
    (
        void_labels,
        _occupied_grid,
        cell_array,
        percolating_labels,
        raw_labels,
        image_offsets,
        distance_field,
    ) = _build_cavity_grid(structure, resolution, cutoff_radii, probe_radius)
    number_of_voids = int(void_labels.max())
    grid_shape = np.array(void_labels.shape, dtype=float)

    if number_of_voids == 0:
        empty: np.ndarray = np.array([], dtype=np.float64)
        return {
            "volumes": empty,
            "surface_areas": empty,
            "asphericities": empty,
            "acylindricities": empty,
            "anisotropies": empty,
            "percolating": np.array([], dtype=bool),
            "max_inscribed_radii": empty,
        }

    # Voxel volume: exact cell volume divided by total voxel count.
    voxel_volume = abs(float(np.linalg.det(cell_array))) / float(np.prod(grid_shape))
    grid_shape_int = np.array(void_labels.shape, dtype=np.int64)

    radii = _atom_radii(structure.get_atomic_numbers(), cutoff_radii)
    surface_points, surface_weights = _exposed_surface_points(
        structure.get_positions(),
        structure.get_atomic_numbers(),
        radii + probe_radius,
        cell_array,
    )
    surface_areas = _assign_points_to_cavities(surface_points, surface_weights, void_labels, cell_array)[1:]

    # Largest included sphere. The field is exact below its cap, so rebuild it with a doubled cap until no
    # cavity reaches the cap; this ends once the cap exceeds the largest distance to an atom.
    cavity_indices = np.arange(1, number_of_voids + 1)
    cap = _voxel_diagonal(void_labels.shape, cell_array)
    deepest = np.asarray(ndimage.maximum(distance_field, void_labels, index=cavity_indices), dtype=np.float64)
    while len(structure) > 0 and np.any(deepest >= np.float32(cap)):
        cap *= 2.0
        deeper_field = _build_distance_field(
            structure.get_scaled_positions(),
            structure.get_atomic_numbers(),
            void_labels.shape,
            cutoff_radii,
            cell_array,
            probe_radius,
            cap,
        )
        deepest = np.asarray(ndimage.maximum(deeper_field, void_labels, index=cavity_indices), dtype=np.float64)
    max_inscribed_radii = deepest + probe_radius if len(structure) > 0 else np.full(number_of_voids, np.inf)

    volumes = np.zeros(number_of_voids, dtype=np.float64)
    asphericities = np.zeros(number_of_voids, dtype=np.float64)
    acylindricities = np.zeros(number_of_voids, dtype=np.float64)
    anisotropies = np.zeros(number_of_voids, dtype=np.float64)
    percolating = np.zeros(number_of_voids, dtype=bool)

    for void_index, voxel_indices, image_shifts in _iter_cavity_voxels(void_labels, raw_labels, image_offsets):
        volumes[void_index - 1] = len(voxel_indices) * voxel_volume

        if void_index in percolating_labels:
            percolating[void_index - 1] = True
            asphericities[void_index - 1] = acylindricities[void_index - 1] = anisotropies[void_index - 1] = np.nan
            continue

        unwrapped_indices = voxel_indices + image_shifts * grid_shape_int
        if len(unwrapped_indices) >= 3:  # noqa: PLR2004
            cavity_voxel_coordinates = (unwrapped_indices / grid_shape) @ cell_array
            (
                asphericities[void_index - 1],
                acylindricities[void_index - 1],
                anisotropies[void_index - 1],
            ) = _compute_gyration_tensor_descriptors(cavity_voxel_coordinates)

    return {
        "volumes": volumes,
        "surface_areas": surface_areas,
        "asphericities": asphericities,
        "acylindricities": acylindricities,
        "anisotropies": anisotropies,
        "percolating": percolating,
        "max_inscribed_radii": max_inscribed_radii,
    }


def visualize_cavities(
    structure: Atoms,
    resolution: int = 128,
    cutoff_radii: dict[str, float] | float | str | None = None,
    show_atoms: bool = True,  # noqa: FBT001
    opacity: float = 0.4,
    excluded_cavities: float | None = None,
    probe_radius: float = 0.0,
) -> object:
    """Render cavity surfaces and optionally atom positions in an interactive 3D plot.

    Builds a marching-cubes mesh (level 0 of the distance to the nearest atom
    surface) for every cavity detected by ``compute_cavities`` and draws it as
    a semi-transparent coloured surface using plotly. Atom positions are
    overlaid as small scatter markers when ``show_atoms=True``.

    Requires ``plotly`` to be installed (``pip install plotly``).

    Args:
        structure: ASE Atoms object with atomic coordinates and cell.
        resolution: Voxel resolution (same meaning as in ``compute_cavities``).
        cutoff_radii: Atomic radius specification (same meaning as in
            ``compute_cavities``). None uses ASE VdW radii (covalent fallback).
        show_atoms: If True, overlay atom positions as scatter markers coloured by element.
        opacity: Opacity of the cavity surface meshes (0 = transparent, 1 = opaque).
        excluded_cavities: Volume threshold in Å³. Cavities with volume greater than
            or equal to this value are not rendered. If None, all cavities are shown.
        probe_radius: Probe radius in A (same meaning as in ``compute_cavities``);
            with a probe the drawn surfaces enclose the space of the probe centre.

    Returns:
        A plotly ``Figure`` object. Call ``.show()`` to display it in a notebook or browser.

    Raises:
        ImportError: If plotly is not installed.

    Examples:
        >>> from ase.io import read
        >>> from amorphouspy.properties.structural.cavities import visualize_cavities
        >>> structure = read('glass.xyz')
        >>> fig = visualize_cavities(structure, excluded_cavities=50.0)
        >>> fig.show()
    """
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError as exc:
        error_message = "plotly is required for visualize_cavities. Install it with: pip install plotly"
        raise ImportError(error_message) from exc

    (
        void_labels,
        _occupied_grid,
        cell_array,
        percolating_labels,
        raw_labels,
        image_offsets,
        distance_field,
    ) = _build_cavity_grid(structure, resolution, cutoff_radii, probe_radius)

    grid_shape = np.array(void_labels.shape, dtype=float)
    voxel_volume = abs(float(np.linalg.det(cell_array))) / float(np.prod(grid_shape))
    mc_to_cart = np.diag(1.0 / grid_shape) @ cell_array

    # Colour palette for distinct cavities (cycles if more cavities than colours)
    cavity_colours = [
        "#e41a1c",
        "#377eb8",
        "#4daf4a",
        "#984ea3",
        "#ff7f00",
        "#a65628",
        "#f781bf",
        "#999999",
        "#66c2a5",
        "#fc8d62",
    ]

    traces: list = []

    rendered_count = 0
    for void_index, voxel_indices, image_shifts in _iter_cavity_voxels(void_labels, raw_labels, image_offsets):
        voxel_count = len(voxel_indices)
        if voxel_count < 2:  # noqa: PLR2004
            continue

        volume = float(voxel_count) * voxel_volume
        if excluded_cavities is not None and volume >= excluded_cavities:
            continue

        try:
            vertices, faces = _display_cavity_mesh(
                voxel_indices, None if void_index in percolating_labels else image_shifts, distance_field
            )
            vertices = vertices @ mc_to_cart
        except (ValueError, RuntimeError):
            continue

        colour = cavity_colours[rendered_count % len(cavity_colours)]
        rendered_count += 1

        traces.append(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=colour,
                opacity=opacity,
                name=f"Cavity {void_index} ({volume:.2f} Å³)",
                showlegend=True,
                flatshading=False,
            )
        )

    if show_atoms:
        wrapped = structure.copy()
        wrapped.wrap()
        positions = wrapped.get_positions()
        atom_symbols = wrapped.get_chemical_symbols()

        element_colour_map: dict[str, str] = {
            "Si": "#f0c040",
            "O": "#e84040",
            "Na": "#6060f0",
            "Al": "#808080",
            "B": "#40c040",
            "Li": "#c040c0",
            "K": "#40c0c0",
            "Ca": "#c08040",
            "Mg": "#80c080",
        }
        default_atom_colour = "#aaaaaa"

        seen_elements: set[str] = set()
        elements_in_structure = sorted(set(atom_symbols))
        for element in elements_in_structure:
            mask = np.array([s == element for s in atom_symbols])
            element_positions = positions[mask]
            atom_colour = element_colour_map.get(element, default_atom_colour)
            traces.append(
                go.Scatter3d(
                    x=element_positions[:, 0],
                    y=element_positions[:, 1],
                    z=element_positions[:, 2],
                    mode="markers",
                    marker={"size": 2, "color": atom_colour, "opacity": 0.5},
                    name=element,
                    showlegend=element not in seen_elements,
                )
            )
            seen_elements.add(element)

    # Cartesian bounding box of the cell (works for orthogonal and triclinic)
    corners = np.array([[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]], dtype=float) @ cell_array
    axis_min = corners.min(axis=0)
    axis_max = corners.max(axis=0)

    figure = go.Figure(data=traces)
    figure.update_layout(
        scene={
            "xaxis": {"range": [float(axis_min[0]), float(axis_max[0])], "title": "x (Å)"},
            "yaxis": {"range": [float(axis_min[1]), float(axis_max[1])], "title": "y (Å)"},
            "zaxis": {"range": [float(axis_min[2]), float(axis_max[2])], "title": "z (Å)"},
            "aspectmode": "data",
        },
        title=(
            f"Cavities ({rendered_count} shown"
            + (f", excluded ≥ {excluded_cavities:.1f} Å³" if excluded_cavities is not None else "")
            + ")"
        ),
        legend={"itemsizing": "constant"},
    )
    return figure
