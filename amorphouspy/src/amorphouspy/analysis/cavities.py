"""Structural analysis functions for multicomponent glass systems.

Author: Achraf Atila (achraf.atila@bam.de)

Identifies and characterises cavities in atomic structures. The simulation
box is discretised into a 3D voxel grid; voxels within the atomic radius
(ASE Van der Waals radii by default) of any atom are marked as occupied.
Connected regions of empty voxels, detected with full 26-connectivity and
periodic-boundary-condition awareness, constitute cavities.

For each cavity the following properties are computed:

- Volume: number of empty voxels x voxel volume (A^3).
- Surface area: area of the cavity-solid interface via marching
  cubes triangulation (A^2).
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

from typing import TYPE_CHECKING

import numpy as np
from ase.data import atomic_numbers as ase_atomic_numbers
from ase.data import covalent_radii as ase_covalent_radii
from ase.data import vdw_radii as ase_vdw_radii
from scipy import ndimage
from skimage.measure import marching_cubes

from amorphouspy.neighbors import cell_perpendicular_heights

if TYPE_CHECKING:
    from ase import Atoms

_PADDING_VOXELS: int = 4
_BORDER_VOXELS: int = _PADDING_VOXELS // 2  # 2-voxel physical boundary inside the padded grid


# ============================================================================
# Internal helpers
# ============================================================================


def _compute_voxel_step(box_lengths: np.ndarray, resolution: int) -> float:
    """Compute the voxel edge length matching sovapy's discretisation scheme.

    Args:
        box_lengths: Side lengths of the simulation box in A, shape (3,).
        resolution: Grid resolution (number of voxels along the longest side
            before padding).

    Returns:
        Voxel edge length in A.

    Examples:
        >>> import numpy as np
        >>> _compute_voxel_step(np.array([20.0, 20.0, 20.0]), resolution=64)
        0.3333333333333333
    """
    return float(np.max(box_lengths)) / (resolution - _PADDING_VOXELS)


def _compute_grid_shape(box_lengths: np.ndarray, voxel_step: float) -> tuple[int, ...]:
    """Compute the 3D grid shape for the given box and voxel size.

    Args:
        box_lengths: Side lengths of the simulation box in A, shape (3,).
        voxel_step: Voxel edge length in A.

    Returns:
        Integer grid dimensions (nx, ny, nz).

    Examples:
        >>> import numpy as np
        >>> lengths = np.array([10.0, 10.0, 10.0])
        >>> _compute_grid_shape(lengths, voxel_step=1.0)
        (14, 14, 14)
    """
    return tuple(int(np.floor(length / voxel_step)) + _PADDING_VOXELS for length in box_lengths)


def _build_occupied_grid(
    positions: np.ndarray,
    atomic_numbers: np.ndarray,
    grid_shape: tuple[int, ...],
    voxel_step: float,
    radii_by_symbol: dict[str, float] | float | None,
    cell: np.ndarray,
) -> np.ndarray:
    """Mark voxels occupied by atoms on a 3D boolean grid.

    For orthogonal cells, replicates sovapy's discretization scheme: the
    physical radius of each atom is first rounded to the nearest integer
    number of voxels (``discrete_radius = round(radius / voxel_step)``),
    then all voxels within that integer radius (Euclidean distance in voxel
    units) are marked as occupied.

    For triclinic cells, a bounding box in voxel space is derived from the
    perpendicular heights, and each candidate voxel is accepted only when
    its actual Cartesian distance to the atom is within the atom radius.

    Periodic boundary conditions are handled by wrapping voxel indices
    modulo ``grid_shape``.

    When ``radii_by_symbol`` is None, the ASE Van der Waals radius for each
    element is used, falling back to the ASE covalent radius for elements
    without VdW data. When a single float is given, that radius is applied
    uniformly to all atoms.

    Args:
        positions: Fractional coordinates of atoms, shape (n_atoms, 3).
            Values are in [0, 1).
        atomic_numbers: Atomic numbers for each atom, shape (n_atoms,).
        grid_shape: Integer grid dimensions (nx, ny, nz).
        voxel_step: Voxel edge length in A.
        radii_by_symbol: Radius specification. None → element-specific ASE VdW
            radii (covalent fallback). A float → uniform radius for all atoms.
            A dict → per-element overrides (missing elements fall back to ASE
            covalent radii).
        cell: (3, 3) cell matrix with lattice vectors as rows.

    Returns:
        Boolean array of shape ``grid_shape``; True = occupied voxel.
    """
    occupied_grid = np.zeros(grid_shape, dtype=bool)
    z_to_symbol = {z: symbol for symbol, z in ase_atomic_numbers.items()}
    grid_shape_array = np.array(grid_shape)

    is_orthogonal = np.allclose(cell - np.diag(np.diag(cell)), 0.0)
    if not is_orthogonal:
        perp_heights = cell_perpendicular_heights(cell)

    for atom_index in range(len(positions)):
        atom_symbol = z_to_symbol.get(int(atomic_numbers[atom_index]), "X")
        z = int(atomic_numbers[atom_index])
        if isinstance(radii_by_symbol, float):
            atom_radius = radii_by_symbol
        elif isinstance(radii_by_symbol, dict):
            atom_radius = radii_by_symbol.get(atom_symbol, float(ase_covalent_radii[z]))
        else:
            vdw = ase_vdw_radii[z]
            atom_radius = float(vdw) if not np.isnan(vdw) else float(ase_covalent_radii[z])

        atom_voxel_position = positions[atom_index] * grid_shape_array

        if is_orthogonal:
            # Fast path: isotropic sphere in voxel space (matches sovapy)
            discrete_radius = max(0, int(np.floor(atom_radius / voxel_step + 0.5)))

            offset_range = np.arange(-discrete_radius, discrete_radius + 1)
            offset_x, offset_y, offset_z = np.meshgrid(offset_range, offset_range, offset_range, indexing="ij")
            offsets = np.stack([offset_x.ravel(), offset_y.ravel(), offset_z.ravel()], axis=1)

            squared_distances = np.sum(offsets.astype(float) ** 2, axis=1)
            inside_sphere = squared_distances <= discrete_radius**2
        else:
            # Triclinic path: bounding box from perpendicular heights, filter
            # by actual Cartesian distance
            voxel_radii = atom_radius / perp_heights * grid_shape_array
            max_voxel_r = int(np.ceil(float(np.max(voxel_radii)))) + 1

            offset_range = np.arange(-max_voxel_r, max_voxel_r + 1)
            offset_x, offset_y, offset_z = np.meshgrid(offset_range, offset_range, offset_range, indexing="ij")
            offsets = np.stack([offset_x.ravel(), offset_y.ravel(), offset_z.ravel()], axis=1)

            delta_cart = (offsets / grid_shape_array) @ cell
            dist_sq = np.sum(delta_cart**2, axis=1)
            inside_sphere = dist_sq <= atom_radius**2

        candidate_indices = (np.round(atom_voxel_position).astype(int) + offsets[inside_sphere]) % grid_shape_array

        occupied_grid[
            candidate_indices[:, 0],
            candidate_indices[:, 1],
            candidate_indices[:, 2],
        ] = True

    return occupied_grid


def _merge_pbc_labels(raw_labels: np.ndarray, raw_label_count: int) -> tuple[np.ndarray, int]:
    """Merge cavity labels that connect across periodic box faces.

    Args:
        raw_labels: Integer label array from scipy.ndimage.label or
            skimage.segmentation.watershed.
        raw_label_count: Number of distinct labels (excluding background 0).

    Returns:
        remapped_labels: Relabelled array with PBC-merged components.
        number_of_voids: Number of distinct voids after merging.
    """
    parent = list(range(raw_label_count + 1))

    def find_root(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union_labels(label_a: int, label_b: int) -> None:
        root_a = find_root(label_a)
        root_b = find_root(label_b)
        if root_a != root_b:
            parent[root_b] = root_a

    grid_shape = raw_labels.shape
    for axis in range(3):
        face_slices_low: list = [slice(None)] * 3
        face_slices_high: list = [slice(None)] * 3
        face_slices_low[axis] = _BORDER_VOXELS
        face_slices_high[axis] = grid_shape[axis] - 1 - _BORDER_VOXELS

        labels_on_low_face = raw_labels[tuple(face_slices_low)].ravel()
        labels_on_high_face = raw_labels[tuple(face_slices_high)].ravel()

        for label_low, label_high in zip(labels_on_low_face, labels_on_high_face, strict=False):
            if label_low > 0 and label_high > 0:
                union_labels(label_low, label_high)

    root_to_new_label: dict[int, int] = {}
    next_label = 1
    remapped_labels = np.zeros_like(raw_labels)

    for original_label in range(1, raw_label_count + 1):
        root = find_root(original_label)
        if root not in root_to_new_label:
            root_to_new_label[root] = next_label
            next_label += 1
        remapped_labels[raw_labels == original_label] = root_to_new_label[root]

    return remapped_labels, next_label - 1


def _label_voids_with_pbc(occupied_grid: np.ndarray) -> tuple[np.ndarray, int]:
    """Find connected empty-voxel components with periodic boundary conditions.

    Uses scipy 26-connectivity labelling followed by a union-find merge of
    components that are connected across opposite faces of the box.

    Args:
        occupied_grid: Boolean array; True = occupied voxel.

    Returns:
        void_labels: Integer label array (0 = occupied; 1..n = void cavities).
        number_of_voids: Total number of distinct void cavities.
    """
    empty_grid = ~occupied_grid
    # Mask the padding border — those voxels are outside the physical simulation cell
    border = _BORDER_VOXELS
    empty_grid[:border, :, :] = False
    empty_grid[-border:, :, :] = False
    empty_grid[:, :border, :] = False
    empty_grid[:, -border:, :] = False
    empty_grid[:, :, :border] = False
    empty_grid[:, :, -border:] = False
    connectivity_structure = np.ones((3, 3, 3), dtype=int)
    raw_labels, raw_label_count = ndimage.label(empty_grid, structure=connectivity_structure)

    if raw_label_count == 0:
        return raw_labels, 0

    return _merge_pbc_labels(raw_labels, raw_label_count)


def _find_percolating_labels(void_labels: np.ndarray) -> set[int]:
    """Return the set of cavity labels that span the simulation cell.

    A cavity is percolating if its voxels appear on both the low and high
    inner periodic face along any of the three axes. Such cavities are
    topologically infinite under PBC and should be excluded from reporting.

    Args:
        void_labels: Integer label array (0 = occupied; 1..n = void cavities).

    Returns:
        Set of integer labels corresponding to percolating (spanning) cavities.
    """
    percolating: set[int] = set()
    for axis in range(3):
        face_slices_low: list = [slice(None)] * 3
        face_slices_high: list = [slice(None)] * 3
        face_slices_low[axis] = _BORDER_VOXELS
        face_slices_high[axis] = void_labels.shape[axis] - 1 - _BORDER_VOXELS
        labels_on_low_face = set(void_labels[tuple(face_slices_low)].ravel()) - {0}
        labels_on_high_face = set(void_labels[tuple(face_slices_high)].ravel()) - {0}
        percolating |= labels_on_low_face & labels_on_high_face
    return percolating


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
    cutoff_radii: dict[str, float] | float | None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Discretize the structure and label void cavities.

    Args:
        structure: ASE Atoms object (will be wrapped internally).
        resolution: Number of voxels along the longest box dimension before padding.
        cutoff_radii: Radius specification passed through to ``_build_occupied_grid``.

    Returns:
        void_labels: Integer label array (0 = occupied; 1..n = void cavities).
        occupied_grid: Boolean array; True = occupied voxel.
        voxel_step: Voxel edge length in A.
        cell_array: (3, 3) cell matrix with lattice vectors as rows.
    """
    wrapped_structure = structure.copy()
    wrapped_structure.wrap()

    cell_array = wrapped_structure.get_cell().array
    box_lengths = cell_perpendicular_heights(cell_array)

    voxel_step = _compute_voxel_step(box_lengths, resolution)
    grid_shape = _compute_grid_shape(box_lengths, voxel_step)

    fractional_positions = wrapped_structure.get_scaled_positions()
    atomic_numbers = wrapped_structure.get_atomic_numbers()

    occupied_grid = _build_occupied_grid(
        fractional_positions,
        atomic_numbers,
        grid_shape,
        voxel_step,
        cutoff_radii,
        cell_array,
    )

    void_labels, _number_of_voids = _label_voids_with_pbc(occupied_grid)
    return void_labels, occupied_grid, voxel_step, cell_array


# ============================================================================
# Public API
# ============================================================================


def compute_cavities(
    structure: Atoms,
    resolution: int = 64,
    cutoff_radii: dict[str, float] | float | None = None,
) -> dict[str, np.ndarray]:
    """Compute cavity distribution and shape properties for a glass structure.

    Replicates sovapy's domain-based cavity detection algorithm. The simulation
    box is discretised into a 3D voxel grid; voxels within the discrete
    (integer-rounded) atomic radius of any atom are marked as occupied.
    Connected regions of empty voxels -- detected with full 26-connectivity
    and periodic-boundary-condition awareness -- constitute cavities (domains).

    Atomic radii are rounded to the nearest integer number of voxels before
    marking, matching sovapy's ``discrete_radius = round(radius / s_step)``
    scheme. This produces higher occupancy than using raw float radii and
    is the key factor that reproduces sovapy's cavity count and volumes.

    Periodic boundary conditions are handled via a union-find merge of cavity
    labels that connect across opposite box faces.

    Args:
        structure: ASE Atoms object with atomic coordinates and cell.
        resolution: Number of voxels along the longest box dimension before
            padding (default 64). Higher values give more accurate surface
            areas and shapes at the cost of memory and runtime.
        cutoff_radii: Atomic radius specification. Three forms are accepted:

            - ``None`` (default): element-specific ASE Van der Waals radii,
              with ASE covalent radii as fallback.
            - ``float``: uniform radius in A applied to every atom
              (e.g. ``cutoff_radii=2.8`` reproduces sovapy's default).
            - ``dict[str, float]``: per-element overrides in A
              (e.g. ``{'Si': 2.1, 'O': 1.52}``); elements not in the dict
              fall back to ASE covalent radii.

    Returns:
        Dictionary with the following keys, each mapping to a 1D float64
        array with one entry per detected cavity:

        - ``'volumes'``: cavity volumes in A^3.
        - ``'surface_areas'``: cavity surface areas in A^2.
        - ``'asphericities'``: asphericity eta in [0, 1].
        - ``'acylindricities'``: acylindricity c in [0, 1].
        - ``'anisotropies'``: relative shape anisotropy kappa in [0, 1].

    Examples:
        >>> from ase.io import read
        >>> structure = read('glass.xyz')
        >>> cavities = compute_cavities(structure, resolution=64)
        >>> print(cavities['volumes'])
    """
    void_labels, _occupied_grid, _voxel_step, cell_array = _build_cavity_grid(structure, resolution, cutoff_radii)
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
        }

    # Voxel volume: exact cell volume divided by total voxel count.
    # For orthogonal cells this equals voxel_step^3 up to padding rounding.
    voxel_volume = abs(float(np.linalg.det(cell_array))) / float(np.prod(grid_shape))

    # Transformation from voxel-index space to Cartesian:
    # voxel index i along axis a → fractional i/na → Cartesian (i/na)*cell[0]
    # matrix form: v_cart = v_voxel @ mc_to_cart  where mc_to_cart = diag(1/grid_shape) @ cell
    mc_to_cart = np.diag(1.0 / grid_shape) @ cell_array

    percolating_labels = _find_percolating_labels(void_labels)
    if percolating_labels:
        import warnings  # noqa: PLC0415

        warnings.warn(
            f"{len(percolating_labels)} percolating cavity/cavities span the simulation cell "
            "and will be excluded from the output. Consider using larger cutoff radii.",
            UserWarning,
            stacklevel=2,
        )

    volumes = np.zeros(number_of_voids, dtype=np.float64)
    surface_areas = np.zeros(number_of_voids, dtype=np.float64)
    asphericities = np.zeros(number_of_voids, dtype=np.float64)
    acylindricities = np.zeros(number_of_voids, dtype=np.float64)
    anisotropies = np.zeros(number_of_voids, dtype=np.float64)

    for void_index in range(1, number_of_voids + 1):
        if void_index in percolating_labels:
            continue

        void_mask = void_labels == void_index
        void_voxel_count = int(np.sum(void_mask))

        if void_voxel_count < 2:  # noqa: PLR2004
            continue

        volumes[void_index - 1] = void_voxel_count * voxel_volume

        void_float_mask = void_mask.astype(np.float32)
        try:
            vertices, faces, _normals, _values = marching_cubes(void_float_mask, level=0.5, spacing=(1.0, 1.0, 1.0))
            vertices_cart = vertices @ mc_to_cart
            surface_areas[void_index - 1] = _compute_triangle_mesh_area(vertices_cart, faces)
        except (ValueError, RuntimeError):
            surface_areas[void_index - 1] = 0.0

        cavity_voxel_indices = np.argwhere(void_mask)

        if len(cavity_voxel_indices) >= 3:  # noqa: PLR2004
            cavity_voxel_coordinates = (cavity_voxel_indices / grid_shape) @ cell_array
            (
                asphericities[void_index - 1],
                acylindricities[void_index - 1],
                anisotropies[void_index - 1],
            ) = _compute_gyration_tensor_descriptors(cavity_voxel_coordinates)

    non_empty = volumes > 0
    return {
        "volumes": volumes[non_empty],
        "surface_areas": surface_areas[non_empty],
        "asphericities": asphericities[non_empty],
        "acylindricities": acylindricities[non_empty],
        "anisotropies": anisotropies[non_empty],
    }


def visualize_cavities(
    structure: Atoms,
    resolution: int = 64,
    cutoff_radii: dict[str, float] | float | None = None,
    show_atoms: bool = True,  # noqa: FBT001
    opacity: float = 0.4,
    excluded_cavities: float | None = None,
) -> object:
    """Render cavity surfaces and optionally atom positions in an interactive 3D plot.

    Builds a marching-cubes mesh for every cavity detected by ``compute_cavities``
    and draws it as a semi-transparent coloured surface using plotly. Atom positions
    are overlaid as small scatter markers when ``show_atoms=True``.

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

    Returns:
        A plotly ``Figure`` object. Call ``.show()`` to display it in a notebook or browser.

    Raises:
        ImportError: If plotly is not installed.

    Examples:
        >>> from ase.io import read
        >>> from amorphouspy.analysis.cavities import visualize_cavities
        >>> structure = read('glass.xyz')
        >>> fig = visualize_cavities(structure, excluded_cavities=50.0)
        >>> fig.show()
    """
    try:
        import plotly.graph_objects as go  # noqa: PLC0415
    except ImportError as exc:
        error_message = "plotly is required for visualize_cavities. Install it with: pip install plotly"
        raise ImportError(error_message) from exc

    void_labels, _occupied_grid, _voxel_step, cell_array = _build_cavity_grid(structure, resolution, cutoff_radii)
    number_of_voids = int(void_labels.max())

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
    for void_index in range(1, number_of_voids + 1):
        void_mask = (void_labels == void_index).astype(np.float32)
        voxel_count = void_mask.sum()
        if voxel_count < 2:  # noqa: PLR2004
            continue

        volume = float(voxel_count) * voxel_volume
        if excluded_cavities is not None and volume >= excluded_cavities:
            continue

        try:
            vertices, faces, _normals, _values = marching_cubes(void_mask, level=0.5, spacing=(1.0, 1.0, 1.0))
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
