"""module to get the neighbors within a cutoff distance.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from collections import defaultdict

import numpy as np

SHIFT_GRID_3D = np.stack(
    np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"),
    axis=-1,
).reshape(-1, 3)


def compute_cell_list(
    coords: np.ndarray,
    box_size: np.ndarray,
    cutoff: float,
) -> tuple[dict[tuple[int, int, int], list[int]], np.ndarray, np.ndarray]:
    """Construct a cell list to accelerate neighbor search.

    Args:
        coords: Atom coordinates.
        box_size: Box size.
        cutoff: Cutoff distance for neighbor searching.

    Returns:
        A tuple containing:
            - cells: Cell to atom index mapping.
            - n_cells: Number of cells in each dimension.
            - inv_cell_size: Inverse of the cell size in each dimension.

    Example:
        >>> cells, n_cells, inv_size = compute_cell_list(coords, box, 5.0)

    """
    cells = defaultdict(list)
    n_cells = np.maximum(1, np.floor(box_size / cutoff)).astype(int)
    inv_cell_size = n_cells / box_size
    atom_cells = np.floor(coords * inv_cell_size).astype(int) % n_cells
    for idx, cell in enumerate(atom_cells):
        cells[tuple(cell)].append(idx)
    return cells, n_cells, inv_cell_size


def get_neighbor_cells(ci: np.ndarray, n_cells: np.ndarray) -> np.ndarray:
    """Generate neighboring cell indices for a given cell index in a 3D grid.

    Args:
        ci: Current cell index as a 3-element array.
        n_cells: Total number of cells in each dimension.

    Returns:
        Array of neighboring cell indices.

    Example:
        >>> neighbor_cells = get_neighbor_cells(np.array([1, 1, 1]), n_cells)

    """
    return (ci + SHIFT_GRID_3D) % n_cells


def compute_distance(rij: np.ndarray, box_size: np.ndarray) -> float:
    """Compute minimum image distance between two atoms.

    Args:
        rij: Vector between two atoms
        box_size: Box dimensions for periodic boundary conditions

    Returns:
        Minimum image distance

    Example:
        >>> dist = compute_distance(coords[0] - coords[1], box_size)

    """
    rij -= box_size * np.round(rij / box_size)
    return np.linalg.norm(rij)


def get_neighbors(
    coords: np.ndarray,
    types: np.ndarray,
    box_size: np.ndarray,
    cutoff: float,
    target_types: list[int] | None = None,
    neighbor_types: list[int] | None = None,
) -> list[list[int]]:
    """Find neighbors of specified type(s) using a cell list.

    Args:
        coords: Atom coordinates.
        types: Atom types.
        box_size: Simulation box dimensions.
        cutoff: Cutoff distance for neighbor searching.
        target_types: Types of atoms to find neighbors for.
        neighbor_types: Acceptable neighbor types (None for all).

    Returns:
        Neighbor indices for each atom.

    Example:
        >>> neighbors = get_neighbors(coords, types, box, 3.0, target_types=[14])

    """
    number_of_atoms = len(coords)
    cells, n_cells, inv_cell_size = compute_cell_list(coords, box_size, cutoff)

    neighbors = [[] for _ in range(number_of_atoms)]
    for i in range(number_of_atoms):
        if target_types is not None and types[i] not in target_types:
            continue

        ci = np.floor(coords[i] * inv_cell_size).astype(int) % n_cells
        for cj in get_neighbor_cells(ci, n_cells):
            for j in cells[tuple(cj)]:
                if i == j:
                    continue
                if neighbor_types is None or types[j] in neighbor_types:
                    rij = coords[i] - coords[j]
                    dist = compute_distance(rij, box_size)
                    if dist <= cutoff:
                        neighbors[i].append(j)
    return neighbors
