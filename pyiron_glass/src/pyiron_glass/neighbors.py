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
        coords (np.ndarray): Atom coordinates.
        box_size (np.ndarray): Box size.
        cutoff (float): Cutoff distance for neighbor searching.

    Returns:
        Tuple containing:
            - cells (Dict[Tuple[int, int, int], List[int]]): Cell to atom index mapping.
            - n_cells (np.ndarray): Number of cells in each dimension.
            - inv_cell_size (np.ndarray): Inverse of the cell size in each dimension.

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
        ci (np.ndarray): Current cell index as a 3-element array.
        n_cells (np.ndarray): Total number of cells in each dimension.

    Returns:
        np.ndarray: Array of neighboring cell indices.

    """
    return (ci + SHIFT_GRID_3D) % n_cells


def compute_distance(rij: np.ndarray, box_size: np.ndarray) -> float:
    """Compute minimum image distance between two atoms.

    Args:
        rij: Vector between two atoms
        box_size: Box dimensions for periodic boundary conditions

    Returns:
        Minimum image distance

    """
    rij -= box_size * np.round(rij / box_size)
    return np.linalg.norm(rij)


def get_neighbors(
    coords: np.ndarray,
    types: np.ndarray,
    box_size: np.ndarray,
    cutoff: float,
    target_type: int,
    neighbor_types: list[int] | None = None,
) -> list[list[int]]:
    """Find neighbors of specified type(s) using a cell list.

    Args:
        coords (np.ndarray): Atom coordinates.
        types (np.ndarray): Atom types.
        box_size (np.ndarray): Simulation box dimensions.
        cutoff (float): Cutoff distance for neighbor searching.
        target_type (int): Type of atom to find neighbors for.
        neighbor_types (List[int] or None): Acceptable neighbor types (None for all).

    Returns:
        List[List[int]]: Neighbor indices for each atom.

    """
    number_of_atoms = len(coords)
    cells, n_cells, inv_cell_size = compute_cell_list(coords, box_size, cutoff)
    neighbors = [[] for _ in range(number_of_atoms)]
    for i in range(number_of_atoms):
        if types[i] not in target_type:
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
