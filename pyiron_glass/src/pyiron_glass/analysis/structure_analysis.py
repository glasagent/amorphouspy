"""Structural analysis functions for multicomponent glass systems.

Author: Achraf Atila (achraf.atila@bam.de)

This script defines functions to be used for analyzing multicomponent glass structure.
Current implementations include analyses of:

- Coordination numbers
- Fraction of bridging oxygens
- Fraction of non-bridging oxygens
- Bond angle distributions
- Qn distributions
- Network connectivity

Note: For now, only LAMMPS dump files can be handled.
It reads a lammps dump file and uses a cell list algorithm for neighbor search
under periodic boundary conditions (PBC).
"""

import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np

# Constants
MIN_NEIGHBORS_FOR_ANGLE = 2
MIN_COORDINATION_FOR_BRIDGING = 2


# See issue #30: Why not use ase.io.read instead of custom parser function?
def read_lammps_dump(
    filepath: str,
    *,
    unwrap: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read a LAMMPS dump file and extract atom IDs, types, coordinates, and box size.

    Args:
        filepath (str): Path to the LAMMPS dump file (can be gzipped).
        unwrap (bool): If True, use unwrapped coordinates.

    Returns:
        Tuple containing:
            - ids (np.ndarray): Array of atom IDs.
            - types (np.ndarray): Array of atom types.
            - coords_out (np.ndarray): Wrapped or unwrapped coordinates.
            - box_size (np.ndarray): Dimensions of the simulation box.

    """
    open_func = gzip.open if str(filepath).endswith(".gz") else open
    with open_func(filepath, "rt") as f:
        # Read line by line until reaching required info
        n_atoms = None
        box_bounds = []
        atom_section = False
        atom_lines = []

        for line in f:
            if "ITEM: NUMBER OF ATOMS" in line:
                n_atoms = int(next(f).strip())
            elif "ITEM: BOX BOUNDS" in line:
                box_bounds = [list(map(float, next(f).split())) for _ in range(3)]
            elif "ITEM: ATOMS" in line:
                atom_section = True
                continue
            elif atom_section:
                atom_lines.append(line)
                if len(atom_lines) == n_atoms:
                    break

    if n_atoms is None or not box_bounds:
        msg = "Missing 'NUMBER OF ATOMS' or 'BOX BOUNDS' section in dump file."
        raise ValueError(msg)

    box_bounds = np.array(box_bounds)
    box_lower = box_bounds[:, 0]
    box_upper = box_bounds[:, 1]
    box_size = box_upper - box_lower

    # Convert atom data to numpy array
    data = np.array([list(map(float, line.split())) for line in atom_lines])
    ids = data[:, 0].astype(int)
    types = data[:, 1].astype(int)
    coords = data[:, 2:5]
    coords_out = coords if unwrap else (coords - box_lower) % box_size

    return ids, types, coords_out, box_size


def remove_atom_type(
    ids: np.ndarray,
    types: np.ndarray,
    coords: np.ndarray,
    box_size: np.ndarray,
    remove_types: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Remove atoms of specified types from the system.

    Args:
        ids (np.ndarray): Atom IDs.
        types (np.ndarray): Atom types.
        coords (np.ndarray): Atom coordinates.
        box_size (np.ndarray): Simulation box dimensions.
        remove_types (List[int]): List of types to remove.

    Returns:
        Tuple containing:
            - Filtered ids (np.ndarray)
            - Filtered types (np.ndarray)
            - Filtered coords (np.ndarray)
            - Original box_size (np.ndarray)

    """
    mask = np.isin(types, remove_types, invert=True)
    return ids[mask], types[mask], coords[mask], box_size


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


SHIFT_GRID_3D = np.stack(
    np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"),
    axis=-1,
).reshape(-1, 3)


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
    N = len(coords)
    cells, n_cells, inv_cell_size = compute_cell_list(coords, box_size, cutoff)
    neighbors = [[] for _ in range(N)]
    for i in range(N):
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


def count_distribution(coord_numbers: dict[int, int]) -> dict[int, int]:
    """Convert coordination numbers to a histogram distribution.

    Args:
        coord_numbers (Dict[int, int]): Mapping from atom ID to coordination number.

    Returns:
        Dict[int, int]: Coordination number frequency histogram.

    """
    dist = {}
    for cn in coord_numbers.values():
        dist[cn] = dist.get(cn, 0) + 1
    return dist


def compute_coordination(
    ids: np.ndarray,
    types: np.ndarray,
    coords: np.ndarray,
    box_size: np.ndarray,
    target_type: list[int],
    cutoff: float,
    neighbor_types: list[int] | None = None,
) -> tuple[dict[int, int], dict[int, int]]:
    """Compute coordination number for atoms of a target type.

    Args:
        ids (np.ndarray): Atom IDs.
        types (np.ndarray): Atom types.
        coords (np.ndarray): Atom coordinates.
        box_size (np.ndarray): Simulation box dimensions.
        target_type (int): Atom type for which to compute coordination.
        cutoff (float): Cutoff radius.
        neighbor_types (List[int] or None): Valid neighbor types.

    Returns:
        Dict[int, int]: Mapping from atom ID to coordination number.

    """
    neighbors = get_neighbors(
        coords,
        types,
        box_size,
        cutoff,
        target_type,
        neighbor_types,
    )
    coord_numbers = {
        ids[idx]: len(neighbors[idx])
        for idx, atom_type in enumerate(types)
        if atom_type == target_type
    }
    coord_numbers_distribution = count_distribution(coord_numbers)
    return dict(sorted(coord_numbers_distribution.items())), coord_numbers


def compute_Qn(
    ids: np.ndarray,
    types: np.ndarray,
    coords: np.ndarray,
    box_size: np.ndarray,
    cutoff: float,
    Former_types: list[int],
    O_type: int,
) -> tuple[dict[int, int], dict[int, dict[int, int]]]:
    """Calculate Qn distribution: number of bridging oxygens per former atom.

    And partial Qn distributions for each former type.

    Args:
        ids (np.ndarray): Atom IDs.
        types (np.ndarray): Atom types.
        coords (np.ndarray): Atom coordinates.
        box_size (np.ndarray): Simulation box dimensions.
        cutoff (float): Cutoff radius for former-O neighbor search.
        Former_types (List[int]): Atom types considered as formers (e.g., Si, B, etc.).
        O_type (int): Atom type considered as oxygen.

    Returns:
        Tuple[
            Dict[int, int],              # Total Qn distribution
            Dict[int, Dict[int, int]]   # Partial Qn per former type
        ]

    """
    neighbors = dict(
        enumerate(
            get_neighbors(coords, types, box_size, cutoff, Former_types, [O_type]),
        ),
    )
    _, coord_numbers_O = compute_coordination(
        ids,
        types,
        coords,
        box_size,
        O_type,
        cutoff,
        neighbor_types=Former_types,
    )

    total_Qn_counts = defaultdict(int)
    partial_Qn_counts = {f_type: defaultdict(int) for f_type in Former_types}

    for idx, atom_type in enumerate(types):
        if atom_type in Former_types:
            bridging_count = 0
            for j in neighbors.get(idx, []):
                if (
                    types[j] == O_type
                    and coord_numbers_O.get(ids[j], 0) >= MIN_COORDINATION_FOR_BRIDGING
                ):
                    bridging_count += 1
            total_Qn_counts[bridging_count] += 1
            partial_Qn_counts[atom_type][bridging_count] += 1

    # Normalize output
    total_Qn_counts = {n: total_Qn_counts.get(n, 0) for n in range(7)}
    for f_type in Former_types:
        partial_Qn_counts[f_type] = {
            n: partial_Qn_counts[f_type].get(n, 0) for n in range(7)
        }

    return total_Qn_counts, partial_Qn_counts


def compute_network_connectivity(Qn_dist: dict[int, int]) -> float:
    """Compute average network connectivity based on Qn distribution.

    Args:
        Qn_dist (Dict[int, int]): Qn distribution histogram.

    Returns:
        float: Average network connectivity.

    Raises:
        ValueError: If Qn_dist is empty or total_formers is zero.

    """
    total_formers = sum(Qn_dist.values())

    if total_formers == 0:
        msg = "total_formers is zero, cannot compute network connectivity."
        raise ValueError(msg)

    return sum(n * (count / total_formers) for n, count in Qn_dist.items())


def write_distribution_to_file(
    composition: float,
    filepath: str,
    dist: dict[int, int],
    label: str,
    *,
    append: bool = False,
) -> None:
    """Write a coordination/Qn histogram to a text file.

    Args:
        composition (float): Composition value to label row.
        filepath (str): Output filepath.
        dist (Dict[int, int]): Histogram data.
        label (str): Prefix for headers (e.g., Si or Q).
        append (bool): Append mode; writes header only if file does not exist.

    """
    max_n = max(dist.keys(), default=0)
    total = sum(dist.values())
    headers = [f"{label}_{i}" for i in range(max_n + 1)] + [f"{label}_tot"]
    values = [dist.get(i, 0) for i in range(max_n + 1)] + [total]
    mode = "a" if append else "w"
    write_header = not append or not Path(filepath).exists()
    with Path(filepath).open(mode, encoding="utf-8") as f:
        if write_header:
            f.write("Composition " + " ".join(headers) + "\n")
        f.write(str(composition) + " " + " ".join(map(str, values)) + "\n")


def compute_angles(
    types: np.ndarray,
    coords: np.ndarray,
    box_size: np.ndarray,
    center_type: int,
    neighbor_type: int,
    cutoff: float,
    bins: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute bond angle distribution between triplets of neighbor_type-center-neighbor_type.

    Args:
        types (np.ndarray): Atom types.
        coords (np.ndarray): Atom coordinates.
        box_size (np.ndarray): Box size.
        center_type (int): Atom type at the center of the angle (e.g., Si).
        neighbor_type (int): Atom type forming the angle with the center (e.g., O).
        cutoff (float): Cutoff for neighbor search.
        bins (int): Number of bins in histogram (default: 180 for 1° resolution).

    Returns:
        Tuple containing:
            - bin_centers (np.ndarray): Centers of angle bins in degrees.
            - angle_hist (np.ndarray): Normalized histogram of angles.

    """
    neighbors = get_neighbors(
        coords,
        types,
        box_size,
        cutoff,
        center_type,
        [neighbor_type],
    )
    angles = []
    for i, atom_type in enumerate(types):
        if atom_type != center_type:
            continue
        neigh_ids = neighbors[i]
        if len(neigh_ids) < MIN_NEIGHBORS_FOR_ANGLE:
            continue
        for j, id_j in enumerate(neigh_ids):
            for k in range(j + 1, len(neigh_ids)):
                id_k = neigh_ids[k]
                v1 = coords[id_j] - coords[i]
                v2 = coords[id_k] - coords[i]
                v1 -= box_size * np.round(v1 / box_size)
                v2 -= box_size * np.round(v2 / box_size)
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                if norm_v1 == 0 or norm_v2 == 0:
                    continue
                cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
                angle = np.arccos(cos_theta) * 180 / np.pi
                angles.append(angle)
    angle_hist, bin_edges = np.histogram(
        angles,
        bins=bins,
        range=(0, 180),
        density=True,
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, angle_hist


def write_angle_distribution(
    bin_centers: np.ndarray,
    angle_hist: np.ndarray,
    composition: float,
    filepath: str,
    *,
    append: bool = False,
) -> None:
    """Write angle distribution to a text file.

    Args:
        bin_centers (np.ndarray): Angle bin centers in degrees.
        angle_hist (np.ndarray): Normalized angle histogram.
        composition (float): Composition value (e.g., % modifier).
        filepath (str): Output filepath.
        append (bool): Whether to append to file.

    """
    mode = "a" if append else "w"
    write_header = not append or not Path(filepath).exists()
    with Path(filepath).open(mode, encoding="utf-8") as f:
        if write_header:
            f.write("Composition " + " ".join(f"{b:.1f}" for b in bin_centers) + "\n")
        f.write(f"{composition} " + " ".join(f"{v:.6f}" for v in angle_hist) + "\n")
