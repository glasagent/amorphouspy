# Author: Achraf Atila (achraf.atila@bam.de)
# Description: This script have function to be used for analyzing multicomponent glass structure this include for now:
# Coordination numbers
# Bridging oxygens
# Non-bridging oxygens
# Bond angle distributions
# Qn distributions
# Network connectivity
# more to come...
# Note: For now, this script is designed to work with LAMMPS dump files.
# It reads a lammps dump file and uses a cell list algorithm for neighbor search under periodic boundary conditions (PBC).


import numpy as np
import gzip
import os
from collections import defaultdict
from typing import Tuple, List, Dict, Union

def read_lammps_dump(filename: str, unwrap=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads a LAMMPS dump file and extracts atom IDs, types, coordinates, and box size.

    Args:
        filename (str): Path to the LAMMPS dump file (can be gzipped).

    Returns:
        Tuple containing:
            - ids (np.ndarray): Array of atom IDs.
            - types (np.ndarray): Array of atom types.
            - wrapped (np.ndarray): Wrapped coordinates within simulation box.
            - box_size (np.ndarray): Dimensions of the simulation box.
    """
    open_func = gzip.open if filename.endswith('.gz') else open
    with open_func(filename, 'rt') as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of file")
            if 'ITEM: NUMBER OF ATOMS' in line:
                n_atoms = int(f.readline().strip())
            elif 'ITEM: BOX BOUNDS' in line:
                box_bounds = np.array([list(map(float, f.readline().split())) for _ in range(3)])
                break
        box_lower, box_upper = box_bounds[:, 0], box_bounds[:, 1]
        box_size = box_upper - box_lower
        while 'ITEM: ATOMS' not in line:
            line = f.readline()
        data = np.empty((n_atoms, 5), dtype=float)
        for i in range(n_atoms):
            parts = f.readline().split()
            data[i, 0] = int(parts[0])
            data[i, 1] = int(parts[1])
            data[i, 2:5] = list(map(float, parts[2:5]))
    ids = data[:, 0].astype(int)
    types = data[:, 1].astype(int)
    coords = data[:, 2:5]
    if unwrap:
        coords_out = data[:, 2:5]
    else:
        coords_out = (coords - box_lower) % box_size
    return ids, types, coords_out, box_size


def remove_atom_type(
    ids: np.ndarray,
    types: np.ndarray,
    coords: np.ndarray,
    box_size: np.ndarray,
    remove_types: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Removes atoms of specified types from the system.

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
    cutoff: float
) -> Tuple[Dict[Tuple[int, int, int], List[int]], np.ndarray, np.ndarray]:
    """
    Constructs a cell list to accelerate neighbor search.

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
    n_cells = np.floor(box_size / cutoff).astype(int)
    inv_cell_size = n_cells / box_size
    atom_cells = np.floor(coords * inv_cell_size).astype(int) % n_cells
    for idx, cell in enumerate(atom_cells):
        cells[tuple(cell)].append(idx)
    return cells, n_cells, inv_cell_size

def get_neighbors(
    coords: np.ndarray,
    types: np.ndarray,
    box_size: np.ndarray,
    cutoff: float,
    target_type: int,
    neighbor_types: Union[List[int], None] = None
) -> List[List[int]]:
    """
    Finds neighbors of specified type(s) using a cell list.

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
    shifts = [-1, 0, 1]
    for i in range(N):
        if types[i] != target_type:
            continue
        ci = np.floor(coords[i] * inv_cell_size).astype(int) % n_cells
        for dx in shifts:
            for dy in shifts:
                for dz in shifts:
                    cj = (ci + [dx, dy, dz]) % n_cells
                    for j in cells[tuple(cj)]:
                        if i == j:
                            continue
                        if neighbor_types is None or types[j] in neighbor_types:
                            rij = coords[i] - coords[j]
                            rij -= box_size * np.round(rij / box_size)
                            dist = np.linalg.norm(rij)
                            if dist <= cutoff:
                                neighbors[i].append(j)
    return neighbors

def count_distribution(coord_numbers: Dict[int, int]) -> Dict[int, int]:
    """
    Converts coordination numbers to a histogram distribution.

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
    target_type: int,
    cutoff: float,
    neighbor_types: Union[List[int], None] = None
) -> Dict[int, int]:
    """
    Computes coordination number for atoms of a target type.

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
    neighbors = get_neighbors(coords, types, box_size, cutoff, target_type, neighbor_types)
    coord_numbers = {ids[idx]: len(neighbors[idx]) for idx, atom_type in enumerate(types) if atom_type == target_type}
    coord_numbers_distribution = count_distribution(coord_numbers)
    return dict(sorted(coord_numbers_distribution.items())), coord_numbers



def compute_Qn(
    ids: np.ndarray,
    types: np.ndarray,
    coords: np.ndarray,
    box_size: np.ndarray,
    cutoff: float,
    Si_types: List[int],
    O_type: int
) -> Dict[int, int]:
    """
    Calculates Qn distribution: number of bridging oxygens per silicon atom.

    Args:
        ids (np.ndarray): Atom IDs.
        types (np.ndarray): Atom types.
        coords (np.ndarray): Atom coordinates.
        box_size (np.ndarray): Simulation box dimensions.
        cutoff (float): Cutoff radius for Si-O neighbor search.
        Si_types (List[int]): Atom types considered as silicon.
        O_type (int): Atom type considered as oxygen.

    Returns:
        Dict[int, int]: Mapping from Qn to count.
    """
    neighbors = {}
    for Si_type in Si_types:
        neighbors.update({i: neigh for i, neigh in enumerate(
            get_neighbors(coords, types, box_size, cutoff, Si_type, [O_type])
        )})
    
    _, coord_numbers_O = compute_coordination(ids, types, coords, box_size, O_type, cutoff, neighbor_types=Si_types)

    Qn_counts = {}
    id_map = {id_: i for i, id_ in enumerate(ids)}
    for idx, atom_type in enumerate(types):
        if atom_type in Si_types:
            bridging_count = 0
            for j in neighbors.get(idx, []):
                if types[j] == O_type and coord_numbers_O.get(ids[j], 0) >= 2:
                    bridging_count += 1
            Qn_counts[bridging_count] = Qn_counts.get(bridging_count, 0) + 1
    full_Qn_counts = {n: Qn_counts.get(n, 0) for n in range(7)}
    return full_Qn_counts

def compute_network_connectivity(Qn_dist: Dict[int, int]) -> float:
    """
    Computes average network connectivity based on Qn distribution.

    Args:
        Qn_dist (Dict[int, int]): Qn distribution histogram.
        total_formers (int): Total number of network former atoms.

    Returns:
        float: Average network connectivity.
    """

    total_formers = sum(Qn_dist.values())

    return sum(n * (count / total_formers) for n, count in Qn_dist.items())


def write_distribution_to_file(composition: float, filename: str, dist: Dict[int, int], label: str, append: bool = False) -> None:
    """
    Writes a coordination/Qn histogram to a text file.

    Args:
        composition (float): Composition value to label row.
        filename (str): Output filename.
        dist (Dict[int, int]): Histogram data.
        label (str): Prefix for headers (e.g., Si or Q).
        append (bool): Append mode; writes header only if file does not exist.
    """
    max_n = max(dist.keys(), default=0)
    total = sum(dist.values())
    headers = [f"{label}_{i}" for i in range(max_n + 1)] + [f"{label}_tot"]
    values = [dist.get(i, 0) for i in range(max_n + 1)] + [total]
    mode = 'a' if append else 'w'
    write_header = not append or not os.path.exists(filename)
    with open(filename, mode) as f:
        if write_header:
            f.write("Composition, " + " ".join(headers) + "\n")
        f.write(str(composition) + " " + " ".join(map(str, values)) + "\n")



def compute_angles(
    ids: np.ndarray,
    types: np.ndarray,
    coords: np.ndarray,
    box_size: np.ndarray,
    center_type: int,
    neighbor_type: int,
    cutoff: float,
    bins: int = 180
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes bond angle distribution between triplets of neighbor_type-center-neighbor_type.

    Args:
        ids (np.ndarray): Atom IDs.
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
    neighbors = get_neighbors(coords, types, box_size, cutoff, center_type, [neighbor_type])
    angles = []
    for i, atom_type in enumerate(types):
        if atom_type != center_type:
            continue
        neigh_ids = neighbors[i]
        if len(neigh_ids) < 2:
            continue
        for j in range(len(neigh_ids)):
            for k in range(j+1, len(neigh_ids)):
                v1 = coords[neigh_ids[j]] - coords[i]
                v2 = coords[neigh_ids[k]] - coords[i]
                v1 -= box_size * np.round(v1 / box_size)
                v2 -= box_size * np.round(v2 / box_size)
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                if norm_v1 == 0 or norm_v2 == 0:
                    continue
                cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
                angle = np.arccos(cos_theta) * 180 / np.pi
                angles.append(angle)
    angle_hist, bin_edges = np.histogram(angles, bins=bins, range=(0, 180), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, angle_hist

def write_angle_distribution(
    bin_centers: np.ndarray,
    angle_hist: np.ndarray,
    composition: float,
    filename: str,
    append: bool = False
) -> None:
    """
    Writes angle distribution to a text file.

    Args:
        bin_centers (np.ndarray): Angle bin centers in degrees.
        angle_hist (np.ndarray): Normalized angle histogram.
        composition (float): Composition value (e.g., % modifier).
        filename (str): Output filename.
        append (bool): Whether to append to file.
    """
    mode = 'a' if append else 'w'
    write_header = not append or not os.path.exists(filename)
    with open(filename, mode) as f:
        if write_header:
            f.write("Composition " + " ".join(f"{b:.1f}" for b in bin_centers) + "\n")
        f.write(f"{composition} " + " ".join(f"{v:.6f}" for v in angle_hist) + "\n")
