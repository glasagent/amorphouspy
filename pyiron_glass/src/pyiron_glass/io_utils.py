"""Structural analysis functions for multicomponent glass systems.

Author: Achraf Atila (achraf.atila@bam.de)


It includes a lightweight parser for LAMMPS dump files and methods
for handling wrapped and unwrapped atomic coordinates.

Currently implemented:

- read_lammps_dump: Efficiently reads LAMMPS dump files (including
  gzipped files), extracts atom IDs, types, coordinates (wrapped or
  unwrapped), and simulation box dimensions. Designed for use in
  structural analyses of disordered materials.

"""

# implement ASE based parser using a wrapper around ase.io.read

from pathlib import Path

import numpy as np
from ase import Atoms


# See issue #30: Why not use ase.io.read instead of custom parser function?
def get_properties_for_structure_analysis(
    atoms: Atoms,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Based on an atoms object, extract atom IDs, types, coordinates, and box size.

    Args:
        atoms (Atoms): ASE Atoms object containing atomic information.

    Returns:
        Tuple containing:
            - ids (np.ndarray): Array of atom IDs.
            - types (np.ndarray): Array of atom types.
            - coords (np.ndarray): Wrapped coordinates.
            - cell (np.ndarray): Dimensions of the simulation box.

    """
    # Use a copy to avoid that the original atoms object is modified by wrap()
    atoms_copy = atoms.copy()
    atoms_copy.wrap()
    coords = atoms_copy.get_positions()
    ids = np.array(range(1, len(atoms_copy) + 1))
    types = atoms_copy.get_atomic_numbers()
    cell = atoms_copy.get_cell()
    # Here, the output of the cell is formatted to match the previous function.
    # But this needs to be changed to something more robust in the future to
    # also be able to handle non-orthorhombic cells
    cell_hack = np.array([cell[0, 0], cell[1, 1], cell[2, 2]])

    return ids, types, coords, cell_hack


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
