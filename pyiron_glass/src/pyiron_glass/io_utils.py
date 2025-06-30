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


import gzip
from pathlib import Path

import numpy as np


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
