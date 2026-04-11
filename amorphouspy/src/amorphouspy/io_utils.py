"""I/O utilities for amorphouspy: reading and writing structural data files.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from pathlib import Path
from typing import cast

import ase.io
import numpy as np
from ase import Atoms


def load_lammps_dump(
    path: str | Path,
    type_map: dict[int, str] | None = None,
    *,
    return_atoms_dict: bool = False,
) -> Atoms | tuple[Atoms, dict]:
    """Read a LAMMPS dump file and return an ASE Atoms object with correct chemical symbols.

    Args:
        path: Path to the LAMMPS dump file.
        type_map: Mapping from LAMMPS integer type to element symbol,
            e.g. ``{1: "O", 2: "Si"}``. When ``None``, the ``element``
            column stored in the dump file is used.
        return_atoms_dict: When ``True``, also return an amorphouspy
            ``atoms_dict`` alongside the ``ase.Atoms`` object (default ``False``).

    Returns:
        - ``ase_atoms`` (``ase.Atoms``) when *return_atoms_dict* is ``False``
        - ``(atoms_dict, ase_atoms)`` when *return_atoms_dict* is ``True``,
          where ``atoms_dict`` contains ``"atoms"``, ``"box"``, and ``"total_atoms"``

    Raises:
        ValueError: If *type_map* is ``None`` and the dump file does not contain
            an ``element`` column.

    Example:
        >>> ase_atoms = load_lammps_dump("final.lammpstrj", type_map={1: "O", 2: "Si"})
        >>> atoms_dict, ase_atoms = load_lammps_dump(
        ...     "final.lammpstrj", type_map={1: "O", 2: "Si"}, return_atoms_dict=True
        ... )

    """
    ase_atoms_result = ase.io.read(str(path), format="lammps-dump-text", index=0)
    ase_atoms = cast("Atoms", ase_atoms_result[-1] if isinstance(ase_atoms_result, list) else ase_atoms_result)

    if type_map is not None:
        symbols = [type_map[int(t)] for t in ase_atoms.arrays["type"]]
        ase_atoms.set_chemical_symbols(symbols)
    elif "element" in ase_atoms.arrays:
        ase_atoms.set_chemical_symbols(list(ase_atoms.arrays["element"]))
    else:
        msg = (
            "type_map is required when the dump file does not contain an 'element' column. "
            "Either pass type_map={1: 'O', 2: 'Si', ...} or add "
            "'dump_modify element O Si ...' to your LAMMPS input script."
        )
        raise ValueError(msg)

    box_length = float(ase_atoms.get_cell()[0][0])
    atoms_list = [
        {"element": sym, "position": list(pos)}
        for sym, pos in zip(ase_atoms.get_chemical_symbols(), ase_atoms.get_positions(), strict=True)
    ]

    if not return_atoms_dict:
        return ase_atoms

    atoms_dict = {
        "atoms": atoms_list,
        "box": box_length,
        "total_atoms": len(atoms_list),
    }

    return ase_atoms, atoms_dict


def write_distribution_to_file(
    composition: float | str,
    filepath: str,
    dist: dict[int, int],
    label: str,
    *,
    append: bool = False,
) -> None:
    """Write a coordination/Qn histogram to a text file.

    Args:
        composition: Composition value to label row.
        filepath: Output filepath.
        dist: Histogram data.
        label: Prefix for headers (e.g., Si or Q).
        append: Append mode; writes header only if file does not exist.

    Example:
        >>> write_distribution_to_file(0.5, "qn.txt", qn_dist, "Q")

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
        bin_centers: Angle bin centers in degrees.
        angle_hist: Normalized angle histogram.
        composition: Composition value (e.g., % modifier).
        filepath: Output filepath.
        append: Whether to append to file.

    Example:
        >>> write_angle_distribution(centers, hist, 0.5, "angles.txt")

    """
    mode = "a" if append else "w"
    write_header = not append or not Path(filepath).exists()
    with Path(filepath).open(mode, encoding="utf-8") as f:
        if write_header:
            f.write("Composition " + " ".join(f"{b:.1f}" for b in bin_centers) + "\n")
        f.write(f"{composition} " + " ".join(f"{v:.6f}" for v in angle_hist) + "\n")


def structure_from_parsed_output(initial_structure: Atoms, parsed_output: dict, *, wrap: bool = False) -> Atoms:
    """Construct an `Atoms` object from parsed output data.

    Args:
        initial_structure: The initial atomic structure to use as a template.
        parsed_output: Parsed output containing atomic positions, cell, and indices.
        wrap: Whether to wrap the atomic positions to the simulation cell (default is False).
            Keeping the unwrapped positions is more beneficial if structures are passed between
            different LAMMPS simulations in one workflow to ensure continuity.

    Returns:
        An `Atoms` object with updated positions and cell.

    Example:
        >>> new_atoms = structure_from_parsed_output(atoms, lammps_output)

    """
    # Take a copy of the initial structure as template and update the relevant properties
    atoms_copy = initial_structure.copy()
    atoms_copy.set_array("indices", parsed_output["generic"]["indices"][-1])
    atoms_copy.set_positions(parsed_output["generic"]["positions"][-1])
    atoms_copy.set_velocities(parsed_output["generic"]["velocities"][-1])
    atoms_copy.set_cell(parsed_output["generic"]["cells"][-1])
    atoms_copy.set_pbc(True)
    if wrap:
        atoms_copy.wrap()

    return atoms_copy


def write_xyz(
    filename: str,
    coords: np.ndarray,
    types: np.ndarray,
    box_size: np.ndarray | None = None,
    type_dict: dict[int, str] | None = None,
) -> None:
    """Write atomic configuration to an XYZ file.

    Args:
        filename: Output XYZ file name.
        coords: Atomic coordinates.
        types: Atomic types as integers.
        box_size: Simulation box size in x, y, z.
        type_dict: Dictionary mapping atomic type integers to element symbols.

    Example:
        >>> write_xyz("output.xyz", coords, types, box, {14: "Si", 8: "O"})

    """
    if type_dict is None:
        msg = "type_dict must be provided"
        raise ValueError(msg)

    n_atoms = coords.shape[0]
    path = Path(filename)
    with path.open("w") as f:
        f.write(f"{n_atoms}\n")
        if box_size is not None:
            f.write(f"CUB {box_size[0]:.8f} {box_size[1]:.8f} {box_size[2]:.8f}\n")
        else:
            f.write("\n")
        for t, (x, y, z) in zip(types, coords, strict=False):
            symbol = type_dict.get(t)
            if symbol is None:
                msg = f"Unknown atomic type: {t}"
                raise ValueError(msg)
            f.write(f"{symbol} {x:.8f} {y:.8f} {z:.8f}\n")
