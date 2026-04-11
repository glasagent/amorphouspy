"""Random structure generation and ASE structure building for glass simulations.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from io import StringIO

import numpy as np
from ase.atoms import Atoms
from ase.io import read

from amorphouspy.mass import get_atomic_mass
from amorphouspy.shared import get_element_types_dict
from amorphouspy.structure.composition import extract_composition, extract_stoichiometry, get_composition
from amorphouspy.structure.planner import (
    _counts_from_n_molecules,
    get_box_from_density,
    plan_system,
    validate_target_mode,
)


def minimum_image_distance(
    pos1: np.ndarray,
    pos2: np.ndarray,
    box_length: float,
) -> float:
    """Calculate the minimum image distance between two points in a cubic periodic box.

    Args:
        pos1: The first position vector (NumPy array).
        pos2: The second position vector (NumPy array).
        box_length: The side length of the cubic box.

    Returns:
        The Euclidean distance between the two points considering periodic boundaries.

    """
    delta = np.abs(pos1 - pos2)
    delta = np.where(delta > 0.5 * box_length, box_length - delta, delta)
    return np.sqrt((delta**2).sum())


def _check_min_distance(
    candidate: np.ndarray,
    positions: np.ndarray,
    box_length: float,
    min_dist: float,
) -> bool:
    """Return True if candidate is at least min_dist from every position (vectorized)."""
    if len(positions) == 0:
        return True
    delta = np.abs(candidate - positions)
    delta = np.where(delta > 0.5 * box_length, box_length - delta, delta)
    return (delta**2).sum(axis=1).min() >= min_dist**2


def _place_atoms(  # noqa: C901, PLR0912
    atom_counts: dict[str, int],
    box_length: float,
    min_distance: float,
    rng: np.random.Generator,
    max_attempts_per_atom: int,
) -> list[dict]:
    """Place atoms in a periodic cubic box using a cell list for O(1) neighbor lookup."""
    n_cells = max(1, int(box_length / min_distance))
    cell_size = box_length / n_cells
    cell_map: dict[tuple[int, int, int], list[np.ndarray]] = {}
    min_dist_sq = min_distance**2
    atoms: list[dict] = []

    _BATCH = 512
    candidates = rng.uniform(0, box_length, size=(_BATCH, 3))
    _ci = 0

    for elem, count in atom_counts.items():
        placed = 0
        attempts = 0
        while placed < count:
            if attempts >= max_attempts_per_atom:
                error_msg = f"Failed to place {elem} atoms: increase box or reduce min_distance"
                raise RuntimeError(error_msg)
            if _ci >= _BATCH:
                candidates = rng.uniform(0, box_length, size=(_BATCH, 3))
                _ci = 0
            pos = candidates[_ci]
            _ci += 1
            cx = int(pos[0] / cell_size) % n_cells
            cy = int(pos[1] / cell_size) % n_cells
            cz = int(pos[2] / cell_size) % n_cells
            ok = True
            for dx in range(-1, 2):
                if not ok:
                    break
                for dy in range(-1, 2):
                    if not ok:
                        break
                    for dz in range(-1, 2):
                        nc = ((cx + dx) % n_cells, (cy + dy) % n_cells, (cz + dz) % n_cells)
                        for q in cell_map.get(nc, ()):
                            d = np.abs(pos - q)
                            d = np.where(d > 0.5 * box_length, box_length - d, d)
                            if (d * d).sum() < min_dist_sq:
                                ok = False
                                break
            if ok:
                atoms.append({"element": elem, "position": pos.tolist()})
                cell_map.setdefault((cx, cy, cz), []).append(pos)
                placed += 1
                attempts = 0
            else:
                attempts += 1

    return atoms


def create_random_atoms(
    composition: dict[str, float],
    n_molecules: int | None = None,
    target_atoms: int | None = None,
    mode: str = "molar",
    stoichiometry: dict[str, dict[str, int]] | None = None,
    box_length: float = 50.0,
    min_distance: float = 1.6,
    seed: int = 42,
    max_attempts_per_atom: int = 100000,
) -> tuple[list[dict[str, str | list[float]]], dict[str, int]]:
    """Generate random atom positions for a glass system in a periodic cubic box.

    Supports specifying system size either by total number of molecules (`n_molecules`)
    or total number of atoms (`target_atoms`).

    Args:
        composition: A dictionary mapping oxide formulas to their fractions,
            e.g. {"CaO": 0.25, "Al2O3": 0.25, "SiO2": 0.5}.
        n_molecules: The desired total number of molecules. Mutually exclusive with `target_atoms`.
        target_atoms: The desired target number of atoms. Mutually exclusive with `n_molecules`.
        mode: The composition interpretation mode ("molar" or "weight"). Defaults to "molar".
        stoichiometry: Optional pre-calculated stoichiometry dictionary.
        box_length: The length of the cubic simulation box in Angstroms. Defaults to 50.0.
        min_distance: The minimum allowed distance between any two atoms in Angstroms. Defaults to 1.6.
        seed: Random seed for reproducibility. Defaults to 42.
        max_attempts_per_atom: Maximum attempts to place a single atom before failing. Defaults to 100000.

    Returns:
        A tuple containing:
            - A list of atom dictionaries, each with "element" and "position".
            - A dictionary of total counts per element.

    Raises:
        ValueError: If composition sum is invalid or target mode is ambiguous.
        RuntimeError: If atom placement fails due to overcrowding (max attempts reached).

    Example:
        >>> atoms_list, atom_counts = create_random_atoms(
        ...     composition={"SiO2": 0.8, "Na2O": 0.2},
        ...     target_atoms=500,
        ...     box_length=20.0
        ... )

    """
    rng = np.random.default_rng(seed)

    validate_target_mode(n_molecules, target_atoms)

    if stoichiometry is None:
        stoichiometry = extract_stoichiometry(composition)

    if target_atoms is not None:
        system_plan = plan_system(composition, target_atoms, mode=mode, target_type="atoms")
        atom_counts = system_plan["element_counts"]
    else:
        assert n_molecules is not None
        _, atom_counts = _counts_from_n_molecules(composition, n_molecules, mode, stoichiometry)

    atoms = _place_atoms(atom_counts, box_length, min_distance, rng, max_attempts_per_atom)
    return atoms, atom_counts


def get_ase_structure(atoms_dict: dict, replicate: tuple[int, int, int] = (1, 1, 1)) -> Atoms:
    """Generate a LAMMPS data file format string and read into an ASE Atoms object.

    Based on the specifications in the provided atoms_dict,
    this function generates a LAMMPS data file
    format string, which is then read into an ASE Atoms object.
    The ASE Atoms object is then returned.
    atoms_dict is expected to specify a cubic box.
    Triclinic boxes are not supported.

    Args:
        atoms_dict: Dictionary that must contain the atom counts and box dimensions under the
            keys "atoms" and "box".
        replicate: Replication factors for the box in x, y, and z directions.
            Default is (1, 1, 1), meaning no replication.

    Returns:
        ASE Atoms object of the specified structure.

    Example:
        >>> struct_dict = get_structure_dict({"CaO": 0.25, "Al2O3": 0.25, "SiO2": 0.5}, target_atoms=1000)
        >>> atoms = get_ase_structure(struct_dict)

    """
    atoms = atoms_dict["atoms"]
    box_length = atoms_dict["box"]
    nx, ny, nz = replicate
    n_atoms_orig = len(atoms)

    element_to_type = get_element_types_dict(atoms)
    n_types = len(element_to_type)

    list_of_lines = []
    list_of_lines.append("LAMMPS data file via create_random_atoms and write_lammps_data\n\n")

    n_atoms = n_atoms_orig * nx * ny * nz
    list_of_lines.append(f"{n_atoms} atoms\n")
    list_of_lines.append(f"{n_types} atom types\n\n")

    new_box_length_x = box_length * nx
    new_box_length_y = box_length * ny
    new_box_length_z = box_length * nz

    list_of_lines.append(f"0.0 {new_box_length_x} xlo xhi\n")
    list_of_lines.append(f"0.0 {new_box_length_y} ylo yhi\n")
    list_of_lines.append(f"0.0 {new_box_length_z} zlo zhi\n\n")

    list_of_lines.append("Masses\n\n")
    for elem, type_id in element_to_type.items():
        mass = get_atomic_mass(elem)
        list_of_lines.append(f"{type_id} {mass} # {elem}\n")

    list_of_lines.append("\nAtoms\n\n")

    atom_id = 1
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for atom in atoms:
                    elem = atom["element"]
                    type_id = element_to_type[elem]
                    x, y, z = atom["position"]
                    x_shifted = x + ix * box_length
                    y_shifted = y + iy * box_length
                    z_shifted = z + iz * box_length
                    q = 0.0
                    list_of_lines.append(
                        f"{atom_id} {type_id} {q:.6f} {x_shifted:.6f} {y_shifted:.6f} {z_shifted:.6f}\n"
                    )
                    atom_id += 1

    atoms_obj = read(
        filename=StringIO("".join(list_of_lines)),
        format="lammps-data",
        atom_style="charge",
    )
    if isinstance(atoms_obj, list):
        return atoms_obj[0]  # type: ignore[return-value]
    return atoms_obj


def get_structure_dict(
    composition: dict[str, float],
    n_molecules: int | None = None,
    target_atoms: int | None = None,
    mode: str = "molar",
    density: float | None = None,
    min_distance: float = 1.6,
    max_attempts_per_atom: int = 10000,
) -> dict:
    """Generate a structure dictionary for a given composition.

    Supports both n_molecules and target_atoms input modes,
    and both molar and weight composition modes.

    Args:
        composition: A dictionary mapping oxide formulas to their fractions,
            e.g. {"CaO": 0.25, "Al2O3": 0.25, "SiO2": 0.5}.
        n_molecules: Total number of molecules (traditional mode).
        target_atoms: Target number of atoms (new mode).
        mode: Composition mode: "molar" for mol%, "weight" for weight%.
        density: Density in g/cm^3, default is calculated from model.
        min_distance: Minimum distance between any two atoms in angstroms, default is 1.6 Å.
        max_attempts_per_atom: Maximum attempts to place an atom before giving up, default is 10000.

    Returns:
        A dictionary containing:
            - "atoms": A list of atom dictionaries with keys "element" and "position"
            - "box": The length of the cubic box in angstroms
            - "formula_units": Dictionary of oxide formula units
            - "total_atoms": Total number of atoms
            - "element_counts": Dictionary of element counts (if target_atoms mode)
            - "mol_fraction": Dictionary of molar fractions (if target_atoms mode)

    Example:
        >>> struct_dict = get_structure_dict(
        ...     composition={"CaO": 0.25, "Al2O3": 0.25, "SiO2": 0.5},
        ...     target_atoms=1000,
        ...     mode="molar"
        ... )

    """
    validate_target_mode(n_molecules, target_atoms)

    stoichiometry = extract_stoichiometry(composition)

    box_length = get_box_from_density(
        composition,
        n_molecules=n_molecules,
        target_atoms=target_atoms,
        mode=mode,
        density=density,
        stoichiometry=stoichiometry,
    )

    atoms_list, _ = create_random_atoms(
        composition,
        n_molecules=n_molecules,
        target_atoms=target_atoms,
        mode=mode,
        box_length=box_length,
        min_distance=min_distance,
        max_attempts_per_atom=max_attempts_per_atom,
        stoichiometry=stoichiometry,
    )

    if target_atoms is not None:
        system_plan = plan_system(composition, target_atoms, mode=mode, target_type="atoms")
        molecule_counts = system_plan["formula_units"]
        total_atoms = system_plan["total_atoms"]
        element_counts = system_plan["element_counts"]
        mol_fraction = system_plan["mol_fraction"]
    else:
        assert n_molecules is not None
        composition_dict = extract_composition(composition)
        molecule_counts = {oxide: round(frac * n_molecules) for oxide, frac in composition_dict.items()}
        diff = n_molecules - sum(molecule_counts.values())
        if diff:
            main = max(composition_dict, key=lambda ox: composition_dict[ox])
            molecule_counts[main] += diff

        total_atoms = 0
        element_counts: dict[str, int] = {}
        for ox, mol_cnt in molecule_counts.items():
            stoich = stoichiometry.get(ox)
            if stoich is None:
                error_msg = f"Unknown oxide formula: {ox}"
                raise KeyError(error_msg)
            for elem, num in stoich.items():
                element_counts[elem] = element_counts.get(elem, 0) + num * mol_cnt
            total_atoms += sum(stoich.values()) * mol_cnt

        mol_fraction = get_composition(composition, mode=mode)

    return {
        "atoms": atoms_list,
        "box": box_length,
        "formula_units": molecule_counts,
        "total_atoms": total_atoms,
        "element_counts": element_counts,
        "mol_fraction": mol_fraction,
    }
