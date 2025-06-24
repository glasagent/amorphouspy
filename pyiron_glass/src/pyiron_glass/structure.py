import re
from io import StringIO

import numpy as np
import scipy
from ase.io import read
from pyiron_base import job

from pyiron_glass.mass import get_atomic_mass
from pyiron_glass.shared import get_element_types_dict

# 1. Compile once: match an element symbol ([A-Z][a-z]*)
#    followed by an optional integer count (\d*)
#    — missing digits will yield an empty string
ELEMENT = re.compile(r"([A-Z][a-z]*)(\d*)")


def extract_composition(composition: str) -> dict[str, float]:
    """Function written to extract the fraction of each element from a given composition.
    The composition can be given as a fraction or in mol%, and the function will return
    the molar fraction in all cases
    Example of usage: extract_composition("0.25CaO-0.25Al2O3-0.5SiO2")
                      extract_composition("79SiO2-13B2O3-3Al2O3-4Na2O-1K2O")

    Example of an output:
    print(extract_composition("0.25CaO-0.25Al2O3-0.5SiO2"))
    Output: {'CaO': 0.25, 'Al2O3': 0.25, 'SiO2': 0.5}
    """
    comp_dict = {}
    total = 0.0
    for segment in composition.split("-"):
        # Find the index of the first letter
        idx = next(i for i, ch in enumerate(segment) if ch.isalpha())
        frac_str, oxide = segment[:idx], segment[idx:]
        frac = float(frac_str) if frac_str else 1.0
        if frac > 1.0:
            frac = frac / 100
        comp_dict[oxide] = frac
        total += frac
    if round(total, 2) < 1.0:
        raise ValueError(f"Component sum ({total:.2f}) is less than 1.00")
    return comp_dict


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a chemical formula (e.g. "Al2O3") and return
    a dict of element counts: {"Al": 2, "O": 3}.
    """
    counts: dict[str, int] = {}
    for elem, cnt_str in ELEMENT.findall(formula):
        # Default to 1 if no digits were captured
        cnt = int(cnt_str) if cnt_str else 1
        counts[elem] = counts.get(elem, 0) + cnt
    return counts


def extract_stoichiometry(composition: str) -> dict[str, dict[str, int]]:
    """Given a composition string, return a mapping:
        { oxide_formula: { element: count, ... }, ... }
    Uses extract_composition() to isolate formulas first.
    """
    comp_dict = extract_composition(composition)
    stoichiometry: dict[str, dict[str, int]] = {}
    for oxide in comp_dict:
        stoichiometry[oxide] = parse_formula(oxide)
    return stoichiometry


def create_random_atoms(
    composition: str,
    n_molecules: int,
    STOICHIOMETRY: dict,
    box_length: float = 50.0,
    min_distance: float = 1.6,
    seed: int = 42,
    max_attempts_per_atom: int = 100000,
):
    """Generate random atom positions in a periodic cubic box, according to a given composition.

    - composition: e.g. "0.25CaO-0.25Al2O3-0.5SiO2"
    - n_molecules: total number of molecules to define atom counts
    - box_length: size of cubic box (calculate automatically from density or provide manually)
    - min_distance: minimum distance between any two atoms
    - seed: random seed for reproducibility
    - max_attempts_per_atom: max attempts to place an atom before giving up

    Returns:
        atoms: list of {"element": str, "position": [x, y, z]}
        atom_counts: dict of total counts per element

    """

    def minimum_image_distance(pos1, pos2, box_length):
        delta = np.abs(pos1 - pos2)
        delta = np.where(delta > 0.5 * box_length, box_length - delta, delta)
        return np.sqrt((delta**2).sum())

    np.random.seed(seed)

    # 1. Determine total atom counts
    comp_dict = extract_composition(composition)
    molecule_counts = {ox: round(frac * n_molecules) for ox, frac in comp_dict.items()}
    diff = n_molecules - sum(molecule_counts.values())
    if diff:
        main = max(comp_dict, key=comp_dict.get)
        molecule_counts[main] += diff

    atom_counts = {}
    for ox, mol_cnt in molecule_counts.items():
        stoich = STOICHIOMETRY.get(ox)
        if stoich is None:
            raise KeyError(f"Unknown oxide formula: {ox}")
        for elem, num in stoich.items():
            atom_counts[elem] = atom_counts.get(elem, 0) + num * mol_cnt

    # 2. Place atoms with min distance using periodic boundary conditions
    atoms = []
    positions = []

    for elem, count in atom_counts.items():
        placed = 0
        attempts = 0
        while placed < count:
            if attempts >= max_attempts_per_atom:
                raise RuntimeError(f"Failed to place {elem} atoms: increase box or reduce min_distance")
            pos = np.random.uniform(0, box_length, size=3)
            if all(minimum_image_distance(pos, p, box_length) >= min_distance for p in positions):
                atoms.append({"element": elem, "position": pos.tolist()})
                positions.append(pos)
                placed += 1
                attempts = 0
            else:
                attempts += 1

    return atoms  # , atom_counts


def get_box_from_density(composition: str, n_molecules: int, STOICHIOMETRY: dict, density: float = 2.65) -> float:
    """Calculate the cubic box length in angstroms needed for a given composition,
    number of molecules, and target density (g/cm^3).
    very straightforward function that calculates the box length from the density
    and the number of molecules.

    Steps:
      1. Parse composition into oxide fractions.
      2. Compute molecule counts and adjust rounding discrepancies.
      3. Tally per-element atom counts via STOICHIOMETRY.
      4. Compute total mass (g) using get_atomic_mass and AVOGADRO.
      5. Derive volume (cm3) from mass/density, convert to angstrom3,
         and return cube root for box length.
    """
    # 1. Determine molecule counts
    comp_dict = extract_composition(composition)
    molecule_counts = {ox: round(frac * n_molecules) for ox, frac in comp_dict.items()}
    # Adjust rounding error
    diff = n_molecules - sum(molecule_counts.values())
    if diff:
        main = max(comp_dict, key=comp_dict.get)
        molecule_counts[main] += diff

    # 2. Compute per-element atom counts
    atom_counts: dict[str, int] = {}
    for oxide, mol_cnt in molecule_counts.items():
        stoich = STOICHIOMETRY[oxide]
        for elem, num in stoich.items():
            atom_counts[elem] = atom_counts.get(elem, 0) + num * mol_cnt

    # 3. Total mass in grams
    #    (sum of atom_counts × atomic_mass) / Avogadro
    total_mass_g = sum(atom_counts[el] * get_atomic_mass(el) for el in atom_counts) / scipy.constants.Avogadro

    # 4. Compute volume (cm3) and convert to \AA3 (1 cm3 = 1e24 \AA3)
    volume_cm3 = total_mass_g / density
    volume_A3 = volume_cm3 * 1e24

    # 5. Box length in \AA
    box_length_A = volume_A3 ** (1 / 3)
    return box_length_A


@job
def get_ase_structure(atoms_dict: dict):
    """Based on the specifications in the provided atoms_dict, this function generates a LAMMPS data file
    format string, which is then read into an ASE Atoms object.The ASE Atoms object is then returned.
    atoms_dict is expected to specify a cubic box. Triclinic boxes are not supported.

    Parameters
    ----------
        atoms_dict : dict
            Dictionary that must contain the atom counts and box dimensions under the
            keys "atoms" and "box".

    Returns
    -------
        ase.Atoms object of the specified structure

    """
    atoms = atoms_dict["atoms"]
    box_length = atoms_dict["box"]
    n_atoms = len(atoms)
    # elements = sorted(set(atom['element'] for atom in atoms))
    # element_to_type = {elem: i+1 for i, elem in enumerate(elements)}  # e.g., {'Al':1, 'Ca':2, 'Na':3, 'O':4, 'Si':5}
    element_to_type = get_element_types_dict(atoms_dict=atoms_dict)
    n_types = len(element_to_type)

    list_of_lines = []
    # with open(filename, 'w') as f:
    # Header
    list_of_lines.append("LAMMPS data file via create_random_atoms and write_lammps_data\n\n")
    list_of_lines.append(f"{n_atoms} atoms\n")
    list_of_lines.append(f"{n_types} atom types\n\n")
    # Box dims
    list_of_lines.append(f"0.0 {box_length} xlo xhi\n")
    list_of_lines.append(f"0.0 {box_length} ylo yhi\n")
    list_of_lines.append(f"0.0 {box_length} zlo zhi\n\n")

    # Masses section
    list_of_lines.append("Masses\n\n")
    for elem, type_id in element_to_type.items():
        mass = get_atomic_mass(elem)  # You can later replace with real atomic masses if needed
        list_of_lines.append(f"{type_id} {mass} # {elem}\n")

    list_of_lines.append("\nAtoms\n\n")

    # Atoms section
    for i, atom in enumerate(atoms, start=1):
        elem = atom["element"]
        type_id = element_to_type[elem]
        x, y, z = atom["position"]
        q = 0.0
        # Charge, I put 0 for simplicity.
        # the real value should be set by the potential parameters either in LAMMPS or in pyiron
        # it can also be calculated automatically here if needed but the potential model should be specified in advance.
        # I wanted to keep these function as general as possible.
        list_of_lines.append(f"{i} {type_id} {q:.6f} {x:.6f} {y:.6f} {z:.6f}\n")
    return read(
        filename=StringIO("".join(list_of_lines)),
        format="lammps-data",
        atom_style="charge",
    )


@job
def get_structure_dict(
    comp,
    n_molecules=100,
    density=2.96 * 1.0,
    min_distance=1.6,
    max_attempts_per_atom=10000,
):
    """Generate a structure dictionary for a given composition, number of molecules, and density.
    This function creates a cubic box of atoms based on the specified composition and density.
    It uses the `create_random_atoms` function to generate atom positions and returns a dictionary
    containing the atoms and box length.

    Parameters
    ----------
        comp : str
            Composition string, e.g. "0.25CaO-0.25Al2O3-0.5SiO2" or "79SiO2-13B2O3-3Al2O3-4Na2O-1K2O"
        n_molecules : int
            Total number of molecules (actually atoms) to define atom counts.
        density : float
            Density in g/cm^3, default is 2.96 g/cm^3.
        min_distance : float
            Minimum distance between any two atoms in angstroms, default is 1.6 Å.
        max_attempts_per_atom : int
            Maximum attempts to place an atom before giving up, default is 10000.

    Returns
    -------
        dict: A dictionary containing:
            - "atoms": A list of atom dictionaries with keys "element" and "position".
            - "box": The length of the cubic box in angstroms.

    """
    STOICHIOMETRY = extract_stoichiometry(comp)
    box_length = get_box_from_density(comp, n_molecules=n_molecules, density=density, STOICHIOMETRY=STOICHIOMETRY)
    atoms_dict = create_random_atoms(
        comp,
        n_molecules=n_molecules,
        box_length=box_length,
        min_distance=min_distance,
        max_attempts_per_atom=max_attempts_per_atom,
        STOICHIOMETRY=STOICHIOMETRY,
    )
    return {"atoms": atoms_dict, "box": box_length}
