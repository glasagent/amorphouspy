# ruff: noqa: C901, PLR0912
"""Structure generation and analysis for oxide glass systems."""

import re
from io import StringIO

import numpy as np
import scipy
from ase.atoms import Atoms
from ase.data import chemical_symbols
from ase.io import read
from pyiron_base import job
from pymatgen.core import Composition

from pyiron_glass.mass import get_atomic_mass
from pyiron_glass.shared import get_element_types_dict

# 1. Compile once: match an element symbol ([A-Z][a-z]*)
#    followed by an optional integer count (\d*)
#    — missing digits will yield an empty string
ELEMENT = re.compile(r"([A-Z][a-z]*)(\d*)")

MAX_ASCII = 127
DENSITY_TOLERANCE = 1e-10
COMPOSITION_TOLERANCE = 1e-10

# Predefined list of trace oxides for the "Remainder" term
TRACE_OXIDES = {
    "Ag2O",
    "Bi2O3",
    "Br",
    "CoxOy",
    "Cr2O3",
    "Cs2O",
    "CuO",
    "Ga2O3",
    "Gd2O3",
    "I",
    "MoO3",
    "Nb2O5",
    "PdO",
    "PrxOy",
    "Rb2O",
    "RexOy",
    "RhxOy",
    "RuO2",
    "SeO2",
    "Sm2O3",
    "SnO2",
    "TeO2",
    "Tl2O3",
    "WO3",
    "Y2O3",
}


def check_neutral_oxide(oxide: str) -> None:
    """Check charge neutrality using pymatgen's oxidation state guesses."""
    try:
        comp = Composition(oxide)
    except Exception as e:
        error_msg = f"Invalid oxide formula: '{oxide}'"
        raise ValueError(error_msg) from e

    # Get oxidation state guesses (context-aware)
    oxi_guesses = comp.oxi_state_guesses()
    if not oxi_guesses:
        error_msg = f"Cannot determine oxidation states for '{oxide}'"
        raise ValueError(error_msg)

    # Calculate total charge using the most probable guess
    total_charge = sum(oxi * comp[el] for el, oxi in oxi_guesses[0].items())

    if total_charge != 0:
        error_msg = f"Oxide '{oxide}' is not charge neutral (net charge {total_charge})"
        raise ValueError(error_msg)


def extract_composition(composition: str) -> dict[str, float]:
    """Extract the fraction of each element from a given composition, checking validity.

    The composition can be given as a fraction or in mol%, and the function will return
    the molar fraction in all cases
    Example of usage: extract_composition("0.25CaO-0.25Al2O3-0.5SiO2")
                      extract_composition("79SiO2-13B2O3-3Al2O3-4Na2O-1K2O").

    Example of an output:
    print(extract_composition("0.25CaO-0.25Al2O3-0.5SiO2"))
    Output: {'CaO': 0.25, 'Al2O3': 0.25, 'SiO2': 0.5}

    - Supports fractions (sum=1.0) or percentages (sum=100.0)
    - Rejects non-ASCII characters
    - Validates element symbols using ASE
    - Checks oxide charge neutrality via check_neutral_oxide()
    """
    if any(ord(ch) > MAX_ASCII for ch in composition):
        error_msg = f"Composition contains non-ASCII characters: {composition!r}"
        raise ValueError(error_msg)

    comp_dict = {}
    total = 0.0
    segments = composition.split("-")

    if not segments:
        error_msg = "Empty composition string"
        raise ValueError(error_msg)

    valid_elements = set(chemical_symbols[1:])

    for segment in segments:
        if not segment:
            continue

        try:
            idx = next(i for i, ch in enumerate(segment) if ch.isalpha())
        except StopIteration as e:
            error_msg = f"Invalid segment: '{segment}' contains no letters"
            raise ValueError(error_msg) from e

        frac_str, oxide = segment[:idx], segment[idx:]

        try:
            frac = float(frac_str) if frac_str else 1.0
        except ValueError as e:
            error_msg = f"Invalid fraction format: '{frac_str}' in segment '{segment}'"
            raise ValueError(error_msg) from e

        if frac < 0:
            error_msg = f"Negative fraction for '{oxide}': {frac}"
            raise ValueError(error_msg)

        # Validate element symbols
        matches = ELEMENT.findall(oxide)
        for element, _ in matches:
            if element not in valid_elements:
                error_msg = f"Invalid element '{element}' in oxide '{oxide}'"
                raise ValueError(error_msg)

        # Check charge neutrality
        check_neutral_oxide(oxide)

        comp_dict[oxide] = frac
        total += frac

    if total == 0:
        error_msg = "Total composition sum is zero"
        raise ValueError(error_msg)

    if abs(total - 100) <= COMPOSITION_TOLERANCE:
        comp_dict = {oxide: frac / 100 for oxide, frac in comp_dict.items()}
    elif 1.0 + COMPOSITION_TOLERANCE < total < 100 - COMPOSITION_TOLERANCE:
        error_msg = f"Invalid composition sum ({total:.4f}). "
        raise ValueError(error_msg)
    elif total > 100 + COMPOSITION_TOLERANCE:
        error_msg = f"Total exceeds 100%: {total:.2f}%"
        raise ValueError(error_msg)
    elif total < 1.0 - COMPOSITION_TOLERANCE:
        error_msg = f"Component sum ({total:.4f}) is less than 1.00"
        raise ValueError(error_msg)

    return comp_dict


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a chemical formula (e.g. "Al2O3") and return a dict of element counts.

    Example: {"Al": 2, "O": 3}.
    """
    counts: dict[str, int] = {}
    for elem, cnt_str in ELEMENT.findall(formula):
        # Default to 1 if no digits were captured
        cnt = int(cnt_str) if cnt_str else 1
        counts[elem] = counts.get(elem, 0) + cnt
    return counts


def extract_stoichiometry(composition: str) -> dict[str, dict[str, int]]:
    """Given a composition string, return a mapping.

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
    stoichiometry: dict[str, dict[str, int]],
    box_length: float = 50.0,
    min_distance: float = 1.6,
    seed: int = 42,
    max_attempts_per_atom: int = 100000,
) -> tuple[list[dict[str, str | list[float]]], dict[str, int]]:
    """Generate random atom positions in a periodic cubic box, according to a given composition.

    - composition: e.g. "0.25CaO-0.25Al2O3-0.5SiO2"
    - n_molecules: total number of molecules to define atom counts
    - box_length: size of cubic box (calculate automatically from density or provide manually)
    - min_distance: minimum distance between any two atoms
    - seed: random seed for reproducibility
    - max_attempts_per_atom: max attempts to place an atom before giving up

    Returns:
        atoms: list of {"element": str, "position": [x, y, z]}

    """

    def minimum_image_distance(
        pos1: np.ndarray,
        pos2: np.ndarray,
        box_length: float,
    ) -> float:
        delta = np.abs(pos1 - pos2)
        delta = np.where(delta > 0.5 * box_length, box_length - delta, delta)
        return np.sqrt((delta**2).sum())

    rng = np.random.default_rng(seed)

    # 1. Determine total atom counts
    comp_dict = extract_composition(composition)
    # In create_random_atoms()
    total_molecules = sum(comp_dict.values())
    if abs(total_molecules - 1.0) > DENSITY_TOLERANCE:
        error_msg = f"Composition sum error: {total_molecules:.10f} != 1.0"
        raise ValueError(error_msg)
    molecule_counts = {ox: round(frac * n_molecules) for ox, frac in comp_dict.items()}
    diff = n_molecules - sum(molecule_counts.values())
    if diff:
        main = max(comp_dict, key=comp_dict.get)
        molecule_counts[main] += diff

    atom_counts = {}
    for ox, mol_cnt in molecule_counts.items():
        stoich = stoichiometry.get(ox)
        if stoich is None:
            msg = f"Unknown oxide formula: {ox}"
            raise KeyError(msg)
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
                msg = f"Failed to place {elem} atoms: increase box or reduce min_distance"
                raise RuntimeError(msg)
            pos = rng.uniform(0, box_length, size=3)
            if all(minimum_image_distance(pos, p, box_length) >= min_distance for p in positions):
                atoms.append({"element": elem, "position": pos.tolist()})
                positions.append(pos)
                placed += 1
                attempts = 0
            else:
                attempts += 1

    return atoms


def get_glass_density_from_model(composition_string: str) -> float:
    """Calculate the room-temperature glass density from its chemical composition.

    The empirical polynomial model described by Fluegel (2007) is used.

    The model is based on mole percent (mol%) of oxides, excluding SiO2 from
    the summations. The coefficients for the calculation are hardcoded from
    Table II of the source paper.

    Paper:
        Title: Global Model for Calculating Room-Temperature Glass Density from the Composition
        Author: Alexander Fluegel
        First published: 26 June 2007
        DOI: https://doi.org/10.1111/j.1551-2916.2007.01751.x

    Args:
        composition_string (str): A string representing the glass composition
                                  in mole fraction or mol%, e.g., '0.25CaO-0.25Al2O3-0.5SiO2'
                                  or '25CaO-25Al2O3-50SiO2'.

    Returns:
        float: The calculated density in g/cm^3.
        None: If the input string format is invalid or a component is not in the model.

    Examples:
        composition = "0.20Na2O-0.80SiO2"  # 20 mol% Na₂O
        density = calculate_glass_density(composition)
        Expected: ~2.391 g/cm³

    """
    COEFFICIENTS = {
        "b0": 2.121560704,
        "Al2O3": 0.010525974,
        "Al2O3_2": -0.000076924,
        "B2O3": 0.00579283,
        "B2O3_2": 0.000129174,
        "B2O3_3": -0.000019887,
        "Li2O": 0.012848733,
        "Li2O_2": -0.000276404,
        "Li2O_3": 0.00000259,
        "Na2O": 0.018129123,
        "Na2O_2": -0.000264838,
        "Na2O_3": 0.000001614,
        "K2O": 0.019177312,
        "K2O_2": -0.000319863,
        "K2O_3": 0.00000191,
        "MgO": 0.01210604,
        "MgO_2": -0.000061159,
        "CaO": 0.017992367,
        "CaO_2": -0.00005478,
        "SrO": 0.034630735,
        "SrO_2": -0.000086939,
        "BaO": 0.049879597,
        "BaO_2": -0.000168063,
        "ZnO": 0.025221567,
        "ZnO_2": 0.000099961,
        "PbO": 0.070020298,
        "PbO_2": 0.000214424,
        "PbO_3": -0.000001502,
        "FexOy": 0.036995747,
        "MnxOy": 0.016648722,
        "TiO2": 0.018820343,
        "ZrO2": 0.043059714,
        "ZrO2_2": -0.000779078,
        "CexOy": 0.061277268,
        "CdO": 0.052945783,
        "La2O3": 0.10643194,
        "Nd2O3": 0.090134135,
        "NiO": 0.024289113,
        "ThO2": 0.090253734,
        "UxOy": 0.063297404,
        "SbxOy": 0.044258719,
        "SO3": -0.044488661,
        "F": 0.00109839,
        "Cl": -0.006092537,
        "Remainder": 0.02514614,
        "Na2O_K2O": -0.000395491,
        "Na2O_Li2O": -0.00031449,
        "K2O_Li2O": -0.000329725,
        "Na2O_B2O3": 0.000242157,
        "K2O_B2O3": 0.000259927,
        "Li2O_B2O3": 0.000106359,
        "MgO_B2O3": -0.000206488,
        "CaO_B2O3": -0.000032258,
        "PbO_B2O3": -0.000186195,
        "FexOy_B2O3": -0.000720268,
        "ZrO2_B2O3": -0.000697195,
        "Al2O3_B2O3": -0.000735749,
        "Li2O_Al2O3": -0.000116227,
        "Na2O_Al2O3": -0.000253454,
        "K2O_Al2O3": -0.000371858,
        "MgO_CaO": 0.000057248,
        "MgO_Al2O3": 0.000167218,
        "MgO_ZnO": 0.000220766,
        "Li2O_CaO": -0.00008792,
        "Na2O_MgO": -0.000300745,
        "Na2O_CaO": -0.000228249,
        "Na2O_SrO": -0.00023137,
        "Na2O_BaO": -0.000171693,
        "K2O_MgO": -0.000337747,
        "K2O_CaO": -0.000349578,
        "K2O_SrO": -0.000425589,
        "K2O_BaO": -0.000392897,
        "Al2O3_CaO": -0.000102444,
        "Al2O3_PbO": -0.000651745,
        "Al2O3_TiO2": -0.000563594,
        "Al2O3_BaO": -0.000273835,
        "Al2O3_SrO": -0.000177761,
        "Al2O3_ZnO": -0.000109968,
        "Al2O3_ZrO2": -0.002381651,
        "Na2O_PbO": -0.000036455,
        "Na2O_TiO2": -0.00014331,
        "Na2O_ZnO": -0.000155275,
        "Na2O_ZrO2": -0.000126728,
        "Na2O_FexOy": -0.000371343,
        "K2O_PbO": -0.000525213,
        "K2O_TiO2": -0.000386587,
        "K2O_ZnO": -0.000329812,
        "CaO_PbO": -0.00084145,
        "ZnO_FexOy": -0.001536804,
        "Na2O_K2O_B2O3": -0.000032967,
        "Na2O_MgO_CaO": -0.000009143,
        "Na2O_MgO_Al2O3": -0.000012286,
        "Na2O_CaO_Al2O3": -0.000005106,
        "Na2O_CaO_PbO": 0.000100796,
        "K2O_MgO_CaO": -0.00001217,
        "K2O_MgO_Al2O3": -0.000041908,
        "K2O_CaO_Al2O3": -0.000012421,
        "K2O_CaO_PbO": 0.000125759,
        "MgO_CaO_Al2O3": -0.000011236,
        "CaO_Al2O3_Li2O": -0.000016177,
        "Al2O3_B2O3_PbO": 0.000030116,
    }
    try:
        mole_fractions = extract_composition(composition_string)
    except (ValueError, TypeError) as e:
        error_msg = f"Invalid composition string '{composition_string}': {e}"
        raise ValueError(error_msg) from e

    # Convert to mol% and separate components
    concentrations = {oxide: frac * 100 for oxide, frac in mole_fractions.items()}
    remainder_conc = 0.0
    main_components = []

    for oxide, conc in concentrations.items():
        if conc < 0:
            error_msg = f"Negative concentration for '{oxide}': {conc}"
            raise ValueError(error_msg)

        if oxide == "SiO2":
            continue

        if oxide in TRACE_OXIDES:
            remainder_conc += conc

        else:
            main_components.append(oxide)

    # Trace oxide validation
    if remainder_conc > 0 and "Remainder" not in COEFFICIENTS:
        error_msg = "Trace oxides present but 'Remainder' coefficient missing"
        raise ValueError(error_msg)

    # Component validation
    valid_components = set(COEFFICIENTS.keys()) - {"b0", "Remainder"}
    for comp in main_components:
        if comp not in valid_components:
            error_msg = f"Component '{comp}' not in density model coefficients"
            raise ValueError(error_msg)

    # Initialize density with intercept and remainder term
    density = COEFFICIENTS.get("b0", 0)
    if remainder_conc > 0:
        density += COEFFICIENTS.get("Remainder", 0) * remainder_conc

    # Process main components
    for comp in main_components:
        conc = concentrations[comp]
        # Linear terms
        if comp in COEFFICIENTS:
            density += COEFFICIENTS[comp] * conc
        # Quadratic terms
        quad_key = f"{comp}_2"
        if quad_key in COEFFICIENTS:
            density += COEFFICIENTS[quad_key] * (conc**2)
        # Cubic terms
        cube_key = f"{comp}_3"
        if cube_key in COEFFICIENTS:
            density += COEFFICIENTS[cube_key] * (conc**3)

    # Precompute keys for interactions
    TWO_WAY_KEY_PARTS = 1
    THREE_WAY_KEY_PARTS = 2
    two_way_keys = {k for k in COEFFICIENTS if k.count("_") == TWO_WAY_KEY_PARTS and not k.endswith(("_2", "_3"))}
    three_way_keys = {k for k in COEFFICIENTS if k.count("_") == THREE_WAY_KEY_PARTS}

    # Two-way interactions
    for key in two_way_keys:
        comp1, comp2 = key.split("_")
        if comp1 in main_components and comp2 in main_components:
            density += COEFFICIENTS[key] * concentrations[comp1] * concentrations[comp2]

    # Three-way interactions
    for key in three_way_keys:
        comp1, comp2, comp3 = key.split("_")
        if comp1 in main_components and comp2 in main_components and comp3 in main_components:
            density += COEFFICIENTS[key] * concentrations[comp1] * concentrations[comp2] * concentrations[comp3]

    return density


def get_box_from_density(
    composition: str,
    n_molecules: int,
    stoichiometry: dict[str, dict[str, int]],
    density: float | None = None,
) -> float:
    """Calculate the cubic box length in angstroms needed for a given composition.

    Very straightforward function that calculates the box length from the density
    and the number of molecules.

    Steps:
      1. Parse composition into oxide fractions.
      2. Compute molecule counts and adjust rounding discrepancies.
      3. Tally per-element atom counts via stoichiometry.
      4. Compute total mass (g) using get_atomic_mass and AVOGADRO.
      5. Derive volume (cm3) from mass/density, convert to angstrom3,
         and return cube root for box length.
    """
    if density is None:
        density = get_glass_density_from_model(composition)

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
        stoich = stoichiometry[oxide]
        for elem, num in stoich.items():
            atom_counts[elem] = atom_counts.get(elem, 0) + num * mol_cnt

    # 3. Total mass in grams
    #    (sum of atom_counts * atomic_mass) / Avogadro
    total_mass_g = sum(count * get_atomic_mass(elem) for elem, count in atom_counts.items()) / scipy.constants.Avogadro

    # 4. Compute volume (cm3) and convert to \AA3 (1 cm3 = 1e24 \AA3)
    volume_cm3 = total_mass_g / density
    volume_ang3 = volume_cm3 * 1e24

    # 5. Box length in \AA
    return volume_ang3 ** (1 / 3)


@job
def get_ase_structure(atoms_dict: dict, replicate: tuple[int, int, int] = (1, 1, 1)) -> Atoms:
    """Generate a LAMMPS data file format string and read into an ASE Atoms object.

    Based on the specifications in the provided atoms_dict,
    this function generates a LAMMPS data file
    format string, which is then read into an ASE Atoms object.
    The ASE Atoms object is then returned.
    atoms_dict is expected to specify a cubic box.
    Triclinic boxes are not supported.

    Parameters
    ----------
    atoms_dict : dict
        Dictionary that must contain the atom counts and box dimensions under the
        keys "atoms" and "box".
    replicate : tuple of int, optional
        Replication factors for the box in x, y, and z directions.
        Default is (1, 1, 1), meaning no replication.

    Returns
    -------
    ase.Atoms
        ASE Atoms object of the specified structure.

    """
    atoms = atoms_dict["atoms"]
    box_length = atoms_dict["box"]
    nx, ny, nz = replicate
    n_atoms_orig = len(atoms)

    element_to_type = get_element_types_dict(atoms_dict=atoms_dict)
    n_types = len(element_to_type)

    list_of_lines = []
    list_of_lines.append("LAMMPS data file via create_random_atoms and write_lammps_data\n\n")

    n_atoms = n_atoms_orig * nx * ny * nz
    list_of_lines.append(f"{n_atoms} atoms\n")
    list_of_lines.append(f"{n_types} atom types\n\n")

    # Adjust box size by replication factors
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
                    # Shift positions according to replication
                    x_shifted = x + ix * box_length
                    y_shifted = y + iy * box_length
                    z_shifted = z + iz * box_length
                    q = 0.0
                    list_of_lines.append(
                        f"{atom_id} {type_id} {q:.6f} {x_shifted:.6f} {y_shifted:.6f} {z_shifted:.6f}\n"
                    )
                    atom_id += 1

    return read(
        filename=StringIO("".join(list_of_lines)),
        format="lammps-data",
        atom_style="charge",
    )


@job
def get_structure_dict(
    comp: str,
    n_molecules: int = 100,
    density: float | None = None,
    min_distance: float = 1.6,
    max_attempts_per_atom: int = 10000,
) -> dict:
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
    stoichiometry = extract_stoichiometry(comp)
    box_length = get_box_from_density(
        comp,
        n_molecules=n_molecules,
        density=density,
        stoichiometry=stoichiometry,
    )
    atoms_dict = create_random_atoms(
        comp,
        n_molecules=n_molecules,
        box_length=box_length,
        min_distance=min_distance,
        max_attempts_per_atom=max_attempts_per_atom,
        stoichiometry=stoichiometry,
    )
    return {"atoms": atoms_dict, "box": box_length}
