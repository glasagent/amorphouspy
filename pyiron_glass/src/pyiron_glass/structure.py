# ruff: noqa: C901, PLR0912
"""Structure generation and analysis for oxide glass systems.

Author: Achraf Atila (achraf.atila@bam.de).
"""

import re
from io import StringIO

import numpy as np
from ase.atoms import Atoms
from ase.data import chemical_symbols
from ase.io import read
from pymatgen.core import Composition
from scipy.constants import Avogadro

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


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a chemical formula and returns a dictionary of element counts.

    Args:
        formula: A string representing the chemical formula (e.g., "Al2O3").

    Returns:
        A dictionary mapping element symbols (str) to their counts (int).
        Example: {"Al": 2, "O": 3} for "Al2O3".

    """
    counts: dict[str, int] = {}
    for elem, cnt_str in ELEMENT.findall(formula):
        # Default to 1 if no digits were captured
        cnt = int(cnt_str) if cnt_str else 1
        counts[elem] = counts.get(elem, 0) + cnt
    return counts


def formula_mass_g_per_mol(formula: str) -> float:
    """Calculate the molar mass of a compound using ASE atomic masses.

    Args:
        formula: A string representing the chemical formula (e.g., "SiO2").

    Returns:
        The molar mass of the compound in grams per mole (float).

    """
    return sum(get_atomic_mass(el) * cnt for el, cnt in parse_formula(formula).items())


def normalize(d: dict[str, float]) -> dict[str, float]:
    """Normalize a dictionary of values so that they sum to 1.0.

    Args:
        d: A dictionary where values are numbers to be normalized.

    Returns:
        A new dictionary with the same keys but normalized values.

    Raises:
        ValueError: If the sum of values is non-positive.

    """
    s = float(sum(d.values()))
    if s <= 0:
        error_msg = "Sum of fractions are non-positive."
        raise ValueError(error_msg)
    return {k: v / s for k, v in d.items()}


def weight_percent_to_mol_fraction(comp_wt_raw: dict[str, float]) -> dict[str, float]:
    """Convert weight fractions to molar fractions.

    The conversion uses the formula:
    x_i = (w_i/M_i) / sum(w_j/M_j)
    where x_i is the mole fraction, w_i is the weight fraction, and M_i is the molar mass.

    Args:
        comp_wt_raw: A dictionary mapping oxide formulas to their weight fractions (or percentages).

    Returns:
        A dictionary mapping oxide formulas to their normalized molar fractions.

    """
    n_i = {ox: comp_wt_raw[ox] / formula_mass_g_per_mol(ox) for ox in comp_wt_raw}
    return normalize(n_i)


def get_composition(comp_str: str, mode: str = "molar") -> dict[str, float]:
    """Parse a composition string into a dictionary of molar fractions.

    Args:
        comp_str: The composition string (e.g., "0.25CaO-0.25Al2O3-0.5SiO2").
        mode: The interpretation mode, either 'molar' or 'weight'. Defaults to "molar".

    Returns:
        A dictionary mapping oxide formulas to their molar fractions.

    Raises:
        ValueError: If `mode` is not 'molar' or 'weight'.

    """
    if mode.lower() not in ("molar", "weight"):
        error_msg = f"Invalid mode: {mode}. Supported modes are 'molar' and 'weight'."
        raise ValueError(error_msg)
    raw = extract_composition(comp_str)
    if mode.lower() == "weight":
        return weight_percent_to_mol_fraction(raw)
    return normalize(raw)


def _atoms_per_fu_map(mol_frac: dict[str, float]) -> dict[str, int]:
    return {ox: sum(parse_formula(ox).values()) for ox in mol_frac}


def _integer_fu_from_total(Nfu_target: int, mol_frac: dict[str, float]) -> dict[str, int]:
    """Allocates integer formula units to oxides using the largest-remainder method.

    This ensures that the total number of formula units sums exactly to `Nfu_target`
    while best approximating the target molar fractions.

    Args:
        Nfu_target: The target total number of formula units (int).
        mol_frac: A dictionary mapping oxide formulas to their target molar fractions.

    Returns:
        A dictionary mapping oxide formulas to their allocated integer formula units.

    """
    x = mol_frac
    w = {ox: x[ox] * Nfu_target for ox in x}
    n = {ox: int(np.floor(w[ox])) for ox in x}
    rem = Nfu_target - sum(n.values())
    if rem > 0:
        order = sorted(x.keys(), key=lambda k: (w[k] - n[k]), reverse=True)
        for i in range(rem):
            n[order[i % len(order)]] += 1
    return n


def allocate_formula_units_to_target_atoms(
    mol_frac: dict[str, float],
    target_atoms: int,
    search_half_width: int = 1000,
) -> tuple[dict[str, int], int]:
    """Allocates formula units to approximate a target total number of atoms.

    This function searches for a total number of formula units (Nfu) that results
    in a total atom count closest to `target_atoms`. It ensures exact stoichiometry
    for each oxide.

    Args:
        mol_frac: A dictionary mapping oxide formulas to their molar fractions.
        target_atoms: The desired total number of atoms in the system.
        search_half_width: The range to search around the estimated Nfu. Defaults to 1000.

    Returns:
        A tuple containing:
            - A dictionary mapping oxide formulas to their allocated integer formula units.
            - The actual total number of atoms resulting from this allocation.

    """
    a = _atoms_per_fu_map(mol_frac)
    avg_atoms_per_fu = sum(mol_frac[ox] * a[ox] for ox in mol_frac)
    Nfu_est = max(1, round(target_atoms / avg_atoms_per_fu))

    best = (None, None, None)  # (abs_diff, N_total_atoms, N_i)
    lo = max(1, Nfu_est - search_half_width)
    hi = Nfu_est + search_half_width + 1

    for Nfu in range(lo, hi):
        Ni = _integer_fu_from_total(Nfu, mol_frac)
        Natoms = sum(Ni[ox] * a[ox] for ox in Ni)
        diff = abs(Natoms - target_atoms)
        if best[0] is None or diff < best[0] or (diff == best[0] and Natoms < best[1]):
            best = (diff, Natoms, Ni)
            if diff == 0:
                break

    return best[2], best[1]


def element_counts_from_formula_units(Ni: dict[str, int]) -> dict[str, int]:
    """Calculate total element counts from oxide formula units.

    Args:
        Ni: A dictionary mapping oxide formulas to their integer counts (formula units).

    Returns:
        A dictionary mapping element symbols to their total counts in the system.

    """
    counts: dict[str, int] = {}
    for ox, nfu in Ni.items():
        sto = parse_formula(ox)
        for el, c in sto.items():
            counts[el] = counts.get(el, 0) + c * nfu
    return counts


def plan_system(comp_str: str, target: int, mode: str = "molar", target_type: str = "atoms") -> dict:
    """Generate a comprehensive plan for the system composition and size.

    This unified planner handles both 'atoms' and 'molecules' target types,
    converting them into a concrete allocation of formula units and atoms.

    Args:
        comp_str: The composition string.
        target: The target count (atoms or molecules depending on `target_type`).
        mode: The composition mode ('molar' or 'weight'). Defaults to "molar".
        target_type: The type of target ('atoms' or 'molecules'). Defaults to "atoms".

    Returns:
        A dictionary containing:
            - "mol_fraction": Dictionary of molar fractions.
            - "formula_units": Dictionary of allocated integer formula units.
            - "total_atoms": The actual total number of atoms.
            - "element_counts": Dictionary of total counts per element.

    Raises:
        ValueError: If `target_type` is not 'atoms' or 'molecules'.

    """
    mol_frac = get_composition(comp_str, mode=mode)

    if target_type == "atoms":
        target_atoms = target
    elif target_type == "molecules":
        # Calculate average atoms per formula unit in the composition
        total_atoms_per_mol = 0
        for oxide, fraction in mol_frac.items():
            # Use parse_formula to get element counts, then sum them
            element_counts = parse_formula(oxide)
            atoms_in_oxide = sum(element_counts.values())
            total_atoms_per_mol += fraction * atoms_in_oxide

        # Convert molecule target to equivalent atom target
        target_atoms = target * total_atoms_per_mol
    else:
        error_msg = f"Invalid target_type: {target_type}. Supported types are 'atoms' and 'molecules'."
        raise ValueError(error_msg)

    Ni, Natoms = allocate_formula_units_to_target_atoms(mol_frac, target_atoms)
    elem_counts = element_counts_from_formula_units(Ni)
    return {
        "mol_fraction": mol_frac,
        "formula_units": Ni,
        "total_atoms": Natoms,
        "element_counts": elem_counts,
    }


def validate_target_mode(n_molecules: int, target_atoms: int) -> None:
    """Validate that exactly one target mode is specified.

    Args:
        n_molecules: The number of molecules (can be None).
        target_atoms: The number of atoms (can be None).

    Raises:
        ValueError: If both or neither are specified.

    """
    if n_molecules is None and target_atoms is None:
        error_msg = "Either n_molecules or target_atoms must be specified"
        raise ValueError(error_msg)
    if n_molecules is not None and target_atoms is not None:
        error_msg = "Only one of n_molecules or target_atoms can be specified"
        raise ValueError(error_msg)


def check_neutral_oxide(oxide: str) -> None:
    """Check if an oxide formula is charge neutral based on standard oxidation states.

    Args:
        oxide: The chemical formula of the oxide (e.g., "Al2O3").

    Raises:
        ValueError: If the oxide is invalid, oxidation states cannot be determined,
            or the net charge is not zero.

    """
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
    """Extract molar fractions from a composition string.

    Handles both fractional (0.0-1.0) and percentage (0-100) inputs. Always returns
    fractions summing to 1.0.

    Args:
        composition: The composition string (e.g., "0.25CaO-0.25Al2O3-0.5SiO2").

    Returns:
        A dictionary mapping oxide formulas to their molar fractions.

    Raises:
        ValueError: If input contains non-ASCII characters, has invalid format,
            invalid elements, non-neutral oxides, or sums to an invalid total.

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


def extract_stoichiometry(composition: str) -> dict[str, dict[str, int]]:
    """Extract the stoichiometry of each component in the composition.

    Args:
        composition: The composition string.

    Returns:
        A dictionary mapping oxide formulas to their stoichiometric dictionaries
        (e.g., {"Al2O3": {"Al": 2, "O": 3}}).

    """
    comp_dict = extract_composition(composition)
    return {oxide: parse_formula(oxide) for oxide in comp_dict}


def create_random_atoms(
    composition: str,
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
        composition: The composition string (e.g., "0.25CaO-0.25Al2O3-0.5SiO2").
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

    """
    rng = np.random.default_rng(seed)

    # 1. Determine total atom counts based on input mode
    validate_target_mode(n_molecules, target_atoms)

    if stoichiometry is None:
        stoichiometry = extract_stoichiometry(composition)

    if target_atoms is not None:
        # Use target atoms mode
        system_plan = plan_system(composition, target_atoms, mode=mode, target_type="atoms")
        molecule_counts = system_plan["formula_units"]
        atom_counts = system_plan["element_counts"]
    else:
        # Use traditional n_molecules mode
        if mode.lower() == "weight":
            # Convert weight% to mol% first, then calculate molecule counts
            mol_frac = get_composition(composition, mode="weight")
            comp_dict = {ox: mol_frac[ox] for ox in mol_frac}
        else:
            # Use molar composition directly
            comp_dict = extract_composition(composition)

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
                error_msg = f"Unknown oxide formula: {ox}"
                raise KeyError(error_msg)
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
                error_msg = f"Failed to place {elem} atoms: increase box or reduce min_distance"
                raise RuntimeError(error_msg)
            pos = rng.uniform(0, box_length, size=3)
            if all(minimum_image_distance(pos, p, box_length) >= min_distance for p in positions):
                atoms.append({"element": elem, "position": pos.tolist()})
                positions.append(pos)
                placed += 1
                attempts = 0
            else:
                attempts += 1

    return atoms, atom_counts


def get_glass_density_from_model(composition_string: str) -> float:
    """Calculate the room-temperature glass density using Fluegel's empirical model.

    The model uses a polynomial expansion based on mole percentages of oxides.
    Source: Fluegel, A. "Global Model for Calculating Room-Temperature Glass Density from the Composition",
    J. Am. Ceram. Soc., 90 [8] 2622-2635 (2007).

    Args:
        composition_string: The glass composition string (molar fractions or mol%).

    Returns:
        The calculated density in g/cm^3.

    Raises:
        ValueError: If the composition contains components unsupported by the model
            or if the format is invalid.

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
    n_molecules: int | None,
    target_atoms: int | None,
    mode: str = "molar",
    density: float | None = None,
    stoichiometry: dict[str, dict[str, int]] | None = None,
) -> float:
    """Calculate the cubic box length in angstroms needed for a given composition.

    Now supports both n_molecules and target_atoms input modes.
    """
    validate_target_mode(n_molecules, target_atoms)

    if stoichiometry is None:
        stoichiometry = extract_stoichiometry(composition)

    if density is None:
        density = get_glass_density_from_model(composition)

    # Determine molecule counts based on input mode
    if target_atoms is not None:
        # Use target atoms mode
        system_plan = plan_system(composition, target_atoms, mode=mode, target_type="atoms")
        molecule_counts = system_plan["formula_units"]
        atom_counts = system_plan["element_counts"]
    else:
        # Use traditional n_molecules mode
        if mode.lower() == "weight":
            # Convert weight% to mol% first, then calculate molecule counts
            mol_frac = get_composition(composition, mode="weight")
            comp_dict = {ox: mol_frac[ox] for ox in mol_frac}
        else:
            # Use molar composition directly
            comp_dict = extract_composition(composition)

        molecule_counts = {ox: round(frac * n_molecules) for ox, frac in comp_dict.items()}
        # Adjust rounding error
        diff = n_molecules - sum(molecule_counts.values())
        if diff:
            main = max(comp_dict, key=comp_dict.get)
            molecule_counts[main] += diff

        # Compute per-element atom counts
        atom_counts: dict[str, int] = {}
        for oxide, mol_cnt in molecule_counts.items():
            stoich = stoichiometry[oxide]
            for elem, num in stoich.items():
                atom_counts[elem] = atom_counts.get(elem, 0) + num * mol_cnt

    # Total mass in grams
    total_mass_g = sum(count * get_atomic_mass(elem) for elem, count in atom_counts.items()) / Avogadro

    # Compute volume (cm3) and convert to \AA3 (1 cm3 = 1e24 \AA3)
    volume_cm3 = total_mass_g / density
    volume_ang3 = volume_cm3 * 1e24

    # Box length in \AA
    return volume_ang3 ** (1 / 3)


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


def get_structure_dict(
    composition: str,
    n_molecules: int | None = None,
    target_atoms: int | None = None,
    mode: str = "molar",
    density: float | None = None,
    min_distance: float = 1.6,
    max_attempts_per_atom: int = 10000,
) -> dict:
    """Generate a structure dictionary for a given composition.

    Now supports both n_molecules and target_atoms input modes,
    and both molar and weight composition modes.

    Parameters
    ----------
    composition : str
        Composition string, e.g. "0.25CaO-0.25Al2O3-0.5SiO2" or "79SiO2-13B2O3-3Al2O3-4Na2O-1K2O"
    n_molecules : int, optional
        Total number of molecules (traditional mode)
    target_atoms : int, optional
        Target number of atoms (new mode)
    mode : str, default "molar"
        Composition mode: "molar" for mol%, "weight" for weight%
    density : float, optional
        Density in g/cm^3, default is calculated from model
    min_distance : float
        Minimum distance between any two atoms in angstroms, default is 1.6 Å
    max_attempts_per_atom : int
        Maximum attempts to place an atom before giving up, default is 10000

    Returns
    -------
    dict: A dictionary containing:
        - "atoms": A list of atom dictionaries with keys "element" and "position"
        - "box": The length of the cubic box in angstroms
        - "formula_units": Dictionary of oxide formula units
        - "total_atoms": Total number of atoms
        - "element_counts": Dictionary of element counts (if target_atoms mode)
        - "mol_fraction": Dictionary of molar fractions (if target_atoms mode)

    """
    validate_target_mode(n_molecules, target_atoms)

    stoichiometry = extract_stoichiometry(composition)

    # Calculate box length
    box_length = get_box_from_density(
        composition,
        n_molecules=n_molecules,
        target_atoms=target_atoms,
        mode=mode,
        density=density,
        stoichiometry=stoichiometry,
    )

    # Generate atom positions - NOTE: create_random_atoms now returns (atoms_list, atom_counts)
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

    # Get comprehensive system information for both modes
    if target_atoms is not None:
        # Use target atoms mode - get full system plan
        system_plan = plan_system(composition, target_atoms, mode=mode, target_type="atoms")
        molecule_counts = system_plan["formula_units"]
        total_atoms = system_plan["total_atoms"]
        element_counts = system_plan["element_counts"]
        mol_fraction = system_plan["mol_fraction"]
    else:
        # Use traditional n_molecules mode - compute missing info
        composition_dict = extract_composition(composition)
        molecule_counts = {oxide: round(frac * n_molecules) for oxide, frac in composition_dict.items()}
        # Adjust rounding error
        diff = n_molecules - sum(molecule_counts.values())
        if diff:
            main = max(composition_dict, key=composition_dict.get)
            molecule_counts[main] += diff

        # Compute additional information for consistency
        total_atoms = 0
        element_counts = {}
        for ox, mol_cnt in molecule_counts.items():
            stoich = stoichiometry.get(ox)
            if stoich is None:
                error_msg = f"Unknown oxide formula: {ox}"
                raise KeyError(error_msg)
            # Calculate element counts
            for elem, num in stoich.items():
                element_counts[elem] = element_counts.get(elem, 0) + num * mol_cnt
            total_atoms += sum(stoich.values()) * mol_cnt

        # Calculate mol fractions for consistency
        mol_fraction = get_composition(composition, mode=mode)

    return {
        "atoms": atoms_list,
        "box": box_length,
        "formula_units": molecule_counts,
        "total_atoms": total_atoms,
        "element_counts": element_counts,
        "mol_fraction": mol_fraction,
    }
