"""Formula unit allocation, system planning, and box sizing for glass simulations.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from collections.abc import Mapping

import numpy as np
from scipy.constants import Avogadro

from amorphouspy.mass import get_atomic_mass
from amorphouspy.structure.composition import extract_composition, extract_stoichiometry, get_composition, parse_formula
from amorphouspy.structure.density import get_glass_density_from_model

DENSITY_TOLERANCE = 1e-10


def _atoms_per_fu_map(mol_frac: dict[str, float]) -> dict[str, int]:
    """Calculate the number of atoms per formula unit for each oxide.

    Args:
        mol_frac: A dictionary mapping oxide formulas to their target molar fractions.

    Returns:
        A dictionary mapping oxide formulas to the total number of atoms in one formula unit.
    """
    return {ox: sum(parse_formula(ox).values()) for ox in mol_frac}


def _integer_fu_from_total(nfu_target: int, mol_frac: dict[str, float]) -> dict[str, int]:
    """Allocate integer formula units to oxides using the largest-remainder method.

    This ensures that the total number of formula units sums exactly to `nfu_target`
    while best approximating the target molar fractions.

    Args:
        nfu_target: The target total number of formula units.
        mol_frac: A dictionary mapping oxide formulas to their target molar fractions.

    Returns:
        A dictionary mapping oxide formulas to their allocated integer formula units.

    """
    w = {ox: mol_frac[ox] * nfu_target for ox in mol_frac}
    n = {ox: int(np.floor(w[ox])) for ox in mol_frac}
    rem = nfu_target - sum(n.values())
    if rem > 0:
        order = sorted(mol_frac.keys(), key=lambda k: w[k] - n[k], reverse=True)
        for i in range(rem):
            n[order[i % len(order)]] += 1
    return n


def allocate_formula_units_to_target_atoms(
    mol_frac: dict[str, float],
    target_atoms: int,
    search_half_width: int = 1000,
) -> tuple[dict[str, int], int]:
    """Allocate formula units to approximate a target total number of atoms.

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
    nfu_est = max(1, round(target_atoms / avg_atoms_per_fu))

    best: tuple[int | None, int | None, dict[str, int] | None] = (None, None, None)
    lo = max(1, nfu_est - search_half_width)
    hi = nfu_est + search_half_width + 1

    for nfu in range(lo, hi):
        ni = _integer_fu_from_total(nfu, mol_frac)
        n_atoms = sum(ni[ox] * a[ox] for ox in ni)
        diff = abs(n_atoms - target_atoms)
        if best[0] is None or diff < best[0] or (diff == best[0] and n_atoms < best[1]):  # type: ignore[operator]
            best = (diff, n_atoms, ni)
            if diff == 0:
                break

    ni, n_atoms = best[2], best[1]
    if ni is None or n_atoms is None:
        msg = "Could not allocate formula units for the given composition and target size."
        raise RuntimeError(msg)
    return ni, n_atoms


def element_counts_from_formula_units(ni: Mapping[str, int | float]) -> dict[str, int | float]:
    """Calculate total element counts from oxide formula units.

    Args:
        ni: A dictionary mapping oxide formulas to their integer counts (formula units).

    Returns:
        A dictionary mapping element symbols to their total counts in the system.

    """
    counts: dict[str, int | float] = {}
    for ox, nfu in ni.items():
        sto = parse_formula(ox)
        for el, c in sto.items():
            counts[el] = counts.get(el, 0) + c * nfu
    return counts


def validate_target_mode(n_molecules: int | None, target_atoms: int | None) -> None:
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


def plan_system(composition: dict[str, float], target: int, mode: str = "molar", target_type: str = "atoms") -> dict:
    """Generate a comprehensive plan for the system composition and size.

    This unified planner handles both 'atoms' and 'molecules' target types,
    converting them into a concrete allocation of formula units and atoms.

    Args:
        composition: A dictionary mapping oxide formulas to their fractions.
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
    mol_frac = get_composition(composition, mode=mode)

    if target_type == "atoms":
        target_atoms = target
    elif target_type == "molecules":
        total_atoms_per_mol = sum(fraction * sum(parse_formula(oxide).values()) for oxide, fraction in mol_frac.items())
        target_atoms = round(target * total_atoms_per_mol)
    else:
        error_msg = f"Invalid target_type: {target_type}. Supported types are 'atoms' and 'molecules'."
        raise ValueError(error_msg)

    ni, n_atoms = allocate_formula_units_to_target_atoms(mol_frac, target_atoms)
    elem_counts = element_counts_from_formula_units(ni)
    return {
        "mol_fraction": mol_frac,
        "formula_units": ni,
        "total_atoms": n_atoms,
        "element_counts": elem_counts,
    }


def _counts_from_n_molecules(
    composition: dict[str, float],
    n_molecules: int,
    mode: str,
    stoichiometry: dict[str, dict[str, int]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Compute molecule and atom counts from a target molecule count."""
    if mode.lower() == "weight":
        mol_frac = get_composition(composition, mode="weight")
        comp_dict = {ox: mol_frac[ox] for ox in mol_frac}
    else:
        comp_dict = extract_composition(composition)

    total_molecules = sum(comp_dict.values())
    if abs(total_molecules - 1.0) > DENSITY_TOLERANCE:
        error_msg = f"Composition sum error: {total_molecules:.10f} != 1.0"
        raise ValueError(error_msg)

    molecule_counts = {ox: round(frac * n_molecules) for ox, frac in comp_dict.items()}
    diff = n_molecules - sum(molecule_counts.values())
    if diff:
        main = max(comp_dict, key=lambda ox: comp_dict[ox])
        molecule_counts[main] += diff

    atom_counts: dict[str, int] = {}
    for ox, mol_cnt in molecule_counts.items():
        stoich = stoichiometry.get(ox)
        if stoich is None:
            error_msg = f"Unknown oxide formula: {ox}"
            raise KeyError(error_msg)
        for elem, num in stoich.items():
            atom_counts[elem] = atom_counts.get(elem, 0) + num * mol_cnt

    return molecule_counts, atom_counts


def get_box_from_density(
    composition: dict[str, float],
    n_molecules: int | None,
    target_atoms: int | None,
    mode: str = "molar",
    density: float | None = None,
    stoichiometry: dict[str, dict[str, int]] | None = None,
) -> float:
    """Calculate the cubic box length in angstroms needed for a given composition.

    Supports both n_molecules and target_atoms input modes.

    Args:
        composition: A dictionary mapping oxide formulas to their fractions.
        n_molecules: Total number of molecules.
        target_atoms: Target total number of atoms.
        mode: Composition mode ("molar" or "weight"). Defaults to "molar".
        density: Optional density in g/cm^3. If None, calculated from model.
        stoichiometry: Optional pre-calculated stoichiometry dictionary.

    Returns:
        The side length of the cubic box in Angstroms.
    """
    validate_target_mode(n_molecules, target_atoms)

    if stoichiometry is None:
        stoichiometry = extract_stoichiometry(composition)

    if density is None:
        density = get_glass_density_from_model(composition)

    if target_atoms is not None:
        system_plan = plan_system(composition, target_atoms, mode=mode, target_type="atoms")
        atom_counts = system_plan["element_counts"]
    else:
        assert n_molecules is not None
        if mode.lower() == "weight":
            mol_frac = get_composition(composition, mode="weight")
            comp_dict = {ox: mol_frac[ox] for ox in mol_frac}
        else:
            comp_dict = extract_composition(composition)

        molecule_counts = {ox: round(frac * n_molecules) for ox, frac in comp_dict.items()}
        diff = n_molecules - sum(molecule_counts.values())
        if diff:
            main = max(comp_dict, key=lambda ox: comp_dict[ox])
            molecule_counts[main] += diff

        atom_counts: dict[str, int] = {}
        for oxide, mol_cnt in molecule_counts.items():
            stoich = stoichiometry[oxide]
            for elem, num in stoich.items():
                atom_counts[elem] = atom_counts.get(elem, 0) + num * mol_cnt

    total_mass_g = sum(count * get_atomic_mass(elem) for elem, count in atom_counts.items()) / Avogadro
    volume_cm3 = total_mass_g / density
    volume_ang3 = volume_cm3 * 1e24
    return volume_ang3 ** (1 / 3)
