"""Composition parsing and normalization for oxide glass systems.

Author: Achraf Atila (achraf.atila@bam.de)
"""

import re

from ase.data import chemical_symbols
from pymatgen.core import Composition

from amorphouspy.mass import get_atomic_mass

# Compile once: match an element symbol ([A-Z][a-z]*) followed by an optional integer count (\d*)
ELEMENT = re.compile(r"([A-Z][a-z]*)(\d*)")

COMPOSITION_TOLERANCE = 0.001


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a chemical formula and returns a dictionary of element counts.

    Args:
        formula: A string representing the chemical formula (e.g., "Al2O3").

    Returns:
        A dictionary mapping element symbols to their counts.
        Example: {"Al": 2, "O": 3} for "Al2O3".

    """
    counts: dict[str, int] = {}
    for elem, cnt_str in ELEMENT.findall(formula):
        cnt = int(cnt_str) if cnt_str else 1
        counts[elem] = counts.get(elem, 0) + cnt
    return counts


def formula_mass_g_per_mol(formula: str) -> float:
    """Calculate the molar mass of a compound using ASE atomic masses.

    Args:
        formula: A string representing the chemical formula (e.g., "SiO2").

    Returns:
        The molar mass of the compound in grams per mole.

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


def get_composition(composition: dict[str, float], mode: str = "molar") -> dict[str, float]:
    """Convert a composition dictionary into normalized molar fractions.

    Args:
        composition: A dictionary mapping oxide formulas to their fractions,
            e.g. {"CaO": 0.25, "Al2O3": 0.25, "SiO2": 0.5}.
        mode: The interpretation mode, either 'molar' or 'weight'. Defaults to "molar".

    Returns:
        A dictionary mapping oxide formulas to their molar fractions.

    Raises:
        ValueError: If `mode` is not 'molar' or 'weight'.

    """
    if mode.lower() not in ("molar", "weight"):
        error_msg = f"Invalid mode: {mode}. Supported modes are 'molar' and 'weight'."
        raise ValueError(error_msg)
    raw = extract_composition(composition)
    if mode.lower() == "weight":
        return weight_percent_to_mol_fraction(raw)
    return normalize(raw)


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

    oxi_guesses = comp.oxi_state_guesses()
    if not oxi_guesses:
        error_msg = f"Cannot determine oxidation states for '{oxide}'"
        raise ValueError(error_msg)

    total_charge = sum(oxi * comp[el] for el, oxi in oxi_guesses[0].items())

    if total_charge != 0:
        error_msg = f"Oxide '{oxide}' is not charge neutral (net charge {total_charge})"
        raise ValueError(error_msg)


def extract_composition(
    composition: dict[str, float],
    tolerance: float = COMPOSITION_TOLERANCE,
) -> dict[str, float]:
    """Validate and normalize a composition dictionary.

    Handles both fractional (0.0-1.0) and percentage (0-100) inputs. Always returns
    fractions summing to 1.0.

    Args:
        composition: A dictionary mapping oxide formulas to their fractions,
            e.g. {"CaO": 0.25, "Al2O3": 0.25, "SiO2": 0.5}.
        tolerance: Maximum allowed fractional deviation from 1.0. Defaults to
            ``COMPOSITION_TOLERANCE``.

    Returns:
        A dictionary mapping oxide formulas to their molar fractions.

    Raises:
        ValueError: If the composition is empty, contains invalid elements,
            non-neutral oxides, or sums to an invalid total.

    """
    if not composition:
        error_msg = "Empty composition"
        raise ValueError(error_msg)

    valid_elements = set(chemical_symbols[1:])
    comp_dict = {}
    total = 0.0

    for oxide, frac in composition.items():
        if frac < 0:
            error_msg = f"Negative fraction for '{oxide}': {frac}"
            raise ValueError(error_msg)

        matches = ELEMENT.findall(oxide)
        for element, _ in matches:
            if element not in valid_elements:
                error_msg = f"Invalid element '{element}' in oxide '{oxide}'"
                raise ValueError(error_msg)

        check_neutral_oxide(oxide)

        comp_dict[oxide] = frac
        total += frac

    if total == 0:
        error_msg = "Total composition sum is zero"
        raise ValueError(error_msg)

    pct_tol = 100 * tolerance

    if abs(total - 100) <= pct_tol or abs(total - 1.0) <= tolerance:
        comp_dict = {oxide: frac / total for oxide, frac in comp_dict.items()}
    elif total > 100 + pct_tol:
        error_msg = f"Total exceeds 100% + {pct_tol}% tolerance: {total:.2f}%"
        raise ValueError(error_msg)
    elif total < 1.0 - tolerance:
        error_msg = f"Component sum ({total:.4f}) is less than allowed minimum"
        raise ValueError(error_msg)
    else:
        error_msg = f"Invalid composition sum ({total:.4f}). Expected ~100 mol% or ~1.0 fractional."
        raise ValueError(error_msg)

    return comp_dict


def extract_stoichiometry(composition: dict[str, float]) -> dict[str, dict[str, int]]:
    """Extract the stoichiometry of each component in the composition.

    Args:
        composition: A dictionary mapping oxide formulas to their fractions.

    Returns:
        A dictionary mapping oxide formulas to their stoichiometric dictionaries
        (e.g., {"Al2O3": {"Al": 2, "O": 3}}).

    """
    comp_dict = extract_composition(composition)
    return {oxide: parse_formula(oxide) for oxide in comp_dict}
