"""LAMMPS potential generation for oxide glass simulations."""

# Author: Achraf Atila (achraf.atila@bam.de)
import pandas as pd

from . import bjp_potential as bjp
from . import pmmcs_potential as pmmcs
from . import shik_potential as shik

# Preference order: pmmcs covers the most elements, shik adds B, bjp is most limited.
POTENTIAL_PREFERENCE = ("pmmcs", "shik", "bjp")

_POTENTIAL_MODULES = {
    "pmmcs": pmmcs,
    "bjp": bjp,
    "shik": shik,
}


def get_supported_elements(potential_type: str) -> set[str]:
    """Return the set of non-oxygen elements supported by *potential_type*.

    Args:
        potential_type: One of ``"pmmcs"``, ``"bjp"``, or ``"shik"``.

    Returns:
        Set of element symbols.

    Raises:
        ValueError: If *potential_type* is not recognised.

    """
    mod = _POTENTIAL_MODULES.get(potential_type.lower())
    if mod is None:
        msg = f"Unsupported potential type: {potential_type}"
        raise ValueError(msg)
    return mod.supported_elements()


def select_potential(elements: set[str]) -> str | None:
    """Choose the best potential that supports all *elements*.

    Potentials are tried in preference order: pmmcs → shik → bjp.

    Args:
        elements: Element symbols required by the composition.

    Returns:
        The name of the best compatible potential, or ``None`` if none can
        handle the full element set.

    """
    for name in POTENTIAL_PREFERENCE:
        if elements <= _POTENTIAL_MODULES[name].supported_elements():
            return name
    return None


def compatible_potentials(elements: set[str]) -> list[str]:
    """Return all potentials that support *elements*, in preference order.

    Args:
        elements: Element symbols required by the composition.

    Returns:
        List of potential names (may be empty).

    """
    return [name for name in POTENTIAL_PREFERENCE if elements <= _POTENTIAL_MODULES[name].supported_elements()]


def generate_potential(atoms_dict: dict, potential_type: str = "pmmcs", *, melt: bool = True) -> pd.DataFrame:
    """Generate LAMMPS potential configuration for glass simulations.

    Args:
        atoms_dict: Dictionary containing atomic structure information.
        potential_type: Type of potential to generate. Options are "pmmcs", "bjp", or "shik".
            (default is "pmmcs").
        melt: Append a Langevin + NVE/limit melt run block (10000 steps). Only used when
            ``potential_type="shik"``.

    Returns:
        DataFrame containing potential configuration.

    Example:
        >>> potential = generate_potential(struct_dict, potential_type="shik")
        >>> potential = generate_potential(struct_dict, potential_type="shik", melt=False)

    """
    if potential_type.lower() == "pmmcs":
        return pmmcs.generate_pmmcs_potential(atoms_dict)
    if potential_type.lower() == "bjp":
        return bjp.generate_bjp_potential(atoms_dict)
    if potential_type.lower() == "shik":
        return shik.generate_shik_potential(atoms_dict, melt=melt)
    msg = f"Unsupported potential type: {potential_type}"
    raise ValueError(msg)
