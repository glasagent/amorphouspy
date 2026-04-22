"""LAMMPS potential generation for oxide glass simulations.

Author: Achraf Atila (achraf.atila@bam.de)
"""

import pandas as pd

from . import bjp_potential as bjp
from . import bmp_potential as bmp
from . import pmmcs_potential as pmmcs
from . import shik_potential as shik
from ._config import DsfConfig, EwaldConfig, InteractionConfig, PppmConfig, WolfConfig

__all__ = ["DsfConfig", "EwaldConfig", "InteractionConfig", "PppmConfig", "WolfConfig"]

# Preference order: pmmcs covers the most elements, shik adds B, bjp is most limited.
POTENTIAL_PREFERENCE = ("pmmcs", "bmp-harmonic", "bmp-screened-harmonic", "shik", "bjp")

_POTENTIAL_MODULES = {
    "pmmcs": pmmcs,
    "bjp": bjp,
    "shik": shik,
    "bmp-harmonic": bmp,
    "bmp-screened-harmonic": bmp,
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

    Potentials are tried in preference order: pmmcs → bmp → shik → bjp.
    When boron (B) is present, ``bmp-screened-harmonic`` is preferred over ``bmp-harmonic``
    because its three-body term explicitly covers B-O-B and B-O-Si interactions.

    Args:
        elements: Element symbols required by the composition.

    Returns:
        The name of the best compatible potential, or ``None`` if none can
        handle the full element set.

    """
    preference = (
        ("pmmcs", "bmp-screened-harmonic", "bmp-harmonic", "shik", "bjp") if "B" in elements else POTENTIAL_PREFERENCE
    )
    for name in preference:
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


def generate_potential(
    atoms_dict: dict,
    potential_type: str = "pmmcs",
    *,
    melt: bool = False,
    electrostatics: InteractionConfig | None = None,
) -> pd.DataFrame:
    """Generate LAMMPS potential configuration for glass simulations.

    Args:
        atoms_dict: Structure dict from ``get_structure_dict()``.
        potential_type: One of ``"pmmcs"``, ``"bjp"``, ``"shik"``,
            ``"bmp-harmonic"``, or ``"bmp-screened-harmonic"``.
        melt: Append a Langevin NVE/limit pre-equilibration block at 4000 K (all potentials).
        electrostatics: Coulomb solver settings. Defaults to DSF with each
            potential's built-in cutoffs and damping parameter.

    Returns:
        DataFrame with columns Name, Filename, Model, Species, Config.

    Example:
        >>> potential = generate_potential(struct_dict, potential_type="shik")
        >>> potential = generate_potential(struct_dict, potential_type="shik", melt=False)
        >>> potential = generate_potential(struct_dict, potential_type="bmp-harmonic")
        >>> potential = generate_potential(struct_dict, potential_type="bmp-screened-harmonic", melt=False)

    """
    if potential_type.lower() == "pmmcs":
        return pmmcs.generate_pmmcs_potential(atoms_dict, melt=melt, electrostatics=electrostatics)
    if potential_type.lower() == "bjp":
        return bjp.generate_bjp_potential(atoms_dict, melt=melt, electrostatics=electrostatics)
    if potential_type.lower() == "shik":
        return shik.generate_shik_potential(atoms_dict, melt=melt, electrostatics=electrostatics)
    if potential_type.lower() in ("bmp-harmonic", "bmp-screened-harmonic"):
        variant = "harmonic" if potential_type.lower() == "bmp-harmonic" else "screened-harmonic"
        return bmp.generate_bmp_potential(atoms_dict, variant=variant, melt=melt, electrostatics=electrostatics)
    msg = f"Unsupported potential type: {potential_type}"
    raise ValueError(msg)
