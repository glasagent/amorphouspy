"""LAMMPS potential generation for oxide glass simulations.

Author: Achraf Atila (achraf.atila@bam.de)
"""

import warnings

import pandas as pd

from amorphouspy.lammps.potentials._config import DsfConfig, EwaldConfig, InteractionConfig, PppmConfig, WolfConfig

from . import bjp_potential as bjp
from . import bmp_potential as bmp
from . import du_teter_potential as du_teter
from . import pmmcs_potential as pmmcs
from . import shik_potential as shik
from . import yang_potential as yang2026

__all__ = ["DsfConfig", "EwaldConfig", "InteractionConfig", "PppmConfig", "WolfConfig"]

# Preference order: pmmcs covers the most elements, shik adds B, bjp is most limited.
POTENTIAL_PREFERENCE = (
    "pmmcs",
    "bmp-harmonic",
    "bmp-screened-harmonic",
    "shik",
    "bjp",
    "du_teter",
    "du_teter_dbx_generalized",
    "yang2026",
)

_POTENTIAL_MODULES = {
    "pmmcs": pmmcs,
    "bjp": bjp,
    "shik": shik,
    "du_teter": du_teter,
    "du_teter_dbx_generalized": du_teter,
    "bmp-harmonic": bmp,
    "bmp-screened-harmonic": bmp,
    "yang2026": yang2026,
}


def get_supported_elements(potential_type: str) -> set[str]:
    """Return the set of non-oxygen elements supported by *potential_type*.

    Args:
        potential_type: One of ``"pmmcs"``, ``"bjp"``, ``"shik"``, ``"du_teter"``, ``"bmp"``, or ``"yang2026"``.

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


# Fallback minimum atoms-per-core used for potentials that do not declare their
# own ``MIN_ATOMS_PER_CORE`` (and for unknown potential names).
DEFAULT_MIN_ATOMS_PER_CORE = 500


def get_min_atoms_per_core(potential_type: str) -> int:
    """Return the minimum atoms-per-core for *potential_type*.

    Each potential module may declare a ``MIN_ATOMS_PER_CORE`` constant
    expressing the smallest number of atoms per MPI core that still keeps it at
    good (~80-90%) scaling efficiency. Potentials without one (and unrecognised
    names) emit a warning and fall back to :data:`DEFAULT_MIN_ATOMS_PER_CORE`.

    Args:
        potential_type: Potential identifier (e.g. ``"pmmcs"``, ``"shik"``).

    Returns:
        The minimum number of atoms per core.

    """
    mod = _POTENTIAL_MODULES.get(potential_type.lower())
    if mod is None:
        warnings.warn(
            f"Unknown potential {potential_type!r}; using default minimum atoms-per-core "
            f"of {DEFAULT_MIN_ATOMS_PER_CORE} for core selection.",
            stacklevel=2,
        )
        return DEFAULT_MIN_ATOMS_PER_CORE
    min_atoms_per_core = getattr(mod, "MIN_ATOMS_PER_CORE", None)
    if min_atoms_per_core is None:
        warnings.warn(
            f"Potential {potential_type!r} does not define MIN_ATOMS_PER_CORE; "
            f"using default of {DEFAULT_MIN_ATOMS_PER_CORE} for core selection.",
            stacklevel=2,
        )
        return DEFAULT_MIN_ATOMS_PER_CORE
    return min_atoms_per_core


def _is_compatible(name: str, elements: set[str]) -> bool:
    """Check whether *elements* are jointly supported by potential *name*.

    If the module exposes an ``is_compatible`` function it is used (this
    allows modules like BMP to enforce multi-element constraints such as
    the boron restriction).  Otherwise falls back to a simple subset
    check against ``supported_elements()``.
    """
    mod = _POTENTIAL_MODULES[name]
    checker = getattr(mod, "is_compatible", None)
    if checker is not None:
        return checker(elements)
    return elements <= mod.supported_elements()


def select_potential(elements: set[str]) -> str | None:
    """Choose the best potential that supports all *elements*.

    Potentials are tried in preference order: pmmcs → bmp → shik → bjp → du_teter → yang2026.
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
        if _is_compatible(name, elements):
            return name
    return None


def compatible_potentials(elements: set[str]) -> list[str]:
    """Return all potentials that support *elements*, in preference order.

    Args:
        elements: Element symbols required by the composition.

    Returns:
        List of potential names (may be empty).

    """
    return [name for name in POTENTIAL_PREFERENCE if _is_compatible(name, elements)]


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
        potential_type: One of ``"pmmcs"``, ``"bjp"``, ``"shik"``, ``"du_teter"``,
            ``"bmp-harmonic"``, ``"bmp-screened-harmonic"``, or ``"yang2026"``.
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
        >>> potential = generate_potential(struct_dict, potential_type="du_teter")
        >>> potential = generate_potential(struct_dict, potential_type="yang2026")

    """
    if potential_type.lower() == "pmmcs":
        return pmmcs.generate_pmmcs_potential(atoms_dict, melt=melt, electrostatics=electrostatics)
    if potential_type.lower() == "bjp":
        return bjp.generate_bjp_potential(atoms_dict, melt=melt, electrostatics=electrostatics)
    if potential_type.lower() == "shik":
        return shik.generate_shik_potential(atoms_dict, melt=melt, electrostatics=electrostatics)
    if potential_type.lower() in ("du_teter", "du_teter_dbx_generalized"):
        n4_model = "dbx_generalized" if potential_type.lower() == "du_teter_dbx_generalized" else "dbx"
        return du_teter.generate_du_teter_potential(atoms_dict, melt=melt, n4_model=n4_model)
    if potential_type.lower() in ("bmp-harmonic", "bmp-screened-harmonic"):
        variant = "harmonic" if potential_type.lower() == "bmp-harmonic" else "screened-harmonic"
        return bmp.generate_bmp_potential(atoms_dict, variant=variant, melt=melt, electrostatics=electrostatics)
    if potential_type.lower() == "yang2026":
        return yang2026.generate_yang2026_potential(atoms_dict, melt=melt, electrostatics=electrostatics)
    msg = f"Unsupported potential type: {potential_type}"
    raise ValueError(msg)
