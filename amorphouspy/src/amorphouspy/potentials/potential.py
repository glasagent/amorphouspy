"""LAMMPS potential generation for oxide glass simulations."""

# Author: Achraf Atila (achraf.atila@bam.de)
import pandas as pd

from . import bjp_potential as bjp
from . import pmmcs_potential as pmmcs
from . import shik_potential as shik


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
