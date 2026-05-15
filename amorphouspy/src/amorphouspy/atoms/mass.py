"""Utilities for atomic mass calculations."""

# Import atomic masses from ASE, which provides a complete and maintained data source
from ase.data import atomic_masses_iupac2016, chemical_symbols


def get_atomic_mass(element: str | int) -> float:
    """Get the atomic mass of an element.

    Args:
        element: Chemical symbol or atomic number. Multi-valence BMP labels like
            "Fe3" or "Ce4" are resolved to their base symbol ("Fe", "Ce").

    Returns:
        Atomic mass in g/mol.

    Example:
        >>> mass = get_atomic_mass("Si")
        >>> print(mass)
        28.085

    """
    if isinstance(element, str):
        symbol = element.rstrip("0123456789") or element
        atomic_number = chemical_symbols.index(symbol)
    else:
        atomic_number = element
    return atomic_masses_iupac2016[atomic_number]
