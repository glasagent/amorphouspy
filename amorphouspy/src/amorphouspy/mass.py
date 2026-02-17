"""Utilities for atomic mass calculations."""

# Import atomic masses from ASE, which provides a complete and maintained data source
from ase.data import atomic_masses_iupac2016, chemical_symbols


def get_atomic_mass(element: str | int) -> float:
    """Get the atomic mass of an element.

    Args:
        element: Chemical symbol or atomic number.

    Returns:
        Atomic mass in g/mol.

    Example:
        >>> mass = get_atomic_mass("Si")
        >>> print(mass)
        28.085

    """
    atomic_number = chemical_symbols.index(element) if isinstance(element, str) else element
    return atomic_masses_iupac2016[atomic_number]
