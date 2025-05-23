# Import atomic masses from ASE, which provides a complete and maintained data source
from ase.data import atomic_masses_iupac2016, chemical_symbols


def get_atomic_mass(element):
    """
    Get the atomic mass of an element.

    Parameters
    ----------
    element : str or int
        Chemical symbol or atomic number

    Returns
    -------
    float
        Atomic mass in g/mol
    """
    if isinstance(element, str):
        # Convert element symbol to atomic number
        atomic_number = chemical_symbols.index(element)
    else:
        atomic_number = element

    return atomic_masses_iupac2016[atomic_number]
