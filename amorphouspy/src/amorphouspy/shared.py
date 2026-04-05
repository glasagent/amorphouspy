"""Shared utilities for amorphouspy package."""

import numpy as np
from ase.data import chemical_symbols


# See issue #31: It could be beneficial to hardcode
# every element type to be able to always identify the elements
def get_element_types_dict(atoms_dict: dict) -> dict[str, int]:
    """Get a dictionary mapping element symbols to unique integer types.

    Elements are ordered alphabetically and assigned a type starting from 1.
    This is useful for setting up LAMMPS simulations where each element needs
    a unique identifier (not to be confused with the position in the periodic table).

    Args:
        atoms_dict: A dictionary containing atom information,
            typically with a key "atoms" that is a list of atom dictionaries.

    Returns:
        A dictionary mapping element symbols to unique integer types.

    Example:
        >>> types = get_element_types_dict(struct_dict)

    """
    atoms = atoms_dict["atoms"]
    elements = sorted({atom["element"] for atom in atoms})
    return {elem: i + 1 for i, elem in enumerate(elements)}


def count_distribution(coord_numbers: dict[int, int]) -> dict[int, int]:
    """Convert coordination numbers to a histogram distribution.

    Args:
        coord_numbers: Mapping from atom ID to coordination number.

    Returns:
        Coordination number frequency histogram.

    Example:
        >>> dist = count_distribution({1: 4, 2: 4, 3: 3})

    """
    dist = {}
    for cn in coord_numbers.values():
        dist[cn] = dist.get(cn, 0) + 1
    return dist


def type_to_dict(types: np.ndarray) -> dict[int, str]:
    """Generate a dictionary mapping atomic numbers (types) to element symbols from an ASE Atoms structure.

    Args:
        types: Array containing atom types in the simulation.

    Returns:
        Dictionary mapping atomic numbers to corresponding element symbols.

    Example:
        >>> type_map = type_to_dict(np.array([14, 8]))

    """
    # Extract unique atomic types from structure
    unique_types = np.unique(types)

    # Map atomic numbers to element symbols using ASE's periodic table
    element_symbols: list[str] = [chemical_symbols[z] for z in unique_types]

    # Create the type-to-symbol dictionary
    type_dict: dict[int, str] = dict(zip(unique_types, element_symbols, strict=True))

    return type_dict


def running_mean(data: list | np.ndarray, N: int) -> np.ndarray:
    """Calculate running mean of an array-like dataset.

    The initial and final values of the returned array are NaN, as the running mean is not defined
    for those points.

    Args:
        data: Input data for which the running mean should be calculated.
        N: Width of the averaging window.

    Returns:
        Array of same size as input data containing the running mean values.

    """
    data = np.asarray(data)
    if N == 1:
        return data
    retArray = np.zeros(data.size) * np.nan
    padL = int(N / 2)
    padR = N - padL - 1
    retArray[padL:-padR] = np.convolve(data, np.ones((N,)) / N, mode="valid")
    return retArray
