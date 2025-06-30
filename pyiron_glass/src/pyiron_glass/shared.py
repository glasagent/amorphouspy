"""Shared utilities for pyiron_glass package."""


# See issue #31: It could be beneficial to hardcode every element type to be able to always identify the elements
def get_element_types_dict(atoms_dict: dict) -> dict[str, int]:
    """Get a dictionary mapping element symbols to unique integer types.

    Elements are ordered alphabetically and assigned a type starting from 1.
    This is useful for setting up LAMMPS simulations where each element needs
    a unique identifier (not to be confused with the position in the periodic table).

    Args:
        atoms_dict (dict): A dictionary containing atom information, typically with a key "atoms" that is a list
        of atom dictionaries.

    Returns:
        dict: A dictionary mapping element symbols to unique integer types.

    """
    atoms = atoms_dict["atoms"]
    elements = sorted({atom["element"] for atom in atoms})
    return {elem: i + 1 for i, elem in enumerate(elements)}


def count_distribution(coord_numbers: dict[int, int]) -> dict[int, int]:
    """Convert coordination numbers to a histogram distribution.

    Args:
        coord_numbers (Dict[int, int]): Mapping from atom ID to coordination number.

    Returns:
        Dict[int, int]: Coordination number frequency histogram.

    """
    dist = {}
    for cn in coord_numbers.values():
        dist[cn] = dist.get(cn, 0) + 1
    return dist
