# See issue #31: It could be beneficial to hardcode every element type to be able to always identify the elements
def get_element_types_dict(atoms_dict: dict):
    """
    Get a dictionary mapping element symbols to unique integer types (not to be confused with the position in
    the periodic table) to be used in the designated LAMMPS simulation. Elements are ordered alphabetically and assigned
    a type starting from 1. This is useful for setting up simulations where each element needs a unique identifier.
    This function assumes that the input dictionary contains a key "atoms" which is a list of dictionaries,
    each representing an atom with an "element" key indicating the element symbol.
    Args:
        atoms_dict (dict): A dictionary containing atom information, typically with a key "atoms" that is a list
        of atom dictionaries.
    Returns:
        dict: A dictionary mapping element symbols to unique integer types.
    """
    atoms = atoms_dict["atoms"]
    elements = sorted(set(atom["element"] for atom in atoms))
    element_to_type = {elem: i + 1 for i, elem in enumerate(elements)}  # e.g., {'Al':1, 'Ca':2, 'Na':3, 'O':4, 'Si':5}
    return element_to_type