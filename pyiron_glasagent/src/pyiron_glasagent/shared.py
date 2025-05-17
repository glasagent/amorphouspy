def get_element_types_dict(atoms_dict: dict):
    atoms = atoms_dict["atoms"]
    elements = sorted(set(atom["element"] for atom in atoms))
    element_to_type = {
        elem: i + 1 for i, elem in enumerate(elements)
    }  # e.g., {'Al':1, 'Ca':2, 'Na':3, 'O':4, 'Si':5}
    return element_to_type
