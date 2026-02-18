"""LAMMPS potential generation for oxide glass simulations."""

# Author: Achraf Atila (achraf.atila@bam.de)
import pandas as pd

from amorphouspy.shared import get_element_types_dict

# Born-Mayer-Huggins parameters from Table I in Bouhadja et al., J. Chem. Phys. 138, 224510 (2013)
# Charges
bjp_charges = {
    "Ca": 1.2,
    "Al": 1.8,
    "Si": 2.4,
    "O": -1.2,
}
# Format: (A [eV], rho [A], sigma [A], C [eV.A^6]), D [eV.A^8]
bjp_params = {
    ("Ca", "Ca"): (0.0035, 0.0800, 2.3440, 20.9856, 0.000),
    ("Ca", "Al"): (0.0032, 0.0740, 1.9572, 17.1710, 0.000),
    ("Ca", "O"): (0.0077, 0.1780, 2.9935, 42.2556, 0.000),
    ("Al", "Al"): (0.0029, 0.0680, 1.5704, 14.0498, 0.000),
    ("Al", "O"): (0.0075, 0.1640, 2.6067, 34.5747, 0.000),
    ("O", "O"): (0.0120, 0.2630, 3.6430, 85.0840, 0.000),
    ("Ca", "Si"): (0.0027, 0.0630, 1.8924, 22.9907, 0.000),
    ("Si", "O"): (0.0070, 0.1560, 2.5419, 46.2930, 0.000),
    ("Al", "Si"): (0.0025, 0.0570, 1.5056, 18.8116, 0.000),
    ("Si", "Si"): (0.0012, 0.0460, 1.4408, 25.1873, 0.000),
}


def generate_bjp_potential(atoms_dict: dict) -> pd.DataFrame:
    """Generate LAMMPS potential configuration for CAS glass simulations (Bouhadja et al. 2013).

    Args:
        atoms_dict: Dictionary containing atomic structure information.

    Returns:
        DataFrame containing potential configuration.

    """
    types = get_element_types_dict(atoms_dict)
    species = list(types.keys())

    # Validate that all pairs exist in BJP parameters
    missing_pairs = [
        (i, j)
        for idx_i, i in enumerate(species)
        for idx_j, j in enumerate(species)
        if idx_j >= idx_i and (i, j) not in bjp_params and (j, i) not in bjp_params
    ]

    if missing_pairs:
        error_msg = f"BJP potential does not include interaction parameters for: {missing_pairs}."
        raise ValueError(error_msg)

    config_lines = [
        "# Bouhadja et al., J. Chem. Phys. 138, 224510 (2013) \n",
        "units metal\n",
        "dimension 3\n",
        "atom_style charge\n",
        "\n",
        "# create groups ###\n",
    ]
    config_lines.extend(f"group {elem} type {types[elem]}\n" for elem in species)

    # Charges
    config_lines.append("\n### set charges ###\n")
    for elem in species:
        q = bjp_charges[elem]
        config_lines.append(f"set type {types[elem]} charge {q}\n")

    # Pair style
    config_lines.extend(
        [
            "\n### Bouhadja Born-Mayer-Huggins + Coulomb Potential Parameters ###\n",
            "pair_style born/coul/dsf 0.25 8.0\n",
        ]
    )

    # Pair coefficients
    done_pairs = set()
    for _i, elem_i in enumerate(species):
        for _j, elem_j in enumerate(species):
            if (elem_i, elem_j) in done_pairs or (elem_j, elem_i) in done_pairs:
                continue
            if (elem_i, elem_j) in bjp_params:
                A, rho, sigma, C, D = bjp_params[(elem_i, elem_j)]
            elif (elem_j, elem_i) in bjp_params:
                A, rho, sigma, C, D = bjp_params[(elem_j, elem_i)]
            else:
                continue  # skip pairs not in table
            type_i = types[elem_i]
            type_j = types[elem_j]
            config_lines.append(f"pair_coeff {type_i} {type_j} {A:.6f} {rho:.6f} {sigma:.6f} {C:.6f} {D:.6f}\n")
            done_pairs.add((elem_i, elem_j))

    config_lines.append("\npair_modify shift yes\n")

    return pd.DataFrame(
        {
            "Name": ["BJP"],
            "Filename": [[]],
            "Model": ["Born-Mayer-Huggins_coulomb_DSF"],
            "Species": [species],
            "Config": [config_lines],
        }
    )
