"""LAMMPS potential generation for oxide glass simulations."""

# Author: Achraf Atila (achraf.atila@bam.de)
import pandas as pd
from pyiron_base import job

from pyiron_glass.shared import get_element_types_dict

# Complete dictionary of Pedone parameters
pedone_potential_params = {
    "Li": {"q": 0.6, "morse": (0.001114, 3.429506, 2.681360), "repulsion": 1.0},
    "Na": {"q": 0.6, "morse": (0.023363, 1.763867, 3.006315), "repulsion": 5.0},
    "K": {"q": 0.6, "morse": (0.011612, 2.062605, 3.305308), "repulsion": 5.0},
    "Be": {"q": 1.2, "morse": (0.239919, 2.527420, 1.815405), "repulsion": 1.0},
    "Mg": {"q": 1.2, "morse": (0.038908, 2.281000, 2.586153), "repulsion": 5.0},
    "Ca": {"q": 1.2, "morse": (0.030211, 2.241334, 2.923245), "repulsion": 5.0},
    "Sr": {"q": 1.2, "morse": (0.019623, 1.886000, 3.328330), "repulsion": 3.0},
    "Ba": {"q": 1.2, "morse": (0.065011, 1.547596, 3.393410), "repulsion": 5.0},
    "Sc": {"q": 1.8, "morse": (0.000333, 3.144445, 3.200000), "repulsion": 2.6},
    "Ti": {"q": 2.4, "morse": (0.024235, 2.254703, 2.708943), "repulsion": 1.0},
    "Zr": {"q": 2.4, "morse": (0.206237, 2.479675, 2.436997), "repulsion": 1.0},
    "Cr": {"q": 1.8, "morse": (0.399561, 1.785079, 2.340810), "repulsion": 1.0},
    "Mn": {"q": 1.2, "morse": (0.029658, 1.997543, 2.852075), "repulsion": 3.0},
    "Fe": {"q": 1.2, "morse": (0.078171, 1.822638, 2.658163), "repulsion": 2.0},
    "Fe3": {"q": 1.8, "morse": (0.418981, 1.620376, 2.382183), "repulsion": 2.0},
    "Co": {"q": 1.2, "morse": (0.012958, 2.361272, 2.756282), "repulsion": 3.0},
    "Ni": {"q": 1.2, "morse": (0.029356, 2.679137, 2.500754), "repulsion": 3.0},
    "Cu": {"q": 0.6, "morse": (0.090720, 3.802168, 2.055405), "repulsion": 1.0},
    "Ag": {"q": 0.6, "morse": (0.088423, 3.439162, 2.265956), "repulsion": 1.0},
    "Zn": {"q": 1.2, "morse": (0.001221, 3.150679, 2.851850), "repulsion": 1.0},
    "Al": {"q": 1.8, "morse": (0.361581, 1.900442, 2.164818), "repulsion": 0.9},
    "Si": {"q": 2.4, "morse": (0.340554, 2.006700, 2.100000), "repulsion": 1.0},
    "Ge": {"q": 2.4, "morse": (0.158118, 2.294230, 2.261313), "repulsion": 5.0},
    "Sn": {"q": 2.4, "morse": (0.079400, 2.156770, 2.633076), "repulsion": 3.0},
    "P": {"q": 3.0, "morse": (0.831326, 2.585833, 1.800790), "repulsion": 1.0},
    "Nd": {"q": 1.8, "morse": (0.014580, 1.825100, 3.398717), "repulsion": 3.0},
    "Gd": {"q": 1.8, "morse": (0.000132, 2.013000, 4.351589), "repulsion": 3.0},
    "Er": {"q": 1.8, "morse": (0.040448, 2.294078, 2.837722), "repulsion": 3.0},
    "O": {"q": -1.2, "morse": (0.042395, 1.379316, 3.618701), "repulsion": 22.0},
}


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


def _generate_pedone_potential(atoms_dict: dict) -> pd.DataFrame:
    """Generate LAMMPS potential configuration for glass simulations.

    Parameters
    ----------
    atoms_dict : dict
        Dictionary containing atomic structure information

    Returns
    -------
    pd.DataFrame
        DataFrame containing potential configuration

    """
    types = get_element_types_dict(atoms_dict)
    species = list(types.keys())

    # Validate that all required X-O pairs exist in Pedone parameters
    missing_pairs = []
    for elem in species:
        if elem == "O":
            continue
        # Each non-oxygen element must have Morse and repulsion parameters defined
        if elem not in pedone_potential_params:
            missing_pairs.append((elem,))
        # Each X-O pair must exist (implicitly true if both elements in dict)
        elif "O" not in pedone_potential_params:
            missing_pairs.append((elem, "O"))

    if missing_pairs:
        error_msg = f"Pedone potential does not include interaction parameters for: {missing_pairs}-O. "
        raise ValueError(error_msg)

    config_lines = [
        "# A. Pedone et.al., JPCB (2006), https://doi.org/10.1021/jp0611018\n",
        "units metal\n",
        "dimension 3\n",
        "atom_style charge\n",
        "\n",
        "# create groups ###\n",
    ]

    config_lines.extend(f"group {elem} type {types[elem]}\n" for elem in species)

    config_lines.append("\n### set charges ###\n")
    for elem in species:
        charge = pedone_potential_params[elem]["q"]
        config_lines.append(f"set type {types[elem]} charge {charge}\n")

    config_lines.extend(
        [
            "\n### Pedone Potential Parameters ###\n",
            "pair_style hybrid/overlay coul/dsf 0.25 8.0 pedone 5.5\n",
            "pair_coeff * * coul/dsf\n",
        ],
    )

    o_type = types.get("O")
    for elem in species:
        if elem == "O":
            i_type = types[elem]
            dij, a, r0 = pedone_potential_params[elem]["morse"]
            cij = pedone_potential_params[elem]["repulsion"]
            config_lines.append(
                f"pair_coeff {i_type} {o_type} pedone {dij} {a} {r0} {cij}\n",
            )

        if elem != "O":
            i_type = types[elem]
            dij, a, r0 = pedone_potential_params[elem]["morse"]
            cij = pedone_potential_params[elem]["repulsion"]
            config_lines.append(
                f"pair_coeff {i_type} {o_type} pedone {dij} {a} {r0} {cij}\n",
            )

    config_lines.append("\npair_modify shift yes\n")

    return pd.DataFrame(
        {
            "Name": ["Pedone"],
            "Filename": [[]],
            "Model": ["Pedone"],
            "Species": [species],
            "Config": [config_lines],
        },
    )


def _generate_bjp_potential(atoms_dict: dict) -> pd.DataFrame:
    """Generate LAMMPS potential configuration for CAS glass simulations (Bouhadja et al. 2013).

    Parameters
    ----------
    atoms_dict : dict
        Dictionary containing atomic structure information

    Returns
    -------
    pd.DataFrame
        DataFrame containing potential configuration

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


@job
def generate_potential(atoms_dict: dict, potential_type: str = "pedone") -> pd.DataFrame:
    """Generate LAMMPS potential configuration for glass simulations.

    Parameters
    ----------
    atoms_dict : dict
        Dictionary containing atomic structure information
    potential_type : str
        Type of potential to generate. Options are "Pedone" or "BJP".

    Returns
    -------
    pd.DataFrame
        DataFrame containing potential configuration

    """
    if potential_type.lower() == "pedone":
        return _generate_pedone_potential(atoms_dict)
    if potential_type.lower() == "bjp":
        return _generate_bjp_potential(atoms_dict)
    msg = f"Unsupported potential type: {potential_type}"
    raise ValueError(msg)
