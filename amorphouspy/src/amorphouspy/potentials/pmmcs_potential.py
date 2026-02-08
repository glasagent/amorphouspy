"""LAMMPS potential generation for oxide glass simulations."""

# Author: Achraf Atila (achraf.atila@bam.de)
import pandas as pd

from amorphouspy.shared import get_element_types_dict

# Complete dictionary of Pmmcs parameters
pmmcs_potential_params = {
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


def generate_pmmcs_potential(atoms_dict: dict) -> pd.DataFrame:
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

    # Validate that all required X-O pairs exist in PMMCS parameters
    missing_pairs = []
    for elem in species:
        if elem == "O":
            continue
        # Each non-oxygen element must have Morse and repulsion parameters defined
        if elem not in pmmcs_potential_params:
            missing_pairs.append((elem,))
        # Each X-O pair must exist (implicitly true if both elements in dict)
        elif "O" not in pmmcs_potential_params:
            missing_pairs.append((elem, "O"))

    if missing_pairs:
        error_msg = f"Pmmcs potential does not include interaction parameters for: {missing_pairs}-O. "
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
        charge = pmmcs_potential_params[elem]["q"]
        config_lines.append(f"set type {types[elem]} charge {charge}\n")

    config_lines.extend(
        [
            "\n### Pmmcs Potential Parameters ###\n",
            "pair_style hybrid/overlay coul/dsf 0.25 8.0 pedone 5.5\n",
            "pair_coeff * * coul/dsf\n",
        ],
    )

    o_type = types.get("O")
    for elem in species:
        if elem == "O":
            i_type = types[elem]
            dij, a, r0 = pmmcs_potential_params[elem]["morse"]
            cij = pmmcs_potential_params[elem]["repulsion"]
            config_lines.append(
                f"pair_coeff {i_type} {o_type} pedone {dij} {a} {r0} {cij}\n",
            )

        if elem != "O":
            i_type = types[elem]
            dij, a, r0 = pmmcs_potential_params[elem]["morse"]
            cij = pmmcs_potential_params[elem]["repulsion"]
            config_lines.append(
                f"pair_coeff {i_type} {o_type} pedone {dij} {a} {r0} {cij}\n",
            )

    config_lines.append("\npair_modify shift yes\n")

    return pd.DataFrame(
        {
            "Name": ["PMMCS"],
            "Filename": [[]],
            "Model": ["PMMCS"],
            "Species": [species],
            "Config": [config_lines],
        },
    )
