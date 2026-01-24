"""LAMMPS potential generation for oxide glass simulations using SHIK parameters."""

# Author: Achraf Atila (achraf.atila@bam.de)
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pyiron_glass.shared import get_element_types_dict

# ================================================================
# SHIK Parameters (Pedone-like Buckingham + r^-24)
# ================================================================
shik_charges = {
    "Li": 0.5727,
    "Na": 0.6018,
    "K": 0.6849,
    "Mg": 1.0850,
    "Ca": 1.4977,
    "B": 1.6126,
    "Al": 1.6334,
    "Si": 1.7755,
    "O": -0.0000,  # to be computed for charge neutrality
}

# Parameter format: (A [eV], B [Å^-1], C [eV·Å^6], D [eV·Å^24])
shik_params = {
    ("O", "O"): (1120.5, 2.8927, 26.132, 16800),
    ("O", "Si"): (23108, 5.0979, 139.70, 66.0),
    ("Si", "Si"): (2798.0, 4.4073, 0.0, 3423204),
    ("O", "Mg"): (139373, 6.0395, 79.562, 16800),
    ("Si", "Mg"): (516227, 5.3958, 0.0, 16800),
    ("Mg", "Mg"): (19669, 4.0000, 0.0, 16800),
    ("O", "Ca"): (146905, 5.6094, 45.073, 16800),
    ("Si", "Ca"): (77366, 5.0770, 0.0, 16800),
    ("Ca", "Ca"): (21633, 3.2562, 0.0, 16800),
    ("O", "B"): (16182, 5.6069, 59.203, 32.0),
    ("B", "B"): (1805.5, 3.8228, 69.174, 6000.0),
    ("Li", "B"): (4148.6, 3.5726, 102.36, 16800),
    ("Na", "B"): (3148.5, 3.6183, 34.000, 16800),
    ("K", "B"): (1548.6, 2.7283, 201.36, 16800),
    ("B", "Si"): (4798.0, 3.6703, 207.0, 16800),
    ("Mg", "B"): (5000.0, 4.0533, 0.736, 16800),
    ("Ca", "B"): (848.55, 5.9826, 81.355, 16800),
    ("O", "Li"): (6745.2, 4.9120, 41.221, 70.0),
    ("Si", "Li"): (17284, 4.3848, 0.0, 16800),
    ("Li", "Li"): (2323.8, 3.9129, 0.0, 3240),
    ("O", "Na"): (1127566, 6.8986, 40.562, 16800),
    ("Si", "Na"): (495653, 5.4151, 0.0, 16800),
    ("Na", "Na"): (1476.9, 3.4075, 0.0, 16800),
    ("O", "K"): (258160, 5.1698, 130.77, 16800),
    ("Si", "K"): (268967, 4.3289, 0.0, 16800),
    ("K", "K"): (3648.0, 4.4207, 0.0, 16800),
    ("O", "Al"): (21740, 5.3054, 65.815, 66.0),
    ("Al", "Al"): (1799.1, 3.6778, 100.0, 16800),
}


# ================================================================
# Core Buckingham + r^-24 potential functions
# ================================================================
def potential_and_force(
    r: np.ndarray, A: float, B: float, C: float, D: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute V(r) and F(r) for the SHIK (Buckingham + r^-24) potential."""
    exp_term = np.exp(-B * r)
    inv_r6 = r**-6
    inv_r24 = inv_r6**4

    V = A * exp_term - C * inv_r6 + D * inv_r24
    F = A * B * exp_term - 6 * C * inv_r6 / r + 24 * D * inv_r24 / r
    return r, V, F


def write_table_file(
    pair: str, params: dict, rmin: float = 0.1, rmax: float = 10.5, npoints: int = 50000, output_dir: str = "."
) -> Path:
    """Write a LAMMPS table file for a given atomic pair in the specified output directory."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    A, B, C, D = params
    r_squared = np.linspace(rmin**2, rmax**2, npoints, dtype=np.float64)
    rs = np.sqrt(r_squared)
    data = np.array([potential_and_force(r, A, B, C, D) for r in rs])

    filename = output_dir / f"table_{pair.replace('-', '_')}.tbl"
    with filename.open("w") as f:
        f.write(f"# LAMMPS potential table for {pair}\n")
        f.write("SHIK_Buck_r24\n")
        f.write(f"N {npoints} RSQ {rmin:.4f} {rmax:.4f}\n\n")
        f.writelines(f"{i} {r:.12e} {V:.12e} {F:.12e}\n" for i, (r, V, F) in enumerate(data, 1))

    return filename


def compute_oxygen_charge(atoms_dict: dict, shik_charges: dict) -> float:
    """Compute composition-dependent oxygen charge.

    qO = -sum(qx * Nx) / NO.

    Args:
        atoms_dict: dictionary of atoms and counts
        shik_charges: dictionary of atomic charges from the potential model

    Returns:
        q_O: composition dependent oxygen charge

    """
    counts = Counter(atom["element"] for atom in atoms_dict["atoms"])
    n_O = counts.get("O", 0)
    if n_O == 0:
        msg = "No oxygen atoms found in structure."
        raise ValueError(msg)

    numerator = sum(shik_charges[elem] * n for elem, n in counts.items() if elem != "O")
    return -numerator / n_O


# ================================================================
# LAMMPS Configuration File Generator
# ================================================================


def generate_shik_potential(atoms_dict: dict, output_dir: str = ".") -> pd.DataFrame:
    """Generate SHIK LAMMPS input configuration with absolute table paths."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    types = get_element_types_dict(atoms_dict)
    species = list(types.keys())

    q_O = compute_oxygen_charge(atoms_dict, shik_charges)
    shik_charges["O"] = q_O

    # --- Validate only X-O pairs ---
    missing_pairs = []
    for elem in species:
        if elem == "O":
            continue
        if (elem, "O") not in shik_params and ("O", elem) not in shik_params:
            missing_pairs.append((elem, "O"))

    if missing_pairs:
        msg = f"Missing SHIK parameters for X-O pairs: {missing_pairs}"
        raise ValueError(msg)

    # --- Build LAMMPS configuration ---
    lines = [
        "# S. Sundararaman et al.,\n"
        "# for silica glass, J. Chem. Phys. 2018, 148, 19, https://doi.org/10.1063/1.5023707\n",
        "# for alkali and alkaline-earth aluminosilicate glasses, J. Chem. Phys. 2019, 150, 15, https://doi.org/10.1063/1.5079663\n",
        "# for borate glasses with mixed network formers, J. Chem. Phys. 2020, 152, 10, https://doi.org/10.1063/1.5142605\n",
        "# for alkaline earth silicate and borate glasses, Shih et al. J. Non-Cryst. Sol. 2021, 565, 120853, https://doi.org/10.1016/j.jnoncrysol.2021.120853\n",
        "\n",
        "units metal\n",
        "dimension 3\n",
        "atom_style charge\n\n",
    ]

    lines.append("\n### Group Definitions ###\n")
    lines.extend([f"group {elem} type {types[elem]}\n" for elem in species])

    lines.append("\n### Charges ###\n")
    lines.extend([f"set type {types[elem]} charge {shik_charges[elem]}\n" for elem in species])

    lines.append("\n### SHIK Potential ###\n")
    lines.append("pair_style hybrid/overlay coul/dsf 0.2 10.0 table spline 10000\n")
    lines.append("pair_coeff * * coul/dsf\n")
    rvdw = 10.0  # cutoff for the SHIK potential

    # --- Generate tables with absolute paths ---
    for i, elem_i in enumerate(species):
        for j, elem_j in enumerate(species):
            if j < i:
                continue
            if (elem_i, elem_j) in shik_params:
                params = shik_params[(elem_i, elem_j)]
            elif (elem_j, elem_i) in shik_params:
                params = shik_params[(elem_j, elem_i)]
            else:
                continue  # skip undefined pairs

            pair_name = f"{elem_i}-{elem_j}"
            filename = write_table_file(pair_name, params, output_dir=output_dir)
            abs_path = Path(filename).resolve()
            lines.append(f'pair_coeff {types[elem_i]} {types[elem_j]} table "{abs_path}" SHIK_Buck_r24 {rvdw}\n')

    lines.append("\npair_modify shift yes\n\n")

    lines.append("\nthermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n")
    lines.append("\nthermo_modify flush yes\n")
    lines.append("\nthermo 100\n")
    lines.append("\nfix langevin all langevin 5000 5000 0.01 48279\n")
    lines.append("\nfix ensemble all nve/limit 0.5\n")

    lines.append("\nrun 10000\n")
    lines.append("\nunfix langevin\n")
    lines.append("\nunfix ensemble\n")

    return pd.DataFrame(
        {
            "Name": ["SHIK"],
            "Filename": [""],
            "Model": ["SHIK"],
            "Species": [species],
            "Config": [lines],
        }
    )
