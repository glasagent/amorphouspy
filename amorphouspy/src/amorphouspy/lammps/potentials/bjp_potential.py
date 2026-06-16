"""LAMMPS potential generation for oxide glass simulations using BJP (Bouhadja) parameters.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from __future__ import annotations

import pandas as pd

from amorphouspy.atoms.shared import get_element_types_dict
from amorphouspy.lammps.potentials._config import DsfConfig, EwaldConfig, InteractionConfig, PppmConfig, WolfConfig

_DEFAULT_SHORT_RANGE_CUTOFF = 8.0
_DEFAULT_PPPM_EWALD_LONG_RANGE_CUTOFF = 12.0
_DEFAULT_ALPHA = 0.25
_MELT_TEMPERATURE = 4000

# Minimum atoms-per-core for good (~80-90%) MPI scaling efficiency.
MIN_ATOMS_PER_CORE = 2000

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


def supported_elements() -> set[str]:
    """Return the set of elements supported by the BJP potential."""
    return set(bjp_charges)


def _build_bjp_pair_coeff_lines(species: list[str], types: dict) -> list[str]:
    done_pairs: set[tuple[str, str]] = set()
    lines = []
    for elem_i in species:
        for elem_j in species:
            if (elem_i, elem_j) in done_pairs or (elem_j, elem_i) in done_pairs:
                continue
            if (elem_i, elem_j) in bjp_params:
                A, rho, sigma, C, D = bjp_params[(elem_i, elem_j)]
            elif (elem_j, elem_i) in bjp_params:
                A, rho, sigma, C, D = bjp_params[(elem_j, elem_i)]
            else:
                continue
            lines.append(f"pair_coeff {types[elem_i]} {types[elem_j]} {A:.6f} {rho:.6f} {sigma:.6f} {C:.6f} {D:.6f}\n")
            done_pairs.add((elem_i, elem_j))
    return lines


def generate_bjp_potential(
    atoms_dict: dict,
    *,
    melt: bool = False,
    electrostatics: InteractionConfig | None = None,
) -> pd.DataFrame:
    """Generate LAMMPS potential configuration for CAS glass simulations (Bouhadja et al. 2013).

    Args:
        atoms_dict: Structure dict from ``get_structure_dict()``.
        melt: Append a Langevin NVE/limit pre-equilibration block at 4000 K.
        electrostatics: Coulomb solver settings. BJP uses a single Born-Mayer cutoff
            controlled by ``short_range_cutoff`` (default 8 Å). ``long_range_cutoff``
            is ignored; kspace handles long range for PPPM/Ewald.

    Returns:
        Single-row DataFrame with LAMMPS config lines in the ``Config`` column.

    """
    types = get_element_types_dict(atoms_dict["atoms"])
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

    electrostatics_cfg = electrostatics if electrostatics is not None else DsfConfig()
    short_range_cutoff = _DEFAULT_SHORT_RANGE_CUTOFF

    if isinstance(electrostatics_cfg, (DsfConfig, WolfConfig)):
        alpha = electrostatics_cfg.alpha or _DEFAULT_ALPHA
        pair_style_line = f"pair_style born/coul/{electrostatics_cfg.lammps_keyword} {alpha} {short_range_cutoff}\n"
        kspace_line = None
    else:  # pppm or ewald
        if not isinstance(electrostatics_cfg, (PppmConfig, EwaldConfig)):
            msg = f"Unsupported electrostatics config: {type(electrostatics_cfg).__name__}"
            raise TypeError(msg)
        long_range_cutoff = electrostatics_cfg.long_range_cutoff or _DEFAULT_PPPM_EWALD_LONG_RANGE_CUTOFF
        pair_style_line = f"pair_style born/coul/long {long_range_cutoff}\n"
        kspace_line = electrostatics_cfg.kspace_line()

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
            pair_style_line,
        ]
    )

    if kspace_line:
        config_lines.append(kspace_line)

    # Pair coefficients
    config_lines.extend(_build_bjp_pair_coeff_lines(species, types))

    config_lines.append("\npair_modify shift yes\n")

    if melt:
        config_lines.extend(
            [
                f"\nfix langevinnve all langevin {_MELT_TEMPERATURE} {_MELT_TEMPERATURE} 0.01 48279\n",
                "\nfix ensemblenve all nve/limit 0.5\n",
                "\nrun 10000\n",
                "\nunfix langevinnve\n",
                "\nunfix ensemblenve\n",
            ]
        )

    coulomb_label = electrostatics_cfg.lammps_keyword.upper()
    return pd.DataFrame(
        {
            "Name": ["BJP"],
            "Filename": [[]],
            "Model": [f"Born-Mayer-Huggins_coulomb_{coulomb_label}"],
            "Species": [species],
            "Config": [config_lines],
        }
    )
