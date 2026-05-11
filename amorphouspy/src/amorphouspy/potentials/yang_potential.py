"""LAMMPS potential generation for oxide glass simulations using Yang et al. parameters.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from __future__ import annotations

import pandas as pd

from amorphouspy._utils import get_element_types_dict
from amorphouspy.potentials._config import DsfConfig, EwaldConfig, InteractionConfig, PppmConfig, WolfConfig

_DEFAULT_SHORT_RANGE_CUTOFF = 11.0
_DEFAULT_DSF_WOLF_LONG_RANGE_CUTOFF = 11.0
_DEFAULT_PPPM_EWALD_LONG_RANGE_CUTOFF = 11.0
_DEFAULT_ALPHA = 0.182
_MELT_TEMPERATURE = 4000

# Potential parameters from Table II and III in Yang et al., J. Non-Cryst. Solids 684, 124104 (2026)
# Charges
yang2026_charges = {
    "Ca": 0.945,
    "Na": 0.4725,
    "B": 1.4175,
    "Si": 1.89,
    "O": -0.945,
}
# Format: Aij (eV) rho_ij (A) Cij (eV*A^6)

yang2026_params = {
    ("O", "O"): (9022.79, 0.265, 85.0921),
    ("Si", "O"): (50306.10, 0.161, 46.2978),
    ("B", "O"): (191757.12, 0.1249, 32.5600),
    ("B", "B"): (532.85, 0.3527, 0.0),
    ("Si", "B"): (337.70, 0.29, 0.0),
    ("Na", "O"): (120303.80, 0.17, 0.0),
    ("Ca", "O"): (155667.70, 0.178, 42.2597),
}


def supported_elements() -> set[str]:
    """Return the set of elements supported by the Yang2026 potential."""
    return set(yang2026_charges)


def _build_yang2026_pair_coeff_lines(species: list[str], types: dict, *, hybrid: bool = False) -> list[str]:
    done_pairs: set[tuple[str, str]] = set()
    lines = []
    prefix = "buck " if hybrid else ""
    for elem_i in species:
        for elem_j in species:
            if (elem_i, elem_j) in done_pairs or (elem_j, elem_i) in done_pairs:
                continue
            if (elem_i, elem_j) in yang2026_params:
                A, rho, C = yang2026_params[(elem_i, elem_j)]
            elif (elem_j, elem_i) in yang2026_params:
                A, rho, C = yang2026_params[(elem_j, elem_i)]
            else:
                continue
            lines.append(f"pair_coeff {types[elem_i]} {types[elem_j]} {prefix}{A:.6f} {rho:.6f} {C:.6f}\n")
            done_pairs.add((elem_i, elem_j))
    return lines


def generate_yang2026_potential(
    atoms_dict: dict,
    *,
    melt: bool = False,
    electrostatics: InteractionConfig | None = None,
) -> pd.DataFrame:
    """Generate LAMMPS potential configuration for CAS glass simulations (Yang et al. 2026).

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

    missing_elem = [e for e in species if e not in yang2026_charges]
    if missing_elem:
        error_msg = f"Yang2026 potential does not include parameters for elements: {missing_elem}."
        raise ValueError(error_msg)

    electrostatics_cfg = electrostatics if electrostatics is not None else DsfConfig()
    src = _DEFAULT_SHORT_RANGE_CUTOFF

    if isinstance(electrostatics_cfg, (DsfConfig, WolfConfig)):
        alpha = electrostatics_cfg.alpha or _DEFAULT_ALPHA
        long_range_cutoff = electrostatics_cfg.long_range_cutoff or _DEFAULT_DSF_WOLF_LONG_RANGE_CUTOFF
        coul_kw = electrostatics_cfg.lammps_keyword
        pair_style_line = f"pair_style hybrid/overlay coul/{coul_kw} {alpha} {long_range_cutoff} buck {src}\n"
        coul_coeff_line = f"pair_coeff * * coul/{coul_kw}\n"
        kspace_line = None
    else:  # pppm or ewald
        if not isinstance(electrostatics_cfg, (PppmConfig, EwaldConfig)):
            msg = f"Unsupported electrostatics config: {type(electrostatics_cfg).__name__}"
            raise TypeError(msg)
        long_range_cutoff = electrostatics_cfg.long_range_cutoff or _DEFAULT_PPPM_EWALD_LONG_RANGE_CUTOFF
        pair_style_line = f"pair_style buck/coul/long {long_range_cutoff}\n"
        coul_coeff_line = None
        kspace_line = electrostatics_cfg.kspace_line()

    config_lines = [
        "# Yang et al., J. Non-Cryst. Solids 684, 124104 (2026) \n",
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
        q = yang2026_charges[elem]
        config_lines.append(f"set type {types[elem]} charge {q}\n")

    # Pair style
    config_lines.extend(
        [
            "\n### Yang et al. 2026 + Coulomb Potential Parameters ###\n",
            pair_style_line,
        ]
    )

    if coul_coeff_line:
        config_lines.append(coul_coeff_line)

    if kspace_line:
        config_lines.append(kspace_line)

    # Pair coefficients
    config_lines.extend(_build_yang2026_pair_coeff_lines(species, types, hybrid=coul_coeff_line is not None))

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
            "Name": ["Yang2026"],
            "Filename": [[]],
            "Model": [f"Born-Mayer-Huggins_coulomb_{coulomb_label}"],
            "Species": [species],
            "Config": [config_lines],
        }
    )
