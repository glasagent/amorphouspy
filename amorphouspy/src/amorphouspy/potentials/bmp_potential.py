"""LAMMPS potential generation for oxide glass simulations using BMP-harm and BMP-shrm parameters.

Author: Achraf Atila (achraf.atila@bam.de)

References:
    Bertani, M., Menziani, M. C., & Pedone, A. (2021). Improved empirical force field for multicomponent
    oxide glasses and crystals. Physical Review Materials, 5(4), 045602.
    https://doi.org/10.1103/physrevmaterials.5.045602

    Malavasi, G., & Pedone, A. (2022). The effect of the incorporation of catalase mimetic activity cations
    on the structural, thermal and chemical durability properties of the 45S5 Bioglass®. Acta Materialia,
    229, 117801. https://doi.org/10.1016/j.actamat.2022.117801

    Bertani, M., Pallini, A., Cocchi, M., Menziani, M. C., & Pedone, A. (2022). A new self-consistent
    empirical potential model for multicomponent borate and borosilicate glasses. Journal of the American
    Ceramic Society, 105(12), 7254-7271. https://doi.org/10.1111/jace.18681

    Bertani, M., Pallini, A., Lodesani, F., Cocchi, M., Menziani, M. C., & Pedone, A. (2023). Erratum for
    a new self-consistent empirical potential model for multicomponent borate and borosilicate glasses.
    Journal of the American Ceramic Society, 106(8), 5104-5105. https://doi.org/10.1111/jace.19158
"""

from __future__ import annotations

import itertools
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from amorphouspy.potentials._config import DsfConfig, EwaldConfig, InteractionConfig, PppmConfig, WolfConfig
from amorphouspy.shared import get_element_types_dict

_DEFAULT_SHORT_RANGE_CUTOFF = 7.0
_DEFAULT_DSF_WOLF_LONG_RANGE_CUTOFF = 8.0
_DEFAULT_PPPM_EWALD_LONG_RANGE_CUTOFF = 12.0
_DEFAULT_ALPHA = 0.25
_MELT_TEMPERATURE = 4000

# Elements active in each three-body style; all others get NULL in pair_coeff
_HARMONIC_3BODY_ELEMENTS = {"Si", "O", "P"}
_SCREENED_HARMONIC_3BODY_ELEMENTS = {"Si", "O", "P", "B", "V"}

# Alkali and alkaline-earth sets for Boron D computation (Dell-Bray model)
_ALKALI = {"Li", "Na", "K"}
_ALKALINE_EARTH = {"Mg", "Ca", "Sr", "Ba"}

# Elements allowed when Boron is present (Dell-Bray D model validity domain)
_BMP_BORON_ALLOWED_ELEMENTS: frozenset[str] = frozenset({"B", "Si", "O"} | _ALKALI | _ALKALINE_EARTH)


# Complete dictionary of BMP-Morse parameters
bmp_morse_potential_params: dict[str, Any] = {
    "B": {"q": 1.8, "morse": (0.000000, 2.643330, 1.436670), "repulsion": 0.0},  # Dij computed from composition
    "Li": {"q": 0.6, "morse": (0.041556, 1.758181, 2.551360), "repulsion": 1.0},
    "Na": {"q": 0.6, "morse": (0.023363, 1.763867, 3.006315), "repulsion": 5.0},
    "K": {"q": 0.6, "morse": (0.016098, 2.067900, 3.180030), "repulsion": 5.0},
    "Be": {"q": 1.2, "morse": (0.239919, 2.527420, 1.815405), "repulsion": 1.0},
    "Mg": {"q": 1.2, "morse": (0.010000, 2.554310, 2.610518), "repulsion": 5.0},
    "Ca": {"q": 1.2, "morse": (0.030211, 2.241334, 2.923245), "repulsion": 5.0},
    "Sr": {"q": 1.2, "morse": (0.019623, 1.886000, 3.328330), "repulsion": 3.0},
    "Ba": {"q": 1.2, "morse": (0.065011, 1.547596, 3.393410), "repulsion": 5.0},
    "Sc": {"q": 1.8, "morse": (0.000333, 3.144445, 3.200000), "repulsion": 2.6},
    "Ti": {"q": 2.4, "morse": (0.024235, 2.254703, 2.708943), "repulsion": 1.0},
    "Zr": {"q": 2.4, "morse": (0.206237, 2.479675, 2.436997), "repulsion": 1.0},
    "Cr": {"q": 1.8, "morse": (0.399561, 1.785079, 2.340810), "repulsion": 1.0},
    "Mn": {"q": 1.2, "morse": (0.029658, 1.997543, 2.852075), "repulsion": 2.0},
    "Mn3": {"q": 1.8, "morse": (0.207600, 1.811907, 2.498827), "repulsion": 1.0},
    "Fe": {"q": 1.2, "morse": (0.078171, 1.822638, 2.608163), "repulsion": 2.0},
    "Fe3": {"q": 1.8, "morse": (0.348991, 1.920376, 2.202183), "repulsion": 1.0},
    "Ce3": {"q": 1.8, "morse": (0.198658, 1.599900, 2.891320), "repulsion": 2.0},
    "Ce4": {"q": 2.4, "morse": (0.115196, 2.144000, 2.723380), "repulsion": 2.0},
    "Cu2": {"q": 1.2, "morse": (0.011856, 1.643080, 3.065264), "repulsion": 3.0},
    "V5": {"q": 3.0, "morse": (0.021911, 1.495955, 3.398507), "repulsion": 1.0},
    "V4": {"q": 2.4, "morse": (0.032832, 2.109308, 2.663618), "repulsion": 1.0},
    "Co": {"q": 1.2, "morse": (0.012958, 2.361272, 2.756282), "repulsion": 2.0},
    "Ni": {"q": 1.2, "morse": (0.029356, 2.679137, 2.500754), "repulsion": 3.0},
    "Cu": {"q": 0.6, "morse": (0.090720, 3.802168, 2.055405), "repulsion": 1.0},
    "Ag": {"q": 0.6, "morse": (0.088423, 3.439162, 2.265956), "repulsion": 1.0},
    "Zn": {"q": 1.2, "morse": (0.001221, 3.150679, 2.851850), "repulsion": 1.0},
    "Al": {"q": 1.8, "morse": (0.361581, 1.900442, 2.164818), "repulsion": 0.9},
    "Si": {"q": 2.4, "morse": (0.340554, 2.006700, 2.100000), "repulsion": 1.0},
    "Ge": {"q": 2.4, "morse": (0.158118, 2.294230, 2.261313), "repulsion": 5.0},
    "Sn": {"q": 2.4, "morse": (0.079400, 2.156770, 2.633076), "repulsion": 3.0},
    "P": {"q": 3.0, "morse": (0.831326, 2.585833, 1.790790), "repulsion": 1.0},
    "Nd": {"q": 1.8, "morse": (0.014580, 1.825100, 3.398717), "repulsion": 3.0},
    "Gd": {"q": 1.8, "morse": (0.000132, 2.013000, 4.351589), "repulsion": 3.0},
    "Er": {"q": 1.8, "morse": (0.040448, 2.294078, 2.837722), "repulsion": 3.0},
    "O": {"q": -1.2, "morse": (0.042395, 1.379316, 3.618701), "repulsion": 100.0},
}


# Cation-cation Buckingham repulsion parameters: Aij (eV), rho_ij (A), Cij (eV A^6)
bmp_buckingham_potential_params: dict[tuple[str, str], tuple[float, float, float]] = {
    ("Si", "Si"): (7.093669, 0.975598, 0.0),
    ("Si", "Al"): (8.090830, 0.521919, 0.0),
    ("Si", "P"): (5.093669, 0.905598, 0.0),
    ("Al", "Al"): (7.059690, 0.919844, 0.0),
    ("P", "P"): (5.093669, 0.905598, 0.0),
    ("B", "B"): (8.9594, 0.8012, 0.0),
    ("B", "Si"): (8.9594, 0.9270, 0.0),
    ("Si", "Ce3"): (7.093669, 0.431639, 0.0),
    ("Si", "Ce4"): (7.093669, 0.975598, 0.0),
    ("Ce3", "Ce3"): (3429.254, 0.369237, 0.0),
    ("Ce3", "Ce4"): (3429.254, 0.369237, 0.0),
    ("Ce4", "Ce4"): (3429.254, 0.369237, 0.0),
    ("Si", "Zr"): (7.093669, 0.975598, 0.0),
    ("Zr", "Zr"): (3429.254, 0.369237, 0.0),
    ("Si", "Cu"): (8.09083, 0.921919, 0.0),
    ("Si", "Fe3"): (8.090830, 0.521919, 0.0),
    ("Cu", "Cu"): (7.059690, 0.919844, 0.0),
    ("Fe3", "Fe3"): (7.059690, 0.919844, 0.0),
    ("Ga", "O"): (10447.35, 0.208, 41.938),
}


# Three-body parameters for BMP-harmonic (nb3b/harmonic)
# Format: (center, side1, side2) -> (K/2 eV/rad2, theta0 deg, cutoff A)
bmp_harmonic_three_body_potential_params: dict[tuple[str, str, str], tuple[float, float, float]] = {
    ("Si", "O", "Si"): (0.73, 109.47, 2.0),
    ("Si", "O", "P"): (2.00, 109.47, 2.0),
    ("P", "O", "P"): (2.00, 109.47, 2.0),
}

# Three-body parameters for BMP-screened-harmonic (nb3b/screened)
# Format: (center, side1, side2) -> (K eV/rad2, theta0 deg, cutoff A)
bmp_screened_harmonic_params: dict[tuple[str, str, str], tuple[float, float, float]] = {
    ("Si", "O", "Si"): (25.0, 109.47, 3.30),
    ("Si", "O", "P"): (120.0, 109.47, 2.00),
    ("P", "O", "P"): (65.0, 109.47, 2.00),
    ("B", "O", "B"): (60.0, 109.47, 3.30),
    ("B", "O", "Si"): (60.0, 109.47, 3.30),
    ("V", "O", "V"): (30.0, 109.00, 2.50),
    ("V", "O", "P"): (120.0, 109.00, 2.00),
    ("V", "O", "Si"): (120.0, 109.00, 2.00),
}


def supported_elements() -> set[str]:
    """Return the set of elements supported by the BMP potential."""
    return set(bmp_morse_potential_params)


# ------------------------------------------------------------------ #
# Dell-Bray model for composition-dependent Boron Dij                 #
# R = ([A2O] + [AEO]) / [B2O3],  K = [SiO2] / [B2O3]               #
# ------------------------------------------------------------------ #


def param_a(ratio_R: float) -> float:
    """Returns the prefactor term in the first part of the d_model equation."""
    return 2.11081 + 1.0 / (ratio_R + 1.0) ** 2


def param_b(ratio_R: float, ratio_K: float) -> float:
    """Returns the exponent term in the first part of the d_model equation."""
    return 0.02063 * ratio_R + 0.06312 * ratio_K


def param_c(ratio_R: float) -> float:
    """Returns the prefactor term in the second part of the d_model equation."""
    return -2.12213 + 1.0 / (ratio_R + 1.0) ** 2


def param_d(ratio_R: float, ratio_K: float) -> float:
    """Returns the exponent term in the second part of the d_model equation."""
    return -7.50152 * ratio_R + 0.32778 * ratio_K


def d_model(ratio_R: float, ratio_K: float) -> float:
    """Composition-dependent Dij for Boron (BMP potential model)."""
    term1 = param_a(ratio_R) * np.exp(param_b(ratio_R, ratio_K))
    term2 = param_c(ratio_R) * np.exp(param_d(ratio_R, ratio_K))

    return term1 + term2 - 0.001665 * (ratio_K**3) - 0.12807 * ratio_R


def _compute_boron_d(atoms_dict: dict) -> float:
    """Compute composition-dependent Dij for Boron from element counts.

    Args:
        atoms_dict: Structure dict from ``get_structure_dict()``.

    Returns:
        Dij for the B-O Morse interaction.

    Raises:
        ValueError: If no B atoms are found.

    """
    counts = Counter(atom["element"] for atom in atoms_dict["atoms"])
    n_b = counts.get("B", 0)
    if n_b == 0:
        msg = "Cannot compute Boron Dij: no B atoms found in structure."
        raise ValueError(msg)

    n_alkali = sum(counts.get(e, 0) for e in _ALKALI)
    n_ae = sum(counts.get(e, 0) for e in _ALKALINE_EARTH)
    n_si = counts.get("Si", 0)

    # moles: A2O = N_alkali/2, AEO = N_AE, B2O3 = N_B/2, SiO2 = N_Si
    ratio_r = ((n_alkali / 2.0) + n_ae) / (n_b / 2.0)
    ratio_k = n_si / (n_b / 2.0)

    return float(d_model(ratio_r, ratio_k))


# ------------------------------------------------------------------ #
# Three-body file writers                                             #
# ------------------------------------------------------------------ #


def write_bmp_harmonic_three_body_potentials(
    elements: list[str],
    params_dict: dict[tuple[str, str, str], tuple[float, float, float]],
    output_dir: str | Path = "three_body_files",
    filename: str = "BMP.nb3b.harmonic",
) -> Path:
    """Write nb3b/harmonic parameter file for all element permutations.

    Args:
        elements: Chemical elements present in the simulation.
        params_dict: Mapping (center, side1, side2) -> (K/2, theta0, cutoff).
        output_dir: Directory to write the file.
        filename: Output filename.

    Returns:
        Path to the generated file.

    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename

    triplets = list(itertools.product(elements, repeat=3))
    with file_path.open("w") as f:
        f.write(f"# {'i':<5} {'j':<5} {'k':<5} {'K/2':<10} {'theta0':<10} {'cutoff':<10}\n")
        for tri in triplets:
            i, j, k = tri
            key = (i, j, k)
            k2, theta0, cutoff = params_dict.get(key, (0.0, 0.0, 0.0))
            f.write(f"{i:<7} {j:<7} {k:<7} {k2:<10.4f} {theta0:<10.4f} {cutoff:<10.4f}\n")

    return file_path


def write_bmp_screened_harmonic_three_body_potentials(
    elements: list[str],
    params_dict: dict[tuple[str, str, str], tuple[float, float, float]],
    output_dir: str | Path = "three_body_files",
    filename: str = "BMP.nb3b.shrm",
) -> Path:
    """Write nb3b/screened parameter file for all element permutations.

    The file convention has 'i' as the central atom. Keys in params_dict
    are (center, side1, side2) — i.e., the center atom is first.

    Args:
        elements: Chemical elements present in the simulation.
        params_dict: Mapping (center, side1, side2) -> (K, theta0, cutoff).
        output_dir: Directory to write the file.
        filename: Output filename.

    Returns:
        Path to the generated file.

    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / filename

    triplets = list(itertools.product(elements, repeat=3))
    rho = 1.00

    with file_path.open("w") as f:
        f.write(f"# {'i':<5} {'j':<5} {'k':<5} {'K':<10} {'theta0':<10} {'rho':<10} {'cutoff':<10}\n")
        for tri in triplets:
            i, j, k = tri
            # In the LAMMPS file, i is the central atom; dict keys are (center, side1, side2)
            # so (j, i, k) maps file-order (center=i, sides=j,k) to dict key (center=j, sides=i,k)
            lookup_key = (j, i, k)
            if lookup_key in params_dict:
                k_val, theta0, cutoff = params_dict[lookup_key]
                k_out = k_val / 2.0
            else:
                k_out, theta0, cutoff = 0.0, 0.0, 0.0
            f.write(f"  {i:<5} {j:<5} {k:<5} {k_out:<10.2f} {theta0:<10.2f} {rho:<10.2f} {cutoff:<10.2f}\n")

    return file_path


# ------------------------------------------------------------------ #
# Coulomb style resolver (mirrors pmmcs_potential)                    #
# ------------------------------------------------------------------ #


def _resolve_coulomb_style(
    electrostatics_cfg: InteractionConfig,
) -> tuple[str, str, str | None]:
    if isinstance(electrostatics_cfg, (DsfConfig, WolfConfig)):
        long_range_cutoff = electrostatics_cfg.long_range_cutoff or _DEFAULT_DSF_WOLF_LONG_RANGE_CUTOFF
        alpha = electrostatics_cfg.alpha or _DEFAULT_ALPHA
        return (
            f"coul/{electrostatics_cfg.lammps_keyword} {alpha} {long_range_cutoff}",
            f"coul/{electrostatics_cfg.lammps_keyword}",
            None,
        )
    assert isinstance(electrostatics_cfg, (PppmConfig, EwaldConfig))
    long_range_cutoff = electrostatics_cfg.long_range_cutoff or _DEFAULT_PPPM_EWALD_LONG_RANGE_CUTOFF
    return (
        f"coul/long {long_range_cutoff}",
        "coul/long",
        f"kspace_style {electrostatics_cfg.lammps_keyword} {electrostatics_cfg.kspace_accuracy}\n",
    )


# ------------------------------------------------------------------ #
# pair_coeff line builders                                            #
# ------------------------------------------------------------------ #


def _build_bmp_pedone_pair_coeff_lines(
    species: list[str],
    types: dict[str, int],
    boron_d: float | None,
) -> list[str]:
    o_type = types["O"]
    lines = []
    for elem in species:
        i_type = types[elem]
        dij, a, r0 = bmp_morse_potential_params[elem]["morse"]
        if elem == "B" and boron_d is not None:
            dij = boron_d
        cij = bmp_morse_potential_params[elem]["repulsion"]
        lines.append(f"pair_coeff {i_type} {o_type} pedone {dij} {a} {r0} {cij}\n")
    return lines


def _build_bmp_buck_pair_coeff_lines(
    species: list[str],
    types: dict[str, int],
    src: float,
) -> list[str]:
    species_set = set(species)
    lines = []
    seen: set[frozenset] = set()
    for (e1, e2), (a_ij, rho_ij, c_ij) in bmp_buckingham_potential_params.items():
        if e1 not in species_set or e2 not in species_set:
            continue
        pair_key = frozenset({e1, e2})
        if pair_key in seen:
            continue
        seen.add(pair_key)
        i, j = types[e1], types[e2]
        if i > j:
            i, j = j, i
        lines.append(f"pair_coeff {i} {j} buck {a_ij} {rho_ij} {c_ij} {src}\n")
    return lines


def _build_bmp_threebody_pair_coeff_line(
    types: dict[str, int],
    variant: str,
    file_path: Path,
) -> str:
    active = _HARMONIC_3BODY_ELEMENTS if variant == "harmonic" else _SCREENED_HARMONIC_3BODY_ELEMENTS
    style = "nb3b/harmonic" if variant == "harmonic" else "nb3b/screened"
    sorted_elems = sorted(types, key=lambda e: types[e])
    elem_list = " ".join(e if e in active else "NULL" for e in sorted_elems)
    return f"pair_coeff * * {style} {file_path} {elem_list}\n"


# ------------------------------------------------------------------ #
# Main generator                                                       #
# ------------------------------------------------------------------ #


def generate_bmp_potential(
    atoms_dict: dict,
    output_dir: str | Path = ".",
    *,
    variant: str = "harmonic",
    melt: bool = False,
    electrostatics: InteractionConfig | None = None,
) -> pd.DataFrame:
    """Generate the BMP potential for the given composition.

    Args:
        atoms_dict: Structure dict from ``get_structure_dict()``.
        output_dir: Directory to write the three-body parameter file.
        variant: Three-body style — ``"harmonic"`` uses ``nb3b/harmonic``,
            ``"screened-harmonic"`` uses ``nb3b/screened``.
        melt: Append a Langevin NVE/limit pre-equilibration block at 4000 K.
        electrostatics: Coulomb solver settings. Defaults to DSF with
            ``alpha=0.25`` and Coulomb cutoff 8.0 Å.

            **DSF / Wolf** — real-space methods; no k-space solve required.
            **PPPM / Ewald** — reciprocal-space; a ``kspace_style`` line is appended.

    Returns:
        Single-row DataFrame with LAMMPS config lines in the ``Config`` column.

    Raises:
        ValueError: If ``variant`` is not ``"harmonic"`` or ``"screened-harmonic"``, or if
            required parameters are missing for any element in the composition.

    Example:
        >>> df = generate_bmp_potential(struct_dict, potential_type="bmp-harmonic")
        >>> df = generate_bmp_potential(struct_dict, potential_type="bmp-screened-harmonic", melt=False)

    """
    if variant not in ("harmonic", "screened-harmonic"):
        msg = f"variant must be 'harmonic' or 'screened-harmonic', got '{variant}'"
        raise ValueError(msg)

    types = get_element_types_dict(atoms_dict["atoms"])
    species = list(types.keys())

    if "B" in species:
        disallowed = {e for e in species if e not in _BMP_BORON_ALLOWED_ELEMENTS}
        if disallowed:
            msg = (
                f"BMP potential with boron is restricted to alkali/alkaline-earth borate and "
                f"borosilicate glasses. Unsupported elements: {sorted(disallowed)}"
            )
            raise ValueError(msg)

    missing = [elem for elem in species if elem not in bmp_morse_potential_params]
    if missing:
        msg = f"BMP potential has no parameters for: {missing}"
        raise ValueError(msg)

    boron_d = _compute_boron_d(atoms_dict) if "B" in species else None

    electrostatics_cfg = electrostatics if electrostatics is not None else DsfConfig()
    src = _DEFAULT_SHORT_RANGE_CUTOFF
    coulomb_style, coulomb_pair_coeff, kspace_line = _resolve_coulomb_style(electrostatics_cfg)

    out_dir = Path(output_dir).resolve()
    if variant == "harmonic":
        tb_file = write_bmp_harmonic_three_body_potentials(species, bmp_harmonic_three_body_potential_params, out_dir)
    else:
        tb_file = write_bmp_screened_harmonic_three_body_potentials(species, bmp_screened_harmonic_params, out_dir)

    tb_style = "nb3b/harmonic" if variant == "harmonic" else "nb3b/screened"
    model_name = f"BMP-{variant}"

    config_lines = [
        "# BMP potential: Pedone (Morse) + Buckingham repulsion + three-body\n",
        "units metal\n",
        "dimension 3\n",
        "atom_style charge\n",
        "\n### Groups ###\n",
    ]
    config_lines.extend(f"group {elem} type {types[elem]}\n" for elem in species)

    config_lines.append("\n### Charges ###\n")
    config_lines.extend(f"set type {types[elem]} charge {bmp_morse_potential_params[elem]['q']}\n" for elem in species)

    config_lines.extend(
        [
            f"\n### {model_name} Potential Parameters ###\n",
            f"pair_style hybrid/overlay {coulomb_style} pedone {src} buck {src} {tb_style}\n",
            f"pair_coeff * * {coulomb_pair_coeff}\n",
        ]
    )

    if kspace_line:
        config_lines.append(kspace_line)

    config_lines.extend(_build_bmp_pedone_pair_coeff_lines(species, types, boron_d))
    config_lines.extend(_build_bmp_buck_pair_coeff_lines(species, types, src))
    config_lines.append(_build_bmp_threebody_pair_coeff_line(types, variant, tb_file))

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

    return pd.DataFrame(
        {
            "Name": [model_name],
            "Filename": [[str(tb_file)]],
            "Model": [model_name],
            "Species": [species],
            "Config": [config_lines],
        }
    )
