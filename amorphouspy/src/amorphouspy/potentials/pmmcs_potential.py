"""LAMMPS potential generation for oxide glass simulations using PMMCS (Pedone) parameters.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from __future__ import annotations

from typing import TypedDict

import pandas as pd

from amorphouspy.potentials._config import ElectrostaticsConfig
from amorphouspy.shared import get_element_types_dict


class _PmmcsEntry(TypedDict):
    q: float
    morse: tuple[float, float, float]
    repulsion: float


_DEFAULT_SHORT_RANGE_CUTOFF = 5.5
_DEFAULT_DSF_WOLF_LONG_RANGE_CUTOFF = 8.0
_DEFAULT_PPPM_EWALD_LONG_RANGE_CUTOFF = 12.0
_DEFAULT_ALPHA = 0.25
_MELT_TEMPERATURE = 4000

# Complete dictionary of Pmmcs parameters
pmmcs_potential_params: dict[str, _PmmcsEntry] = {
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


def supported_elements() -> set[str]:
    """Return the set of elements supported by the PMMCS potential."""
    return set(pmmcs_potential_params)


def _resolve_coulomb_style(
    electrostatics_cfg: ElectrostaticsConfig,
) -> tuple[str, str, str | None]:
    """Return (coulomb_style, coulomb_pair_coeff, kspace_line) for the given config."""
    if electrostatics_cfg.method in ("dsf", "wolf"):
        long_range_cutoff = electrostatics_cfg.long_range_cutoff or _DEFAULT_DSF_WOLF_LONG_RANGE_CUTOFF
        alpha = electrostatics_cfg.alpha or _DEFAULT_ALPHA
        return (
            f"coul/{electrostatics_cfg.method} {alpha} {long_range_cutoff}",
            f"coul/{electrostatics_cfg.method}",
            None,
        )
    long_range_cutoff = electrostatics_cfg.long_range_cutoff or _DEFAULT_PPPM_EWALD_LONG_RANGE_CUTOFF
    return (
        f"coul/long {long_range_cutoff}",
        "coul/long",
        f"kspace_style {electrostatics_cfg.method} {electrostatics_cfg.kspace_accuracy}\n",
    )


def _build_pmmcs_pair_coeff_lines(species: list[str], types: dict) -> list[str]:
    o_type = types.get("O")
    lines = []
    for elem in species:
        i_type = types[elem]
        dij, a, r0 = pmmcs_potential_params[elem]["morse"]
        cij = pmmcs_potential_params[elem]["repulsion"]
        lines.append(f"pair_coeff {i_type} {o_type} pedone {dij} {a} {r0} {cij}\n")
    return lines


def generate_pmmcs_potential(
    atoms_dict: dict,
    *,
    melt: bool = True,
    electrostatics: ElectrostaticsConfig | None = None,
) -> pd.DataFrame:
    """Generate the PMMCS (Pedone) potential for the given composition.

    Args:
        atoms_dict: Structure dict from ``get_structure_dict()``.
        melt: If ``True``, appends a high-temperature pre-equilibration block
            at 4000 K using a Langevin thermostat (``fix langevin``) combined
            with ``fix nve/limit`` to prevent runaway atom velocities during the
            first 10 000 steps. This helps relax unfavourable contacts in the
            random initial structure before the main melt-quench protocol. Set
            to ``False`` when the starting structure is already equilibrated or
            when you want full control over the thermostat schedule.
        electrostatics: Controls the Coulomb solver and associated cutoffs.
            When ``None`` (default), DSF is used with ``alpha=0.25``,
            short-range Morse cutoff 5.5 Å, and Coulomb cutoff 8.0 Å.

            **DSF / Wolf** - damped shifted force / Wolf summation. These are
            real-space methods and do not require a k-space solve. ``alpha``
            (Å⁻¹) damps the interaction; larger values decay faster and allow a
            shorter ``long_range_cutoff``. Emitted as
            ``coul/dsf <alpha> <long_range_cutoff>`` (or ``coul/wolf``).

            **PPPM / Ewald** — reciprocal-space methods. ``alpha`` is ignored.
            ``long_range_cutoff`` defaults to 12.0 Å (wider than DSF because the
            real-space part decays more slowly without an explicit damping term).
            A ``kspace_style <method> <kspace_accuracy>`` line is appended after
            the pair coefficients. ``kspace_accuracy`` (default ``1e-5``) trades
            cost against accuracy of the long-range sum.

            ``short_range_cutoff`` sets the Morse (pedone) pair cutoff
            independently of the Coulomb cutoff (default 5.5 Å). Increasing it
            captures slightly longer-ranged Morse interactions but raises the
            pair-list cost quadratically.

    Returns:
        Single-row DataFrame with LAMMPS config lines in the ``Config`` column.

    """
    types = get_element_types_dict(atoms_dict["atoms"])
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

    electrostatics_cfg = electrostatics or ElectrostaticsConfig()
    short_range_cutoff = electrostatics_cfg.short_range_cutoff or _DEFAULT_SHORT_RANGE_CUTOFF
    coulomb_style, coulomb_pair_coeff, kspace_line = _resolve_coulomb_style(electrostatics_cfg)

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
            f"pair_style hybrid/overlay {coulomb_style} pedone {short_range_cutoff}\n",
            f"pair_coeff * * {coulomb_pair_coeff}\n",
        ],
    )

    if kspace_line:
        config_lines.append(kspace_line)

    config_lines.extend(_build_pmmcs_pair_coeff_lines(species, types))

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
            "Name": ["PMMCS"],
            "Filename": [[]],
            "Model": ["PMMCS"],
            "Species": [species],
            "Config": [config_lines],
        },
    )
