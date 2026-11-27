"""LAMMPS potential generation for the GRACE-FS universal machine-learned potential.

Wraps an already-exported GRACE-FS native LAMMPS model (a YAML file produced by the
grace-tensorpotential CLI tools) into the same Config-line-list contract as the classical
potentials in this package, so it can be dispatched through ``generate_potential()`` like
any other potential. ``select_potential()`` never auto-selects it; it is opt-in.

This module performs no network access and no subprocess calls -- model acquisition and
export is a one-time manual step, documented in the error message raised by
``_resolve_model_path`` when the exported model is missing.

Only single-layer/local GRACE checkpoints (exportable to this TensorFlow-free native YAML
format and runnable via ``pair_style grace/fs``) are supported. Two-layer/three-layer
"semilocal" checkpoints require a TensorFlow-linked or KOKKOS LAMMPS build, which this
package does not provide.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from ase.data import chemical_symbols

from amorphouspy.atoms.shared import get_element_types_dict

DEFAULT_MODEL_NAME = "GRACE-FS-OAM"

# Minimum atoms-per-core for good MPI scaling efficiency. GRACE is a graph-neural-network
# pair style with a substantially different per-atom cost than the classical Buckingham/
# Pedone/Born-Mayer potentials this constant was tuned for elsewhere in this package.
# Kept as a placeholder pending a real scaling benchmark -- not yet verified.
MIN_ATOMS_PER_CORE = 500


def supported_elements() -> set[str]:
    """Return the elements covered by the OMat24/Alexandria/MPtrj training data.

    GRACE-OAM models are pretrained on OMat24 and fine-tuned on sAlex and MPtrj, whose
    combined elemental coverage is Z=1-83 (H-Bi) plus Z=89-94 (Ac-Pu) -- every element
    except Po/At/Rn/Fr/Ra and everything heavier than Pu (89 elements total).

    Source: Figure 1c, Barroso-Luque et al., "Open Materials 2024 (OMat24) Inorganic
    Materials Dataset and Models", arXiv:2410.12771.
    """
    return set(chemical_symbols[1:84]) | set(chemical_symbols[89:95])


def _resolve_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve and validate the path to an already-exported GRACE-FS model.

    Args:
        model_path: Explicit path to a native LAMMPS GRACE-FS YAML file. If None, defaults
            to ``~/.cache/grace/checkpoints/GRACE-FS-OAM/FS_model.yaml``. Must come from a
            single-layer/local checkpoint -- the only tier exportable to the native,
            TensorFlow-free YAML format used by ``pair_style grace/fs``.

    Returns:
        Resolved, existing path to the FS model YAML file.

    Raises:
        FileNotFoundError: If the resolved path does not exist. The message contains the
            one-time export commands to run first.

    """
    checkpoint_dir = Path("~/.cache/grace/checkpoints").expanduser() / DEFAULT_MODEL_NAME
    if model_path is not None:
        resolved = Path(model_path).expanduser().resolve()
    else:
        resolved = (checkpoint_dir / "FS_model.yaml").resolve()

    if not resolved.exists():
        msg = (
            f"GRACE-FS model file not found at '{resolved}'. "
            "Export it once from a checkpoint using the grace-tensorpotential CLI tools "
            "(conda-forge package 'grace-tensorpotential'):\n\n"
            f"    grace_models checkpoint {DEFAULT_MODEL_NAME}\n"
            f"    grace_utils -p {checkpoint_dir}/model.yaml -c {checkpoint_dir}/checkpoint "
            f"export -sf -n {resolved}\n\n"
            f"Note: '{DEFAULT_MODEL_NAME}' must be a single-layer/local checkpoint -- the only tier "
            "exportable to this native YAML format and runnable with pair_style grace/fs. "
            "Two-layer/three-layer checkpoints need a TensorFlow- or KOKKOS-linked LAMMPS "
            "build this package does not provide.\n\n"
            "If the checkpoint name above is not recognised, run `grace_models list` to see "
            "the currently available checkpoints (exact names drift across "
            "grace-tensorpotential releases). generate_grace_potential() does not download "
            "or export models itself; pass an explicit model_path= to point at a file "
            "exported elsewhere."
        )
        raise FileNotFoundError(msg)

    return resolved


def generate_grace_potential(
    atoms_dict: dict,
    *,
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    """Generate a LAMMPS ``pair_style grace/fs`` configuration.

    Args:
        atoms_dict: Structure dict from ``get_structure_dict()``.
        model_path: Explicit path to an already-exported native GRACE-FS YAML file. If
            None, resolved from the default cache location (see ``_resolve_model_path``).

    Returns:
        Single-row DataFrame with LAMMPS config lines in the ``Config`` column.

    Raises:
        FileNotFoundError: If no exported model file is found (see ``_resolve_model_path``
            for the exact remediation commands).

    Example:
        >>> grace_pot = generate_grace_potential(struct_dict, model_path="/data/models/FS_model.yaml")

    """
    fs_model_path = _resolve_model_path(model_path)

    types = get_element_types_dict(atoms_dict["atoms"])
    species = list(types.keys())

    lines = [
        "# GRACE-FS universal machine-learned potential (grace-tensorpotential)\n",
        "# https://gracemaker.readthedocs.io/\n",
        "\n",
        "units metal\n",
        "dimension 3\n",
        "atom_style atomic\n\n",
    ]

    lines.append("### Group Definitions ###\n")
    lines.extend(f"group {elem} type {types[elem]}\n" for elem in species)

    lines.append("\n### GRACE-FS Potential ###\n")
    lines.append("pair_style grace/fs\n")
    # LAMMPS splits input lines on whitespace, so the path is double-quoted to survive spaces.
    lines.append(f'pair_coeff * * "{fs_model_path}" {" ".join(species)}\n')

    lines.append("\nthermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n")
    lines.append("thermo_modify flush no\n")
    lines.append("thermo 100\n")

    return pd.DataFrame(
        {
            "Name": ["GRACE"],
            "Filename": [[str(fs_model_path)]],
            "Model": ["GRACE-FS"],
            "Species": [species],
            "Config": [lines],
        }
    )
