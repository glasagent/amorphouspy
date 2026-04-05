"""Elastic constant workflows for glass systems using LAMMPS and Finite Differences.

Implements molecular dynamics workflows and post-processing utilities for
elastic constant calculations based on the stress-strain method.

Author
------
Achraf Atila (achraf.atila@bam.de)
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ase.atoms import Atoms

from amorphouspy.workflows.shared import _run_lammps_md


# =========================
# STRAIN / STRESS UTILITIES
# =========================
def apply_strain(atoms: Atoms, eps: np.ndarray) -> Atoms:
    """Apply a strain tensor to an atomic structure.

    Args:
        atoms: The input atomic structure.
        eps: A 3x3 strain tensor to apply to the cell.

    Returns:
        A new Atoms object with the strained cell and scaled atomic positions.

    """
    atoms_new = atoms.copy()
    cell = atoms_new.cell.array
    # Apply strain: h_new = h_old @ (I + eps)
    atoms_new.set_cell(cell @ (np.eye(3) + eps), scale_atoms=True)

    # Ensure atoms are inside the new primary cell
    atoms_new.wrap()

    return atoms_new


def _run_strained_md(structure: Atoms, strain_tensor: np.ndarray, base_kwargs: dict) -> np.ndarray:
    """Calculate the stress response difference between positive and negative strains.

    This helper function applies a central difference approach to determine the
    change in the stress tensor. It runs two independent molecular dynamics
    simulations: one with the applied strain tensor and one with the inverse
    (negative) strain tensor.

    Args:
        structure: The base atomic structure to which the strain is applied.
        strain_tensor: A 3x3 matrix representing the infinitesimal strain tensor to be applied.
        base_kwargs: A dictionary containing the simulation parameters for `_run_lammps_md`.
            Must include:
            - "n_ionic_steps" (int): Total MD steps.
            - "potential" (str): Path to the LAMMPS potential.
            - "temperature" (float): Target temperature.
            - "timestep" (float): Integration time step.
            - "tmp_working_directory" (str | Path | None): Temporary working directory.
            - "n_print" (int): Frequency of output during the simulation.
            - "initial_temperature" (float): Initial temperature for the simulation.
            - "pressure" (float | None): Target pressure for NPT simulations.
            - "langevin" (bool): Whether to use Langevin dynamics.
            - "seed" (int): Random seed for velocity initialization.
            - "server_kwargs" (dict | None): Additional keyword arguments for the server.

    Returns:
        The difference in the mean stress tensors (stress_plus - stress_minus).
        The stress is calculated as the negative of the pressure tensor and
        averaged over the second half of the production run to ensure
        equilibration.

    """
    # Positive strain
    s_pos = apply_strain(structure, strain_tensor)
    _, res_p = _run_lammps_md(structure=s_pos, **base_kwargs)
    gen_p = res_p.get("generic")
    assert gen_p is not None
    press_p = gen_p["pressures"]
    stress_plus = -np.mean(press_p[base_kwargs["n_ionic_steps"] // 2 :], axis=0)

    # Negative strain
    s_neg = apply_strain(structure, -strain_tensor)
    _, res_n = _run_lammps_md(structure=s_neg, **base_kwargs)
    gen_n = res_n.get("generic")
    assert gen_n is not None
    press_n = gen_n["pressures"]
    stress_minus = -np.mean(press_n[base_kwargs["n_ionic_steps"] // 2 :], axis=0)

    return stress_plus - stress_minus


# =========================
# ISOTROPIC MODULI
# =========================
def isotropic_moduli_from_Cij(cij: np.ndarray) -> dict[str, float]:
    """Calculate isotropic elastic moduli from the stiffness tensor using Voigt-Reuss-Hill averaging.

    Assumes the material system is cubic.

    Args:
        cij: 6x6 Elastic stiffness matrix (Voigt notation).

    Returns:
        Dictionary containing:
        - "B": Bulk modulus
        - "G": Shear modulus
        - "E": Young's modulus
        - "nu": Poisson's ratio

    """
    cij_sym = 0.5 * (cij + cij.T)
    c11 = np.mean([cij_sym[0, 0], cij_sym[1, 1], cij_sym[2, 2]])
    c12 = np.mean([cij_sym[0, 1], cij_sym[0, 2], cij_sym[1, 2]])
    c44 = np.mean([cij_sym[3, 3], cij_sym[4, 4], cij_sym[5, 5]])

    bulk = (c11 + 2.0 * c12) / 3.0
    gv = (c11 - c12 + 3.0 * c44) / 5.0
    gr = 5.0 * c44 * (c11 - c12) / (4.0 * c44 + 3.0 * (c11 - c12))
    shear = 0.5 * (gv + gr)

    youngs = 9.0 * bulk * shear / (3.0 * bulk + shear)
    poisson = (3.0 * bulk - 2.0 * shear) / (2.0 * (3.0 * bulk + shear))

    return {"B": float(bulk), "G": float(shear), "E": float(youngs), "nu": float(poisson)}


def elastic_simulation(
    structure: Atoms,
    potential: pd.DataFrame,
    temperature_sim: float = 300.0,
    pressure: float | None = None,
    timestep: float = 1.0,
    equilibration_steps: int = 1_000_000,
    production_steps: int = 10_000,
    n_print: int = 1,
    strain: float = 1e-3,
    server_kwargs: dict[str, Any] | None = None,
    *,
    langevin: bool = False,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
) -> dict[str, Any]:  # pylint: disable=too-many-positional-arguments
    """Perform a LAMMPS-based elastic constant simulation via the stress-strain method.

    This workflow calculates the elastic stiffness tensor (Cij) using finite differences.
    It equilibrates the structure, applies small normal and shear strains, and measures
    the resulting stress response to determine the elastic constants.

    Args:
        structure: Input structure (assumed pre-equilibrated).
        potential: LAMMPS potential file.
        temperature_sim: Simulation temperature in Kelvin (default 5000.0 K).
        pressure: Target pressure for equilibration (default None, i.e., NVT).
        timestep: MD integration timestep in femtoseconds (default 1.0 fs).
        equilibration_steps: Number of steps for the initial equilibration phase (default 1,000,000).
        production_steps: Number of MD steps for the production run (default 10,000).
        n_print: Thermodynamic output frequency (default 1).
        strain: Magnitude of the strain applied for finite differences (default 1e-3).
        server_kwargs: Additional server configuration arguments.
        langevin: Whether to use Langevin dynamics (default False).
        seed: Random seed for velocity initialization (default 12345).
        tmp_working_directory: Temporary directory for job execution.

    Returns:
        Dictionary containing the results. Key "result" contains the "Cij" 6x6 matrix.

    Notes:
        - The structure is first equilibrated (NPT/NVT).
        - Positive and negative strains are applied to cancel lower-order errors (central difference).
        - Calculated Cij values assume Voigt notation.
        - For production simulations, system size, cooling rate, equilibration time,
        and strain magnitude should be tested to ensure the robustness of the results.

    Example:
        >>> result = elastic_simulation(
        ...     structure=my_atoms,
        ...     potential=my_potential,
        ...     temperature_sim=300.0,
        ...     strain=0.001
        ... )

    """
    potential_name = potential.loc[0, "Name"]

    if potential_name.lower() == "shik":
        exclude_patterns = [
            "fix langevin all langevin 5000 5000 0.01 48279",
            "fix ensemble all nve/limit 0.5",
            "run 10000",
            "unfix langevin",
            "unfix ensemble",
        ]

        potential["Config"] = potential["Config"].apply(
            lambda lines: [line for line in lines if not any(p in line for p in exclude_patterns)]
        )

    # Stage 0: INITIAL EQUILIBRATION
    structure0, res = _run_lammps_md(
        structure=structure,
        potential=potential,
        tmp_working_directory=tmp_working_directory,
        temperature=temperature_sim,
        n_ionic_steps=equilibration_steps,
        timestep=timestep,
        n_print=n_print,
        initial_temperature=temperature_sim,
        pressure=pressure,
        seed=seed,
        langevin=langevin,
        server_kwargs=server_kwargs,
    )

    structure_npt = structure0.copy()

    # Enforce cubic average cell

    gen0 = res.get("generic")
    assert gen0 is not None
    vol_array = np.array(gen0["volume"])
    slice_start = len(vol_array) // 2
    avg_l = np.mean(vol_array[slice_start:]) ** (1.0 / 3.0)

    structure_npt.set_cell([[avg_l, 0, 0], [0, avg_l, 0], [0, 0, avg_l]], scale_atoms=True)

    # Shared settings for all production runs
    md_kwargs = {
        "potential": potential,
        "tmp_working_directory": tmp_working_directory,
        "temperature": temperature_sim,
        "n_ionic_steps": production_steps,
        "timestep": timestep,
        "n_print": n_print,
        "initial_temperature": temperature_sim,
        "pressure": pressure,
        "langevin": langevin,
        "seed": seed,
        "server_kwargs": server_kwargs,
    }

    cij = np.zeros((6, 6))
    denom = 2 * strain

    # Normal strains (C11, C22, C33 and off-diagonals)
    for i in range(3):
        strain_mat = np.zeros((3, 3))
        strain_mat[i, i] = strain
        stress_diff = _run_strained_md(structure_npt, strain_mat, md_kwargs)
        cij[i, 0] = stress_diff[0, 0] / denom
        cij[i, 1] = stress_diff[1, 1] / denom
        cij[i, 2] = stress_diff[2, 2] / denom

    # Shear strains (C44, C55, C66)
    shear_indices = [(1, 2), (0, 2), (0, 1)]  # yz, xz, xy
    for i, (idx1, idx2) in enumerate(shear_indices):
        strain_mat = np.zeros((3, 3))
        strain_mat[idx1, idx2] = strain / 2
        strain_mat[idx2, idx1] = strain / 2
        stress_diff = _run_strained_md(structure_npt, strain_mat, md_kwargs)
        voigt = 3 + i
        cij[voigt, voigt] = stress_diff[idx1, idx2] / denom

    moduli = isotropic_moduli_from_Cij(cij)

    result = {"Cij": cij, "moduli": moduli}

    # After calculating all Cij
    if not np.allclose(cij[0, 0], cij[1, 1]) or not np.allclose(cij[1, 1], cij[2, 2]):
        warnings.warn("System may not be cubic: C11 != C22 != C33", stacklevel=2)
    if not np.allclose(cij[3, 3], cij[4, 4]) or not np.allclose(cij[4, 4], cij[5, 5]):
        warnings.warn("System may not be cubic: C44 != C55 != C66", stacklevel=2)

    return result
