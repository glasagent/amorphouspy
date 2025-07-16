"""Structural analysis functions for multicomponent glass systems.

Author: Achraf Atila (achraf.atila@bam.de)

Compute ring size distribution.
The current implementation is wrapper around the functions implemented in sovapy
Only Guttman rings are analyzed.


If this code is used please cite the relevant papers below

sovapy paper:
Shiga, M., Hirata, A., Onodera, Y. et al.
Ring-originated anisotropy of local structural ordering in amorphous and crystalline silicon dioxide.
Commun Mater 4, 91 (2023). https://doi.org/10.1038/s43246-023-00416-w
Source code: https://github.com/MotokiShiga/sova-cui/tree/main


Guttman Rings paper:
Ring structure of the crystalline and amorphous forms of silicon dioxide
Lester Guttman, J. Non-Cryst. Solids doi:https://doi.org/10.1016/0022-3093(90)90686-G

"""

import tempfile
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from ase import Atoms
from sovapy.computation.rings import RINGs
from sovapy.core.file import File

from pyiron_glass.io_utils import get_properties_for_structure_analysis
from pyiron_glass.shared import type_to_dict


def write_xyz(
    filename: str,
    coords: np.ndarray,
    types: np.ndarray,
    box_size: np.ndarray = None,
    type_dict: dict[int, str] | None = None,
) -> None:
    """Write atomic configuration to an XYZ file.

    Parameters
    ----------
    filename : str
        Output XYZ file name.
    coords : np.ndarray, shape (N, 3)
        Atomic coordinates.
    types : np.ndarray, shape (N,)
        Atomic types as integers.
    box_size : array-like of float, shape (3,), optional
        Simulation box size in x, y, z.
    type_dict : dict[int, str], optional
        Dictionary mapping atomic type integers to element symbols.

    """
    if type_dict is None:
        msg = "type_dict must be provided"
        raise ValueError(msg)

    N = coords.shape[0]
    path = Path(filename)
    with path.open("w") as f:
        f.write(f"{N}\n")
        if box_size is not None:
            f.write(f"CUB {box_size[0]:.8f} {box_size[1]:.8f} {box_size[2]:.8f}\n")
        else:
            f.write("\n")
        for t, (x, y, z) in zip(types, coords, strict=False):
            symbol = type_dict.get(t)
            if symbol is None:
                msg = f"Unknown atomic type: {t}"
                raise ValueError(msg)
            f.write(f"{symbol} {x:.8f} {y:.8f} {z:.8f}\n")


def compute_guttmann_rings(
    types: np.ndarray,
    coords: np.ndarray,
    box_size: np.ndarray,
    bond_lengths: dict[tuple[str, str], float],
    max_size: int = 24,
    n_cpus: int = 1,
) -> tuple[dict[int, int], float]:
    """Compute the Guttman ring size distribution and mean ring size for a given atomic configuration.

    Rings are detected using the sovapy implementation of Guttman's algorithm.
    Only closed Guttman rings are considered in the analysis.

    Ring structure of the crystalline and amorphous forms of silicon dioxide
    Lester Guttman, J. Non-Cryst. Solids doi:https://doi.org/10.1016/0022-3093(90)90686-G.

    Parameters
    ----------
    types : np.ndarray of shape (N,)
        Array of integer atomic types.
    coords : np.ndarray of shape (N, 3)
        Array of atomic coordinates in Cartesian space.
    box_size : np.ndarray or array-like of shape (3,)
        Simulation box dimensions along x, y, z directions.
    bond_lengths : dict[(str, str), float]
        Dictionary specifying maximum bond lengths for each element pair.
    max_size : int, optional
        Maximum ring size to consider (default is 24).
    n_cpus : int, optional
        Number of CPUs to use for parallel ring search (default is 1).

    Returns
    -------
    hist : dict[int, int]
        Dictionary mapping ring size to frequency (number of occurrences).
    mean_ring_size : float
        Mean ring size computed as sum over (s / 2) * p(s), where s is ring size and p(s) is its probability.

    """
    type_dict = type_to_dict(types)
    with tempfile.NamedTemporaryFile("w+", suffix=".xyz", delete=True) as tmp:
        write_xyz(filename=tmp.name, coords=coords, types=types, box_size=box_size, type_dict=type_dict)
        tmp.flush()
        f = File.open(tmp.name)

        atoms = f.get_atoms()
        atoms.set_bond_lengths(bond_lengths)

        rings = RINGs(atoms).calculate(ring_type=RINGs.RingType.GUTTMAN, num_parallel=n_cpus, cutoff_size=max_size)

    closed = [r for r in rings if r.closed]
    sizes = np.array([r.size for r in closed])
    hist = {int(s): int(np.sum(sizes == s)) for s in np.unique(sizes)}

    if sizes.size == 0:
        return hist, 0.0

    s_max = sizes.max()
    s_num = np.arange(s_max + 1)
    hist_size = np.array([np.sum(sizes == s) for s in s_num])
    prob = hist_size / hist_size.sum()
    mean_ring_size = np.sum((s_num / 2) * prob)

    return hist, float(mean_ring_size)


def generate_bond_length_dict(
    atoms: Atoms, specific_cutoffs: dict[tuple[str, str], float] | None = None, default_cutoff: float = -1.0
) -> dict[tuple[str, str], float]:
    """Generate all symmetric element pairs from a list of elements.

    and assign bond lengths using specific or default cutoffs.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object containing atomic types and coordinates.
    specific_cutoffs : dict[(str, str), float]
        Optional cutoff overrides for specific pairs.
    default_cutoff : float
        Default bond length for unspecified pairs.

    Returns
    -------
    dict[(str, str), float]
        Dictionary of bond length thresholds.

    """
    if specific_cutoffs is None:
        specific_cutoffs = {}
    _, types, _, _ = get_properties_for_structure_analysis(atoms)
    type_dict = type_to_dict(types)
    elements = list(type_dict.values())
    bond_dict = {}
    for a, b in combinations_with_replacement(elements, 2):
        cutoff = specific_cutoffs.get((a, b), specific_cutoffs.get((b, a), default_cutoff))
        bond_dict[(a, b)] = cutoff
    return bond_dict
