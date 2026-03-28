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

import numpy as np
from ase import Atoms
from sovapy.computation.rings import RINGs
from sovapy.core.file import File

from amorphouspy.io_utils import write_xyz
from amorphouspy.shared import type_to_dict


def compute_guttmann_rings(
    structure: Atoms,
    bond_lengths: dict[tuple[str, str], float],
    max_size: int = 24,
    n_cpus: int = 1,
) -> tuple[dict[int, int], float]:
    """Compute the Guttman ring size distribution and mean ring size for a given atomic configuration.

    Rings are detected using the sovapy implementation of Guttman's algorithm.
    Only closed Guttman rings are considered in the analysis.

    Ring structure of the crystalline and amorphous forms of silicon dioxide
    Lester Guttman, J. Non-Cryst. Solids doi:https://doi.org/10.1016/0022-3093(90)90686-G.

    Args:
        structure: ASE Atoms object containing atomic coordinates and types.
        bond_lengths: Dictionary specifying maximum bond lengths for each element pair.
        max_size: Maximum ring size to consider (default is 24).
        n_cpus: Number of CPUs to use for parallel ring search (default is 1).

    Returns:
        A tuple containing:
            hist: Mapping from ring size to frequency.
            mean_ring_size: Mean ring size.

    Example:
        >>> structure = read('glass.xyz')
        >>> bond_lengths = {('Si', 'O'): 1.8}
        >>> hist, mean_size = compute_guttmann_rings(
        ...     structure, bond_lengths, max_size=12
        ... )

    """
    _atoms = structure.copy()
    _atoms.wrap()
    coords = _atoms.get_positions()
    types = _atoms.get_atomic_numbers()
    cell = _atoms.get_cell()
    box_size = np.array([cell[0, 0], cell[1, 1], cell[2, 2]])
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

    Args:
        atoms: ASE Atoms object containing atomic types and coordinates.
        specific_cutoffs: Optional cutoff overrides for specific pairs.
        default_cutoff: Default bond length for unspecified pairs.

    Returns:
        Dictionary of bond length thresholds.

    Example:
        >>> structure = read('glass.xyz')
        >>> bond_lengths = generate_bond_length_dict(
        ...     structure, default_cutoff=2.0
        ... )

    """
    if specific_cutoffs is None:
        specific_cutoffs = {}
    types = atoms.get_atomic_numbers()
    type_dict = type_to_dict(types)
    elements = list(type_dict.values())
    bond_dict = {}
    for a, b in combinations_with_replacement(elements, 2):
        cutoff = specific_cutoffs.get((a, b), specific_cutoffs.get((b, a), default_cutoff))
        bond_dict[(a, b)] = cutoff
    return bond_dict
