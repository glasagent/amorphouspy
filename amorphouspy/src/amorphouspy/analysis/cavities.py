"""Structural analysis functions for multicomponent glass systems.

Author: Achraf Atila (achraf.atila@bam.de)

Compute cavity distribution and associated properties.
The current implementation is wrapper around the functions implemented in sovapy
Only Guttman rings are analyzed.


If this code is used please cite the relevant papers below

sovapy paper:
Shiga, M., Hirata, A., Onodera, Y. et al.
Ring-originated anisotropy of local structural ordering in amorphous and crystalline silicon dioxide.
Commun Mater 4, 91 (2023). https://doi.org/10.1038/s43246-023-00416-w
Source code: https://github.com/MotokiShiga/sova-cui/tree/main


Cavity analysis paper:
I. Meyer, F. Rhiem, F. Beule, D. Knodt, J. Heinen, R. O. Jones,
pyMolDyn: Identification, structure, and properties of cavities/vacancies in condensed matter,
J. Comput. Chem., 38, 389-394 (2017). https://doi.org/10.1002/jcc.24697

"""

import tempfile

import numpy as np
from ase import Atoms
from sovapy.computation.cavity import Cavity
from sovapy.core.file import File

from amorphouspy.io_utils import get_properties_for_structure_analysis, write_xyz
from amorphouspy.shared import type_to_dict


def compute_cavities(
    structure: Atoms, resolution: int = 64, cutoff_radii: dict[str, float] | None = None
) -> dict[str, np.ndarray]:
    """Compute cavity distribution and associated properties for a glass structure via grid-based void analysis.

    Args:
        structure: The ASE Atoms object to analyze.
        resolution: Grid resolution for the cavity calculation.
        cutoff_radii: Optional dictionary mapping element symbols to radii.
                     If None, default radii from sovapy are used.

    Returns:
        A dictionary containing cavity attributes volumes,
        surface_areas, asphericities, acylindricities, anisotropies.

    Example:
        >>> structure = read('glass.xyz')
        >>> cavities = compute_cavities(structure, resolution=128)
        >>> print(cavities['volumes'])

    """
    # Extract properties using the provided helper
    ids, types, coords, box_size = get_properties_for_structure_analysis(structure)
    type_dict = type_to_dict(types)

    # Use a context manager to ensure the temporary file is cleaned up
    with tempfile.NamedTemporaryFile("w+", suffix=".xyz", delete=True) as tmp:
        # Write the structure to the temp file using the provided write_xyz function
        write_xyz(filename=tmp.name, coords=coords, types=types, box_size=box_size, type_dict=type_dict)
        tmp.flush()

        # Load file into sovapy
        f = File.open(tmp.name)
        sovapy_atoms = f.get_atoms()

        # Initialize and calculate
        cavity = Cavity(sovapy_atoms)
        cavity.calculate(resolution=resolution, cutoff_radii=cutoff_radii)

        # List of attributes to extract
        attrs = ["volumes", "surface_areas", "asphericities", "acylindricities", "anisotropies"]

        return {attr: getattr(cavity.domains, attr) for attr in attrs}
