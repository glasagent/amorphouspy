"""Structure generation and analysis for oxide glass systems.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from amorphouspy.structure.composition import (
    COMPOSITION_TOLERANCE,
    ELEMENT,
    check_neutral_oxide,
    extract_composition,
    extract_stoichiometry,
    formula_mass_g_per_mol,
    get_composition,
    normalize,
    parse_formula,
    weight_percent_to_mol_fraction,
)
from amorphouspy.structure.density import TRACE_OXIDES, get_glass_density_from_model
from amorphouspy.structure.geometry import (
    create_random_atoms,
    get_ase_structure,
    get_structure_dict,
    minimum_image_distance,
)
from amorphouspy.structure.planner import (
    DENSITY_TOLERANCE,
    allocate_formula_units_to_target_atoms,
    element_counts_from_formula_units,
    get_box_from_density,
    plan_system,
    validate_target_mode,
)

__all__ = [
    "COMPOSITION_TOLERANCE",
    "DENSITY_TOLERANCE",
    "ELEMENT",
    "TRACE_OXIDES",
    "allocate_formula_units_to_target_atoms",
    "check_neutral_oxide",
    "create_random_atoms",
    "element_counts_from_formula_units",
    "extract_composition",
    "extract_stoichiometry",
    "formula_mass_g_per_mol",
    "get_ase_structure",
    "get_box_from_density",
    "get_composition",
    "get_glass_density_from_model",
    "get_structure_dict",
    "minimum_image_distance",
    "normalize",
    "parse_formula",
    "plan_system",
    "validate_target_mode",
    "weight_percent_to_mol_fraction",
]
