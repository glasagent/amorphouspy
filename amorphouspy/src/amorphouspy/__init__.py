"""amorphouspy - workflows for atomistic modeling of oxide glasses."""

from amorphouspy.analysis.bond_angle_distribution import compute_angles
from amorphouspy.analysis.cavities import compute_cavities
from amorphouspy.analysis.cte import (
    cte_from_npt_fluctuations,
    cte_from_volume_temperature_data,
)
from amorphouspy.analysis.qn_network_connectivity import (
    classify_oxygens,
    compute_network_connectivity,
    compute_qn,
    compute_qn_and_classify,
)
from amorphouspy.analysis.radial_distribution_functions import compute_coordination, compute_rdf
from amorphouspy.analysis.rings import compute_guttmann_rings, generate_bond_length_dict
from amorphouspy.analysis.structure_factor import compute_structure_factor
from amorphouspy.io_utils import (
    structure_from_parsed_output,
    write_angle_distribution,
    write_distribution_to_file,
    write_xyz,
)
from amorphouspy.mass import get_atomic_mass
from amorphouspy.neighbors import get_neighbors
from amorphouspy.potentials.potential import generate_potential
from amorphouspy.shared import count_distribution, running_mean, type_to_dict
from amorphouspy.structure import (
    check_neutral_oxide,
    create_random_atoms,
    extract_composition,
    formula_mass_g_per_mol,
    get_ase_structure,
    get_composition,
    get_glass_density_from_model,
    get_structure_dict,
    parse_formula,
    plan_system,
)
from amorphouspy.workflows.cte import cte_from_fluctuations_simulation, temperature_scan_simulation
from amorphouspy.workflows.elastic_mod import elastic_simulation
from amorphouspy.workflows.md import md_simulation
from amorphouspy.workflows.meltquench import melt_quench_simulation
from amorphouspy.workflows.structural_analysis import analyze_structure, find_rdf_minimum, plot_analysis_results_plotly
from amorphouspy.workflows.viscosity import fit_vft, get_viscosity, viscosity_ensemble, viscosity_simulation

__all__ = [
    "analyze_structure",
    "check_neutral_oxide",
    "classify_oxygens",
    "compute_angles",
    "compute_cavities",
    "compute_coordination",
    "compute_guttmann_rings",
    "compute_network_connectivity",
    "compute_qn",
    "compute_qn_and_classify",
    "compute_rdf",
    "compute_structure_factor",
    "count_distribution",
    "create_random_atoms",
    "cte_from_fluctuations_simulation",
    "cte_from_npt_fluctuations",
    "cte_from_volume_temperature_data",
    "elastic_simulation",
    "extract_composition",
    "find_rdf_minimum",
    "fit_vft",
    "formula_mass_g_per_mol",
    "generate_bond_length_dict",
    "generate_potential",
    "get_ase_structure",
    "get_atomic_mass",
    "get_composition",
    "get_glass_density_from_model",
    "get_neighbors",
    "get_structure_dict",
    "get_viscosity",
    "md_simulation",
    "melt_quench_simulation",
    "parse_formula",
    "plan_system",
    "plot_analysis_results_plotly",
    "running_mean",
    "structure_from_parsed_output",
    "temperature_scan_simulation",
    "type_to_dict",
    "viscosity_ensemble",
    "viscosity_simulation",
    "write_angle_distribution",
    "write_distribution_to_file",
    "write_xyz",
]
