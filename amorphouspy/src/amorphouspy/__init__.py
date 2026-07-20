"""amorphouspy - workflows for atomistic modeling of oxide glasses."""

from amorphouspy.atoms.io import write_angle_distribution, write_distribution_to_file, write_xyz
from amorphouspy.atoms.mass import get_atomic_mass
from amorphouspy.atoms.neighbors import get_neighbors
from amorphouspy.atoms.shared import count_distribution, running_mean, type_to_dict
from amorphouspy.fabrication import (
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
from amorphouspy.fabrication.meltquench import extract_equilibration_frames, melt_quench_simulation
from amorphouspy.lammps.io import frames_from_melt_quench_result, load_lammps_dump, structure_from_parsed_output
from amorphouspy.lammps.md import md_simulation
from amorphouspy.lammps.potentials._config import DsfConfig, EwaldConfig, InteractionConfig, PppmConfig, WolfConfig
from amorphouspy.lammps.potentials.potential import generate_potential
from amorphouspy.pipelines.meltquench import generate_structure, run_melt_quench
from amorphouspy.properties.cte import cte_from_fluctuations_simulation, temperature_scan_simulation
from amorphouspy.properties.cte_analysis import (
    cte_from_npt_fluctuations,
    cte_from_volume_temperature_data,
)
from amorphouspy.properties.diffusion import (
    compute_msd,
    diffusion_simulation,
    fit_arrhenius,
    get_diffusion,
    log_linear_dump_steps,
    nernst_einstein_conductivity,
    save_frames,
)
from amorphouspy.properties.elastic import elastic_simulation
from amorphouspy.properties.structural.all import (
    analyze_structure,
    find_rdf_minimum,
    plot_analysis_results_plotly,
    run_structural_analysis,
)
from amorphouspy.properties.structural.averaging import average_over_frames
from amorphouspy.properties.structural.bond_angles import compute_angles
from amorphouspy.properties.structural.cavities import compute_cavities
from amorphouspy.properties.structural.projected_rdf import compute_projected_rdf
from amorphouspy.properties.structural.qn import (
    classify_oxygens,
    compute_network_connectivity,
    compute_qn,
    compute_qn_and_classify,
)
from amorphouspy.properties.structural.rdf import compute_coordination, compute_rdf
from amorphouspy.properties.structural.rings import compute_guttmann_rings, generate_bond_length_dict
from amorphouspy.properties.structural.structure_factor import compute_structure_factor
from amorphouspy.properties.viscosity import fit_vft, get_viscosity, viscosity_ensemble, viscosity_simulation

__all__ = [
    "DsfConfig",
    "EwaldConfig",
    "InteractionConfig",
    "PppmConfig",
    "WolfConfig",
    "analyze_structure",
    "average_over_frames",
    "check_neutral_oxide",
    "classify_oxygens",
    "compute_angles",
    "compute_cavities",
    "compute_coordination",
    "compute_guttmann_rings",
    "compute_msd",
    "compute_network_connectivity",
    "compute_projected_rdf",
    "compute_qn",
    "compute_qn_and_classify",
    "compute_rdf",
    "compute_structure_factor",
    "count_distribution",
    "create_random_atoms",
    "cte_from_fluctuations_simulation",
    "cte_from_npt_fluctuations",
    "cte_from_volume_temperature_data",
    "diffusion_simulation",
    "elastic_simulation",
    "extract_composition",
    "extract_equilibration_frames",
    "find_rdf_minimum",
    "fit_arrhenius",
    "fit_vft",
    "formula_mass_g_per_mol",
    "frames_from_melt_quench_result",
    "generate_bond_length_dict",
    "generate_potential",
    "generate_structure",
    "get_ase_structure",
    "get_atomic_mass",
    "get_composition",
    "get_diffusion",
    "get_glass_density_from_model",
    "get_neighbors",
    "get_structure_dict",
    "get_viscosity",
    "load_lammps_dump",
    "log_linear_dump_steps",
    "md_simulation",
    "melt_quench_simulation",
    "nernst_einstein_conductivity",
    "parse_formula",
    "plan_system",
    "plot_analysis_results_plotly",
    "run_melt_quench",
    "run_structural_analysis",
    "running_mean",
    "save_frames",
    "structure_from_parsed_output",
    "temperature_scan_simulation",
    "type_to_dict",
    "viscosity_ensemble",
    "viscosity_simulation",
    "write_angle_distribution",
    "write_distribution_to_file",
    "write_xyz",
]
