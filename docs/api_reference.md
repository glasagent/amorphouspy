# API Reference

All symbols listed here are part of the public API and importable directly from the top-level package:

```python
import amorphouspy as am
```

The source of truth for what is public is [`__all__`](https://github.com/pyiron/amorphouspy/blob/main/amorphouspy/src/amorphouspy/__init__.py) in `amorphouspy/__init__.py`. This page is auto-generated from docstrings — it cannot drift from the code.

---

## Structure Generation

::: amorphouspy.structure.parse_formula

---

::: amorphouspy.structure.formula_mass_g_per_mol

---

::: amorphouspy.structure.get_composition

---

::: amorphouspy.structure.extract_composition

---

::: amorphouspy.structure.check_neutral_oxide

---

::: amorphouspy.structure.plan_system

---

::: amorphouspy.structure.get_glass_density_from_model

---

::: amorphouspy.structure.create_random_atoms

---

::: amorphouspy.structure.get_structure_dict

---

::: amorphouspy.structure.get_ase_structure

---

## Potentials

::: amorphouspy.potentials.potential.generate_potential

---

## Simulation Workflows

::: amorphouspy.workflows.meltquench.melt_quench_simulation

---

::: amorphouspy.workflows.md.md_simulation

---

::: amorphouspy.workflows.elastic_mod.elastic_simulation

---

::: amorphouspy.workflows.viscosity.viscosity_simulation

---

::: amorphouspy.workflows.viscosity.get_viscosity

---

::: amorphouspy.workflows.viscosity.fit_vft

---

::: amorphouspy.workflows.cte.cte_from_fluctuations_simulation

---

::: amorphouspy.workflows.cte.temperature_scan_simulation

---

::: amorphouspy.workflows.structural_analysis.analyze_structure

---

::: amorphouspy.workflows.structural_analysis.find_rdf_minimum

---

::: amorphouspy.workflows.structural_analysis.plot_analysis_results_plotly

---

## Analysis Functions

::: amorphouspy.analysis.radial_distribution_functions.compute_rdf

---

::: amorphouspy.analysis.radial_distribution_functions.compute_coordination

---

::: amorphouspy.analysis.qn_network_connectivity.compute_qn

---

::: amorphouspy.analysis.qn_network_connectivity.compute_network_connectivity

---

::: amorphouspy.analysis.bond_angle_distribution.compute_angles

---

::: amorphouspy.analysis.rings.compute_guttmann_rings

---

::: amorphouspy.analysis.rings.generate_bond_length_dict

---

::: amorphouspy.analysis.cavities.compute_cavities

---

::: amorphouspy.analysis.structure_factor.compute_structure_factor

---

::: amorphouspy.analysis.cte.cte_from_npt_fluctuations

---

::: amorphouspy.analysis.cte.cte_from_volume_temperature_data

---

## Neighbor Search

::: amorphouspy.neighbors.get_neighbors

---

## Utilities

::: amorphouspy.shared.count_distribution

---

::: amorphouspy.shared.type_to_dict

---

::: amorphouspy.shared.running_mean

---

::: amorphouspy.mass.get_atomic_mass

---

::: amorphouspy.io_utils.get_properties_for_structure_analysis

---

::: amorphouspy.io_utils.structure_from_parsed_output

---

::: amorphouspy.io_utils.write_xyz

---

::: amorphouspy.io_utils.write_distribution_to_file

---

::: amorphouspy.io_utils.write_angle_distribution
