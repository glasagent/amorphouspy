# Tutorial

This guide covers common concepts used throughout `amorphouspy` and explains how our core functions typically work.

## Core Concepts

Before diving into the workflows, it is important to understand some key concepts that `amorphouspy` relies on.

### The Executor Concept

Many simulation tasks in `amorphouspy` (especially those involving LAMMPS) are executed via an "executor". This allows simulations to run asynchronously, across multiple cores via MPI, or even on high-performance computing clusters seamlessly. 

When you call a function that requires a simulation, such as `melt_quench_simulation` or `elastic_simulation`, the function typically:
1. Prepares the LAMMPS input script.
2. Submits the task using an executor (e.g., using `executorlib` or standard subprocesses).
3. Waits for the result or allows you to query it later via a job reference.

### Standard Function Patterns

Most high-level functions in `amorphouspy` follow a consistent pattern:
- **Inputs**: They take an [ASE `Atoms`](https://wiki.fysik.dtu.dk/ase/ase/atoms.html) object representing your structure, alongside a pandas DataFrame representing the [Interatomic Potential](theory/potentials/index.md). 
- **Outputs**: They return structured dictionaries, pandas DataFrames, or custom namedtuples (like `StructuralAnalysisData`).

By maintaining this standard, you can easily pipe the output of one function into another.

---

## Quick Start

### 1. Generate a glass structure

The first step in any simulation is creating an initial atomic configuration. `amorphouspy` takes a composition dict and generates random atom positions in a periodic cubic box with the correct stoichiometry and a physically realistic density.

```python
from amorphouspy import get_structure_dict, get_ase_structure

# Define a soda-lime silicate composition (mol%)
composition = {"SiO2": 75, "Na2O": 15, "CaO": 10}

# Generate structure with ~3000 atoms
# Density is auto-calculated using Fluegel's empirical model
structure_dict = get_structure_dict(composition, target_atoms=3000)

# Convert to ASE Atoms object for visualization and manipulation
atoms = get_ase_structure(structure_dict)

print(f"Generated {len(atoms)} atoms in a {structure_dict['box']:.2f} Å box")
print(f"Formula units: {structure_dict['formula_units']}")
print(f"Element counts: {structure_dict['element_counts']}")
```

### 2. Set up an interatomic potential

Choose from three built-in classical force fields. The potential generator returns a DataFrame containing all LAMMPS configuration lines:

```python
from amorphouspy import generate_potential

# Generate PMMCS potential (broadest element coverage)
potential = generate_potential(structure_dict, potential_type="pmmcs")

# Other options:
# potential = generate_potential(structure_dict, potential_type="bjp")   # CAS glasses
# potential = generate_potential(structure_dict, potential_type="shik")  # Si/Al/B glasses
```

### 3. Run a melt-quench simulation

Transform the random initial structure into a realistic amorphous glass through a heating-equilibration-cooling cycle:

```python
from amorphouspy import melt_quench_simulation

result = melt_quench_simulation(
    structure=atoms,
    potential=potential,
    temperature_high=5000.0,   # Melt at 5000 K
    temperature_low=300.0,     # Quench to 300 K
    heating_rate=1e12,         # K/s (typical for MD)
    cooling_rate=1e12,         # K/s
)

glass_structure = result["structure"]  # Quenched glass
```

> **Tip:** For production runs, use the potential-specific protocols which include optimized multi-stage temperature programs. See [Simulation Workflows](how_to_guides/index.md) for details.

### 4. Analyze the glass structure

Run a comprehensive structural analysis with a single function call:

```python
from amorphouspy import analyze_structure
from amorphouspy.workflows.structural_analysis import plot_analysis_results_plotly

# Compute all structural properties
data = analyze_structure(glass_structure)

# Inspect results
print(f"Density: {data.density:.3f} g/cm³")
print(f"Network connectivity: {data.network.connectivity:.2f}")
print(f"Qⁿ distribution: {data.network.Qn_distribution}")
print(f"Si coordination: {data.coordination.formers}")

# Generate interactive Plotly visualization
fig = plot_analysis_results_plotly(data)
fig.show()
```

### 5. Compute material properties

```python
from amorphouspy import elastic_simulation

# Calculate elastic constants via stress-strain method
elastic_result = elastic_simulation(
    structure=glass_structure,
    potential=potential,
    temperature=300.0,
    strain=1e-3,
    production_steps=10_000,
)

print(f"Young's modulus: {elastic_result['E']:.1f} GPa")
print(f"Bulk modulus: {elastic_result['B']:.1f} GPa")
print(f"Poisson's ratio: {elastic_result['nu']:.3f}")
```
