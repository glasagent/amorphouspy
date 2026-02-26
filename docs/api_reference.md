# API Reference

Complete function signatures for the `amorphouspy` public API. All functions listed here are importable directly from the top-level package.

```python
import amorphouspy
```

---

## Structure Generation

### `amorphouspy.structure`

```python
parse_formula(formula: str) -> dict[str, int]
```
Parse a chemical formula into element counts. E.g., `"Al2O3"` → `{"Al": 2, "O": 3}`.

---

```python
formula_mass_g_per_mol(formula: str) -> float
```
Calculate molar mass of a compound in g/mol.

---

```python
get_composition(comp_str: str, mode: str = "molar") -> dict[str, float]
```
Parse composition string into normalized molar fractions. `mode` is `"molar"` or `"weight"`.

---

```python
extract_composition(composition: str) -> dict[str, float]
```
Strict composition parser with charge neutrality and element validation.

---

```python
check_neutral_oxide(oxide: str) -> None
```
Raise `ValueError` if oxide is not charge-neutral.

---

```python
plan_system(
    comp_str: str,
    target: int,
    mode: str = "molar",
    target_type: str = "atoms",
) -> dict
```
Generate integer formula-unit allocation for a target system size.

---

```python
get_glass_density_from_model(composition_string: str) -> float
```
Estimated room-temperature density (g/cm³) using Fluegel's empirical model.

---

```python
create_random_atoms(
    composition: str,
    n_molecules: int | None = None,
    target_atoms: int | None = None,
    mode: str = "molar",
    stoichiometry: dict | None = None,
    box_length: float = 50.0,
    min_distance: float = 1.6,
    seed: int = 42,
    max_attempts_per_atom: int = 100000,
) -> tuple[list[dict], dict[str, int]]
```
Generate random atom positions in a periodic cubic box with minimum distance constraints.

---

```python
get_structure_dict(
    composition: str,
    n_molecules: int | None = None,
    target_atoms: int | None = None,
    mode: str = "molar",
    density: float | None = None,
    min_distance: float = 1.6,
    max_attempts_per_atom: int = 10000,
) -> dict
```
Generate a complete structure dictionary ready for simulation.

---

```python
get_ase_structure(
    atoms_dict: dict,
    replicate: tuple[int, int, int] = (1, 1, 1),
) -> Atoms
```
Convert structure dictionary to ASE Atoms object with optional supercell replication.

---

## Potentials

### `amorphouspy.potentials.potential`

```python
generate_potential(
    atoms_dict: dict,
    potential_type: str = "pmmcs",
) -> pd.DataFrame
```
Generate LAMMPS potential configuration. `potential_type` is `"pmmcs"`, `"bjp"`, or `"shik"`.

---

## Workflows

### `amorphouspy.workflows.meltquench`

```python
melt_quench_simulation(
    structure: Atoms,
    potential: str,
    temperature_high: float = 5000.0,
    temperature_low: float = 300.0,
    timestep: float = 1.0,
    heating_rate: float = 1e12,
    cooling_rate: float = 1e12,
    n_print: int = 1000,
    *,
    server_kwargs: dict | None = None,
    langevin: bool = False,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
) -> dict
```
Multi-stage melt-quench simulation with potential-specific protocols.

---

### `amorphouspy.workflows.md`

```python
md_simulation(
    structure: Atoms,
    potential: str,
    temperature_sim: float = 5000.0,
    timestep: float = 1.0,
    production_steps: int = 10_000_000,
    n_print: int = 1000,
    server_kwargs: dict | None = None,
    *,
    pressure: float | None = None,
    langevin: bool = False,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
) -> dict
```
Single-point constant-T/P molecular dynamics simulation.

---

### `amorphouspy.workflows.elastic_mod`

```python
elastic_simulation(
    structure: Atoms,
    potential: str,
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
) -> dict[str, Any]
```
Elastic stiffness tensor via stress-strain finite differences.

---

### `amorphouspy.workflows.viscosity`

```python
viscosity_simulation(
    structure: Atoms,
    potential: str,
    temperature_sim: float = 5000.0,
    timestep: float = 1.0,
    production_steps: int = 10_000_000,
    n_print: int = 1,
    server_kwargs: dict[str, Any] | None = None,
    *,
    langevin: bool = False,
    seed: int = 12345,
    tmp_working_directory: str | Path | None = None,
) -> dict[str, Any]
```
Run MD for viscosity computation via Green-Kubo formalism.

---

```python
get_viscosity(
    result: dict,
    timestep: float = 1.0,
    max_lag: int | None = 1_000_000,
) -> dict
```
Post-process stress tensor data to compute viscosity from stress autocorrelation.

---

```python
fit_vft(
    T_data: ArrayLike,
    log10_eta_data: ArrayLike,
    initial_guess: tuple = (-3, 1000, 200),
) -> tuple[np.ndarray, np.ndarray]
```
Fit viscosity-temperature data to the Vogel-Fulcher-Tammann model.

---

### `amorphouspy.workflows.cte`

```python
cte_from_fluctuations_simulation(
    structure: Atoms,
    potential: str,
    temperature: float | list[int | float] = 300,
    pressure: float = 1e-4,
    timestep: float = 1.0,
    equilibration_steps: int = 100_000,
    production_steps: int = 200_000,
    min_production_runs: int = 2,
    max_production_runs: int = 25,
    CTE_uncertainty_criterion: float = 1e-6,
    n_dump: int = 100000,
    n_log: int = 10,
    server_kwargs: dict[str, Any] | None = None,
    *,
    aniso: bool = False,
    seed: int | None = 12345,
    tmp_working_directory: str | Path | None = None,
) -> dict[str, Any]
```
CTE simulation via H-V fluctuations with iterative convergence checking.

---

```python
temperature_scan_simulation(
    structure: Atoms,
    potential: str,
    temperature: list[int | float] | None = None,
    pressure: float = 1e-4,
    timestep: float = 1.0,
    equilibration_steps: int = 100_000,
    production_steps: int = 200_000,
    n_dump: int = 100000,
    n_log: int = 10,
    server_kwargs: dict[str, Any] | None = None,
    *,
    aniso: bool = False,
    seed: int | None = 12345,
    tmp_working_directory: str | Path | None = None,
) -> dict[Any, Any]
```
Temperature scan to collect structural data for CTE calculations via V-T curves.

---

### `amorphouspy.workflows.structural_analysis`

```python
analyze_structure(atoms: Atoms) -> StructureData
```
Comprehensive structural analysis returning a Pydantic model with density, coordination, network connectivity, RDFs, distributions, and element info.

---

```python
find_rdf_minimum(
    r: np.ndarray,
    rdf_data: np.ndarray,
    sigma: float = 2.0,
    window_length: int = 21,
    polyorder: int = 3,
) -> float
```
Detect the first minimum in an RDF curve for automatic cutoff determination.

---

```python
plot_analysis_results_plotly(structure_data: StructureData) -> go.Figure
```
Generate interactive Plotly figure with all analysis results.

---

## Analysis Functions

### `amorphouspy.analysis.radial_distribution_functions`

```python
compute_rdf(
    structure: Atoms,
    r_max: float = 10.0,
    n_bins: int = 500,
    type_pairs: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, dict, dict]
```
Compute partial radial distribution functions and cumulative coordination numbers.

---

```python
compute_coordination(
    structure: Atoms,
    target_type: int,
    cutoff: float,
    neighbor_types: list[int] | None = None,
) -> tuple[dict[int, int], dict[int, int]]
```
Compute coordination distribution and per-atom coordination numbers.

---

### `amorphouspy.analysis.qn_network_connectivity`

```python
compute_qn(
    structure: Atoms,
    cutoff: float,
    former_types: list[int],
    o_type: int,
) -> tuple[dict[int, int], dict[int, dict[int, int]]]
```
Calculate total and partial Qⁿ distributions from bridging oxygen analysis.

---

```python
compute_network_connectivity(qn_dist: dict[int, int]) -> float
```
Average network connectivity: Σ(n × count_n) / Σ(count_n).

---

### `amorphouspy.analysis.bond_angle_distribution`

```python
compute_angles(
    structure: Atoms,
    center_type: int,
    neighbor_type: int,
    cutoff: float,
    bins: int = 180,
) -> tuple[np.ndarray, np.ndarray]
```
Bond angle distribution for neighbor–center–neighbor triplets.

---

### `amorphouspy.analysis.rings`

```python
compute_guttmann_rings(
    structure: Atoms,
    bond_lengths: dict[tuple[str, str], float],
    max_size: int = 24,
    n_cpus: int = 1,
) -> tuple[dict[int, int], float]
```
Guttman ring size distribution and mean ring size via sovapy.

---

```python
generate_bond_length_dict(
    atoms: Atoms,
    specific_cutoffs: dict | None = None,
    default_cutoff: float = -1.0,
) -> dict[tuple[str, str], float]
```
Generate bond length dictionary for all element pairs.

---

### `amorphouspy.analysis.cavities`

```python
compute_cavities(
    structure: Atoms,
    resolution: int = 64,
    cutoff_radii: dict[str, float] | None = None,
) -> dict[str, np.ndarray]
```
Grid-based cavity (void) analysis via sovapy. Returns volumes, surface areas, and shape descriptors.

---

### `amorphouspy.analysis.cte`

```python
cte_from_npt_fluctuations(
    temperature: float | list | np.ndarray,
    enthalpy: list | np.ndarray,
    volume: list | np.ndarray,
    N_points: int = 1000,
    use_running_mean: bool = False,
) -> float
```
CTE from enthalpy-volume cross-correlations in a single NPT run.

---

```python
cte_from_volume_temperature_data(
    temperature: list | np.ndarray,
    volume: list | np.ndarray,
    reference_volume: float | None = None,
) -> tuple[float, float]
```
CTE and R² from linear fit of volume vs. temperature across multiple NPT runs.

---

## Neighbor Search

### `amorphouspy.neighbors`

```python
get_neighbors(
    coords: np.ndarray,
    types: np.ndarray,
    box_size: np.ndarray,
    cutoff: float,
    target_types: list[int] | None = None,
    neighbor_types: list[int] | None = None,
) -> list[list[int]]
```
Cell-list neighbor search with PBC. Returns neighbor index lists for each atom.

---

## Utilities

### `amorphouspy.shared`

```python
count_distribution(coord_numbers: dict[int, int]) -> dict[int, int]
```
Convert per-atom coordination numbers to a frequency histogram.

---

```python
type_to_dict(types: np.array) -> dict[int, str]
```
Map atomic numbers to element symbols.

---

```python
running_mean(data: list | np.ndarray, N: int) -> np.ndarray
```
Calculate running mean of an array-like dataset.

---

### `amorphouspy.mass`

```python
get_atomic_mass(element: str | int) -> float
```
Atomic mass in g/mol from ASE data (IUPAC 2016).

---

### `amorphouspy.io_utils`

```python
get_properties_for_structure_analysis(atoms: Atoms) -> tuple
```
Extract IDs, types, wrapped coordinates, and cell dimensions from an ASE Atoms object.

---

```python
structure_from_parsed_output(
    initial_structure: Atoms,
    parsed_output: dict,
    wrap: bool = False,
) -> Atoms
```
Reconstruct ASE Atoms from LAMMPS parsed output.

---

```python
write_xyz(
    filename: str,
    coords: np.ndarray,
    types: np.ndarray,
    box_size: np.ndarray = None,
    type_dict: dict[int, str] | None = None,
) -> None
```
Write atomic configuration to an XYZ file.

---

```python
write_distribution_to_file(
    composition: float,
    filepath: str,
    dist: dict[int, int],
    label: str,
    *,
    append: bool = False,
) -> None
```
Write a coordination or Qⁿ histogram to a text file.

---

```python
write_angle_distribution(
    bin_centers: np.ndarray,
    angle_hist: np.ndarray,
    composition: float,
    filepath: str,
    *,
    append: bool = False,
) -> None
```
Write angle distribution to a text file.
