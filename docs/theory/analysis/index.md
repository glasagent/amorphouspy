# Structural Analysis

Tools for computing structural descriptors of amorphous materials. All analysis functions accept ASE `Atoms` objects and handle periodic boundary conditions automatically.

---

## Comprehensive Analysis

### `analyze_structure(structure, ...)`

Runs all available analyses in one call and returns a structured `StructureData` object containing RDF, coordination, $Q^n$, bond angles, ring statistics, and cavity data.

```python
from amorphouspy import analyze_structure
from amorphouspy.workflows.structural_analysis import plot_analysis_results_plotly

data = analyze_structure(glass_structure)

# Access individual results
print(f"Density: {data.density:.3f} g/cm³")
print(f"Network connectivity: {data.network.connectivity:.2f}")

# Generate interactive visualization
fig = plot_analysis_results_plotly(data)
fig.show()
```

The `StructureData` object groups results into logical categories:

| Attribute | Contents |
|---|---|
| `data.rdf` | Partial and total RDFs, distances, pair labels |
| `data.coordination` | Coordination numbers for formers and modifiers |
| `data.network` | $Q^n$ distribution, network connectivity, bridging oxygen fraction |
| `data.angles` | O-X-O and X-O-X bond angle distributions |
| `data.rings` | Ring size statistics (Guttman algorithm) |
| `data.cavities` | Void volumes and size distributions |
| `data.density` | Calculated density in g/cm³ |

### `plot_analysis_results_plotly(data)`

Generates a multi-panel interactive Plotly figure with:

- All partial RDFs $g_{\alpha\beta}(r)$ overlaid
- Coordination numbers table
- $Q^n$ distribution bar chart per former species
- Bond angle histograms for O-X-O and X-O-X
- Ring size distribution histogram
- Cavity volume distribution histogram

---

## Individual Analysis Tools

Each analysis method has its own dedicated page with full parameter documentation, theory background, and code examples:

| Method | Page | Description |
|---|---|---|
| **Radial Distribution Function** | [RDF & Coordination](rdf.md) | $g(r)$, partial RDFs, running coordination numbers |
| **$Q^n$ Distribution** | [$Q^n$ & Network Connectivity](qn.md) | Bridging oxygen analysis, network connectivity |
| **Bond Angles** | [Bond Angle Distribution](bond_angles.md) | O-X-O and X-O-X angle histograms |
| **Ring Statistics** | [Ring Analysis](rings.md) | Guttman ring counting via sovapy |
| **Cavity Analysis** | [Cavity / Void Analysis](cavities.md) | Void volume and size distributions via sovapy |
| **Thermal Expansion** | [CTE Analysis](cte_analysis.md) | From NPT enthalpy-volume fluctuations |

---

## Neighbor Search

All analysis methods depend on neighbor finding. `amorphouspy` provides a cell-list based neighbor search that handles periodic boundary conditions efficiently.

### `compute_neighbors(structure, cutoff)`

Uses a cell-list algorithm (with Numba JIT compilation for performance) to find all pairs of atoms within a given cutoff distance. The algorithm:

1. Divides the simulation box into cells of size ≥ cutoff
2. For each atom, only checks atoms in the same cell and 26 neighboring cells
3. Applies the minimum image convention for periodic boundaries
4. Returns neighbor lists as arrays of pairs and distances

```python
from amorphouspy.neighbors import compute_neighbors

neighbors = compute_neighbors(glass_structure, cutoff=3.0)
# Returns arrays of (i, j, distance) for all pairs within cutoff
```

This scales as $O(N)$ rather than the naïve $O(N^2)$, making it efficient for systems with thousands of atoms.
