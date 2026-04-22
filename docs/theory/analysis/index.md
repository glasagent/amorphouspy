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
| `data.density` | Calculated density in g/cm³ |
| `data.rdfs` | Partial RDFs, distances, cumulative coordination numbers |
| `data.coordination` | Coordination distributions for oxygen, formers, and modifiers |
| `data.network` | $Q^n$ distribution, partial $Q^n$ by former species, network connectivity |
| `data.distributions` | Bond angle histograms (`bond_angles`), ring statistics (`rings`) |
| `data.structure_factor` | Neutron and X-ray $S(q)$, partial $S_{\alpha\beta}(q)$ |
| `data.elements` | Classified formers, modifiers, cutoffs, oxygen class counts/IDs |

---

## Element Classification

`analyze_structure` automatically classifies every element in the structure into one of three roles using the sets defined in `structural_analysis.py`:

| Role | Elements | Analysis treatment |
|---|---|---|
| **Formers** | Si, B, P, Ge, As, Sb, Te, V | Network-forming cations: coordination, $Q^n$, bond angles, rings, Former-O RDF panel |
| **Intermediates** | Al, Ti, Zr, Be, Zn, Pb, Bi, Nb, Ta, W, Mo, Ga, In, Sn, Fe, Cr | Treated as formers in all analysis |
| **Modifiers** | Li, Na, K, Rb, Cs, Mg, Ca, Sr, Ba, La, Y, Cd, Tl | Coordination only, Modifier-O RDF panel |

Oxygen is handled separately. Any element not in the above sets triggers a `warnings.warn` and falls back to the modifier role.

The resolved classification is available on the result:

```python
data = analyze_structure(glass_structure)
print(data.elements.formers)    # e.g. ["Si", "Al"]
print(data.elements.modifiers)  # e.g. ["Na", "Ca"]
print(data.elements.cutoffs)    # e.g. {"Si": 2.0, "Al": 2.1, "Na": 3.4, "O": 1.9}
```

---

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
