# Cavity / Void Analysis

Cavity analysis quantifies the void spaces in glass structures — regions not occupied by atoms. Void volume and connectivity affect properties like gas diffusion, ion transport, and density.

---

## Theory

The analysis identifies void regions in the atomic structure by:

1. **Discretizing** the simulation cell into a fine grid of voxels
2. **Marking** voxels that fall within any atomic radius as "occupied"
3. **Clustering** contiguous unoccupied voxels into distinct cavities
4. **Measuring** the volume of each cavity cluster

This is implemented via the **sovapy** library, which provides an efficient voxel-based cavity detection algorithm.

### Key Metrics

| Metric | Description |
|---|---|
| **Total void fraction** | Fraction of cell volume that is unoccupied |
| **Cavity count** | Number of distinct void regions |
| **Cavity size distribution** | Histogram of individual cavity volumes |
| **Largest cavity** | Volume of the single largest void region |

---

## Usage

### `compute_cavities(structure, grid_resolution)`

```python
from amorphouspy import compute_cavities

cavities = compute_cavities(
    structure=glass_structure,
    grid_resolution=0.2,      # Voxel size in Å
)

print(f"Number of cavities: {cavities['n_cavities']}")
print(f"Total void fraction: {cavities['void_fraction']:.3f}")
print(f"Largest cavity: {cavities['max_volume']:.1f} ų")
print(f"Mean cavity volume: {cavities['mean_volume']:.1f} ų")
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | ASE Atoms object |
| `grid_resolution` | `float` | `0.2` | Voxel edge length (Å). Smaller = more accurate but slower |

**Returns:** A dictionary with:

| Key | Type | Description |
|---|---|---|
| `"n_cavities"` | `int` | Number of distinct void regions |
| `"void_fraction"` | `float` | Total void volume / total cell volume |
| `"volumes"` | `np.ndarray` | Array of individual cavity volumes (ų) |
| `"max_volume"` | `float` | Largest cavity volume (ų) |
| `"mean_volume"` | `float` | Mean cavity volume (ų) |

### Visualization

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=cavities["volumes"],
    nbinsx=50,
    name="Cavity volumes",
))
fig.update_layout(
    xaxis_title="Cavity volume (ų)",
    yaxis_title="Count",
    title="Cavity Size Distribution",
)
fig.show()
```

---

## Interpretation

### Void fraction vs. composition

- **Pure SiO₂**: ~30–35% void fraction (open tetrahedral network)
- **Soda-lime glass**: ~25–30% (modifiers fill voids)
- **Dense borosilicates**: ~20–25%

Higher modifier content generally reduces the void fraction as modifier cations and NBOs fill the network interstices.

### Cavity connectivity

- **Isolated small cavities**: typical in well-packed structures, limit diffusion
- **Percolating void networks**: connected channels that enable ionic transport
- **Large cavities**: may indicate incomplete quenching or structural defects

> **Note:** Cavity analysis requires the **sovapy** package. The grid resolution trades off between accuracy and computational cost. A value of 0.2 Å is a good balance; reducing to 0.1 Å gives better accuracy but is ~8× slower.
