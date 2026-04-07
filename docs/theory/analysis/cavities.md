# Cavity / Void Analysis

Cavity analysis quantifies the void spaces in glass structures — regions not occupied by atoms. Void volume and connectivity affect properties like gas diffusion, ion transport, and density.

---

## Theory

The implementation follows the domain-based algorithm described in the **pyMolDyn** paper (Meyer et al., *J. Comput. Chem.* **38**, 389–394, 2017).

The analysis pipeline:

1. **Discretize** the simulation cell into a 3D voxel grid. The voxel edge length is derived from the requested resolution: `s = max(L) / (resolution - 4)`.
2. **Mark occupied voxels** — voxels within the (integer-rounded) atomic radius of any atom are labelled as occupied. By default, ASE Van der Waals radii are used, with ASE covalent radii as a fallback for elements without VdW data.
3. **Label connected cavities** — contiguous empty voxels are grouped into distinct cavities using 26-connectivity (face, edge, and corner neighbours). Periodic boundary conditions are handled via a union-find merge of components that appear on opposite faces of the box.
4. **Exclude percolating cavities** — cavities whose voxels span the full simulation cell along any axis are topologically infinite under PBC and are excluded from the output with a `UserWarning`.
5. **Compute cavity properties** for each physical cavity.

### Cavity Properties

| Property | Description |
|---|---|
| **Volume** (Å³) | Number of empty voxels × voxel volume |
| **Surface area** (Å²) | Area of the cavity–solid interface via marching-cubes triangulation |
| **Asphericity** η | Deviation from spherical shape; 0 = sphere, 1 = rod |
| **Acylindricity** *c* | Deviation from cylindrical symmetry; 0 = cylinder |
| **Anisotropy** κ | Combined shape anisotropy; 0 = sphere, 1 = collinear |

Shape descriptors are derived from the eigenvalues λ₁ ≥ λ₂ ≥ λ₃ of the volume-weighted gyration tensor 
$\mathbf{R} = \frac{1}{V_C} \sum_{j} v_j , \mathbf{r}_j \mathbf{r}_j^{\mathrm T}$ computed over all voxels inside the cavity (pyMolDyn definition):

$$\eta = \frac{\lambda_1 - \tfrac{1}{2}(\lambda_2+\lambda_3)}{R_g^2}, \quad
c = \frac{\lambda_2 - \lambda_3}{R_g^2}, \quad
\kappa = \sqrt{\eta^2 + \tfrac{3}{4}c^2}$$

### Atomic Radii

The default uses element-specific ASE Van der Waals radii (e.g. Si = 2.10 Å, O = 1.52 Å, Na = 2.27 Å), which are more physically motivated than a single uniform value. The `cutoff_radii` parameter allows overriding these.

---

## Usage

### `compute_cavities(structure, resolution, cutoff_radii)`

```python
from amorphouspy.analysis.cavities import compute_cavities

result = compute_cavities(
    structure=glass_structure,
    resolution=64,       # voxels along the longest box dimension (before padding)
    cutoff_radii=None,   # None → ASE VdW radii (default)
)

print(f"Number of cavities: {len(result['volumes'])}")
print(f"Largest cavity:     {result['volumes'].max():.2f} Å³")
print(f"Mean cavity volume: {result['volumes'].mean():.2f} Å³")
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | ASE Atoms object |
| `resolution` | `int` | `64` | Voxels along the longest dimension (before 4-voxel padding) |
| `cutoff_radii` | `None`, `float`, or `dict[str, float]` | `None` | Atomic radius specification (see below) |

**`cutoff_radii` modes:**

| Value | Behaviour |
|---|---|
| `None` | Element-specific ASE VdW radii; covalent radii as fallback |
| `2.8` (float) | Uniform 2.8 Å for every atom |
| `{'Si': 2.1, 'O': 1.52}` (dict) | Per-element overrides; missing elements fall back to ASE covalent radii |

**Returns:** A dictionary with one 1D `float64` array per cavity property:

| Key | Description |
|---|---|
| `'volumes'` | Cavity volumes (Å³) |
| `'surface_areas'` | Cavity surface areas (Å²) |
| `'asphericities'` | Asphericity η ∈ [0, 1] |
| `'acylindricities'` | Acylindricity *c* ∈ [0, 1] |
| `'anisotropies'` | Relative shape anisotropy κ ∈ [0, 1] |

### Histogram with plotly

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=result["volumes"],
    nbinsx=50,
    name="Cavity volumes",
))
fig.update_layout(
    xaxis_title="Cavity volume (Å³)",
    yaxis_title="Count",
    title="Cavity Size Distribution",
)
fig.show()
```

### 3D Visualization

```python
from amorphouspy.analysis.cavities import visualize_cavities

fig = visualize_cavities(
    structure=glass_structure,
    resolution=64,
    show_atoms=True,   # overlay atom positions
    opacity=0.4,
)
fig.show()
```

**Parameters for `visualize_cavities`:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | ASE Atoms object |
| `resolution` | `int` | `64` | Same as `compute_cavities` |
| `cutoff_radii` | same as above | `None` | Same as `compute_cavities` |
| `show_atoms` | `bool` | `True` | Overlay atom positions coloured by element |
| `opacity` | `float` | `0.4` | Mesh surface opacity (0–1) |
| `excluded_cavities` | `float` or `None` | `None` | Hide cavities with volume ≥ this value (Å³); `None` shows all |

Percolating cavities are **not** automatically excluded from the 3D plot; use `excluded_cavities` with a large volume threshold to hide them manually. The figure title shows the number of rendered cavities and, if set, the `excluded_cavities` volume threshold.

---

## Interpretation

### Void fraction vs. composition

- **Pure SiO₂**: ~30–35% void fraction (open tetrahedral network)
- **Soda-lime glass**: ~25–30% (modifiers fill voids)
- **Dense borosilicates**: ~20–25%

Higher modifier content generally reduces the void fraction as modifier cations and NBOs fill the network interstices.

### Percolating cavities

A **percolating cavity** spans the entire simulation cell along at least one periodic direction. Such cavities are topologically infinite under PBC (their image wraps back into themselves) and cannot be assigned a meaningful finite volume. They are excluded with a `UserWarning`:

```
UserWarning: 6 percolating cavity/cavities span the simulation cell
and will be excluded from the output. Consider using larger cutoff radii.
```

If you see many percolating cavities, consider increasing `cutoff_radii` or the `resolution`.

### Shape descriptors

| κ value | Interpretation |
|---|---|
| κ ≈ 0 | Spherical cavity |
| η ≫ c | Prolate (rod-like) cavity |
| c ≫ η | Oblate (disc-like) cavity |
| κ ≈ 1 | Highly elongated/anisotropic |

---

## References

Meyer, I. et al. pyMolDyn: Identification, structure, and properties of cavities/vacancies in condensed matter. *J. Comput. Chem.* **38**, 389–394 (2017). <https://doi.org/10.1002/jcc.24697>
