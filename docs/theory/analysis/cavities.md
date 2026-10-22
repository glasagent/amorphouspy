# Cavity / Void Analysis

Cavity analysis quantifies the void spaces in glass structures — regions not occupied by atoms. Void volume and connectivity affect properties like gas diffusion, ion transport, and density.

---

## Theory

The implementation follows the domain-based algorithm described in the **pyMolDyn** paper (Meyer et al., *J. Comput. Chem.* **38**, 389–394, 2017). It does not reproduce pyMolDyn's or sovapy's numbers: those round atom positions and radii to whole voxels, and pyMolDyn also rounds the periodic cell vectors to whole voxels, whereas amorphouspy tests every voxel against the true atom positions and radii and wraps the periodic boundaries exactly.

The analysis pipeline:

1. **Discretize** the simulation cell into a periodic 3D voxel grid that spans exactly one cell (no padding). The voxel edge length is derived from the requested resolution: `s = max(L) / resolution`.
2. **Mark occupied voxels** — a voxel is occupied when its centre lies within the atomic radius of any atom, measured from the atom's true position across the periodic boundaries; the same rule is used for orthogonal and triclinic cells. By default, ASE Van der Waals radii are used, with ASE covalent radii as a fallback for elements without VdW data. Because atoms are not snapped to the grid and radii are not rounded, the free volume is unbiased at any resolution. Cavity counts, percolation and surface areas converge more slowly — channels and crevices narrower than about one voxel are not resolved — so check them at two resolutions.
3. **Label connected cavities** — contiguous empty voxels are grouped into distinct cavities using 26-connectivity (face, edge, and corner neighbours). Components that touch across opposite cell faces (including edges and corners) are merged with a union-find that tracks which periodic image of each component is used, so a cavity is found whole wherever the cell boundary cuts it.
4. **Flag percolating cavities** — a cavity that connects to its own periodic image wraps around the cell and is infinite under PBC. It is kept (its volume and surface area per cell are well defined), flagged in `'percolating'`, and its shape descriptors are set to NaN.
5. **Compute cavity properties** for each cavity. Boundary-crossing cavities are unwrapped before computing shape descriptors.

Every empty voxel belongs to exactly one cavity, so `result['volumes'].sum() / structure.get_volume()` is the total free-volume fraction of the cell.

### Cavity Properties

| Property | Description |
|---|---|
| **Volume** (Å³) | Number of empty voxels × voxel volume |
| **Surface area** (Å²) | Area of the exposed atom-sphere patches that bound the cavity (see below) |
| **Asphericity** η | Deviation from spherical shape; 0 = sphere, 1 = rod |
| **Acylindricity** *c* | Deviation from cylindrical symmetry; 0 = cylinder |
| **Anisotropy** κ | Combined shape anisotropy; 0 = sphere, 1 = collinear |

Shape descriptors are derived from the eigenvalues λ₁ ≥ λ₂ ≥ λ₃ of the volume-weighted gyration tensor 
$\mathbf{R} = \frac{1}{V_C} \sum_{j} v_j , \mathbf{r}_j \mathbf{r}_j^{\mathrm T}$ computed over all voxels inside the cavity (pyMolDyn definition):

$$\eta = \frac{\lambda_1 - \tfrac{1}{2}(\lambda_2+\lambda_3)}{R_g^2}, \quad
c = \frac{\lambda_2 - \lambda_3}{R_g^2}, \quad
\kappa = \sqrt{\eta^2 + \tfrac{3}{4}c^2}$$

### Surface area

The boundary of the empty space consists of the parts of the atom spheres that no other atom covers. Each atom carries 20 Shrake–Rupley points per Å² of its sphere; every atom's point set gets its own random rotation from a fixed seed, so results are reproducible. A fixed orientation aliases with regular atom arrangements: −1.7 % on a simple-cubic lattice at 10 points/Å². Points inside another atom (periodic images included) are discarded. Each remaining point is assigned to the cavity of the nearest empty voxel within 3 voxels along each axis, and each point carries 4πr²/(number of points on its atom).

Measured accuracy of the summed area against analytic geometries (`amorphouspy/src/tests/test_cavities.py`):

| Geometry | Exact area (Å²) | Error |
|---|---|---|
| Single sphere, r = 3 Å | 113.10 | 0 (all points exposed) |
| Two overlapping spheres, r = 1.5 and 2.0 Å, d = 2.5 Å | 67.86 | +0.04 % |
| One atom, r = 2.5 Å, in a 4 Å cell (six overlapping self-images) | 31.42 | +0.10 % |
| Simple-cubic lattice, a = 3 Å, r = 1.8 Å, 125 atoms (cubic and triclinic cell, resolution 32 and 64) | 2544.69 | −0.035 % |

The resolution decides which cavity a patch belongs to and affects the total only through the dropped area. A point with no empty voxel within 3 voxels borders a crevice the grid does not resolve, and its area is dropped. For the 10,020-atom tutorial glass that is 0.002 % of the total at resolution 64, 0.07 % at 128 and 0.05 % at 256.

Per-cavity values carry point-sampling noise. On the tutorial glass at resolution 128, compared with an 80 points/Å² reference, the 90th-percentile error is 19 % for cavities below 2 Å², 7 % for 2–10 Å², 2.7 % for 10–50 Å² and 1.1 % above 50 Å².

Marching cubes is used only to draw cavities (`visualize_cavities`). As an area estimator it fails on glasses: necks and crevices thinner than a voxel are missed. On the tutorial glass at resolution 128, marching cubes on the distance field underestimates the total by 25 % and small cavities by up to 100 %. Marching cubes on the binary voxel mask also overestimates smooth surfaces by about 8 % at any resolution.

### Resolution dependence and limitations

- **Free volume** is unbiased at any resolution, and its scatter shrinks as resolution grows. On the tutorial glass it is within 0.035 percentage points of a Monte Carlo reference from resolution 40 to 256, in both cubic and sheared cells.
- **Cavity count, percolation and the split of volume and area between cavities** depend on resolution. A channel percolates only if voxel centres fall inside its narrowest neck. A neck narrower than half a voxel diagonal may connect or not depending on a sub-voxel shift of the atoms. With resolution 64 on a 20 Å cell this happened for a 0.1 Å neck. Compare results at two resolutions.
- **Small cavities** (a few voxels) are resolution artefacts as much as physical voids, and their surface areas carry the sampling noise quoted above.
- **Runtime** at resolution 128 is about 8 s for 10,020 atoms with VdW radii: the grid, the surface points and the largest-included-sphere search (which rebuilds the distance field with a larger cap) take a few seconds each. Resolution 256 takes about 40 s.

### Probe-accessible volume (e.g. hydrogen)

To find the space that can hold at least one atom or molecule of radius r_p, pass `probe_radius=r_p`. For example, r_p = 1.20 Å is the Van der Waals radius of a neutral H atom. Every atom radius is then increased by r_p, so each cavity is a region where the probe **centre** fits:

- every cavity holds at least one probe;
- a percolating cavity lets the probe cross the cell;
- `volumes` and `surface_areas` describe the space of the probe centre (accessible volume and accessible surface area, as in Zeo++);
- `max_inscribed_radii` is the largest sphere that fits in each cavity, measured with the original radii.

A cavity found without a probe can hold the probe if its `max_inscribed_radii` ≥ r_p.

```python
result = compute_cavities(glass_structure, cutoff_radii="ionic", probe_radius=1.20)
print(f"H sites: {len(result['volumes'])}, percolating: {result['percolating'].any()}")
print(f"Accessible volume: {result['volumes'].sum() / glass_structure.get_volume():.4%}")
```

The result depends on the atom radii far more than on anything else. The table shows the tutorial glass (`notebooks/data/SiONa_25.extxyz`, 10,020 atoms). The Monte Carlo reference uses 2,000,000 random points (seed 2024) with the same inflated radii.

| Atom radii | r_p (Å) | Accessible volume, res 128 / 256 | Monte Carlo | Sites (percolating), res 128 / 256 | Largest included sphere |
|---|---|---|---|---|---|
| ASE VdW (O 1.52, Si 2.10, Na 2.27 Å) | 0 | 6.598 / 6.596 % | 6.618 ± 0.018 % | 829 (1) / 970 (1) | 1.33 Å |
| ASE VdW | 1.20 | 0.0001 / 0.0002 % | 0.0003 ± 0.0001 % | 1 / 5 | 1.33 Å |
| Shannon ionic (O 1.35, Si 0.26, Na 1.02 Å) | 0 | 49.65 / 49.64 % | 49.62 ± 0.04 % | 6 (1) / 1 (1) | 1.78 / 1.83 Å |
| Shannon ionic | 1.20 | 0.0793 / 0.0798 % | 0.0795 ± 0.0020 % | 180 / 236 | 1.78 / 1.83 Å |

The accessible volume converges with resolution. The number of sites does not converge in dense glasses: most H sites are pockets smaller than a voxel. With ionic radii and r_p = 1.20 Å there are 180, 215, 236 and 301 sites at resolutions 128, 192, 256 and 384. So treat site counts as lower bounds, and report the accessible volume and the largest included sphere instead. The largest included sphere is sampled at voxel centres, so it is low by at most half a voxel diagonal.

The volume that the probe *body* can fill (accessible region dilated by r_p) is not reported. On the grid it was dominated by the sub-voxel pockets: 2.2 % at resolution 128 and 3.6 % at resolution 384 for the ionic case, still rising.

`cutoff_radii="ionic"` uses Shannon (1976) ionic radii from pymatgen. It assumes two-coordinated O²⁻, tetrahedral Si, B, Al, P, Ge, Zn and Li, and octahedral Ti, Na, K, Mg, Ca, Sr and Ba. Other elements raise `ValueError`; pass a dict for them or for other coordinations.

### Domains vs. pyMolDyn's cavity definitions

pyMolDyn reports three kinds of cavity:

- **Domains** — connected empty space farther than r from every atom centre. This is what amorphouspy computes, without pyMolDyn's rounding of positions, radii and cell vectors to whole voxels.
- **Center-based cavities** — each domain extended to the points inside the atom spheres that are nearer to the domain's centre than to the nearest atom.
- **Surface-based cavities** — each domain extended to the points inside the atom spheres that are nearer to a point on the domain's surface than to the nearest atom. Domains whose extensions overlap are merged into multicavities.

Center-based and surface-based cavities are larger than domains and need no atomic radius for the extended part. They are not implemented here, so amorphouspy values are not comparable to pyMolDyn's center- or surface-based results.

### Atomic Radii

The default uses element-specific ASE Van der Waals radii (e.g. Si = 2.10 Å, O = 1.52 Å, Na = 2.27 Å), which are more physically motivated than a single uniform value. The `cutoff_radii` parameter allows overriding these.

---

## Usage

### `compute_cavities(structure, resolution, cutoff_radii)`

```python
from amorphouspy.properties.structural.cavities import compute_cavities

result = compute_cavities(
    structure=glass_structure,
    resolution=128,      # voxels along the longest box dimension
    cutoff_radii=None,   # None → ASE VdW radii (default)
)

print(f"Number of cavities: {len(result['volumes'])}")
print(f"Largest cavity:     {result['volumes'].max():.2f} Å³")
print(f"Mean cavity volume: {result['volumes'].mean():.2f} Å³")
print(f"Free-volume fraction: {result['volumes'].sum() / glass_structure.get_volume():.3%}")
print(f"Percolating cavities: {result['percolating'].sum()}")
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | ASE Atoms object |
| `resolution` | `int` | `128` | Voxels along the longest dimension of the cell |
| `cutoff_radii` | `None`, number, `"ionic"`, or `dict[str, float]` | `None` | Atomic radius specification (see below) |
| `probe_radius` | `float` | `0.0` | Probe sphere radius in Å added to every atom radius (e.g. 1.20 for H) |

**`cutoff_radii` modes:**

| Value | Behaviour |
|---|---|
| `None` | Element-specific ASE VdW radii; covalent radii as fallback |
| `"ionic"` | Shannon ionic radii (see *Probe-accessible volume*) |
| `2.8` or `3` (int or float) | Uniform radius in Å for every atom (a `bool` raises `TypeError`) |
| `{'Si': 2.1, 'O': 1.52}` (dict) | Per-element overrides; missing elements fall back to ASE covalent radii |

**Returns:** A dictionary with one 1D array per cavity property (one entry per cavity):

| Key | Description |
|---|---|
| `'volumes'` | Cavity volumes (Å³); per cell for percolating cavities |
| `'surface_areas'` | Exposed atom-sphere area bounding each cavity (Å²); per cell for percolating cavities |
| `'asphericities'` | Asphericity η ∈ [0, 1]; NaN for percolating cavities |
| `'acylindricities'` | Acylindricity *c* ∈ [0, 1]; NaN for percolating cavities |
| `'anisotropies'` | Relative shape anisotropy κ ∈ [0, 1]; NaN for percolating cavities |
| `'percolating'` | `bool`; True for cavities that wrap around the periodic cell |
| `'max_inscribed_radii'` | Radius of the largest sphere that fits in the cavity (Å), sampled at voxel centres |

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
from amorphouspy.properties.structural.cavities import visualize_cavities

fig = visualize_cavities(
    structure=glass_structure,
    resolution=128,
    show_atoms=True,   # overlay atom positions
    opacity=0.4,
)
fig.show()
```

**Parameters for `visualize_cavities`:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | ASE Atoms object |
| `resolution` | `int` | `128` | Same as `compute_cavities` |
| `cutoff_radii` | same as above | `None` | Same as `compute_cavities` |
| `show_atoms` | `bool` | `True` | Overlay atom positions coloured by element |
| `opacity` | `float` | `0.4` | Mesh surface opacity (0–1) |
| `excluded_cavities` | `float` or `None` | `None` | Hide cavities with volume ≥ this value (Å³); `None` shows all |
| `probe_radius` | `float` | `0.0` | Same as `compute_cavities`; surfaces then enclose the space of the probe centre |

Surfaces are drawn with marching cubes on the distance to the nearest atom surface. Percolating cavities are drawn over exactly one cell, as their in-cell fragments; all other cavities are drawn as one contiguous surface, even when they cross the cell boundary. Use `excluded_cavities` with a large volume threshold to hide large (e.g. percolating) cavities. The figure title shows the number of rendered cavities and, if set, the `excluded_cavities` volume threshold.

---

## Interpretation

### Void fraction vs. composition

- **Pure SiO₂**: ~30–35% void fraction (open tetrahedral network)
- **Soda-lime glass**: ~25–30% (modifiers fill voids)
- **Dense borosilicates**: ~20–25%

Higher modifier content generally reduces the void fraction as modifier cations and NBOs fill the network interstices.

### Percolating cavities

A **percolating cavity** connects to its own periodic image, i.e. it wraps around the cell along at least one periodic direction and forms an infinite channel network under PBC. A cavity that merely crosses a cell face is *not* percolating. Percolating cavities are kept in the output with their volume and surface area per cell and flagged in `result['percolating']`; their shape descriptors are NaN because a gyration tensor is undefined for an infinite object.

In open glass networks with small radii, most of the empty space can form a single percolating cavity. Report the total free-volume fraction (`volumes.sum()`) and, if needed, the enclosed part (`volumes[~percolating].sum()`) separately.

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
