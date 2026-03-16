# Bond Angle Distribution

Bond angle distributions provide insight into the local polyhedral geometry of network-forming cations and the inter-polyhedral connectivity structure.

---

## Theory

### Intra-polyhedral angles: O–X–O

The O–X–O angle distribution measures the angles between oxygen atoms bonded to the same network-forming cation X. For perfect tetrahedral coordination (e.g., Si in SiO₂), the ideal angle is:

$$
\theta_{\text{tet}} = \arccos\left(-\frac{1}{3}\right) \approx 109.47°
$$

Deviations from this value indicate distorted tetrahedra. The width of the distribution reflects the degree of structural disorder.

Common ideal angles:

| Geometry | Coordination | Ideal angle |
|---|---|---|
| Tetrahedral | 4 | 109.47° |
| Trigonal planar | 3 | 120.00° |
| Octahedral | 6 | 90.00° |

### Inter-polyhedral angles: X–O–X

The X–O–X angle distribution measures the angles at bridging oxygens connecting two network-forming polyhedra. This angle controls the medium-range order and ring topology.

For SiO₂ glass, the Si–O–Si angle distribution is centered around ~144° with a broad range of ~120–180°. This is a key structural parameter that distinguishes glass polymorphs and relates to the density anomaly.

---

## Usage

### `compute_bond_angle_distribution(structure, cutoff, center_types, ligand_type, nbins)`

```python
from amorphouspy import compute_bond_angle_distribution

# O-Si-O angles (intra-tetrahedral)
osi_o = compute_bond_angle_distribution(
    structure=glass_structure,
    cutoff=2.0,
    center_types=["Si"],      # Central atom
    ligand_type="O",           # Ligand atoms
    nbins=180,                 # Number of histogram bins
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | ASE Atoms object |
| `cutoff` | `float` | — | Bond cutoff for center-ligand pairs (Å) |
| `center_types` | `list[str]` | — | Element symbols of central atoms |
| `ligand_type` | `str` | — | Element symbol of ligand atoms |
| `nbins` | `int` | `180` | Number of bins for the angle histogram (1°–180°) |

**Returns:** A dictionary with:

| Key | Type | Description |
|---|---|---|
| `"angles"` | `np.ndarray` | Bin centers in degrees |
| `"counts"` | `np.ndarray` | Histogram counts (or normalized probability) |
| `"mean"` | `float` | Mean angle in degrees |
| `"std"` | `float` | Standard deviation in degrees |

### Example: Complete bond angle analysis

```python
from amorphouspy import compute_bond_angle_distribution
import plotly.graph_objects as go

# Intra-tetrahedral: O-Si-O
o_si_o = compute_bond_angle_distribution(
    glass_structure, cutoff=2.0,
    center_types=["Si"], ligand_type="O",
)

# Inter-tetrahedral: Si-O-Si
si_o_si = compute_bond_angle_distribution(
    glass_structure, cutoff=2.0,
    center_types=["O"], ligand_type="Si",
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=o_si_o["angles"], y=o_si_o["counts"],
    name=f"O-Si-O (mean={o_si_o['mean']:.1f}°)",
))
fig.add_trace(go.Scatter(
    x=si_o_si["angles"], y=si_o_si["counts"],
    name=f"Si-O-Si (mean={si_o_si['mean']:.1f}°)",
))
fig.update_layout(xaxis_title="Angle (°)", yaxis_title="P(θ)")
fig.show()
```

---

## Typical Results for Oxide Glasses

| Angle | Expected peak | Width | Significance |
|---|---|---|---|
| O-Si-O | ~109° | Narrow (~5°) | Nearly perfect tetrahedra |
| O-Al-O | ~109° | Broader (~10°) | More distorted tetrahedra |
| O-B-O (III) | ~120° | Narrow | Trigonal planar BO₃ |
| O-B-O (IV) | ~109° | Moderate | Tetrahedral BO₄ |
| Si-O-Si | ~144° | Very broad (120–180°) | Inter-tetrahedral flexibility |
| Si-O-Al | ~130–140° | Broad | Mixed linkage |

> **Tip:** A bimodal O-B-O distribution (peaks near 109° and 120°) indicates a mix of 3-fold and 4-fold coordinated boron — the ratio relates to the glass composition and modifier content.
