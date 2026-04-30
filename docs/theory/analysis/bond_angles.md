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

### `compute_angles(structure, center_type, neighbor_type, cutoff, bins)`

```python
from amorphouspy import compute_angles

# O-Si-O angles (intra-tetrahedral): center=Si (14), neighbor=O (8)
bin_centers, angle_hist = compute_angles(
    structure=glass_structure,
    center_type=14,    # Atomic number of central atom (Si)
    neighbor_type=8,   # Atomic number of ligand atoms (O)
    cutoff=2.0,        # Bond cutoff (Å)
    bins=180,          # Number of histogram bins
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | ASE Atoms object |
| `center_type` | `int` | — | Atomic number of the central atom |
| `neighbor_type` | `int` | — | Atomic number of the ligand atoms |
| `cutoff` | `float` | — | Bond cutoff for center-neighbor pairs (Å) |
| `bins` | `int` | `180` | Number of bins for the angle histogram (1°–180°) |

**Returns:** A tuple `(bin_centers, angle_hist)`:

| Variable | Type | Description |
|---|---|---|
| `bin_centers` | `np.ndarray` | Bin centers in degrees |
| `angle_hist` | `np.ndarray` | Normalized angle histogram |

### Example: Complete bond angle analysis

```python
from amorphouspy import compute_angles
import plotly.graph_objects as go

# Intra-tetrahedral: O-Si-O (center=Si=14, neighbor=O=8)
o_si_o_bins, o_si_o_hist = compute_angles(glass_structure, center_type=14, neighbor_type=8, cutoff=2.0)

# Inter-tetrahedral: Si-O-Si (center=O=8, neighbor=Si=14)
si_o_si_bins, si_o_si_hist = compute_angles(glass_structure, center_type=8, neighbor_type=14, cutoff=2.0)

fig = go.Figure()
fig.add_trace(go.Scatter(x=o_si_o_bins, y=o_si_o_hist, name="O-Si-O"))
fig.add_trace(go.Scatter(x=si_o_si_bins, y=si_o_si_hist, name="Si-O-Si"))
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
