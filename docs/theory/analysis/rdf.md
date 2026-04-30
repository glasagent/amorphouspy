# Radial Distribution Function & Coordination

The radial distribution function (RDF) is the most fundamental structural descriptor for amorphous materials. It describes how atomic density varies as a function of distance from a reference atom.

---

## Theory

### Pair RDF

The partial radial distribution function $g_{\alpha\beta}(r)$ for species $\alpha$ and $\beta$ is defined as:

$$
g_{\alpha\beta}(r) = \frac{V}{4\pi r^2 \Delta r \, N_\alpha N_\beta} \sum_{i \in \alpha} \sum_{j \in \beta} \delta(r - r_{ij})
$$

where $V$ is the volume, $N_\alpha$ and $N_\beta$ are the number of atoms of each species, and $\Delta r$ is the bin width.

In practice, we histogram pair distances into bins and normalize by the ideal gas density:

$$
g_{\alpha\beta}(r) = \frac{n_{\alpha\beta}(r)}{4\pi r^2 \Delta r \, \rho_\beta}
$$

where $n_{\alpha\beta}(r)$ is the number of $\beta$ atoms in the shell $[r, r + \Delta r)$ around each $\alpha$ atom, and $\rho_\beta = N_\beta / V$ is the number density of species $\beta$.

### Interpretation

- $g(r) = 1$ → atoms are at the same density as the average
- $g(r) > 1$ → atoms are more likely to be found at this distance (peaks = coordination shells)
- $g(r) < 1$ → atoms are less likely to be found (troughs = exclusion zones)
- $g(r) \to 1$ as $r \to \infty$ → long-range disorder (amorphous structure confirmed)

### Running Coordination Number

The running coordination number $n_{\alpha\beta}(R)$ counts the average number of $\beta$ atoms within distance $R$ of each $\alpha$ atom:

$$
n_{\alpha\beta}(R) = 4\pi \rho_\beta \int_0^R g_{\alpha\beta}(r) \, r^2 \, dr
$$

The coordination number is usually read at the first minimum of $g(r)$, which corresponds to the boundary between the first and second coordination shells.

---

## Computing RDFs

### `compute_rdf(structure, r_max, n_bins, type_pairs)`

```python
from amorphouspy import compute_rdf

# Compute all partial RDFs up to 8 Å with 500 bins
r, rdfs, cn = compute_rdf(
    structure=glass_structure,
    r_max=8.0,         # Maximum distance (Å)
    n_bins=500,        # Number of radial bins
    type_pairs=None,   # None → all unique pairs
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | ASE Atoms object with periodic boundaries |
| `r_max` | `float` | `10.0` | Maximum distance for RDF calculation (Å) |
| `n_bins` | `int` | `500` | Number of radial bins |
| `type_pairs` | `list[tuple[int, int]]` or `None` | `None` | Specific pairs as atomic numbers, e.g. `[(14, 8), (11, 8)]`. `None` = all pairs |

**Returns:** A tuple `(r, rdfs, cn_cumulative)`:

| Variable | Type | Description |
|---|---|---|
| `r` | `np.ndarray` | Bin center positions (Å) |
| `rdfs` | `dict[tuple[int,int], np.ndarray]` | Partial RDFs keyed by atomic number pair |
| `cn_cumulative` | `dict[tuple[int,int], np.ndarray]` | Running coordination numbers |

### Example: Typical oxide glass RDF analysis

```python
from amorphouspy import compute_rdf
import plotly.graph_objects as go

r, rdfs, cn = compute_rdf(glass_structure, r_max=8.0, n_bins=500)

fig = go.Figure()
for pair, g_r in rdfs.items():
    fig.add_trace(go.Scatter(x=r, y=g_r, name=str(pair)))

fig.update_layout(
    xaxis_title="r (Å)",
    yaxis_title="g(r)",
    title="Partial Radial Distribution Functions",
)
fig.show()

# Access specific pairs by atomic numbers (e.g. Si=14, O=8)
g_SiO = rdfs[(14, 8)]
cn_SiO = cn[(14, 8)]
```

### Typical peak positions for oxide glasses

| Pair | First peak (Å) | Coordination |
|---|---|---|
| Si-O | ~1.62 | 4 (tetrahedral) |
| Al-O | ~1.75 | 4–5 |
| B-O | ~1.37 (III) / ~1.47 (IV) | 3 or 4 |
| Ca-O | ~2.35 | 6–7 |
| Na-O | ~2.30 | 5–6 |
| O-O | ~2.63 | — |

---

## Computing Coordination Numbers

### `compute_coordination(structure, target_type, cutoff, neighbor_types)`

Computes coordination number distribution for atoms of a target type.

```python
from amorphouspy import compute_coordination

# Si (atomic number 14) coordinated by O (atomic number 8), cutoff 2.0 Å
coord_dist, per_atom_cn = compute_coordination(
    glass_structure,
    target_type=14,       # Atomic number of central atom (Si)
    cutoff=2.0,           # Cutoff radius (Å)
    neighbor_types=[8],   # Atomic numbers of neighbors (O); None = all types
)

# coord_dist: {cn: count} e.g. {4: 195, 5: 5}
# per_atom_cn: {atom_id: cn}
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `structure` | `Atoms` | ASE Atoms object |
| `target_type` | `int` | Atomic number of the central atom type |
| `cutoff` | `float` | Cutoff radius in Å |
| `neighbor_types` | `list[int]` or `None` | Atomic numbers of valid neighbors; `None` = all types |

> **`r_max` clamping:** If `r_max` exceeds half the smallest perpendicular cell height, it is automatically reduced to the largest integer that stays within the limit and a `UserWarning` is emitted. To suppress it: `warnings.filterwarnings("ignore", category=UserWarning, module="amorphouspy")`.

> **Tip:** The cutoff should be set at the first minimum of the corresponding partial RDF. Compute the RDF first, then read the minimum position. A common default for Si-O is 2.0 Å and for most modifier-O pairs is 2.8–3.2 Å.
