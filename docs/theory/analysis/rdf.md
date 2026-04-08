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

### `compute_rdf(structure, r_max, dr, pairs)`

```python
from amorphouspy import compute_rdf

# Compute all partial RDFs up to 8 Å with 0.02 Å bin width
rdfs = compute_rdf(
    structure=glass_structure,
    r_max=8.0,       # Maximum distance (Å)
    dr=0.02,         # Bin width (Å)
    pairs=None,      # None → all unique pairs
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | ASE Atoms object with periodic boundaries |
| `r_max` | `float` | `8.0` | Maximum distance for RDF calculation (Å) |
| `dr` | `float` | `0.02` | Bin width (Å) — smaller = smoother RDF |
| `pairs` | `list[tuple]` or `None` | `None` | Specific element pairs, e.g. `[("Si", "O"), ("Na", "O")]`. `None` = all pairs |

**Returns:** A dictionary with:

| Key | Type | Description |
|---|---|---|
| `"r"` | `np.ndarray` | Bin center positions (Å) |
| `"g_r"` | `dict[str, np.ndarray]` | Partial RDFs keyed by pair label (e.g. `"Si-O"`) |
| `"n_r"` | `dict[str, np.ndarray]` | Running coordination numbers |

### Example: Typical oxide glass RDF analysis

```python
from amorphouspy import compute_rdf
import plotly.graph_objects as go

rdfs = compute_rdf(glass_structure, r_max=8.0, dr=0.02)

fig = go.Figure()
for pair, g_r in rdfs["g_r"].items():
    fig.add_trace(go.Scatter(x=rdfs["r"], y=g_r, name=pair))

fig.update_layout(
    xaxis_title="r (Å)",
    yaxis_title="g(r)",
    title="Partial Radial Distribution Functions",
)
fig.show()
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

### `compute_coordination(structure, cutoff_dict)`

Extracts integer coordination numbers from neighbor lists using element-pair-specific cutoff distances.

```python
from amorphouspy import compute_coordination

# Define cutoffs for each pair (first minimum of g(r))
cutoffs = {
    ("Si", "O"): 2.0,   # Å
    ("Al", "O"): 2.2,
    ("Na", "O"): 3.0,
    ("Ca", "O"): 3.0,
}

coord = compute_coordination(glass_structure, cutoff_dict=cutoffs)

# Returns average coordination numbers
print(f"Si-O coordination: {coord['Si-O']:.2f}")  # Should be ~4.0
print(f"Al-O coordination: {coord['Al-O']:.2f}")  # Should be ~4.0–5.0
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `structure` | `Atoms` | ASE Atoms object |
| `cutoff_dict` | `dict[tuple, float]` | Cutoff distance for each (element1, element2) pair |

> **`r_max` clamping:** If `r_max` exceeds half the smallest perpendicular cell height, it is automatically reduced to the largest integer that stays within the limit and a `UserWarning` is emitted. To suppress it: `warnings.filterwarnings("ignore", category=UserWarning, module="amorphouspy")`.

> **Tip:** The cutoff should be set at the first minimum of the corresponding partial RDF. Compute the RDF first, then read the minimum position. A common default for Si-O is 2.0 Å and for most modifier-O pairs is 2.8–3.2 Å.
