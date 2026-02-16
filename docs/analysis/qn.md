# $Q^n$ Distribution & Network Connectivity

The $Q^n$ distribution characterizes the connectivity of network-forming polyhedra (typically tetrahedra) in oxide glasses by counting bridging oxygens (BOs).

---

## Theory

### $Q^n$ Species

Each network-forming cation (e.g., Si, Al, B in 4-fold coordination) is bonded to a certain number of oxygen atoms. Some of these oxygens are **bridging** (shared with another network former) and some are **non-bridging** (NBOs, bonded to only one former).

The $Q^n$ label indicates the number of bridging oxygens:

| Species | Bridging O | Non-bridging O | Network role |
|---|---|---|---|
| $Q^0$ | 0 | 4 | Isolated tetrahedron |
| $Q^1$ | 1 | 3 | Chain end |
| $Q^2$ | 2 | 2 | Chain middle |
| $Q^3$ | 3 | 1 | Sheet/branching |
| $Q^4$ | 4 | 0 | Fully cross-linked (vitreous silica) |

### Bridging Oxygen Definition

An oxygen is classified as **bridging** if it is bonded to $\geq 2$ network-forming cations (within the specified cutoff distance).

### Network Connectivity

The network connectivity (NC) is the average number of bridging oxygens per network former:

$$
\text{NC} = \frac{\sum_{n=0}^{4} n \cdot f(Q^n)}{\sum_{n=0}^{4} f(Q^n)} = \sum_{n=0}^{4} n \cdot x(Q^n)
$$

where $f(Q^n)$ is the fraction of formers with $n$ bridging oxygens and $x(Q^n)$ is the normalized $Q^n$ distribution.

**Typical values:**
- Pure SiO₂ glass: NC ≈ 4.0 (all $Q^4$)
- 75SiO₂-25Na₂O: NC ≈ 3.0 (mainly $Q^3$ and $Q^4$)
- 50SiO₂-50Na₂O: NC ≈ 2.0 (mainly $Q^2$ and $Q^3$)

The NC provides a single number that correlates with many glass properties including viscosity, fragility, glass transition temperature, and elastic moduli.

---

## Usage

### `compute_qn(structure, cutoff, former_types, o_type)`

```python
from amorphouspy import compute_qn, compute_network_connectivity

# Compute Qn distribution
qn = compute_qn(
    structure=glass_structure,
    cutoff=2.0,                  # Si-O bond cutoff (Å)
    former_types=["Si"],         # Network formers to analyze
    o_type="O",                  # Oxygen symbol
)

# Returns: {'Q0': 0.01, 'Q1': 0.05, 'Q2': 0.15, 'Q3': 0.45, 'Q4': 0.34}

# Compute network connectivity from Qn distribution
nc = compute_network_connectivity(qn)
print(f"Network connectivity: {nc:.2f}")
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | ASE Atoms object |
| `cutoff` | `float` | — | Bond cutoff distance for former-O bonds (Å) |
| `former_types` | `list[str]` | — | Element symbols of network formers (e.g., `["Si", "Al"]`) |
| `o_type` | `str` | `"O"` | Element symbol for oxygen |

**Returns:** A dictionary mapping `"Q0"` through `"Q4"` to fractions (summing to 1.0).

### Multi-former analysis

When multiple network formers are present, the $Q^n$ distribution is computed **separately for each former species**:

```python
qn = compute_qn(
    structure=glass_structure,
    cutoff=2.2,
    former_types=["Si", "Al"],
    o_type="O",
)

# Returns separate distributions:
# {'Si': {'Q0': 0.00, 'Q1': 0.02, 'Q2': 0.10, 'Q3': 0.48, 'Q4': 0.40},
#  'Al': {'Q0': 0.00, 'Q1': 0.05, 'Q2': 0.15, 'Q3': 0.55, 'Q4': 0.25}}
```

> **Note:** The cutoff should be chosen to match the first minimum in the former-O RDF. For Si-O this is typically ~2.0 Å; for Al-O it is ~2.2 Å. Using a single cutoff for multiple formers is an approximation — choose a value that works reasonably for all.

---

## Relationship to Composition

For binary alkali silicate glasses $x\text{M}_2\text{O} \cdot (1-x)\text{SiO}_2$, the theoretical $Q^n$ distribution can be predicted from composition using the **binary model**:

$$
\text{NC} = 4 - \frac{2x}{1-x}
$$

This assumes each M₂O adds one non-bridging oxygen, converting one $Q^n$ → $Q^{n-1}$.

Deviations from this model (measured by the actual $Q^n$ distribution) reveal important structural features:
- **Disproportionation**: $2Q^3 \to Q^2 + Q^4$ indicates local unmixing
- **Aluminum avoidance**: Al prefers $Q^4$ to satisfy Loewenstein's rule
