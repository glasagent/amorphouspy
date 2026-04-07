# Ring Statistics

Ring analysis determines the distribution of closed loops in the atomic network, revealing medium-range order that is invisible to pair correlation functions like the RDF.

---

## Theory

### Guttman Rings

A **Guttman ring** is a closed path through the T-O-T network that satisfies the *primitiveness* (shortest-path) criterion: no shortcut exists through the rest of the network between any two non-adjacent ring nodes. Specifically, for every pair of non-adjacent ring atoms, the shortest path in the full network graph equals their arc distance along the ring.

The ring size is counted in terms of the number of **network-forming cation nodes** (T atoms, e.g. Si, Al) — not total atoms — in the loop.

For example, in SiO₂:
- A **3-membered ring** contains 3 Si atoms connected by bridging oxygens: Si-O-Si-O-Si-O-Si
- A **6-membered ring** (most common in vitreous silica) contains 6 Si atoms

#### Algorithm

The implementation is a pure-Python networkx-based BFS approach:

1. Build a **T-T connectivity graph** where two network formers share an edge if they are both bonded to the same bridging oxygen (coordination ≥ 2).
2. For every edge (u, v): temporarily remove it, find all shortest paths from u back to v, restore the edge.
3. Each candidate ring is tested against the **Guttman primitiveness criterion** using shortest-path lengths in the full graph.
4. Canonical ring forms (rotation- and reflection-invariant) prevent double-counting.

### Physical Significance

Ring statistics connect structure to properties:

| Ring size | Structural feature |
|---|---|
| 3-membered | Associated with the D₂ Raman band (~606 cm⁻¹) in SiO₂ |
| 4-membered | Associated with the D₁ Raman band (~492 cm⁻¹) in SiO₂ |
| 5–7 | Dominant in vitreous silica; peak at 6 |
| Large (>8) | Less strained; common in open network structures |

Small rings (3, 4) are energetically strained but kinetically trapped during the quench. Their population is sensitive to:
- Cooling rate (faster quench → more small rings)
- Composition (modifiers break rings)
- Temperature (high T → more small rings)

---

## Usage

### `compute_guttmann_rings(structure, bond_lengths, max_size)`

```python
from amorphouspy.analysis.rings import compute_guttmann_rings, generate_bond_length_dict

# Generate bond length cutoffs for all element pairs
bond_lengths = generate_bond_length_dict(
    glass_structure,
    specific_cutoffs={('Si', 'O'): 1.8, ('Al', 'O'): 1.95},
    default_cutoff=2.0,
)

# Compute ring statistics
histogram, mean_size = compute_guttmann_rings(
    structure=glass_structure,
    bond_lengths=bond_lengths,
    max_size=12,
)

print(f"Mean ring size: {mean_size:.2f}")
print(histogram)
# Example: {3: 12, 4: 45, 5: 120, 6: 210, 7: 98, 8: 30}
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | ASE Atoms object |
| `bond_lengths` | `dict[tuple[str, str], float]` | — | Cutoff distances per element pair in Å |
| `max_size` | `int` | `24` | Maximum ring size (number of T atoms) to search for |

**Returns:** `(histogram, mean_ring_size)` where:

| Value | Type | Description |
|---|---|---|
| `histogram` | `dict[int, int]` | Mapping from ring size to ring count |
| `mean_ring_size` | `float` | Mean ring size weighted by count |

### `generate_bond_length_dict(atoms, specific_cutoffs, default_cutoff)`

Generates all symmetric element-pair combinations from the structure and assigns cutoff values.

```python
from amorphouspy.analysis.rings import generate_bond_length_dict

bond_lengths = generate_bond_length_dict(
    glass_structure,
    specific_cutoffs={('Si', 'O'): 1.8},
    default_cutoff=-1.0,   # -1.0 marks pairs to ignore (e.g. T-T, O-O)
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `atoms` | `Atoms` | — | ASE Atoms object (determines element set) |
| `specific_cutoffs` | `dict` or `None` | `None` | Per-pair cutoff overrides |
| `default_cutoff` | `float` | `-1.0` | Fallback for unspecified pairs; negative values are ignored by the ring finder |

---

## Typical Results

### Vitreous SiO₂ (MD simulation)

| Ring size | Count (fraction) |
|---|---|
| 3 | ~1–3% |
| 4 | ~5–10% |
| 5 | ~20–25% |
| **6** | **~30–35%** (peak) |
| 7 | ~15–20% |
| 8 | ~5–10% |
| 9+ | ~2–5% |

### Effect of modifiers

Adding network modifiers (Na₂O, CaO) to SiO₂:
- Reduces the average ring size
- Broadens the distribution
- Decreases the 6-membered ring population
- Can increase the fraction of small (3, 4) rings in some compositions

---

## References

Guttman, L. Ring structure of the crystalline and amorphous forms of silicon dioxide.
*J. Non-Cryst. Solids* **116**, 145–147 (1990).
<https://doi.org/10.1016/0022-3093(90)90686-G>
