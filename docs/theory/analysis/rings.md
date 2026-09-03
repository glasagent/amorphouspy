# Ring Statistics

Ring analysis determines the distribution of closed loops in the atomic network, revealing medium-range order that is invisible to pair correlation functions like the RDF.

---

## Theory

### Guttman Rings

A **Guttman ring** is defined per bond: the ring belonging to a given T-O bond is the *shortest closed path* that contains it. This is the shortest-path criterion, and it is weaker than the *primitive* (King/Franzblau) criterion — a Guttman ring may still have a shortcut across it, so the two definitions do not select the same set of rings.

The ring size is counted in terms of the number of **network-forming cation nodes** (T atoms, e.g. Si, Al) — not total atoms — in the loop. Every ring alternates T and O, so an *n*-membered ring contains 2*n* atoms.

For example, in SiO₂:
- A **2-membered ring** is two Si sharing a polyhedron edge: Si-O-Si-O, two tetrahedra bridged by two oxygens
- A **3-membered ring** contains 3 Si atoms connected by bridging oxygens: Si-O-Si-O-Si-O
- A **6-membered ring** (most common in vitreous silica) contains 6 Si atoms

#### Algorithm

The search runs on the **bipartite T-O atom network** — formers and oxygens are both nodes, and the only edges are T-O bonds:

1. Build the T-O network from the neighbour list, keeping the minimum-image vector of every bond. Nodes with fewer than two bonds are stripped iteratively, which removes non-bridging oxygens and dangling formers before any search starts.
2. For every bond (u, v): suppress it and sweep breadth-first from u, stopping as soon as the level containing v is complete. Every shortest path found closes the bond into a candidate ring.
3. Keep only candidates that **close in real space**: the minimum-image bond vectors summed around the loop must vanish. A path that leaves the cell and re-enters through the opposite face returns to the same atom index but is a helix, not a ring. If every shortest closure through a bond fails this test, the search deepens by two atoms at a time until it finds the shortest one that does close.
4. Canonical ring forms (rotation- and reflection-invariant, over the full T-O cycle) prevent double-counting.

Working at the atom level rather than on a contracted T-T graph matters physically. An oxygen bonded to three or more formers — a tricluster, common in aluminosilicates and borosilicates — is a single node, so it cannot be traversed twice and does not masquerade as a small ring. Two formers bridged by two distinct oxygens form a genuine four-node cycle, so edge-sharing polyhedra are detected instead of collapsing onto one edge.

### Physical Significance

Ring statistics connect structure to properties:

| Ring size | Structural feature |
|---|---|
| 2-membered | Edge-sharing polyhedra — two formers bridged by two oxygens. Rare in equilibrium silicates; a marker of strongly non-equilibrium or high-pressure structures |
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
from amorphouspy.properties.structural.rings import compute_guttmann_rings, generate_bond_length_dict

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
| `max_size` | `int` | `24` | Maximum ring size (number of T atoms) to search for. It bounds the breadth-first sweep itself, so raising it costs almost nothing — the sweep still stops at each ring's own radius |
| `n_cpus` | `int` | `1` | Worker processes. Only networks above ~100 000 T-O bonds benefit; below that the search stays sequential regardless, because starting workers costs more than it saves |

**Returns:** `(histogram, mean_ring_size)` where:

| Value | Type | Description |
|---|---|---|
| `histogram` | `dict[int, float]` | Mapping from ring size to ring count. The smallest reportable size is 2 |
| `mean_ring_size` | `float` | Mean ring size weighted by count |

### `generate_bond_length_dict(atoms, specific_cutoffs, default_cutoff)`

Generates all symmetric element-pair combinations from the structure and assigns cutoff values.

```python
from amorphouspy.properties.structural.rings import generate_bond_length_dict

bond_lengths = generate_bond_length_dict(
    glass_structure,
    specific_cutoffs={('Si', 'O'): 1.8},
    default_cutoff=0.0,   # 0.0 (the default) marks pairs as never bonded, e.g. T-T and O-O
)
```

A cutoff of `0.0` means "these two species never bond", so only the pairs named in
`specific_cutoffs` can form an edge in the ring graph. Any non-positive value is treated
the same way, so an explicit `-1.0` still works, but `0.0` is preferred: it reads as a
zero bonding radius rather than a negative distance.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `atoms` | `Atoms` | — | ASE Atoms object (determines element set) |
| `specific_cutoffs` | `dict` or `None` | `None` | Per-pair cutoff overrides |
| `default_cutoff` | `float` | `0.0` | Fallback for unspecified pairs. Non-positive values mark a pair as never bonded, so the default excludes every pair not named in `specific_cutoffs` |

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
