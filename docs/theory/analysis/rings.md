# Ring Statistics

Ring analysis determines the distribution of closed loops in the atomic network, revealing medium-range order that is invisible to pair correlation functions like the RDF.

---

## Theory

### Guttman Rings

A ring is defined as the **shortest closed path** through the network graph starting from a given atom and returning to it via bonded neighbors. The ring size is counted in terms of the number of **network-forming cation nodes** (not total atoms) in the loop.

For example, in SiO₂:
- A **3-membered ring** consists of 3 Si atoms connected by bridging oxygens (Si-O-Si-O-Si-O-Si, closing back to the start)
- A **6-membered ring** (most common in vitreous silica) consists of 6 Si atoms

The Guttman algorithm finds the shortest path ring for each atom, providing a complete ring size distribution. This implementation uses the **sovapy** library which wraps an efficient C++ graph search.

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

### `compute_guttmann_rings(structure, bond_length_dict)`

```python
from amorphouspy import compute_guttmann_rings, generate_bond_length_dict

# Generate default bond length cutoffs from the structure
bond_lengths = generate_bond_length_dict(glass_structure)
# Returns: {('Si', 'O'): 2.0, ('Al', 'O'): 2.2, ...}

# Compute ring statistics
rings = compute_guttmann_rings(
    structure=glass_structure,
    bond_length_dict=bond_lengths,
)

# Returns ring size distribution
print(rings)
# Example: {3: 0.02, 4: 0.08, 5: 0.22, 6: 0.35, 7: 0.20, 8: 0.10, 9: 0.03}
```

### `generate_bond_length_dict(structure)`

Automatically determines bond length cutoffs for all former-oxygen pairs in the structure, typically using the first minimum of the RDF.

```python
from amorphouspy import generate_bond_length_dict

bond_lengths = generate_bond_length_dict(glass_structure)
# Returns: {('Si', 'O'): 2.0, ('Na', 'O'): 3.0, ...}
```

**Parameters for `compute_guttmann_rings`:**

| Parameter | Type | Description |
|---|---|---|
| `structure` | `Atoms` | ASE Atoms object |
| `bond_length_dict` | `dict[tuple, float]` | Cutoff distances per pair |

**Returns:** A dictionary mapping ring size (int) to fraction of atoms participating in rings of that size.

---

## Typical Results

### Vitreous SiO₂ (MD simulation)

| Ring size | Fraction |
|---|---|
| 3 | ~0.01–0.03 |
| 4 | ~0.05–0.10 |
| 5 | ~0.20–0.25 |
| **6** | **~0.30–0.35** (peak) |
| 7 | ~0.15–0.20 |
| 8 | ~0.05–0.10 |
| 9+ | ~0.02–0.05 |

### Effect of modifiers

Adding network modifiers (Na₂O, CaO) to SiO₂:
- Reduces the average ring size
- Broadens the distribution
- Decreases the 6-membered ring population
- Can increase the fraction of small (3, 4) rings in some compositions

> **Note:** Ring analysis requires the **sovapy** package. If sovapy is not installed, `compute_guttmann_rings` will raise an `ImportError`.
