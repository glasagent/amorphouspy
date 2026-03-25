# BJP Potential (Bouhadja)

The BJP potential, developed by Bouhadja et al., is a Born-Mayer-Huggins (BMH) force field specifically parameterized for calcium aluminosilicate (CAS) glass systems. It provides accurate structural and thermodynamic properties for Ca-Al-Si-O compositions.

---

## Reference

> Y. Bouhadja, N. Jakse, A. Pasturel. "Structural and dynamic properties of calcium aluminosilicate melts: a molecular dynamics study", *J. Chem. Phys.* **138**, 224510 (2013). [DOI:10.1063/1.4809523](https://doi.org/10.1063/1.4809523)

---

## Functional Form

The total pairwise interaction energy uses the Born-Mayer-Huggins form:

$$
V(r_{ij}) = A_{ij} \, e^{-r_{ij}/\rho_{ij}} - \frac{C_{ij}}{r_{ij}^6} + \frac{D_{ij}}{r_{ij}^8} + \frac{q_i q_j}{r_{ij}}
$$

where:

| Symbol | Description |
|---|---|
| $A_{ij}$ | Repulsion energy prefactor (eV) |
| $\rho_{ij}$ | Repulsion range parameter (Å) |
| $C_{ij}$ | Van der Waals attraction coefficient (eV·Å⁶) |
| $D_{ij}$ | Dipole-quadrupole correction coefficient (eV·Å⁸) |
| $q_i, q_j$ | Partial atomic charges |

The Coulomb interactions are handled within the LAMMPS `born/coul/dsf` pair style with a DSF damping parameter of 0.25 Å⁻¹ and a cutoff of 8.0 Å.

**LAMMPS pair style:** `born/coul/dsf 0.25 8.0`

---

## Charges

All charges are **fixed** at:

| Element | Charge ($e$) |
|---|---|
| O | −1.2 |
| Ca | +1.2 |
| Al | +1.8 |
| Si | +2.4 |

> **Note:** These are the same partial charges as in the PMMCS potential, consistent with the factor-of-0.6 scaling from formal charges.

---

## Supported Elements

BJP supports a focused set of **4 elements**:

| Element | Role in glass |
|---|---|
| Si | Network former |
| Al | Network former / intermediate |
| Ca | Network modifier |
| O | Anion |

This makes BJP **specifically designed for CAS glasses** such as:
- Anorthite (CaAl₂Si₂O₈)
- Gehlenite (Ca₂Al₂SiO₇)
- Wollastonite (CaSiO₃)
- General Ca-Al-Si-O compositions

---

## Usage

```python
from amorphouspy import get_structure_dict, generate_potential

# BJP works only with Ca-Al-Si-O compositions
structure_dict = get_structure_dict(
    {"CaO": 0.25, "Al2O3": 0.25, "SiO2": 0.5},
    target_atoms=3000,
)

potential = generate_potential(structure_dict, potential_type="bjp")
```

### What the generator produces

The BJP generator creates LAMMPS configuration lines that:
1. Define the `born/coul/dsf` pair style (BMH + DSF Coulomb in one style)
2. Set atomic charges
3. Define BMH parameters ($A$, $\rho$, $\sigma$, $C$, $D$) for all pair interactions

---

## Technical Details

### Born-Mayer-Huggins vs. Buckingham

The BMH form used by BJP differs from the standard Buckingham potential in two ways:
- It includes a **$D/r^8$ dipole-quadrupole correction** term
- The repulsion uses the form $A \exp(-r/\rho)$ rather than $A \exp(-Br)$

These extra terms provide better reproduction of melt structure and dynamics for CAS systems.

### When to use BJP

- **Pure CAS glass studies** — Ca-Al-Si-O compositions exclusively
- **Structural properties** — well-validated for RDFs, coordination, and Qⁿ distributions in CAS melts
- **High-temperature melt dynamics** — parameterized against ab initio MD of CAS melts

### Limitations

- **Only 4 elements** — cannot model glasses with Na, K, Mg, B, or other modifiers
- Adding additional elements requires reparameterization
- Not suitable for borosilicate or alkali silicate glasses
