# SHIK Potential (Sundararaman)

The SHIK potential is a Buckingham-type force field with an additional $r^{-24}$ repulsive term, developed through a series of papers by Sundararaman et al. Its distinguishing feature is **composition-dependent oxygen charges**, which improves accuracy for mixed-modifier glass systems.

---

## References

The SHIK potential was developed incrementally across several publications:

1. S. Sundararaman, L. Huang, S. Ispas, W. Kob. "New optimization scheme to obtain interaction potentials for oxide glasses", *J. Chem. Phys.* **148**, 194504 (2018) — **Silica**
2. S. Sundararaman, L. Huang, S. Ispas, W. Kob. "New interaction potentials for alkali and alkaline-earth aluminosilicate glasses", *J. Chem. Phys.* **150**, 154505 (2019) — **Alkali/AE aluminosilicates**
3. S. Sundararaman, L. Huang, S. Ispas, W. Kob. "New interaction potentials for borate glasses with mixed network formers", *J. Chem. Phys.* **152**, 104501 (2020) — **Borates**
4. S. Shih, L. Huang, S. Ispas, W. Kob. "Interaction potentials for alkaline earth silicate and borate glasses", *J. Non-Cryst. Solids* **565**, 120853 (2021) — **AE silicates/borates**

---

## Functional Form

The total pairwise interaction energy is:

$$
V(r_{ij}) = A_{ij} \, e^{-B_{ij} \, r_{ij}} - \frac{C_{ij}}{r_{ij}^6} + \frac{D_{ij}}{r_{ij}^{24}} + \frac{q_i q_j}{r_{ij}}
$$

where:

| Symbol | Description |
|---|---|
| $A_{ij}$ | Repulsion prefactor (eV) |
| $B_{ij}$ | Repulsion decay rate (Å⁻¹) |
| $C_{ij}$ | Dispersion coefficient (eV·Å⁶) |
| $D_{ij}$ | Short-range repulsive wall coefficient (eV·Å²⁴) |
| $q_i, q_j$ | Partial atomic charges |

### The $r^{-24}$ term

The $D/r^{24}$ term is a **steep repulsive wall** that prevents atoms from approaching too closely. This is much steeper than the typical $r^{-12}$ Lennard-Jones repulsion and was specifically optimized to reproduce ab initio RDFs at short distances. It improves the description of the first coordination shell while having virtually no effect beyond ~2 Å.

The Coulomb term uses DSF with a damping parameter of 0.2 Å⁻¹ and a cutoff of 10.0 Å.

**LAMMPS pair style:** `hybrid/overlay coul/dsf 0.2 10.0 table spline 10000`

---

## Composition-Dependent Charges

Unlike PMMCS and BJP where the oxygen charge is fixed at −1.2$e$, the SHIK potential computes the oxygen charge from charge neutrality:

$$
q_{\text{O}} = -\frac{\sum_{X \neq \text{O}} q_X \cdot N_X}{N_{\text{O}}}
$$

where $q_X$ are the fixed cation charges and $N_X$, $N_{\text{O}}$ are atom counts.

**Fixed cation charges:**

| Element | Charge ($e$) |
|---|---|
| Li | +0.5727 |
| Na | +0.6018 |
| K | +0.6849 |
| Mg | +1.0850 |
| Ca | +1.4977 |
| B | +1.6126 |
| Al | +1.6334 |
| Si | +1.7755 |

> **Key insight:** Unlike PMMCS which uses simple multiples of 0.6$e$, the SHIK charges are individually fitted values. This allows each element to have a more physically accurate charge. The oxygen charge is then computed from global charge neutrality, capturing the effect that the electron density around oxygen atoms changes with the local cation environment.

**Example:** For a SiO₂ glass (750 Si, 1500 O):

$$
q_{\text{O}} = -\frac{1.7755 \times 750}{1500} = -0.8878 \, e
$$

For a CaO-Al₂O₃-SiO₂ glass, the oxygen charge will differ because different cations contribute different charge densities per oxygen atom.

---

## Supported Elements

| Element | Role | Charge ($e$) |
|---|---|---|
| Li | Alkali modifier | +0.5727 |
| Na | Alkali modifier | +0.6018 |
| K | Alkali modifier | +0.6849 |
| Mg | Alkaline earth modifier | +1.0850 |
| Ca | Alkaline earth modifier | +1.4977 |
| B | Network former | +1.6126 |
| Al | Intermediate / former | +1.6334 |
| Si | Network former | +1.7755 |
| O | Anion | Composition-dependent |

---

## Tabulated Potentials

Because LAMMPS does not have a built-in pair style for the Buckingham + $r^{-24}$ form, the SHIK potential is implemented as **tabulated pair interactions**. The generator:

1. Evaluates $V(r)$ and $F(r)$ on a fine grid of 50,000 points (using $r^2$ spacing from 0.1–10.5 Å)
2. Writes LAMMPS table files to the specified output directory
3. References these files with absolute paths in the pair style configuration

The table files are automatically created when you call `generate_shik_potential()`.

---

## Usage

```python
from amorphouspy import get_structure_dict
from amorphouspy.potentials import generate_potential

structure_dict = get_structure_dict(
    {"SiO2": 75, "Na2O": 15, "CaO": 10},
    target_atoms=3000,
)

# With melt pre-equilibration (default)
potential = generate_potential(structure_dict, potential_type="shik")

# Without melt pre-equilibration
potential = generate_potential(structure_dict, potential_type="shik", melt=False)
```

### Langevin pre-equilibration (`melt` parameter)

By default (`melt=True`), the SHIK configuration appends a **Langevin dynamics + NVE/limit pre-equilibration** stage. This handles initial atomic overlaps in random starting configurations, which cause extremely large forces with the steep $r^{-24}$ repulsion:

```
fix langevin all langevin 5000 5000 0.01 48279
fix ensemble all nve/limit 0.5
run 10000
unfix langevin
unfix ensemble
```

This runs 10,000 steps of velocity-limited NVE with Langevin damping at 5000 K, allowing atoms to move apart gently (max 0.5 Å per step) before the main simulation begins.

Pass `melt=False` to omit this block — useful when the starting configuration is already well-equilibrated or when you want full control over the pre-equilibration protocol.

---

## Defined Pair Interactions

The SHIK potential defines parameters for specific element pairs (not all combinations). The following pairs have tabulated interactions:

| Pair | $A$ (eV) | $B$ (Å⁻¹) | $C$ (eV·Å⁶) | $D$ (eV·Å²⁴) |
|---|---|---|---|---|
| O–O | 1120.5 | 2.8927 | 26.132 | 16800 |
| O–Si | 23108 | 5.0979 | 139.70 | 66.0 |
| Si–Si | 2798.0 | 4.4073 | 0.0 | 3423204 |
| O–Al | 21740 | 5.3054 | 65.815 | 66.0 |
| Al–Al | 1799.1 | 3.6778 | 100.0 | 16800 |
| O–B | 16182 | 5.6069 | 59.203 | 32.0 |
| B–B | 1805.5 | 3.8228 | 69.174 | 6000.0 |
| O–Na | 1127566 | 6.8986 | 40.562 | 16800 |
| Na–Na | 1476.9 | 3.4075 | 0.0 | 16800 |
| O–Ca | 146905 | 5.6094 | 45.073 | 16800 |
| Ca–Ca | 21633 | 3.2562 | 0.0 | 16800 |
| ... | | | | |

> The full set includes cross-terms (e.g., Si–Ca, Si–Na, Li–B, etc.) totaling 27 defined pair interactions.

---

## Technical Details

### Cutoff distance

The SHIK potential uses a cutoff of **10.0 Å** for both the Coulomb (DSF) and tabulated pair interactions. This is longer than PMMCS (8.0 Å) and BJP (8.0 Å).

### When to use SHIK

- **Alkali and alkaline-earth silicate glasses** — well-validated for Li, Na, K, Mg, Ca silicates
- **Aluminosilicate glasses** — good accuracy for Al coordination and $Q^n$ distributions
- **Borosilicate glasses** — supports B₂O₃ mixed former systems
- **Mixed-modifier effects** — composition-dependent charging captures modifier mixing
- **Accurate structural properties** — $r^{-24}$ term improves short-range structure

### Limitations

- **Fewer elements** (9) than PMMCS (28) — no transition metals, rare earths, or Zn/Pb
- **Tabulated potentials** — slightly slower than analytical pair styles
- **Pre-equilibration required** — the steep $r^{-24}$ term requires careful handling of initial configurations
- **Longer cutoff** (10 Å) — more expensive per timestep than PMMCS (8 Å)
