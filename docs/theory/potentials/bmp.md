# BMP Potential (Bertani–Menziani–Pedone)

!!! tip "Interactive example"
    See the [Melt-Quench Tutorial notebook](../../notebooks/MeltquenchTutorial.ipynb) for worked examples with both `bmp-harmonic` and `bmp-screened-harmonic`.

The BMP potential is a Morse-based force field extended with cation–cation Buckingham repulsion and explicit three-body angular terms. It is available in two variants that differ only in how the three-body interactions are handled:

- **`bmp-harmonic`** — uses `nb3b/harmonic` (harmonic angle potential). Simpler form, covers Si–O–P and Si–O–Si triplets.
- **`bmp-screened-harmonic`** — uses `nb3b/screened` (screened harmonic). Adds distance-dependent screening and covers B and V in addition to Si and P.

Both variants use identical Morse, Buckingham, and electrostatic parameters. The choice of variant controls which LAMMPS three-body style is activated and which element triplets are parameterised.

---

## References

1. M. Bertani, M. C. Menziani, A. Pedone. "Improved empirical force field for multicomponent oxide glasses and crystals", *Phys. Rev. Materials* **5**, 045602 (2021)
2. G. Malavasi, A. Pedone. "The effect of the incorporation of catalase mimetic activity cations on the structural, thermal and chemical durability properties of the 45S5 Bioglass®", *Acta Materialia* **229**, 117801 (2022)
3. M. Bertani, A. Pallini, M. Cocchi, M. C. Menziani, A. Pedone. "A new self-consistent empirical potential model for multicomponent borate and borosilicate glasses", *J. Am. Ceram. Soc.* **105**, 7254–7271 (2022)
4. M. Bertani et al. "Erratum for a new self-consistent empirical potential model for multicomponent borate and borosilicate glasses", *J. Am. Ceram. Soc.* **106**, 5104–5105 (2023)

---

## Functional Form

### Morse (pair) interaction

$$
V_{\text{Morse}}(r_{ij}) = D_{ij}\left[\left(1 - e^{-\alpha_{ij}(r_{ij}-r_{0,ij})}\right)^2 - 1\right] + \frac{A_{ij}}{r_{ij}^{12}} + \frac{q_i q_j}{r_{ij}}
$$

where $D_{ij}$ (eV) is the well depth, $\alpha_{ij}$ (Å⁻¹) the stiffness, $r_{0,ij}$ (Å) the equilibrium distance, and $A_{ij}$ (eV·Å¹²) the short-range repulsive wall.

For **Boron**, $D_{\text{B-O}}$ is not a fixed constant — it is computed from the glass composition via the Dell–Bray model (see [Boron restriction](#boron-restriction) below).

### Cation–cation Buckingham repulsion

$$
V_{\text{Buck}}(r_{ij}) = A_{ij}^{\text{Buck}} \, e^{-r_{ij}/\rho_{ij}}
$$

Only specific cation–cation pairs (Si–Si, Si–Al, B–Si, etc.) carry Buckingham parameters; most cross-terms are zero.

### Three-body angular terms

**`bmp-harmonic` (`nb3b/harmonic`):**

$$
V_{\text{3b}}(\theta_{ijk}) = \frac{K}{2}\left(\theta_{ijk} - \theta_0\right)^2
$$

**`bmp-screened-harmonic` (`nb3b/screened`):**

$$
V_{\text{3b}}(r_{ij}, r_{ik}, \theta_{ijk}) = K\left(\theta_{ijk} - \theta_0\right)^2 e^{-r_{ij}/\rho} e^{-r_{ik}/\rho}
$$

The screened form damps the angular penalty as bond lengths increase, which is more physically appropriate for the disordered glass network and allows parameterisation of B and V triplets.

**LAMMPS pair style:**

```
pair_style hybrid/overlay coul/dsf <alpha> <cutoff> pedone 7.0 buck 7.0 nb3b/harmonic   # bmp-harmonic
pair_style hybrid/overlay coul/dsf <alpha> <cutoff> pedone 7.0 buck 7.0 nb3b/screened   # bmp-screened-harmonic
```

---

## Three-body Parameters

### `bmp-harmonic` (`nb3b/harmonic`)

| Triplet (center–O–center) | $K/2$ (eV/rad²) | $\theta_0$ (°) | Cutoff (Å) |
|---|---|---|---|
| Si–O–Si | 0.73 | 109.47 | 2.0 |
| Si–O–P | 2.00 | 109.47 | 2.0 |
| P–O–P | 2.00 | 109.47 | 2.0 |

### `bmp-screened-harmonic` (`nb3b/screened`)

| Triplet (center–O–center) | $K$ (eV/rad²) | $\theta_0$ (°) | Cutoff (Å) |
|---|---|---|---|
| Si–O–Si | 25.0 | 109.47 | 3.30 |
| Si–O–P | 120.0 | 109.47 | 2.00 |
| P–O–P | 65.0 | 109.47 | 2.00 |
| B–O–B | 60.0 | 109.47 | 3.30 |
| B–O–Si | 60.0 | 109.47 | 3.30 |
| V–O–V | 30.0 | 109.00 | 2.50 |
| V–O–P | 120.0 | 109.00 | 2.00 |
| V–O–Si | 120.0 | 109.00 | 2.00 |

All triplets not in the table receive zero parameters (effectively inactive).

---

## Boron Restriction

When Boron is present in the composition, the B–O well depth $D_{\text{B-O}}$ is computed from the glass composition using the **Dell–Bray model**, which expresses $D$ as a function of two ratios:

$$
R = \frac{[\text{alkali oxide}] + [\text{alkaline-earth oxide}]}{[\text{B}_2\text{O}_3]}, \qquad K = \frac{[\text{SiO}_2]}{[\text{B}_2\text{O}_3]}
$$

This model is only valid for systems containing **B, Si, O and alkali/alkaline-earth modifiers**. If any other element is present alongside Boron, `generate_potential` raises a `ValueError`.

**Elements allowed with Boron:**

| Category | Elements |
|---|---|
| Network formers | B, Si, O |
| Alkali modifiers | Li, Na, K |
| Alkaline-earth modifiers | Mg, Ca, Sr, Ba |

Compositions such as B + Al, B + Ti, or B + Fe will raise an error. Remove Boron or use `bmp-harmonic` with a B-free composition in those cases.

> **`bmp-harmonic` vs. boron:** The harmonic three-body style has no B triplets in its parameter set, so `bmp-harmonic` works correctly with B-free glasses. To simulate borosilicate or borate systems, use `bmp-screened-harmonic`.

---

## Supported Elements

The BMP potential covers 36 elements:

| Element | Charge ($e$) | Role |
|---|---|---|
| Li, Na, K | +0.6 | Alkali modifiers |
| Be, Mg, Ca, Sr, Ba | +1.2 | Alkaline-earth modifiers |
| B | +1.8 | Network former (composition-dependent $D$) |
| Al | +1.8 | Intermediate / former |
| Si | +2.4 | Network former |
| P | +3.0 | Network former |
| Ge, Sn | +2.4 | Network formers |
| Ti, Zr, Sc | +2.4 / +1.8 | Intermediate |
| Fe²⁺, Fe³⁺ | +1.2 / +1.8 | Transition metal |
| Mn, Mn³⁺, Mn⁴⁺ | +1.2 / +1.8 | Transition metal |
| Cr | +1.8 | Transition metal |
| Cu, Cu²⁺ | +0.6 / +1.2 | Transition metal |
| Zn, Ni, Co | +1.2 | Transition metal |
| Ag | +0.6 | Noble metal |
| Ce³⁺, Ce⁴⁺ | +1.8 / +2.4 | Rare earth |
| Nd, Gd, Er | +1.8 | Rare earth |
| V⁴⁺, V⁵⁺ | +2.4 / +3.0 | Vanadium (shrm only for 3b) |
| O | −1.2 | Anion (fixed) |

---

## Usage

```python
from amorphouspy import get_structure_dict
from amorphouspy.potentials import generate_potential

structure_dict = get_structure_dict(
    {"SiO2": 70, "Na2O": 20, "CaO": 10},
    target_atoms=3000,
)

# BMP with harmonic three-body (Si/P glasses, no boron)
potential = generate_potential(structure_dict, potential_type="bmp-harmonic")

# BMP with screened three-body (borosilicate/borate glasses)
structure_dict_b = get_structure_dict(
    {"SiO2": 60, "B2O3": 20, "Na2O": 20},
    target_atoms=3000,
)
potential_b = generate_potential(structure_dict_b, potential_type="bmp-screened-harmonic")
```

### Melt pre-equilibration

Pass `melt=True` when the structure was generated by `get_structure_dict`. The BMP potential can produce large forces on random configurations, particularly from the $r^{-12}$ repulsive wall. The pre-equilibration stage runs 10,000 steps of Langevin dynamics with a maximum displacement of 0.5 Å per step at 4000 K to gently relax atomic overlaps.

```python
potential = generate_potential(structure_dict, potential_type="bmp-harmonic", melt=True)
```

When `melt=False` (default), the pre-equilibration block is omitted — useful when starting from an already-equilibrated structure.

### Electrostatics options

BMP supports all four Coulomb solvers. The default is DSF with $\alpha = 0.25$ Å⁻¹ and a cutoff of 8.0 Å.

```python
from amorphouspy import DsfConfig, WolfConfig, PppmConfig, EwaldConfig

# DSF (default) — real-space, no k-space
potential = generate_potential(structure_dict, potential_type="bmp-harmonic",
                               electrostatics=DsfConfig(alpha=0.25, long_range_cutoff=8.0))

# Wolf summation
potential = generate_potential(structure_dict, potential_type="bmp-harmonic",
                               electrostatics=WolfConfig(alpha=0.25, long_range_cutoff=8.0))

# PPPM — reciprocal-space, more accurate for large systems
potential = generate_potential(structure_dict, potential_type="bmp-harmonic",
                               electrostatics=PppmConfig(long_range_cutoff=12.0))

# Ewald — reciprocal-space
potential = generate_potential(structure_dict, potential_type="bmp-harmonic",
                               electrostatics=EwaldConfig(long_range_cutoff=12.0))
```

The short-range (Morse + Buckingham) cutoff is fixed at **7.0 Å**.

---

## When to Use Each Variant

| | `bmp-harmonic` | `bmp-screened-harmonic` |
|---|---|---|
| **Three-body style** | `nb3b/harmonic` | `nb3b/screened` |
| **Boron support** | No | Yes (with alkali/AE only) |
| **Vanadium 3b** | No | Yes |
| **Angular screening** | None | Distance-dependent |
| **Best for** | Multi-modifier silicate/phosphate oxides | Borate and borosilicate glasses |

Both variants support the full 36-element library for non-three-body interactions. The choice only affects which angular triplets are active.
