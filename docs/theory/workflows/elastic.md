# Elastic Moduli Calculation

This workflow computes the elastic constants of a glass structure using the **stress-strain finite differences** method. Small uniaxial and shear strains are applied, the resulting stress tensor is measured, and elastic constants are extracted.

---

## Method

### Stress-Strain Approach

The elastic stiffness tensor $C_{ijkl}$ relates stress $\sigma$ to strain $\varepsilon$:

$$
\sigma_{ij} = \sum_{kl} C_{ijkl} \, \varepsilon_{kl}
$$

For an isotropic material like glass, only two independent constants are needed, and the full tensor reduces to three measurable constants:

| Constant | Strain applied | Measurement |
|---|---|---|
| $C_{11}$ | Uniaxial $\varepsilon_{xx}$ | $\sigma_{xx}$ response |
| $C_{12}$ | Uniaxial $\varepsilon_{xx}$ | $\sigma_{yy}$ response |
| $C_{44}$ | Shear $\varepsilon_{xy}$ | $\sigma_{xy}$ response |

The workflow applies small strains ($\pm \varepsilon$) and uses **central differences** to compute the elastic constants:

$$
C_{11} = \frac{\sigma_{xx}(+\varepsilon) - \sigma_{xx}(-\varepsilon)}{2\varepsilon}
$$

### Derived Moduli

From $C_{11}$, $C_{12}$, and $C_{44}$, the engineering elastic moduli are calculated using Voigt-Reuss-Hill averaging:

| Modulus | Formula | Unit |
|---|---|---|
| **Bulk modulus** $B$ | $B = \frac{C_{11} + 2C_{12}}{3}$ | GPa |
| **Shear modulus** $G$ | $G = C_{44}$ or $G = \frac{C_{11} - C_{12}}{2}$ (Voigt/Reuss) | GPa |
| **Young's modulus** $E$ | $E = \frac{9BG}{3B + G}$ | GPa |
| **Poisson's ratio** $\nu$ | $\nu = \frac{3B - 2G}{2(3B + G)}$ | — |

---

## Usage

### `elastic_simulation(structure, potential, ...)`

```python
from amorphouspy import elastic_simulation

result = elastic_simulation(
    structure=glass_structure,
    potential=potential,
    temperature_sim=300.0,         # K
    strain=1e-3,                   # Applied strain magnitude
    production_steps=10_000,       # MD steps per strain state
    equilibration_steps=1_000_000, # Initial equilibration
    timestep=1.0,                  # fs
    n_repeats=3,                   # Independent repeats for uncertainty quantification
)

# Mean moduli and uncertainties
moduli = result["moduli"]
print(f"Young's modulus E = {moduli['E']:.1f} ± {moduli['E_std']:.1f} GPa")
print(f"Bulk modulus B    = {moduli['B']:.1f} ± {moduli['B_std']:.1f} GPa")
print(f"Shear modulus G   = {moduli['G']:.1f} ± {moduli['G_std']:.1f} GPa")
print(f"Poisson's ratio ν = {moduli['nu']:.3f} ± {moduli['nu_std']:.3f}")
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | Equilibrated glass structure |
| `potential` | `str` | — | Path to LAMMPS potential file |
| `temperature_sim` | `float` | `300.0` | Temperature (K) |
| `strain` | `float` | `1e-3` | Strain magnitude (dimensionless) |
| `production_steps` | `int` | `10_000` | Steps per deformed state for stress averaging |
| `equilibration_steps` | `int` | `1_000_000` | Initial equilibration phase steps |
| `timestep` | `float` | `1.0` | MD timestep (fs) |
| `n_repeats` | `int` | `1` | Independent production runs per strain component; each repeat uses `seed + r`. With `n_repeats=1` no uncertainty is computed. |
| `seed` | `int` | `12345` | Base random seed for velocity initialisation |

**Returns:**

Returns a dictionary with:

| Key | Description |
|---|---|
| `"Cij"` | Mean 6×6 stiffness tensor (GPa) |
| `"Cij_std"` | Per-element standard deviation across repeats (zero when `n_repeats=1`) |
| `"Cij_samples"` | `(n_repeats, 6, 6)` array of per-repeat tensors (only present when `n_repeats > 1`) |
| `"moduli"` | Dict with mean keys `B`, `G`, `E`, `nu` and std keys `B_std`, `G_std`, `E_std`, `nu_std` |
| `"n_repeats"` | Number of repeats used |

---

## Workflow Details

The simulation runs the following steps automatically:

1. **Equilibrate** the undeformed structure at temperature $T$ (NVT)
2. For each repeat $r = 0, \ldots, n\_repeats - 1$ (seed $= $ base seed $+ r$):
   - For each strain direction ($xx$, $yy$, $zz$, $xy$, $xz$, $yz$):
     - Apply $+\varepsilon$ strain → run MD → measure average stress tensor
     - Apply $-\varepsilon$ strain → run MD → measure average stress tensor
     - Compute $C_{ij}^{(r)}$ from finite differences
3. Average $C_{ij}^{(r)}$ across repeats; compute per-element std and VRH moduli with uncertainties

This results in **$12 \times n\_repeats$ LAMMPS runs** (6 directions × 2 signs × repeats), plus the initial equilibration.

---

## Typical Values for Oxide Glasses

| Glass | $E$ (GPa) | $B$ (GPa) | $G$ (GPa) | $\nu$ |
|---|---|---|---|---|
| SiO₂ | 72 | 36 | 31 | 0.17 |
| Soda-lime | 70 | 45 | 28 | 0.22 |
| Borosilicate | 63 | 37 | 26 | 0.20 |
| Aluminosilicate | 85 | 52 | 35 | 0.22 |

> **Note:** MD values may differ from experiment by 5–20% depending on the potential. Trends with composition are generally well-reproduced.

---

## Tips

- **Strain magnitude**: Use $\varepsilon = 10^{-3}$ for the linear elastic regime. Larger strains may probe non-linear response.
- **Production steps**: More steps = better stress averaging. 10,000 steps is a minimum; 50,000 gives smoother results.
- **System size**: 3000+ atoms recommended. Smaller systems have large fluctuations in the stress tensor.
- **Temperature**: Elastic constants are temperature-dependent. Compute at the same temperature as the experimental comparison.
- **Uncertainty quantification**: Use `n_repeats=3` or higher to obtain statistical uncertainties. Each repeat reruns all strain directions with a different seed (`seed + r`), so the cost scales linearly with `n_repeats`.
