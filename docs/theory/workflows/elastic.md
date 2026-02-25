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
    temperature=300.0,          # K
    strain=1e-3,                # Applied strain magnitude
    production_steps=10_000,    # MD steps per strain state
    equilibration_steps=5_000,  # Equilibration before measurement
    timestep=1.0,               # fs
)

print(f"C11 = {result['C11']:.1f} GPa")
print(f"C12 = {result['C12']:.1f} GPa")
print(f"C44 = {result['C44']:.1f} GPa")
print(f"Young's modulus E = {result['E']:.1f} GPa")
print(f"Bulk modulus B = {result['B']:.1f} GPa")
print(f"Shear modulus G = {result['G']:.1f} GPa")
print(f"Poisson's ratio ν = {result['nu']:.3f}")
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | Equilibrated glass structure |
| `potential` | `DataFrame` | — | Potential configuration |
| `temperature` | `float` | `300.0` | Temperature (K) |
| `strain` | `float` | `1e-3` | Strain magnitude (dimensionless) |
| `production_steps` | `int` | `10_000` | Steps per deformed state for stress averaging |
| `equilibration_steps` | `int` | `5_000` | Steps to equilibrate after applying strain |
| `timestep` | `float` | `1.0` | MD timestep (fs) |

**Returns:**

| Key | Type | Description |
|---|---|---|
| `"C11"`, `"C12"`, `"C44"` | `float` | Elastic constants (GPa) |
| `"B"` | `float` | Bulk modulus (GPa) |
| `"G"` | `float` | Shear modulus (GPa) |
| `"E"` | `float` | Young's modulus (GPa) |
| `"nu"` | `float` | Poisson's ratio |

---

## Workflow Details

The simulation runs the following steps automatically:

1. **Equilibrate** the undeformed structure at temperature $T$ (NVT)
2. For each strain direction ($xx$, $yy$, $zz$, $xy$, $xz$, $yz$):
   - Apply $+\varepsilon$ strain → equilibrate → measure average stress tensor
   - Apply $-\varepsilon$ strain → equilibrate → measure average stress tensor
   - Compute $C_{ij}$ from finite differences
3. Average symmetric components and compute VRH moduli

This results in **12 LAMMPS runs** per elastic calculation (6 directions × 2 signs), plus the initial equilibration.

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
