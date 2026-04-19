# PMMCS Potential (Pedone)

The PMMCS potential, developed by Pedone et al., is the most broadly applicable force field in `amorphouspy`. It uses a Morse-type short-range interaction combined with a repulsive $r^{-12}$ wall and damped-shifted-force (DSF) Coulomb interactions.

---

## Reference

> A. Pedone, G. Malavasi, M.C. Menziani, A.N. Cormack, U. Segre. "A New Self-Consistent Empirical Interatomic Potential Model for Oxides, Silicates, and Silica-Based Glasses", *J. Phys. Chem. B* **110**, 11780–11795 (2006). [DOI:10.1021/jp0611018](https://doi.org/10.1021/jp0611018)

---

## Functional Form

The total pairwise interaction energy is:

$$
V(r_{ij}) = D_{ij} \left[1 - e^{-a_{ij}(r_{ij} - r_0)}\right]^2 - D_{ij} + \frac{C_{ij}}{r_{ij}^{12}} + \frac{q_i q_j}{r_{ij}}
$$

where:

| Symbol | Description |
|---|---|
| $D_{ij}$ | Morse potential well depth (eV) |
| $a_{ij}$ | Morse potential width parameter (Å⁻¹) |
| $r_0$ | Morse equilibrium distance (Å) |
| $C_{ij}$ | Repulsive wall coefficient (eV·Å¹²) |
| $q_i, q_j$ | Partial atomic charges |

The Coulomb term is configurable. By default, `amorphouspy` uses the **damped shifted force (DSF)** method with a damping parameter of 0.25 Å⁻¹ and a Coulomb cutoff of 8.0 Å. DSF provides accurate electrostatic energies without the expense of Ewald summation, making it efficient for amorphous systems.

**Default LAMMPS pair style:** `hybrid/overlay coul/dsf 0.25 8.0 pedone 5.5`

---

## Charges

All atomic charges are **fixed** (composition-independent):

| Element | Charge ($e$) |
|---|---|
| O | −1.2 |
| Si | +2.4 |
| Al | +1.8 |
| Na | +0.6 |
| Ca | +1.2 |
| Mg | +1.2 |
| K | +0.6 |
| Li | +0.6 |

> **Note:** The oxygen charge is always −1.2 regardless of composition. This is a defining feature of the PMMCS potential.

---

## Supported Elements

The PMMCS potential supports 28 elements (plus oxygen), making it the broadest of the three potentials:

| Category | Elements |
|---|---|
| **Alkali metals** | Li, Na, K |
| **Alkaline earth** | Be, Mg, Ca, Sr, Ba |
| **Transition metals** | Sc, Ti, Zr, Cr, Mn, Fe, Fe3+, Co, Ni, Cu, Ag, Zn |
| **Post-transition** | Al, Si, Ge, Sn |
| **Pnictogens** | P |
| **Rare earth** | Nd, Gd, Er |
| **Anion** | O |

---

## Usage

```python
from amorphouspy import get_structure_dict, generate_potential, ElectrostaticsConfig

# Works with any composition using supported elements
structure_dict = get_structure_dict(
    {"SiO2": 60, "Al2O3": 10, "Na2O": 15, "CaO": 10, "MgO": 5},
    target_atoms=3000,
)

# Default: DSF electrostatics, melt pre-equilibration enabled
potential = generate_potential(structure_dict, potential_type="pmmcs")

# PPPM with custom cutoffs, no melt block
potential = generate_potential(
    structure_dict,
    potential_type="pmmcs",
    melt=False,
    electrostatics=ElectrostaticsConfig(
        method="pppm",
        short_range_cutoff=6.0,
        long_range_cutoff=12.0,
        kspace_accuracy=1e-5,
    ),
)
```

### `melt` — high-temperature pre-equilibration

When `melt=True` (default), the generator appends a 10 000-step Langevin NVE/limit block at 4000 K:

```lammps
fix langevinnve all langevin 4000 4000 0.01 48279
fix ensemblenve all nve/limit 0.5
run 10000
unfix langevinnve
unfix ensemblenve
```

This relaxes unfavourable atomic contacts that are common in randomly packed starting structures before the main melt–quench run. The `nve/limit 0.5` cap prevents runaway atom velocities if two atoms are placed too close together. Set `melt=False` when the starting structure is already equilibrated or when you want full control over the thermostat schedule.

### Electrostatics options

`amorphouspy` supports four Coulomb solvers via `ElectrostaticsConfig`. The choice affects which LAMMPS directives are emitted:

| Method | `pair_style` fragment | `kspace_style` | Recommended cutoff |
|---|---|---|---|
| `dsf` (default) | `coul/dsf <alpha> <cutoff>` | — | 8.0 Å |
| `wolf` | `coul/wolf <alpha> <cutoff>` | — | 8.0 Å |
| `pppm` | `coul/long <cutoff>` | `pppm <accuracy>` | 12.0 Å |
| `ewald` | `coul/long <cutoff>` | `ewald <accuracy>` | 12.0 Å |

**DSF and Wolf** are real-space methods. They require a damping parameter `alpha` (Å⁻¹, default 0.25) that controls how quickly the interaction is damped. Larger `alpha` allows a shorter `long_range_cutoff` but at the cost of accuracy near the cutoff. DSF shifts both the potential and the force to zero at the cutoff; Wolf shifts only the potential.

**PPPM and Ewald** are reciprocal-space methods. The `alpha` parameter is ignored. The default `long_range_cutoff` widens to 12.0 Å because without DSF-style damping the real-space part decays more slowly. A `kspace_style` line is appended after the pair coefficients; `kspace_accuracy` (default `1e-5`) tunes the accuracy of the long-range Fourier sum at the cost of k-space solver time.

`short_range_cutoff` controls the Morse + repulsive wall cutoff independently of the Coulomb cutoff (default 5.5 Å). Increasing it captures slightly longer-ranged Morse interactions but raises the pair-list cost roughly quadratically.

### What the generator produces

The PMMCS generator creates LAMMPS configuration lines that:
1. Define the `hybrid/overlay coul/* pedone` pair style
2. Set atomic charges via `set type ... charge ...`
3. Define Morse parameters for all element pairs via `pair_coeff`
4. Set the repulsive wall coefficient $C_{ij}$ for close-range interactions
5. Optionally append the melt pre-equilibration block

---

## Technical Details

### Short-range cutoff

The Morse + repulsive term uses a default cutoff of **5.5 Å**, configurable via `ElectrostaticsConfig(short_range_cutoff=...)`. This is shorter than BJP (8.0 Å) and SHIK (10.0 Å), making PMMCS simulations somewhat faster per timestep.

### When to use PMMCS

- **Multi-component glasses** with elements beyond Ca-Al-Si-O
- **Exploratory studies** where element coverage matters more than potential accuracy for a specific system
- **Rapid screening** of compositions (fast short-range cutoff)
- **Systems with rare earth or transition metal dopants**

### Limitations

- Fixed oxygen charge may not accurately capture composition-dependent charge transfer effects
- Parameters for some element pairs may be less well-validated than others
- The $r^{-12}$ repulsive wall is a simplification compared to more physically motivated forms
