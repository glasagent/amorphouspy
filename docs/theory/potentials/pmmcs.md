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

The Coulomb term uses the **damped shifted force (DSF)** method with a damping parameter of 0.25 Å⁻¹ and a cutoff of 8.0 Å. DSF provides accurate electrostatic energies without the expense of Ewald summation, making it efficient for amorphous systems.

**LAMMPS pair style:** `hybrid/overlay coul/dsf 0.25 8.0 pedone 5.5`

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
from amorphouspy import get_structure_dict, generate_potential

# Works with any composition using supported elements
structure_dict = get_structure_dict(
    "0.60SiO2-0.10Al2O3-0.15Na2O-0.10CaO-0.05MgO",
    target_atoms=3000,
)

potential = generate_potential(structure_dict, potential_type="pmmcs")

# The potential DataFrame contains LAMMPS configuration
print(potential["Config"].iloc[0][:5])  # First 5 LAMMPS commands
```

### What the generator produces

The PMMCS generator creates LAMMPS configuration lines that:
1. Define the `hybrid/overlay coul/dsf pedone` pair style
2. Set atomic charges via `set type ... charge ...`
3. Define Morse parameters for all element pairs via `pair_coeff`
4. Set the repulsive wall coefficient $C_{ij}$ for close-range interactions

---

## Technical Details

### Short-range cutoff

The Morse + repulsive term uses a cutoff of **5.5 Å**. This is shorter than BJP (8.0 Å) and SHIK (10.0 Å), making PMMCS simulations somewhat faster per timestep.

### When to use PMMCS

- **Multi-component glasses** with elements beyond Ca-Al-Si-O
- **Exploratory studies** where element coverage matters more than potential accuracy for a specific system
- **Rapid screening** of compositions (fast short-range cutoff)
- **Systems with rare earth or transition metal dopants**

### Limitations

- Fixed oxygen charge may not accurately capture composition-dependent charge transfer effects
- Parameters for some element pairs may be less well-validated than others
- The $r^{-12}$ repulsive wall is a simplification compared to more physically motivated forms
