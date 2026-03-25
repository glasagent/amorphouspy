# Structure Generation

This module handles everything from validating compositions to generating ready-to-simulate atomic structures with physically realistic densities. It is the starting point for any simulation workflow.

---

## Overview

Creating a glass structure involves several steps:

1. **Define the composition** — provide a dict mapping oxide formulas to mol%, e.g. `{"SiO2": 75, "Na2O": 15, "CaO": 10}`
2. **Validate the chemistry** — check that each oxide is charge-neutral and elements are valid
3. **Plan the system** — determine how many formula units of each oxide are needed to reach the target size
4. **Estimate the density** — use Fluegel's empirical model to predict the glass density
5. **Calculate the box size** — determine the cubic simulation box dimensions
6. **Place atoms randomly** — distribute atoms in the box with a minimum distance constraint

All of these steps are handled automatically by the high-level `get_structure_dict()` function, but each individual step is also available separately for fine-grained control.

---

## Composition Input

Compositions are specified as a dict mapping oxide formulas to mol% values. Values are automatically rescaled to sum to 1.0 internally:

```python
# Molar fractions (sum to ~1.0)
{"CaO": 0.25, "Al2O3": 0.25, "SiO2": 0.5}

# Molar percentages (sum to ~100)
{"SiO2": 75, "Na2O": 15, "CaO": 10}

# Weight percentages (requires mode="weight")
{"SiO2": 79, "B2O3": 13, "Al2O3": 3, "Na2O": 4, "K2O": 1}
```

### `get_composition(composition, mode="molar")`

Normalizes a composition dict into molar fractions. When `mode="weight"`, it first converts weight percentages to molar fractions using the molar masses of each oxide.

```python
from amorphouspy.structure import get_composition

# Molar composition
mol_frac = get_composition({"CaO": 0.25, "Al2O3": 0.25, "SiO2": 0.5})
# Returns: {'CaO': 0.25, 'Al2O3': 0.25, 'SiO2': 0.5}

# Weight% to mol% conversion
mol_frac = get_composition({"SiO2": 75, "Na2O": 15, "CaO": 10}, mode="weight")
# Returns molar fractions (different from weight fractions due to molar masses)
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `composition` | `dict[str, float]` | — | Oxide formula → mol% mapping |
| `mode` | `str` | `"molar"` | `"molar"` for mol fractions/%, `"weight"` for weight% |

### `extract_composition(composition)`

A stricter validator that performs additional checks on top of `get_composition`. Use this when processing user input that may contain errors.

**Validation steps:**

1. **Element validation** — verifies each element symbol against the periodic table
2. **Charge neutrality** — uses pymatgen's oxidation state guessing to ensure each oxide is charge-neutral
3. **Sum validation** — checks that fractions sum to ~1.0 or percentages sum to ~100

```python
from amorphouspy.structure import extract_composition

comp = extract_composition({"SiO2": 0.75, "Na2O": 0.15, "CaO": 0.10})
# Returns: {'SiO2': 0.75, 'Na2O': 0.15, 'CaO': 0.1}

# These will raise ValueError:
# extract_composition({"XyZ": 0.5, "SiO2": 0.5})    # Invalid element
# extract_composition({"NaCl": 0.5, "SiO2": 0.5})   # Not charge-neutral for oxide
```

---

## Density Estimation

### `get_glass_density_from_model(composition)`

Calculates room-temperature glass density using Fluegel's empirical polynomial model. This is a widely used regression model trained on a large database of measured glass densities.

> **Reference:** Fluegel, A. "Global Model for Calculating Room-Temperature Glass Density from the Composition", *J. Am. Ceram. Soc.* **90** [8] 2622–2635 (2007).

The model uses a polynomial expansion in mole percentages with linear, quadratic, cubic, and cross-interaction terms:

$$
\rho = b_0 + \sum_i b_i \cdot x_i + \sum_i b_{i,2} \cdot x_i^2 + \sum_i b_{i,3} \cdot x_i^3 + \sum_{i<j} b_{ij} \cdot x_i \cdot x_j + \sum_{i<j<k} b_{ijk} \cdot x_i \cdot x_j \cdot x_k
$$

where $x_i$ is the mole percentage of oxide $i$ and $b$ values are fitted coefficients.

**Supported oxide components:**

| Category | Oxides |
|---|---|
| **Glass formers** | SiO₂, B₂O₃, Al₂O₃ |
| **Alkali oxides** | Li₂O, Na₂O, K₂O |
| **Alkaline earth oxides** | MgO, CaO, SrO, BaO |
| **Transition metal oxides** | ZnO, PbO, TiO₂, ZrO₂, NiO, MnO, FeO |
| **Rare earth oxides** | La₂O₃, Nd₂O₃, CeO₂ |
| **Others** | CdO, ThO₂, UO, SbO, SO₃, F, Cl |

```python
from amorphouspy.structure import get_glass_density_from_model

density = get_glass_density_from_model({"SiO2": 75, "Na2O": 15, "CaO": 10})
print(f"Predicted density: {density:.4f} g/cm³")
# Typical soda-lime glass: ~2.49 g/cm³
```

> **Note:** If your composition contains oxides not in the model, they will be treated as "remainder" components with a generic coefficient. For best accuracy, use compositions within the model's training domain.

---

## Structure Generation

### High-level: `get_structure_dict()`

The main entry point for generating complete structure dictionaries. This function chains all sub-steps automatically: composition parsing → density estimation → box calculation → atom placement.

```python
from amorphouspy import get_structure_dict

# Mode 1: Specify target atom count
structure_dict = get_structure_dict(
    composition={"SiO2": 75, "Na2O": 15, "CaO": 10},
    target_atoms=3000,       # Target ~3000 atoms (may differ slightly due to stoichiometry)
    mode="molar",            # Interpret fractions as mol% (default)
    density=None,            # Auto-calculate density (Fluegel model)
    min_distance=1.6,        # Minimum inter-atomic distance (Å)
    max_attempts_per_atom=10000,  # Max placement attempts per atom
)

# Mode 2: Specify molecule (formula unit) count
structure_dict = get_structure_dict(
    composition={"CaO": 0.25, "Al2O3": 0.25, "SiO2": 0.5},
    n_molecules=100,         # 100 total formula units
)

# Mode 3: Weight percentages
structure_dict = get_structure_dict(
    composition={"SiO2": 79, "B2O3": 13, "Al2O3": 3, "Na2O": 4, "K2O": 1},
    target_atoms=5000,
    mode="weight",           # Interpret as weight%
)
```

> **Important:** You must specify exactly one of `target_atoms` or `n_molecules` — not both and not neither.

**Returns** a dictionary with:

| Key | Type | Description |
|---|---|---|
| `"atoms"` | `list[dict]` | List of `{"element": str, "position": [x, y, z]}` dicts |
| `"box"` | `float` | Cubic box side length in Å |
| `"formula_units"` | `dict[str, int]` | Integer formula units per oxide (e.g. `{"SiO2": 250, "Na2O": 50}`) |
| `"total_atoms"` | `int` | Actual total atom count (may differ slightly from target) |
| `"element_counts"` | `dict[str, int]` | Total count per element symbol |
| `"mol_fraction"` | `dict[str, float]` | Normalized molar fractions |

### Converting to ASE Atoms

The `get_ase_structure()` function converts the structure dictionary into an ASE `Atoms` object by generating a LAMMPS data file in memory and reading it back with ASE's LAMMPS reader. This ensures proper atom type assignment and charge columns.

```python
from amorphouspy import get_ase_structure

# Basic conversion
atoms = get_ase_structure(structure_dict)
print(f"Number of atoms: {len(atoms)}")
print(f"Cell: {atoms.get_cell()}")
print(f"Chemical symbols: {set(atoms.get_chemical_symbols())}")

# With supercell replication (2×2×2 → 8× more atoms)
atoms_supercell = get_ase_structure(structure_dict, replicate=(2, 2, 2))
print(f"Supercell atoms: {len(atoms_supercell)}")  # 8 × original count
```

> **Note:** The `replicate` parameter creates a supercell by tiling the box in each direction. A `(2, 2, 2)` replication doubles the box in x, y, and z, resulting in 8× the original number of atoms.

### Low-level: `create_random_atoms()`

For direct control over atom placement. This function generates random positions in a periodic cubic box, ensuring that no two atoms are closer than `min_distance` using the minimum image convention.

The placement algorithm:
1. For each element type and count, attempt to place atoms one by one
2. Generate a random position uniformly in `[0, box_length]³`
3. Check the minimum image distance to all previously placed atoms
4. If distance ≥ `min_distance`, accept the position; otherwise retry
5. Raise `RuntimeError` after `max_attempts_per_atom` consecutive failures

```python
from amorphouspy.structure import create_random_atoms

atoms_list, atom_counts = create_random_atoms(
    composition={"SiO2": 75, "Na2O": 15, "CaO": 10},
    target_atoms=3000,
    box_length=35.0,      # Å
    min_distance=1.6,     # Å (typical for oxide glasses)
    seed=42,              # Reproducible results
)

# atoms_list → [{"element": "Si", "position": [1.2, 3.4, 5.6]}, ...]
# atom_counts → {"Si": 750, "Na": 100, "Ca": 50, "O": 2100}
```

> **Tip:** If you get a `RuntimeError` about placement failure, either increase `box_length` or decrease `min_distance`. A `min_distance` of 1.6 Å works well for most oxide glasses; the melt-quench simulation will relax any remaining stress.

---

## System Planning

### `plan_system(composition, target, mode, target_type)`

Generates a comprehensive plan that converts a composition and target size into concrete integer formula units. This is the core allocation algorithm used internally by `get_structure_dict()`.

The algorithm uses the **largest-remainder method** to fairly distribute formula units:

1. Calculate the ideal (fractional) number of formula units for each oxide based on composition and target
2. Assign the integer floor to each oxide
3. Distribute remaining units one-by-one to oxides with the largest fractional remainder
4. This minimizes the deviation from ideal composition while ensuring integer counts

```python
from amorphouspy.structure import plan_system

# Plan for ~3000 atoms
plan = plan_system(
    composition={"CaO": 0.25, "Al2O3": 0.25, "SiO2": 0.5},
    target=3000,
    target_type="atoms",    # or "molecules"
    mode="molar",
)

print(plan["formula_units"])   # {'CaO': 150, 'Al2O3': 150, 'SiO2': 300}
print(plan["total_atoms"])     # 3000
print(plan["element_counts"])  # {'Ca': 150, 'O': 1500, 'Al': 300, 'Si': 300}
print(plan["mol_fraction"])    # {'CaO': 0.25, 'Al2O3': 0.25, 'SiO2': 0.5}
```

---

## Box Size Calculation

### `get_box_from_density(composition, n_molecules, target_atoms, mode, density, stoichiometry)`

Calculates the cubic box side length needed to achieve a target density. If `density` is not provided, it is estimated using `get_glass_density_from_model()`.

The calculation:

$$
L = \left(\frac{V_{\text{Å}^3}}{1}\right)^{1/3} \quad \text{where} \quad V_{\text{Å}^3} = \frac{m_{\text{total}}}{N_A \cdot \rho} \times 10^{24}
$$

```python
from amorphouspy.structure import get_box_from_density

box_length = get_box_from_density(
    composition={"SiO2": 75, "Na2O": 15, "CaO": 10},
    target_atoms=3000,
    n_molecules=None,
    density=2.5,     # g/cm³ (or None for auto)
)

print(f"Box length: {box_length:.2f} Å")
```

---

## Utility Functions

### Formula parsing

| Function | Description | Example |
|---|---|---|
| `parse_formula(formula)` | Parse oxide into element counts | `"Al2O3"` → `{"Al": 2, "O": 3}` |
| `formula_mass_g_per_mol(formula)` | Molar mass of a compound | `"SiO2"` → `60.08` g/mol |
| `extract_stoichiometry(composition)` | Stoichiometry of all composition components | See below |

```python
from amorphouspy.structure import parse_formula, extract_stoichiometry

# Single formula
parse_formula("Ca3(PO4)2")  # {'Ca': 3, 'P': 2, 'O': 8}

# Full composition
extract_stoichiometry({"SiO2": 0.75, "Na2O": 0.15, "CaO": 0.10})
# {'SiO2': {'Si': 1, 'O': 2}, 'Na2O': {'Na': 2, 'O': 1}, 'CaO': {'Ca': 1, 'O': 1}}
```

### Composition conversion

| Function | Description |
|---|---|
| `weight_percent_to_mol_fraction(comp)` | Convert weight% dictionary → molar fraction dictionary |
| `check_neutral_oxide(oxide)` | Validate an oxide formula is charge-neutral (raises `ValueError` if not) |
| `validate_target_mode(n_molecules, target_atoms)` | Ensure exactly one target mode is specified |

### Geometry utilities

| Function | Description |
|---|---|
| `minimum_image_distance(pos1, pos2, box_length)` | Minimum image distance in a cubic periodic box |
| `get_box_from_density(...)` | Calculate box length from composition and density |
