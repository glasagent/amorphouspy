# Interatomic Potentials

`amorphouspy` provides built-in support for three classical interatomic potentials widely used in oxide glass simulations. Each potential generates complete LAMMPS input configuration ready for simulation.

---

## Unified Interface

All potentials are accessed through a single function:

### `generate_potential(atoms_dict, potential_type="pmmcs")`

```python
from amorphouspy import get_structure_dict, generate_potential

structure_dict = get_structure_dict({"SiO2": 75, "Na2O": 15, "CaO": 10}, target_atoms=3000)
potential = generate_potential(structure_dict, potential_type="pmmcs")
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `atoms_dict` | `dict` | — | Structure dictionary from `get_structure_dict()` |
| `potential_type` | `str` | `"pmmcs"` | One of `"pmmcs"`, `"bjp"`, or `"shik"` |

**Returns:** A `pd.DataFrame` with columns:

| Column | Description |
|---|---|
| `Name` | Potential name identifier |
| `Filename` | Associated table files (if any) |
| `Model` | Model type descriptor |
| `Species` | List of element species |
| `Config` | List of LAMMPS input lines |

The `Config` column contains a list of strings — each string is a LAMMPS input command that sets up the pair style, pair coefficients, charges, and any additional parameters needed for the simulation.

---

## Comparison

| Feature | PMMCS | BJP | SHIK |
|---|---|---|---|
| **Element coverage** | 28 elements | 4 elements (CAS) | 9 elements |
| **Functional form** | Morse + $r^{-12}$ | Born-Mayer-Huggins | Buckingham + $r^{-24}$ |
| **Coulomb solver** | coul/dsf | born/coul/dsf | coul/dsf |
| **Cation charges** | Simple (multiples of 0.6) | Simple (multiples of 0.6) | Individually fitted |
| **Oxygen charge** | Fixed ($-1.2$) | Fixed ($-1.2$) | Composition-dependent |
| **Table files** | No | No | Yes (auto-generated) |
| **Best for** | General oxide glasses | CAS systems | Si/Al/B oxide glasses |
| **Cutoff** | 8.0 Å | 8.0 Å | 10.0 Å |

### Choosing a Potential

- **PMMCS** — Use for general multi-component oxide glasses. Broadest element support (28 elements). Good default choice.
- **BJP** — Use for calcium aluminosilicate (CAS) systems. Specifically parameterized for Ca-Al-Si-O.
- **SHIK** — Use for silicate, aluminosilicate, and borosilicate glasses. Composition-dependent oxygen charge provides better accuracy for mixed-modifier systems.

---

## Detailed Guides

Each potential has its own detailed page:

- [**PMMCS (Pedone)**](pmmcs.md) — Morse + repulsive wall + Coulomb
- [**BJP (Bouhadja)**](bjp.md) — Born-Mayer-Huggins + Coulomb
- [**SHIK (Sundararaman)**](shik.md) — Buckingham + $r^{-24}$ + Coulomb
