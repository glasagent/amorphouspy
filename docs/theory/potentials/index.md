# Interatomic Potentials

`amorphouspy` provides built-in support for classical interatomic potentials widely used in oxide glass simulations. Each potential generates complete LAMMPS input configuration ready for simulation.

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
| `potential_type` | `str` | `"pmmcs"` | One of `"pmmcs"`, `"bjp"`, `"shik"`, `"bmp-harmonic"`, or `"bmp-screened-harmonic"` |

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

| Feature | PMMCS | BJP | SHIK | BMP-harm | BMP-shrm |
|---|---|---|---|---|---|
| **Element coverage** | 28 elements | 4 elements (CAS) | 9 elements | 36 elements | 36 elements |
| **Functional form** | Morse + $r^{-12}$ | Born-Mayer-Huggins | Buckingham + $r^{-24}$ | Morse + Buck + harmonic 3b | Morse + Buck + screened 3b |
| **Coulomb solver** | coul/dsf | born/coul/dsf | coul/dsf | DSF / Wolf / PPPM / Ewald | DSF / Wolf / PPPM / Ewald |
| **Cation charges** | Simple (multiples of 0.6) | Simple (multiples of 0.6) | Individually fitted | Simple (multiples of 0.6) | Simple (multiples of 0.6) |
| **Oxygen charge** | Fixed ($-1.2$) | Fixed ($-1.2$) | Composition-dependent | Fixed ($-1.2$) | Fixed ($-1.2$) |
| **Three-body terms** | No | No | No | Si/P triplets | Si/P/B/V triplets |
| **Boron support** | Yes | No | Yes | B-free only | Yes (alkali/AE systems) |
| **Table files** | No | No | Yes (auto-generated) | Yes (auto-generated) | Yes (auto-generated) |
| **Best for** | General oxide glasses | CAS systems | Si/Al/B oxide glasses | Multi-modifier silicates/phosphates | Borates and borosilicates |
| **Cutoff** | 8.0 Å | 8.0 Å | 10.0 Å | 7.0 Å (SR), 8.0 Å (Coul) | 7.0 Å (SR), 8.0 Å (Coul) |

### Choosing a Potential

- **PMMCS** — Use for general multi-component oxide glasses. Broadest element support (28 elements). Good default choice.
- **BJP** — Use for calcium aluminosilicate (CAS) systems. Specifically parameterized for Ca-Al-Si-O.
- **SHIK** — Use for silicate, aluminosilicate, and borosilicate glasses. Composition-dependent oxygen charge provides better accuracy for mixed-modifier systems.
- **BMP-harm** — Use for multi-modifier silicate or phosphosilicate glasses without boron. Explicit harmonic three-body terms improve angular ordering around Si and P.
- **BMP-shrm** — Use for borate and borosilicate glasses. The screened three-body form covers B and V triplets and is the recommended choice when boron is present.

---

## Detailed Guides

Each potential has its own detailed page:

- [**PMMCS (Pedone)**](pmmcs.md) — Morse + repulsive wall + Coulomb
- [**BJP (Bouhadja)**](bjp.md) — Born-Mayer-Huggins + Coulomb
- [**SHIK (Sundararaman)**](shik.md) — Buckingham + $r^{-24}$ + Coulomb
- [**BMP (Bertani–Menziani–Pedone)**](bmp.md) — Morse + Buckingham + three-body angular terms
