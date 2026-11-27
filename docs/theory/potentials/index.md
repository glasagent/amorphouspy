# Interatomic Potentials

`amorphouspy` provides built-in support for classical interatomic potentials widely used in oxide glass simulations, plus the opt-in GRACE-FS machine-learned potential. Each potential generates complete LAMMPS input configuration ready for simulation.

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
| `potential_type` | `str` | `"pmmcs"` | One of `"pmmcs"`, `"bjp"`, `"shik"`, `"bmp-harmonic"`, `"bmp-screened-harmonic"`, `"yang2026"`, or `"grace"` |
| `model_path` | `str \| Path \| None` | `None` | GRACE only: path to an exported `FS_model.yaml`; `None` uses the default cache location |

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

| Feature | PMMCS | BJP | SHIK | BMP-harm | BMP-shrm | Yang2026 | GRACE-FS |
|---|---|---|---|---|---|---|---|
| **Element coverage** | 28 elements | 4 elements (CAS) | 9 elements | 36 elements | 36 elements | 5 elements (Na/Ca borosilicate) | 89 elements (H–Bi, Ac–Pu) |
| **Functional form** | Morse + $r^{-12}$ | Born-Mayer-Huggins | Buckingham + $r^{-24}$ | Morse + Buck + harmonic 3b | Morse + Buck + screened 3b | Buckingham | Graph ACE (machine-learned) |
| **Coulomb solver** | coul/dsf | born/coul/dsf | coul/dsf | DSF / Wolf / PPPM / Ewald | DSF / Wolf / PPPM / Ewald | DSF (hybrid/overlay) / PPPM / Ewald | None (no charges) |
| **Cation charges** | Simple (multiples of 0.6) | Simple (multiples of 0.6) | Individually fitted | Simple (multiples of 0.6) | Simple (multiples of 0.6) | Scaled (×0.5 from formal) | — |
| **Oxygen charge** | Fixed ($-1.2$) | Fixed ($-1.2$) | Composition-dependent | Fixed ($-1.2$) | Fixed ($-1.2$) | Fixed ($-0.945$) | — |
| **Three-body terms** | No | No | No | Si/P triplets | Si/P/B/V triplets | No | Many-body (learned) |
| **Boron support** | Yes | No | Yes | B-free only | Yes (alkali/AE systems) | Yes | Yes |
| **Table files** | No | No | Yes (auto-generated) | Yes (auto-generated) | Yes (auto-generated) | No | Exported model (`FS_model.yaml`) |
| **Best for** | General oxide glasses | CAS systems | Si/Al/B oxide glasses | Multi-modifier silicates/phosphates | Borates and borosilicates | Na/Ca borosilicate glasses | Compositions no classical potential covers |
| **Cutoff** | 8.0 Å | 8.0 Å | 10.0 Å | 7.0 Å (SR), 8.0 Å (Coul) | 7.0 Å (SR), 8.0 Å (Coul) | 11.0 Å | Set by the model |

### Choosing a Potential

- **PMMCS** — Use for general multi-component oxide glasses. Broadest element support (28 elements). Good default choice.
- **BJP** — Use for calcium aluminosilicate (CAS) systems. Specifically parameterized for Ca-Al-Si-O.
- **SHIK** — Use for silicate, aluminosilicate, and borosilicate glasses. Composition-dependent oxygen charge provides better accuracy for mixed-modifier systems.
- **BMP-harm** — Use for multi-modifier silicate or phosphosilicate glasses without boron. Explicit harmonic three-body terms improve angular ordering around Si and P.
- **BMP-shrm** — Use for borate and borosilicate glasses. The screened three-body form covers B and V triplets and is the recommended choice when boron is present.
- **Yang2026** — Use for Na/Ca borosilicate glasses (Na, Ca, B, Si, O). Transferable classical potential validated against machine-learning-optimized parameters.
- **GRACE-FS** — Use when no classical potential covers the composition, or as a machine-learned reference. Much more expensive, requires a one-time model export, and is never auto-selected.

---

## Detailed Guides

Each potential has its own detailed page:

- [**PMMCS (Pedone)**](pmmcs.md) — Morse + repulsive wall + Coulomb
- [**BJP (Bouhadja)**](bjp.md) — Born-Mayer-Huggins + Coulomb
- [**SHIK (Sundararaman)**](shik.md) — Buckingham + $r^{-24}$ + Coulomb
- [**BMP (Bertani–Menziani–Pedone)**](bmp.md) — Morse + Buckingham + three-body angular terms
- [**Yang2026 (Yang et al.)**](yang2026.md) — Buckingham + Coulomb for Na/Ca borosilicate glasses
- [**GRACE-FS**](grace.md) — universal machine-learned potential (opt-in)
