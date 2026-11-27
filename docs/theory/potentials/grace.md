# GRACE-FS Potential

GRACE (Graph Atomic Cluster Expansion) is a machine-learned interatomic potential. `amorphouspy` supports its single-layer "FS" variant through LAMMPS's native `pair_style grace/fs`, using the pretrained universal foundation model `GRACE-FS-OAM`. Unlike the classical potentials, it needs no per-element charges or pair parameters, but it is considerably more expensive per atom and step.

GRACE is **opt-in**: `select_potential()` never chooses it automatically. Use it by passing `potential_type="grace"` explicitly.

---

## Reference

> A. Bochkarev, Y. Lysogorskiy, R. Drautz. "Graph Atomic Cluster Expansion for Semilocal Interactions beyond Equivariant Message Passing", *Phys. Rev. X* **14**, 021036 (2024). [DOI:10.1103/PhysRevX.14.021036](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.14.021036)

Model documentation: [gracemaker.readthedocs.io](https://gracemaker.readthedocs.io/)

---

## Prerequisite: Exporting the Model

`amorphouspy` does not download or export models. It only reads an already-exported native `FS_model.yaml`. The export tools (`grace-tensorpotential`) depend on TensorFlow, which requires Python ≤ 3.12, so run this one-time step in a **separate** environment:

```bash
conda create -n grace-export -c conda-forge python=3.12 grace-tensorpotential
conda activate grace-export

model_name=GRACE-FS-OAM
checkpoint_dir=~/.cache/grace/checkpoints/$model_name
grace_models checkpoint $model_name
grace_utils -p $checkpoint_dir/model.yaml -c $checkpoint_dir/checkpoint export -sf -n $checkpoint_dir/FS_model.yaml
```

The default lookup path is `~/.cache/grace/checkpoints/GRACE-FS-OAM/FS_model.yaml`. If the file is missing, `generate_potential` raises `FileNotFoundError` with these commands. If a checkpoint name is not recognised, `grace_models list` shows the names available in your `grace-tensorpotential` version.

Only single-layer (local) checkpoints are supported. Two-layer and three-layer "semilocal" checkpoints require a TensorFlow- or KOKKOS-linked LAMMPS build, which `amorphouspy` does not provide. `pair_style grace/fs` requires LAMMPS ≥ 2025.07.22 (pinned in `pixi.toml`).

---

## Supported Elements

GRACE-OAM models are trained on OMat24 and fine-tuned on sAlex and MPtrj, which cover **89 elements**: Z = 1–83 (H–Bi) and Z = 89–94 (Ac–Pu). Po, At, Rn, Fr, Ra and elements heavier than Pu are not supported.

---

## Usage

```python
from amorphouspy import get_structure_dict, get_ase_structure, generate_potential, melt_quench_simulation

structure_dict = get_structure_dict({"SiO2": 70, "Na2O": 30}, target_atoms=3000)
atoms = get_ase_structure(structure_dict)

# Default model location
potential = generate_potential(structure_dict, potential_type="grace")

# Model exported elsewhere
potential = generate_potential(structure_dict, potential_type="grace", model_path="/data/models/FS_model.yaml")

result = melt_quench_simulation(structure=atoms, potential=potential, thermostat="nose_hoover")
```

`model_path` is only valid for `potential_type="grace"`. The generated configuration uses `units metal`, `atom_style atomic`, and `pair_coeff * * "<model path>" <elements>`; there are no charges or electrostatics settings.

---

## Melt-Quench Protocol

The `grace_protocol` uses a default melt temperature of **3500 K** and a default cooling rate of **10 K/ps** (`1e13` K/s). All NPT stages run at 0 GPa.

| Stage | Ensemble | Temperature | Duration |
|---|---|---|---|
| 0 | Langevin + `nve/limit` | $T_\text{melt}$ | 10,000 steps (skipped when `pre_equilibrate=False`) |
| 1 | NVT | $T_\text{melt}$ | 10 ps |
| 2 | NPT | $T_\text{melt}$ | 50 ps |
| 3 | NPT | $T_\text{melt}$ → 300 K | cooling at 10 K/ps (320 ps by default) |
| 4 | NPT | 300 K | 100 ps |
| 5 | NVT | 300 K | 10 ps |

`equilibration_steps` overrides the two NPT holds (stages 2 and 4) together; the NVT holds are fixed. For independent control of each stage, chain `md_simulation` calls instead (see the GRACE tutorial notebook).

---

## Limitations

- **Cost**: much slower per atom and step than the classical potentials; plan system sizes and run lengths accordingly. The minimum atoms-per-core used for automatic core selection (500) is a placeholder that has not yet been benchmarked.
- **Model export** is a manual step in a separate Python ≤ 3.12 environment.
- **Single-layer checkpoints only** (`pair_style grace/fs`).
- **API**: the API always sends its own `quench_rate` (default `1e12` K/s), so GRACE jobs submitted through the API cool at 1 K/ps unless `quench_rate` is set.
