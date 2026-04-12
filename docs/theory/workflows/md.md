# Molecular Dynamics Simulation

The MD workflow provides a general-purpose interface for running NVT/NPT molecular dynamics simulations. It is used for equilibration, production runs, and as a building block for property calculations.

---

## Usage

### `md_simulation(structure, potential, ...)`

```python
from amorphouspy import md_simulation

result = md_simulation(
    structure=glass_structure,
    potential=potential,
    temperature_sim=300.0,    # K (start temperature)
    production_steps=10_000_000,
    timestep=1.0,             # fs
    n_print=1000,             # Write output every N steps
)

final_structure = result["structure"]
thermo = result["result"]
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | Input atomic structure |
| `potential` | `DataFrame` | — | Potential configuration |
| `temperature_sim` | `float` | `5000.0` | Start temperature in K (constant if `temperature_end` is None) |
| `production_steps` | `int` | `10_000_000` | Number of MD steps |
| `timestep` | `float` | `1.0` | MD timestep in femtoseconds |
| `n_print` | `int` | `1000` | Output frequency in steps |
| `temperature_end` | `float \| None` | `None` | End temperature for a linear ramp; None = constant temperature |
| `pressure` | `float \| None` | `None` | Start pressure in GPa; None = NVT, float value = NPT |
| `pressure_end` | `float \| None` | `None` | End pressure in GPa for a linear ramp; requires `pressure` to be set |
| `langevin` | `bool` | `False` | Use Langevin dynamics instead of Nosé-Hoover |
| `seed` | `int` | `12345` | Random seed for velocity initialization |
| `server_kwargs` | `dict \| None` | `None` | Additional arguments passed to the LAMMPS server |
| `tmp_working_directory` | `str \| Path \| None` | `None` | Directory for temporary simulation files |

**Returns:** `dict` with keys:

| Key | Type | Description |
|---|---|---|
| `"structure"` | `Atoms` | Final structure after MD |
| `"result"` | `dict` | Thermodynamic data (step, temp, pressure, energy, volume) |

---

## Ensembles

Ensemble selection is controlled by the `pressure` parameter:

- **NVT** (constant volume): leave `pressure=None` (the default)
- **NPT** (constant pressure): set `pressure` to a float value in GPa, e.g. `pressure=0.0`

### NVT example

```python
result = md_simulation(
    structure=glass_structure,
    potential=potential,
    temperature_sim=300.0,
    production_steps=50_000,
)
```

Use NVT for equilibration at a fixed density, structural analysis, or property calculations that require constant volume.

### NPT example

```python
result = md_simulation(
    structure=glass_structure,
    potential=potential,
    temperature_sim=300.0,
    production_steps=100_000,
    pressure=0.0,    # GPa — ambient pressure
)
```

Use NPT for density relaxation, pressure equilibration after quenching, or CTE calculations.

---

## Temperature and pressure ramps

Linear ramps are supported for both temperature and pressure:

```python
# Cool from 3000 K → 300 K while ramping pressure from 5 GPa → 0 GPa
result = md_simulation(
    structure=glass_structure,
    potential=potential,
    temperature_sim=3000.0,
    temperature_end=300.0,
    production_steps=500_000,
    pressure=5.0,
    pressure_end=0.0,
)
```

`pressure_end` requires `pressure` to be set; omitting it holds pressure constant.

---

## Timestep Selection

| System type | Recommended dt | Notes |
|---|---|---|
| Oxide glass (standard) | 1.0 fs | Safe for most potentials |
| SHIK potential (initial) | 0.25 fs | Steep $r^{-24}$ requires smaller dt |
| High-temperature melt | 0.5–1.0 fs | Fast dynamics benefit from smaller dt |
| Production at 300 K | 1.0–2.0 fs | Slower dynamics allow larger dt |

> **Warning:** Using a timestep that is too large will cause energy drift and eventually simulation instability. If you see the total energy drifting upward, reduce the timestep.

---

## Output Analysis

### Thermodynamic data

```python
import plotly.graph_objects as go

thermo = result["result"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=thermo["step"], y=thermo["temp"], name="Temperature"))
fig.update_layout(xaxis_title="Step", yaxis_title="T (K)")
fig.show()
```

### Saving the final structure

```python
from ase.io import write

write("final_structure.xyz", result["structure"])
```
