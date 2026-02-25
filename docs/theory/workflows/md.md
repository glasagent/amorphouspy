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
    temperature=300.0,       # K
    n_steps=50_000,          # Total MD steps
    timestep=1.0,            # fs
    ensemble="nvt",          # "nvt" or "npt"
    pressure=0.0,            # bar (only for "npt")
    dump_interval=100,       # Write trajectory every N steps
    thermo_interval=10,      # Write thermodynamics every N steps
)

final_structure = result["structure"]
trajectory = result["trajectory"]
thermo = result["thermo"]
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | Input atomic structure |
| `potential` | `DataFrame` | — | Potential configuration |
| `temperature` | `float` | — | Simulation temperature (K) |
| `n_steps` | `int` | — | Number of MD steps |
| `timestep` | `float` | `1.0` | MD timestep in femtoseconds |
| `ensemble` | `str` | `"nvt"` | `"nvt"` (constant volume) or `"npt"` (constant pressure) |
| `pressure` | `float` | `0.0` | Target pressure in bar (NPT only) |
| `dump_interval` | `int` | `100` | Trajectory dump frequency (steps) |
| `thermo_interval` | `int` | `10` | Thermodynamic output frequency (steps) |

**Returns:**

| Key | Type | Description |
|---|---|---|
| `"structure"` | `Atoms` | Final structure after MD |
| `"trajectory"` | `list[Atoms]` | Trajectory frames at dump intervals |
| `"thermo"` | `dict` | Thermodynamic data: step, temp, pressure, energy, volume |

---

## Ensembles

### NVT (Canonical)

Fixed temperature, constant volume. Uses the Nosé-Hoover thermostat with a damping parameter of 100 timesteps.

```python
result = md_simulation(
    structure=glass_structure,
    potential=potential,
    temperature=300.0,
    n_steps=50_000,
    ensemble="nvt",
)
```

Use NVT for:
- Equilibration at a fixed density
- Production runs for structural analysis
- Property calculations that require constant volume

### NPT (Isothermal-Isobaric)

Fixed temperature and pressure. Uses Nosé-Hoover thermostat + barostat with damping parameters of 100 and 1000 timesteps respectively.

```python
result = md_simulation(
    structure=glass_structure,
    potential=potential,
    temperature=300.0,
    n_steps=100_000,
    ensemble="npt",
    pressure=0.0,        # 0 bar ≈ ambient pressure
)
```

Use NPT for:
- Density relaxation
- Pressure equilibration after quenching
- CTE calculations (volume fluctuations)
- Simulating at experimental conditions

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

thermo = result["thermo"]

fig = go.Figure()
fig.add_trace(go.Scatter(x=thermo["step"], y=thermo["temp"], name="Temperature"))
fig.update_layout(xaxis_title="Step", yaxis_title="T (K)")
fig.show()
```

### Trajectory

```python
from ase.io import write

# Write trajectory to XYZ for visualization
write("trajectory.xyz", result["trajectory"])
```
