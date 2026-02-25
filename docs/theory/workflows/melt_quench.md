# Melt-Quench Simulation

The melt-quench workflow transforms a random initial atomic configuration into a realistic amorphous glass structure by simulating the glass formation process: melting at high temperature, equilibrating the liquid, and rapidly cooling (quenching) to below the glass transition temperature.

---

## Process Overview

```mermaid
graph TD
    A[Random structure] --> B[Energy minimization]
    B --> C[Heat to melt temperature]
    C --> D[Equilibrate liquid]
    D --> E[Cool to glass temperature]
    E --> F[Final equilibration]
    F --> G[Quenched glass structure]
```

### Stages

1. **Energy minimization** — Relax the random initial configuration to remove atomic overlaps
2. **Heating** — Ramp temperature from low to the melting temperature (typically 3000–6000 K)
3. **Melt equilibration** — Hold at high temperature to equilibrate the liquid (lose memory of initial config)
4. **Cooling** — Ramp temperature down to the target glass temperature (typically 300 K)
5. **Final equilibration** — Hold at the target temperature to equilibrate the glass

---

## Basic Usage

### `melt_quench_simulation(structure, potential, ...)`

```python
from amorphouspy import melt_quench_simulation

result = melt_quench_simulation(
    structure=atoms,
    potential=potential,
    temperature_high=5000.0,     # Melt temperature (K)
    temperature_low=300.0,       # Quench target (K)
    heating_rate=1e12,           # K/s
    cooling_rate=1e12,           # K/s
    equilibration_steps=10_000,  # Steps at melt temperature
    timestep=1.0,                # fs
)

glass = result["structure"]     # Quenched ASE Atoms
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | Initial structure (from `get_ase_structure()`) |
| `potential` | `DataFrame` | — | Potential configuration (from `generate_potential()`) |
| `temperature_high` | `float` | — | Maximum (melting) temperature in K |
| `temperature_low` | `float` | — | Final (glass) temperature in K |
| `heating_rate` | `float` | `1e12` | Heating rate in K/s |
| `cooling_rate` | `float` | `1e12` | Cooling rate in K/s |
| `equilibration_steps` | `int` | `10_000` | Number of steps at the melt temperature |
| `timestep` | `float` | `1.0` | MD timestep in femtoseconds |

**Returns:** A dictionary with:

| Key | Type | Description |
|---|---|---|
| `"structure"` | `Atoms` | Final quenched glass structure |
| `"trajectory"` | `list[Atoms]` | Structures at each stage |
| `"thermo"` | `dict` | Thermodynamic data (T, P, E, V vs. step) |

---

## Potential-Specific Protocols

Each interatomic potential has an optimized multi-stage protocol that has been validated to produce high-quality glass structures.

### `melt_quench_protocol(structure, potential, potential_type, ...)`

```python
from amorphouspy import melt_quench_protocol

# Automatically selects the right protocol for the potential
result = melt_quench_protocol(
    structure=atoms,
    potential=potential,
    potential_type="pmmcs",  # or "bjp" or "shik"
)
```

### PMMCS Protocol

Multi-stage cooling with holds at intermediate temperatures:

| Stage | Temperature range | Ensemble | Duration |
|---|---|---|---|
| 1. Minimize | — | Conjugate gradient | — |
| 2. Heat | 300 → 5000 K | NVT | Variable (heating rate) |
| 3. Equilibrate | 5000 K | NVT | 50,000 steps |
| 4. Cool stage 1 | 5000 → 3000 K | NPT | Variable (cooling rate) |
| 5. Hold | 3000 K | NPT | 20,000 steps |
| 6. Cool stage 2 | 3000 → 300 K | NPT | Variable (cooling rate) |
| 7. Equilibrate | 300 K | NPT | 50,000 steps |

### SHIK Protocol

Includes a pressure ramp during cooling and Langevin pre-equilibration to handle the steep $r^{-24}$ repulsion:

| Stage | Temperature range | Ensemble | Special |
|---|---|---|---|
| 1. Langevin | — | Langevin | Small dt=0.25 fs, 10,000 steps |
| 2. Heat | 300 → 5000 K | NVT | Ramp temperature |
| 3. Equilibrate | 5000 K | NVT | 50,000 steps |
| 4. Cool | 5000 → 400 K | NPT | Pressure ramp: 1000 → 0 bar |
| 5. Equilibrate | 400 K | NPT | 50,000 steps |

The pressure ramp in stage 4 (`iso 1000.0 0.0`) helps the system densify correctly during cooling.

### BJP Protocol

Standard two-stage cooling optimized for CAS glasses with holds at intermediate temperatures.

---

## Cooling Rate Effects

The cooling rate is a critical parameter in MD glass simulations:

| Cooling rate (K/s) | MD equivalent | Notes |
|---|---|---|
| $10^{14}$ | Very fast | Highest fictive T, lowest density |
| $10^{13}$ | Fast | Standard rapid quench |
| $10^{12}$ | Moderate | Better structures, longer computation |
| $10^{11}$ | Slow | Closer to experimental, very expensive |
| $10^{0}$ (experiment) | Not accessible | MD cannot reach experimental rates |

> **Tip:** For production studies, use cooling rates of $10^{12}$–$10^{13}$ K/s. Slower rates give more realistic structures but the computational cost scales linearly. Generate multiple independent samples to assess statistical uncertainty.

---

## Tips

- **System size**: 3000–10,000 atoms is adequate for most structural properties. Use larger systems (~100,000 atoms) for ring statistics and long-range correlations.
- **Multiple samples**: Generate 3–5 independent glasses per composition using different random seeds for statistical averaging.
- **Density validation**: Compare the final glass density to the Fluegel model prediction or experimental values.
- **Structure inspection**: Always visualize the quenched structure (e.g., with ASE's `view()`) to catch obvious issues like phase separation or incomplete mixing.
