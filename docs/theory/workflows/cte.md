# CTE Simulation

This workflow computes the coefficient of thermal expansion (CTE) from NPT molecular dynamics simulations using the enthalpy-volume cross-correlation method.

---

## Method

### NPT Fluctuation Approach

The volumetric CTE is computed from the cross-correlation of enthalpy and volume fluctuations in a single NPT simulation:

$$
\alpha_V = \frac{\langle \delta H \cdot \delta V \rangle}{k_B \, T^2 \, \langle V \rangle}
$$

The linear CTE is then:

$$
\alpha_L = \frac{\alpha_V}{3}
$$

This approach requires only one NPT run per temperature point, making it computationally efficient.

### Alternative: Direct Volume Method

The CTE can also be extracted from the slope of the volume-temperature curve:

$$
\alpha_L = \frac{1}{3V_0} \frac{dV}{dT}
$$

This requires running NPT at multiple closely-spaced temperatures and fitting the $V(T)$ curve.

The fluctuation method (default in `amorphouspy`) is generally preferred for single-point calculations, while the direct method is useful for identifying the glass transition temperature from the $V(T)$ plot.

---

## Usage

### `cte_from_fluctuations_simulation(structure, potential, ...)`

```python
from amorphouspy import cte_from_fluctuations_simulation

result = cte_from_fluctuations_simulation(
    structure=glass_structure,
    potential=potential,
    temperature=300.0,           # K
    production_steps=200_000,    # Default production steps
    equilibration_steps=100_000, # Default equilibration steps
    timestep=1.0,                # fs
    pressure=1e-4,               # 1 bar in GPa
)

# Extract results from nested summary
summary = result["summary"]
print(f"Volumetric CTE: {summary['CTE_V_mean']:.2e} K⁻¹")
print(f"Linear CTE: {summary['CTE_x_mean']:.2e} K⁻¹")
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | Equilibrated glass structure |
| `potential` | `str` | — | Path to LAMMPS potential file |
| `temperature` | `float` | `300.0` | Simulation temperature (K) |
| `pressure` | `float` | `1e-4` | Target pressure (GPa) |
| `production_steps` | `int` | `200_000` | Steps for individual production runs |
| `equilibration_steps` | `int` | `100_000` | Equilibration steps |
| `timestep` | `float` | `1.0` | MD timestep (fs) |

**Returns:**

Returns a nested dictionary with `"summary"` and `"data"` keys. The `"summary"` contains averaged values and uncertainties.

---

### `temperature_scan_simulation(structure, potential, ...)`

For a complete thermal expansion study across a temperature range, use `temperature_scan_simulation`:

```python
import numpy as np
from amorphouspy import temperature_scan_simulation

temperatures = np.arange(100, 1200, 100).tolist()  # 100 to 1100 K

result = temperature_scan_simulation(
    glass_structure, 
    potential,
    temperature=temperatures,
    production_steps=200_000,
)

# Access data for each temperature
# Result structure: result["data"]["01_100K"]["run01"]...
```

The glass transition temperature ($T_g$) appears as a change in slope of the $V(T)$ curve:
- **Below $T_g$**: lower CTE (glassy state)  
- **Above $T_g$**: higher CTE (liquid-like state)

---

## Convergence

The H-V cross-correlation is sensitive to convergence. Guidelines:

| Duration (ps) | Reliability |
|---|---|
| < 100 | Poor — large fluctuations |
| 100–500 | Moderate — acceptable for trends |
| 500–1000 | Good — reliable single-point values |
| > 1000 | Excellent — publication quality |

The CTE workflow includes built-in convergence checking that monitors the running average of $\langle \delta H \cdot \delta V \rangle$ and reports whether the simulation has converged.

> **Tip:** If you see very noisy CTE values, increase the production run length. The H-V cross-correlation has higher statistical noise than simple volume averaging.
