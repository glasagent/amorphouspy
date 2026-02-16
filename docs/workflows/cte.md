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

### `cte_simulation(structure, potential, ...)`

```python
from amorphouspy import cte_simulation

result = cte_simulation(
    structure=glass_structure,
    potential=potential,
    temperature=300.0,           # K
    n_steps=500_000,             # Long run for convergence
    equilibration_steps=50_000,  # Equilibrate at T
    timestep=1.0,                # fs
    pressure=0.0,                # bar
)

print(f"Volumetric CTE: {result['alpha_v']:.2e} K⁻¹")
print(f"Linear CTE: {result['alpha_l']:.2e} K⁻¹")
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | Equilibrated glass structure |
| `potential` | `DataFrame` | — | Potential configuration |
| `temperature` | `float` | — | Simulation temperature (K) |
| `n_steps` | `int` | `500_000` | Number of production NPT steps |
| `equilibration_steps` | `int` | `50_000` | Equilibration steps |
| `timestep` | `float` | `1.0` | MD timestep (fs) |
| `pressure` | `float` | `0.0` | Target pressure (bar) |

**Returns:**

| Key | Type | Description |
|---|---|---|
| `"alpha_v"` | `float` | Volumetric CTE (K⁻¹) |
| `"alpha_l"` | `float` | Linear CTE (K⁻¹) |
| `"mean_volume"` | `float` | Mean volume (ų) |
| `"mean_enthalpy"` | `float` | Mean enthalpy (eV) |
| `"structure"` | `Atoms` | Final structure after simulation |

---

## Multi-Temperature CTE Study

For a complete thermal expansion study, run CTE simulations at multiple temperatures:

```python
import numpy as np

temperatures = np.arange(100, 1200, 100)  # 100 to 1100 K
cte_values = []
volumes = []

for T in temperatures:
    result = cte_simulation(
        glass_structure, potential,
        temperature=T,
        n_steps=500_000,
    )
    cte_values.append(result["alpha_l"])
    volumes.append(result["mean_volume"])

# Plot V(T) to identify glass transition
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=temperatures, y=volumes, mode="lines+markers"))
fig.update_layout(xaxis_title="Temperature (K)", yaxis_title="Volume (ų)")
fig.show()
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
