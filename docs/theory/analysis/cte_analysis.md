# CTE Analysis (Thermal Expansion)

The coefficient of thermal expansion (CTE) measures how a material's volume changes with temperature. In `amorphouspy`, CTE is computed from enthalpy-volume cross-correlations in NPT molecular dynamics simulations.

---

## Theory

### Statistical Mechanical Definition

The volumetric CTE $\alpha_V$ can be computed from NPT ensemble fluctuations using the cross-correlation between enthalpy ($H$) and volume ($V$):

$$
\alpha_V = \frac{\langle \delta H \cdot \delta V \rangle}{k_B \, T^2 \, \langle V \rangle}
$$

where:

| Symbol | Description |
|---|---|
| $\delta H = H - \langle H \rangle$ | Instantaneous enthalpy fluctuation |
| $\delta V = V - \langle V \rangle$ | Instantaneous volume fluctuation |
| $k_B$ | Boltzmann constant |
| $T$ | Temperature |
| $\langle V \rangle$ | Time-averaged volume |

The linear CTE is approximately:

$$
\alpha_L \approx \frac{\alpha_V}{3}
$$

### Why H-V Cross-Correlation?

This approach is statistically more efficient than directly fitting volume vs. temperature curves, because:
- It requires only a **single NPT simulation** at each temperature
- The cross-correlation converges faster than volume fluctuations alone
- It avoids the need for multiple simulations at slightly different temperatures

---

## Usage

### `cte_from_npt_fluctuations(temperature, enthalpy, volume, ...)`

This is the analysis function that computes CTE from pre-collected time series data.

```python
from amorphouspy.analysis.cte import cte_from_npt_fluctuations
import numpy as np

# enthalpy and volume are 1D arrays from NPT simulation output
# (e.g., from LAMMPS thermo output)

cte = cte_from_npt_fluctuations(
    temperature=300.0,            # K
    enthalpy=enthalpy_array,      # Shape: (n_steps,), units: eV
    volume=volume_array,          # Shape: (n_steps,), units: ų
)

print(f"Volumetric CTE: {cte:.2e} K⁻¹")
print(f"Linear CTE: {cte/3:.2e} K⁻¹")
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `temperature` | `float` | Simulation temperature (K) |
| `enthalpy` | `np.ndarray` | Time series of enthalpy values (eV) |
| `volume` | `np.ndarray` | Time series of volume values (ų) |

**Returns:**

| Type | Description |
|---|---|
| `float` | Volumetric CTE ($K^{-1}$) |

---

## Typical CTE Values for Oxide Glasses

| Glass system | Linear CTE ($\times 10^{-6}$ K⁻¹) |
|---|---|
| Vitreous SiO₂ | 0.5–0.6 |
| Borosilicate (Pyrex-type) | 3.2–3.5 |
| Soda-lime silicate | 8.5–9.5 |
| Lead silicate | 8–10 |
| Aluminosilicate | 4–6 |

> **Note:** MD-computed CTE values may differ from experimental values due to the high cooling rates used in simulation and the limitations of classical potentials. Trends across compositions are generally well-reproduced.

---

## Convergence

The H-V cross-correlation requires sufficiently long NPT trajectories for convergence. Guidelines:

- **Minimum**: 100 ps of production data (after equilibration)
- **Recommended**: 500 ps – 1 ns for reliable values
- **Check**: Plot the running average of $\langle \delta H \cdot \delta V \rangle$ to verify convergence

The CTE workflow module (`amorphouspy.workflows.cte`) includes built-in convergence checking.
