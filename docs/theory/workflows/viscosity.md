# Viscosity Calculation

This workflow computes the shear viscosity of glass-forming melts using the **Green-Kubo method**, which integrates the stress autocorrelation function (SACF) from equilibrium MD simulations.

---

## Theory

### Green-Kubo Formalism

The shear viscosity $\eta$ is computed from the integral of the off-diagonal stress autocorrelation function:

$$
\eta = \frac{V}{k_B T} \int_0^{\infty} \langle \sigma_{xy}(0) \cdot \sigma_{xy}(t) \rangle \, dt
$$

where $V$ is the volume, $T$ is temperature, and $\sigma_{xy}$ is a component of the off-diagonal stress tensor.

In practice, we average over all three independent off-diagonal components ($xy$, $xz$, $yz$) and optionally the three normal stress differences for better statistics:

$$
\eta = \frac{V}{6 k_B T} \int_0^{\infty} \sum_{\alpha\beta} \langle \sigma_{\alpha\beta}(0) \cdot \sigma_{\alpha\beta}(t) \rangle \, dt
$$

### SACF Fitting

The stress autocorrelation function (SACF) decays from its initial value to zero. In practice it is noisy, so a fit is applied. The integrated SACF (running integral) typically shows a plateau — the plateau value is the viscosity.

The fitting uses a stretched exponential (Kohlrausch-Williams-Watts) model:

$$
C(t) = A \, \exp\left[-\left(\frac{t}{\tau}\right)^\beta\right]
$$

where $\tau$ is the relaxation time and $\beta$ is the stretching exponent.

---

## Usage

### `viscosity_simulation(structure, potential, ...)`

```python
from amorphouspy import viscosity_simulation

result = viscosity_simulation(
    structure=glass_structure,
    potential=potential,
    temperature=2000.0,         # K (must be above Tg!)
    n_steps=1_000_000,          # Long run for statistics
    equilibration_steps=50_000, # Equilibrate before measuring
    timestep=1.0,               # fs
    correlation_length=10_000,  # Max correlation lag (steps)
    dump_stress_interval=10,    # Stress sampling frequency
)

print(f"Viscosity: {result['viscosity']:.3f} Pa·s")
print(f"Viscosity: {result['viscosity_poise']:.3f} Poise")
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | Equilibrated structure (ideally at the target T) |
| `potential` | `DataFrame` | — | Potential configuration |
| `temperature` | `float` | — | Simulation temperature in K (must be above $T_g$) |
| `n_steps` | `int` | `1_000_000` | Number of production MD steps |
| `equilibration_steps` | `int` | `50_000` | Equilibration steps |
| `timestep` | `float` | `1.0` | MD timestep (fs) |
| `correlation_length` | `int` | `10_000` | Maximum lag for SACF (steps) |
| `dump_stress_interval` | `int` | `10` | Stress tensor output frequency |

**Returns:**

| Key | Type | Description |
|---|---|---|
| `"viscosity"` | `float` | Viscosity in Pa·s |
| `"viscosity_poise"` | `float` | Viscosity in Poise |
| `"sacf"` | `np.ndarray` | Stress autocorrelation function |
| `"sacf_integral"` | `np.ndarray` | Running integral of SACF |
| `"fit_params"` | `dict` | KWW fit parameters ($A$, $\tau$, $\beta$) |

---

## VFT Fitting

For temperature-dependent viscosity studies, the Vogel-Fulcher-Tammann (VFT) model is commonly used:

$$
\log_{10}(\eta) = A + \frac{B}{T - T_0}
$$

where $A$, $B$, and $T_0$ are fitting parameters.

```python
from scipy.optimize import curve_fit
import numpy as np

# Run viscosity at multiple temperatures
temperatures = [2000, 2500, 3000, 3500, 4000]
viscosities = []

for T in temperatures:
    result = viscosity_simulation(glass, potential, temperature=T, n_steps=500_000)
    viscosities.append(result["viscosity"])

# Fit VFT model
def vft(T, A, B, T0):
    return A + B / (T - T0)

popt, pcov = curve_fit(vft, temperatures, np.log10(viscosities))
print(f"VFT params: A={popt[0]:.2f}, B={popt[1]:.0f}, T0={popt[2]:.0f}")
```

---

## Practical Considerations

### Temperature range

- Green-Kubo only works **above the glass transition** ($T > T_g$) where the system is ergodic
- Typical range: 1500–5000 K for oxide glass melts
- At lower temperatures, the SACF does not decay to zero within accessible simulation times

### Run length

- The SACF must decay to zero for accurate viscosity
- Higher viscosity → longer decay → longer simulation needed
- Rule of thumb: total simulation time should be $\geq 100\tau$ where $\tau$ is the decay time

| Temperature | Typical $\tau$ | Minimum run |
|---|---|---|
| 5000 K | ~1 ps | 100 ps |
| 3000 K | ~10 ps | 1 ns |
| 2000 K | ~100 ps | 10 ns |
| Near $T_g$ | ~1 ns+ | Infeasible |

### System size

- 3000 atoms minimum
- Larger systems (5000–10,000) reduce stress tensor noise
- Very small systems (<1000) give unreliable viscosities

> **Tip:** Run long equilibration periods (≥50 ps) before production to ensure the system has reached the target temperature and density. Use NPT for equilibration, then switch to NVT for the production run.
