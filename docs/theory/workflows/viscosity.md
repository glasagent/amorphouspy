# Viscosity Calculation

This workflow computes the shear viscosity of glass-forming melts from equilibrium NVT
molecular dynamics.  Two complementary methods are implemented:

- **Helfand moment method** (default) — integrates the stress tensor directly to form
  Helfand moments and extracts viscosity from the linear regime of their mean-square
  displacement.  Robust when the SACF decay time approaches the trajectory length.
- **Green-Kubo method** — integrates the stress autocorrelation function (SACF) up to
  an automatically detected cutoff time.

---

## Theory

### Helfand Moment Method

The Helfand moments are defined as the running time-integral of each off-diagonal stress
component:

$$
\chi_{\alpha\beta}(t) = \int_0^t \sigma_{\alpha\beta}(t') \, dt'
$$

The viscosity is extracted from the long-time linear slope of the mean-square displacement
of $\chi$:

$$
\eta = \frac{V}{2 k_B T} \lim_{t \to \infty} \frac{d}{dt}
      \left\langle \left| \chi_{\alpha\beta}(t) - \chi_{\alpha\beta}(0) \right|^2 \right\rangle
$$

This formulation is equivalent to Green-Kubo but avoids the need to choose an explicit
SACF integration cutoff, making it more robust for long-$\tau$ systems.

### Green-Kubo Method

The shear viscosity can also be computed directly from the SACF:

$$
\eta = \frac{V}{k_B T} \int_0^{\tau_{\mathrm{cut}}} \langle \sigma_{\alpha\beta}(0) \, \sigma_{\alpha\beta}(t) \rangle \, dt
$$

The cutoff $\tau_{\mathrm{cut}}$ is selected automatically via one of three strategies
(`noise_threshold`, `cumulative_integral`, or `self_consistent`).

### Vogel-Fulcher-Tammann (VFT) Model

Temperature-dependent viscosity data are typically described by the VFT equation:

$$
\log_{10}(\eta) = \log_{10}(\eta_0) + \frac{B \log_{10}(e)}{T - T_0}
$$

where $\eta_0$, $B$, and $T_0$ are fitting parameters.

---

## Usage

### High-level: `viscosity_simulation`

Runs an initial NVT production simulation, then iteratively extends it until the Helfand
viscosity converges.

```python
from amorphouspy import viscosity_simulation

result = viscosity_simulation(
    structure=glass_structure,      # pre-equilibrated ASE Atoms
    potential=potential_df,         # LAMMPS potential DataFrame
    temperature_sim=3000.0,         # K (must be above Tg)
    timestep=1.0,                   # fs
    initial_production_steps=1_000_000,  # ~1 ns at 1 fs
    max_total_time_ns=50.0,         # hard time-budget cap
    max_iterations=40,              # extension iterations
    eta_rel_tol=0.05,               # 5 % relative-change tolerance
    eta_stable_iters=3,             # stable iterations required
)

print(f"Viscosity : {result['viscosity_data']['viscosity']:.3e} Pa·s")
print(f"Converged : {result['converged']}")
print(f"Iterations: {result['iterations']}")
```

**Return keys:**

| Key | Description |
|---|---|
| `viscosity_data` | Output of `helfand_viscosity` from the final iteration |
| `result` | Accumulated raw MD arrays (`pressures`, `volume`, `temperature`) |
| `structure` | Final ASE `Atoms` object |
| `total_production_steps` | Total steps completed |
| `iterations` | Number of iterations run |
| `converged` | `True` if both convergence criteria were satisfied |

---

### Ensemble: `viscosity_ensemble`

Runs `n_replicas` independent trajectories with different random seeds and returns
ensemble-averaged viscosity with uncertainty.

```python
from amorphouspy import viscosity_ensemble

out = viscosity_ensemble(
    structure=glass_structure,
    potential=potential_df,
    n_replicas=3,
    temperature_sim=3000.0,
    timestep=1.0,
    initial_production_steps=1_000_000,
    server_kwargs={"cores": 4},     # MPI cores per replica
    parallel=False,                 # set True to run all replicas simultaneously
)

print(f"η = {out['viscosity']:.3e} ± {out['viscosity_sem']:.3e} Pa·s")
```

Replicas can also be dispatched to an HPC cluster via an
[executorlib](https://github.com/jan-janssen/executorlib) executor:

```python
from executorlib import SlurmJobExecutor

with SlurmJobExecutor(max_workers=12) as exe:
    out = viscosity_ensemble(..., executor=exe)
```

**Return keys:**

| Key | Description |
|---|---|
| `viscosity` | Mean shear viscosity (Pa·s) |
| `viscosity_fit_residual` | Sample std across replicas (Pa·s, ddof=1) |
| `viscosity_sem` | Standard error of the mean (Pa·s) |
| `shear_modulus_inf` | Mean infinite-frequency shear modulus (Pa) |
| `bulk_viscosity` | Mean bulk viscosity (Pa·s) |
| `maxwell_relaxation_time_ps` | Mean Maxwell relaxation time (ps) |
| `mean_pressure_gpa` | Mean pressure averaged across replicas (GPa) |
| `temperature` | Mean temperature (K) |
| `n_replicas` | Number of replicas run |
| `seeds` | Seeds actually used |
| `viscosities` | Per-replica shear viscosity values |
| `converged` | Per-replica convergence flags |
| `results` | Full `viscosity_simulation` output per replica |

---

### Low-level analysis: `helfand_viscosity` and `get_viscosity`

Both functions accept the `result` dict returned by `viscosity_simulation` (or
`_viscosity_simulation`) and perform post-processing only.

```python
from amorphouspy.workflows.viscosity import helfand_viscosity, get_viscosity

# Helfand method (recommended)
helfand = helfand_viscosity(result, timestep=1.0, output_frequency=1)
print(helfand["viscosity"])          # Pa·s
print(helfand["shear_modulus_inf"])  # Pa
print(helfand["bulk_viscosity"])     # Pa·s

# Green-Kubo method (legacy)
gk = get_viscosity(result, timestep=1.0, cutoff_method="noise_threshold")
print(gk["viscosity"])               # Pa·s
print(gk["sacf"])                    # normalised SACF list
```

---

### VFT fitting: `fit_vft` and `vft_model`

```python
import numpy as np
from amorphouspy.workflows.viscosity import fit_vft, vft_model

temperatures = np.array([2000, 2500, 3000, 3500, 4000], dtype=float)
log10_eta = np.array([-0.5, -1.0, -1.8, -2.5, -3.1])

popt, pcov = fit_vft(temperatures, log10_eta)
log10_eta0, B, T0 = popt
print(f"log10(η₀)={log10_eta0:.2f}, B={B:.0f} K, T0={T0:.0f} K")

# Evaluate the fitted model on a fine grid
T_fine = np.linspace(2000, 5000, 200)
log10_eta_fit = vft_model(T_fine, *popt)
```

---

## Convergence Criteria

`viscosity_simulation` considers the result converged only when **both** conditions hold
simultaneously for `eta_stable_iters` consecutive iterations:

1. **η-stability** — `|η_new − η_prev| / |η_prev| < eta_rel_tol`
2. **MSD linearity** — the local slope of the Helfand-moment MSD is flat in the last
   30 % of the lag window, confirming the diffusive regime has been reached.

---

## Practical Considerations

### Temperature range

- Both methods require the system to be ergodic, i.e. **above the glass transition** ($T > T_g$).
- Typical range for oxide glass melts: 1 500–5 000 K.
- Below $T_g$ the SACF does not decay within accessible simulation times.

### Run length and system size

| Temperature | Typical $\tau$ | Recommended run |
|---|---|---|
| 5 000 K | ~1 ps | 100 ps – 1 ns |
| 3 000 K | ~10 ps | 5–10 ns |
| 2 000 K | ~100 ps | ≥50 ns |
| Near $T_g$ | ~1 ns+ | Impractical |

- Minimum 3 000 atoms; 5 000–10 000 atoms recommended for low noise.
- Use the Helfand method or `viscosity_ensemble` for temperatures where $\tau \gtrsim 10$ ps.

> **Tip:** Pre-equilibrate the structure at the target temperature and density (NPT) before
> passing it to `viscosity_simulation`.  The built-in equilibration stages use Langevin
> dynamics followed by NVT, but a well-equilibrated input structure reduces the required
> equilibration time significantly.
