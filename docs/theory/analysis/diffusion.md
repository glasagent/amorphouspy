# Self-Diffusion & Mean-Squared Displacement

Self-diffusion quantifies how far atoms wander over time. It characterises ionic mobility in melts and glasses (e.g. alkali transport), sets the time scale for structural relaxation, and — together with formal charges — yields an estimate of the ionic conductivity.

---

## Theory

### Mean-squared displacement

The self part of atomic motion is captured by the mean-squared displacement (MSD) of species $s$,

$$
\mathrm{MSD}_s(t) = \frac{1}{N_s}\sum_{i\in s}\big\langle\,|\mathbf{r}_i(t_0+t) - \mathbf{r}_i(t_0)|^2\,\big\rangle_{t_0},
$$

averaged over all atoms of the species and over all available time origins $t_0$. The displacements use **unwrapped** coordinates, so an atom crossing a periodic boundary contributes its true path, not an artificial jump back into the box.

### Einstein relation

In the diffusive regime the MSD grows linearly with time, and the self-diffusion coefficient follows from the Einstein relation in $d$ dimensions:

$$
\mathrm{MSD}(t) = 2\,d\,D\,t \qquad\Rightarrow\qquad D = \frac{1}{2d}\,\frac{\mathrm{d}\,\mathrm{MSD}}{\mathrm{d}t}
\quad (d = 3 \Rightarrow \mathrm{MSD} = 6 D t).
$$

$D$ is obtained from the slope of a straight-line fit over the diffusive window. At short times motion is ballistic ($\mathrm{MSD}\propto t^2$) and at long times the single-origin average becomes noisy, so the fit excludes both ends. The log–log slope of the MSD over the fit window is reported; a value far from 1 signals that the trajectory has not reached the diffusive regime.

### Temperature dependence (Arrhenius)

Diffusion in glass-forming oxides is thermally activated. Fitting $D(T)$ at several temperatures to an Arrhenius law

$$
D(T) = D_0\,\exp\!\left(-\frac{E_a}{k_\mathrm{B} T}\right)
$$

(a linear fit of $\ln D$ versus $1/T$) gives the activation energy $E_a$ and prefactor $D_0$.

### Ionic conductivity (Nernst–Einstein)

Assuming uncorrelated ionic motion (Haven ratio $H_R = 1$), the diffusion coefficients map onto an ionic conductivity through the Nernst–Einstein relation,

$$
\sigma = \frac{1}{H_R}\sum_i \frac{N_i}{V}\,\frac{(z_i e)^2 D_i}{k_\mathrm{B} T},
$$

summed over mobile species $i$ with formal charge $z_i$. Because cross-correlations between ions are ignored, this is an upper-bound estimate.

---

## Implementation

`diffusion_simulation` runs an NVT production stage that dumps on a **log-then-linear** schedule (below) and computes a **single-origin** MSD — $\langle |r(t) - r(0)|^2\rangle$ referenced to the first frame — because the non-uniform time grid rules out a multi-time-origin estimator. It uses LAMMPS' image-flag-unwrapped coordinates (`xsu ysu zsu`) and removes the per-frame mass-weighted centre-of-mass drift by default.

For *uniformly*-sampled trajectories, `compute_msd(method="fft")` offers a multi-time-origin **Fast Correlation Algorithm** (FFT, Calandrini et al. 2011) that averages over every time origin in $O(N\log N)$ for better statistics; when frames lack unwrapped coordinates it unwraps on the fly from wrapped positions and per-frame cells.

### Log-then-linear dump schedule

To keep production trajectories small, the production stage dumps on a **log-then-linear** schedule:
decade-log below `crossover_ps` (with `points_per_decade=9` the dumped steps are 1, 2, …, 9, 10, 20, …, 90, 100, …, resolving the ballistic→diffusive crossover) and uniform at `linear_interval_ps` above it.
This shrinks the trajectory by one to two orders of magnitude (e.g. a ~6 GB uniform 1 ns run for 6000 atoms becomes ~80 MB).

Implementation: an equal-style LAMMPS variable drives `dump_modify <id> every v_...`, and the identical schedule is reproduced in Python (`log_linear_dump_steps`) to recover each frame's exact time (LAMMPS' parser reports thermo, not dump, steps). Fit the diffusive (linear) region above the crossover.

### Usage

```python
from amorphouspy import (
    diffusion_simulation, compute_msd, get_diffusion,
    fit_arrhenius, nernst_einstein_conductivity, save_frames,
)
import ase.io, gzip

# Run an MD production trajectory (log-then-linear dump) and save it as gzipped extXYZ.
out = diffusion_simulation(
    structure, potential, temperature_sim=3000.0,
    crossover_ps=100.0, linear_interval_ps=10.0,
    save_trajectory="trajectory.xyz.gz",
)
D_na = out["diffusion"]["per_species"]["Na"]["diffusion_cm2_s"]

# Reload the saved trajectory and re-analyse it — no need to re-run the MD.
with gzip.open("trajectory.xyz.gz", "rt") as fh:
    frames = ase.io.read(fh, index=":", format="extxyz")
diffusion = get_diffusion(compute_msd(frames, method="single_origin"))

# Activation energy from D(T) collected at several temperatures.
(D0, Ea_eV), _ = fit_arrhenius(temperatures, diffusivities)

# Ionic conductivity from per-species D (m^2/s) and formal charges.
sigma = nernst_einstein_conductivity(
    {"Na": D_na_m2_s}, {"Na": 1.0},
    n_per_species={"Na": n_na}, volume_a3=volume, temperature=3000.0,
)
```

### Saving & reloading

`save_trajectory=".../traj.xyz.gz"` (or a direct `save_frames(frames, path)` call) writes a **gzipped extXYZ** trajectory. It preserves the unwrapped coordinates and per-frame step/temperature, so it reloads straight back into `compute_msd(method="single_origin")`.

### Units & conventions

- Positions in Å, time in ps; MSD in Å²; $D$ reported in both cm²/s and m²/s
  ($1\,\text{Å}^2/\text{ps} = 10^{-4}\,\text{cm}^2/\text{s} = 10^{-8}\,\text{m}^2/\text{s}$).
- Production trajectories are dumped id-sorted (`dump_modify ... sort id`) so an atom keeps its identity across frames.

---

## References

Calandrini, V., Pellegrini, E., Calligari, P., Hinsen, K. & Kneller, G. R. nMoldyn - Interfacing spectroscopic experiments, molecular dynamics simulations and models for time correlation functions. *Collection SFN* **12**, 201–232 (2011). <https://doi.org/10.1051/sfn/201112010>
