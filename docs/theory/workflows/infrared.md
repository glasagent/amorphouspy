# Infrared Response

The infrared (IR) response says which vibrations couple to light: a mode absorbs only if it moves charge. It is
computed by the same two routes as the [vibrational density of states](vdos.md), from the harmonic modes or from
the dynamics of an NVE run, and both give the complex dielectric function $\varepsilon(\nu)$, from which the
absorption coefficient and the reflectivity follow.

Both routes are options of the VDOS workflows, so the IR response comes from the same LAMMPS runs:

```python
harmonic = harmonic_vdos_simulation(glass, potential, infrared=True, eps_inf=2.1)
dynamic = vdos_simulation(glass, potential, thermostat="langevin", infrared=True, eps_inf=2.1)
harmonic["infrared"]["epsilon"], dynamic["infrared"]["epsilon"]
```

---

## Conventions

- Charges are in units of the elementary charge $e$, masses in amu, volumes in Å$^3$.
- Frequencies are ordinary frequencies $\nu$ in THz, as on the VDOS page.
- Fields vary as $e^{-i 2\pi\nu t}$, so absorption is $\operatorname{Im} \varepsilon \ge 0$.
- The response is the isotropic average over the three directions, as appropriate for a glass.

---

## Charges

Every potential of `generate_potential()` gives each element one fixed charge. This is the **rigid-ion model**:
the charge moves with the atom and nothing polarises. `charges_from_potential()` reads the charges back from the
potential, including a charge that depends on the composition (the oxygen charge of SHIK), and both workflows use
it unless `charges` is passed.

The rigid-ion model puts the peaks where the potential puts the modes, but its intensities are qualitative: the
electronic polarisation of the oxygen ions, which it leaves out, redistributes the intensity between the bands of
silica (Wilson *et al.*, 1996). Two arguments bring the missing physics in from outside:

- `charges` also accepts one Born effective charge tensor per atom, shape $(N, 3, 3)$, e.g. from a
  density-functional calculation (Pasquarello and Car, 1997). $Z_{i,\alpha\beta}$ is the dipole along $\alpha$
  created by a unit displacement of atom $i$ along $\beta$; a scalar charge $q_i$ stands for
  $q_i \delta_{\alpha\beta}$.
- `eps_inf` is the electronic dielectric constant $\varepsilon_\infty$, the square of the optical refractive
  index (about 2.1 for silica). The default 1 leaves it out.

The charges must sum to zero over the cell (the acoustic sum rule), or a rigid translation would carry a dipole;
a warning is issued otherwise.

---

## Harmonic route

### Oscillator strengths

The eigenvector $e_{3i+\beta,m}$ of the mass-weighted dynamical matrix displaces atom $i$ by
$e_{3i+\beta,m} / \sqrt{m_i}$ per unit normal coordinate. The dipole it creates per unit normal coordinate is the
**mode effective charge**, and its orientational average the **oscillator strength** (Gonze and Lee, 1997):

$$
\tilde Z_{m,\alpha} = \sum_i \sum_\beta \frac{Z_{i,\alpha\beta}\, e_{3i+\beta,m}}{\sqrt{m_i}},
\qquad f_m = \frac{1}{3} \sum_\alpha \tilde Z_{m,\alpha}^2 .
$$

$f_m$ is in $e^2/\mathrm{amu}$. Because the eigenvectors are orthonormal, the strengths obey a sum rule that does
not depend on the force constants: for a neutral cell

$$
\sum_m f_m = \frac{1}{3} \sum_i \frac{\operatorname{Tr}\left(Z_i Z_i^T\right)}{m_i},
$$

which is $\sum_i q_i^2 / m_i$ for scalar charges. The potential only decides how this total is shared between the
modes.

### Dielectric function

Each mode responds to a field as a damped oscillator. For a cell of volume $V$

$$
\varepsilon(\nu) = \varepsilon_\infty + \frac{e^2}{4\pi^2 \varepsilon_0 V}
\sum_m \frac{f_m}{\nu_m^2 - \nu^2 - i\gamma\nu},
$$

with the prefactor $e^2 / 4\pi^2\varepsilon_0 = 44224.5\ \mathrm{THz^2\,Å^3\,amu}/e^2$. The peaks of
$\operatorname{Im} \varepsilon$ are the transverse optical (TO) modes. The static limit is
$\varepsilon(0) = \varepsilon_\infty + (e^2 / 4\pi^2\varepsilon_0 V) \sum_m f_m / \nu_m^2$.

The damping $\gamma$ (`ir_damping_thz`, default 0.5 THz) is the full width at half maximum of the peaks. In a
harmonic calculation it is not a physical linewidth: it joins the discrete modes of a finite cell into a
spectrum, like the Gaussian broadening of the harmonic VDOS. Modes with $\nu_m \le 0$ are left out and counted.

---

## MD route

The charges moving with the velocities carry the current $J(t) = \sum_i Z_i\, v_i(t)$. It is the time derivative
of the dipole moment of the cell and, unlike the dipole itself, well defined in a periodic box. Linear response
(Kubo, 1957) gives the conductivity from the equilibrium fluctuations of the current, and the dielectric function
from the conductivity:

$$
\sigma(\nu) = \frac{1}{3 V k_B T} \int_0^\infty \langle J(0) \cdot J(t) \rangle\, e^{i 2\pi\nu t}\, dt,
\qquad \varepsilon(\nu) = \varepsilon_\infty + \frac{i\,\sigma(\nu)}{2\pi\nu\,\varepsilon_0}.
$$

- **Temperature.** $k_B T$ is taken from the same velocities,
  $k_B T = \langle \sum_i m_i |v_i|^2 \rangle / (3N - 3)$ after the centre-of-mass velocity is removed. The unit
  of the velocities therefore cancels, as it does in the VDOS.
- **Harmonic limit.** At equipartition every mode contributes $f_m$ to
  $\langle J \cdot J \rangle / 3 k_B T$, and the formula reduces to the harmonic dielectric function. The response
  of a harmonic oscillator is the same in classical and quantum mechanics, so no quantum correction factor
  applies. What the MD route adds is anharmonicity: the positions and widths of the peaks at the simulated
  temperature.
- **Sum rule.** The integral of $\operatorname{Re} \sigma$ over all frequencies is fixed by the equal-time value
  $\langle J \cdot J \rangle$ alone, so it does not depend on the window.
- **Estimate.** Each NVE block gives the unbiased correlation function over all time origins for the first half
  of its frames. It is multiplied by the decaying half of the `window` of the workflow and transformed. The
  cosine part uses the frames themselves. The sine part, which gives $\operatorname{Re} \varepsilon$, uses the
  band-limited interpolation of the correlation function on a 16 times finer time grid: on the frames
  themselves $\operatorname{Re} \varepsilon$ below a mode at $\nu_0$ would be too large by about
  $(2\pi\nu_0 \Delta t)^2 / 12$, 11% for a 37 THz mode sampled every 5 fs.
- **Zero frequency.** $\varepsilon(0)$ is returned as NaN. There $\sigma$ is the DC ionic conductivity, which is
  finite when modifier ions diffuse during the run, and $\varepsilon$ diverges. Close to zero
  $\operatorname{Im} \varepsilon$ divides a small $\operatorname{Re} \sigma$ by $\nu$ and is noisy.

The workflow returns the mean over the blocks and its standard error, for the real and imaginary parts
separately.

---

## Optical constants

Both routes pass their dielectric function to `compute_optical_response()`. With the complex refractive index
$n + ik = \sqrt{\varepsilon}$:

| Quantity | Formula | Meaning |
|---|---|---|
| Absorption coefficient | $\alpha = 4\pi\tilde\nu k$ | $I(x) = I_0 e^{-\alpha x}$, in cm$^{-1}$ with the wavenumber $\tilde\nu = \nu / c$ |
| Reflectivity | $R = \lvert(\sqrt{\varepsilon} - 1) / (\sqrt{\varepsilon} + 1)\rvert^2$ | Thick sample, normal incidence, in vacuum |
| Loss function | $\operatorname{Im}(-1/\varepsilon)$ | Peaks at the longitudinal optical (LO) modes |

For one undamped mode $\varepsilon$ crosses zero at $\nu_\mathrm{LO} = \nu_\mathrm{TO} \sqrt{\varepsilon(0) /
\varepsilon_\infty}$, the Lyddane-Sachs-Teller relation, so the LO-TO splitting grows with the strength of the
mode and shrinks with $\varepsilon_\infty$.

---

## Partial contributions

Restricting the sum over atoms to a group $G$ gives the group's mode effective charge $\tilde Z^G_m$ (harmonic) or
current $J_G$ (MD). The response is quadratic in them, so it splits into pairs of groups rather than into groups:

$$
f^{GG'}_m = \frac{2 - \delta_{GG'}}{3} \sum_\alpha \tilde Z^G_{m,\alpha}\, \tilde Z^{G'}_{m,\alpha} .
$$

The cross terms ($G \neq G'$) measure whether two groups move their charges together or against each other and
can be negative. Both workflows return `"partial_susceptibility"`, the share of every pair of labels in
$\varepsilon - \varepsilon_\infty$, keyed `"A-B"` for the groupings of `classify_vibrational_groups()`. The pairs
of `"element"` sum to the total, because every atom has one element; `"oxygen"` and `"qn"` cover only some atoms
and do not.

---

## Usage

### High-level

```python
from amorphouspy import generate_potential, harmonic_vdos_simulation, vdos_simulation

harmonic = harmonic_vdos_simulation(glass, potential, infrared=True, eps_inf=2.1, ir_damping_thz=0.5)
infrared = harmonic["infrared"]
infrared["strengths"]             # f_m per mode, e^2/amu
infrared["epsilon"]               # complex, on harmonic["frequencies_thz"]
infrared["absorption_cm"], infrared["reflectivity"], infrared["loss_function"]
infrared["partial_susceptibility"]["element"]["O-Si"]

dynamic = vdos_simulation(harmonic["structure"], potential, thermostat="langevin", infrared=True, eps_inf=2.1)
infrared = dynamic["infrared"]
infrared["frequencies_thz"], infrared["epsilon"], infrared["epsilon_sem"]
```

### Low-level

```python
from amorphouspy import (
    charges_from_potential,
    compute_dielectric_from_velocities,
    compute_dielectric_function,
    compute_ir_oscillator_strengths,
    compute_optical_response,
)

charges = charges_from_potential(potential, structure)

# Harmonic: from the modes of diagonalize_dynamical_matrix()
strengths = compute_ir_oscillator_strengths(modes["eigenvectors"], masses, charges, groups=groups)
dielectric = compute_dielectric_function(
    modes["frequencies_thz"], strengths["strengths"], structure.cell.volume, eps_inf=2.1
)

# MD: from a velocity trajectory of shape (n_frames, N, 3)
dielectric = compute_dielectric_from_velocities(
    velocities, masses, charges, dt_fs=5.0, volume_a3=structure.cell.volume, eps_inf=2.1
)

optics = compute_optical_response(dielectric["frequencies_thz"], dielectric["epsilon"])
```

---

## Parameters

| Parameter | Workflow | Default | Meaning |
|---|---|---|---|
| `infrared` | both | `False` | Compute the infrared response and return it in `"infrared"` |
| `charges` | both | `None` | Charges $(N,)$ or Born tensors $(N, 3, 3)$ in $e$; `None` reads the potential |
| `eps_inf` | both | 1.0 | Electronic dielectric constant $\varepsilon_\infty$ |
| `ir_damping_thz` | harmonic | 0.5 | Damping $\gamma$ of the modes in THz |
| `window` | MD | `"hann"` | Lag window of the current correlation function (shared with the VDOS) |

Missing or malformed charges raise before any LAMMPS run.

---

## Validation

The functions are tested against exact results: the completeness sum rule of the strengths, $q^2/\mu$ for the
stretch of a $\pm q$ pair, the static limit and the Lyddane-Sachs-Teller zero of one oscillator, the f-sum rule of
the conductivity (to rounding), and the agreement of the MD route with the harmonic route on a synthetic harmonic
trajectory.

A run with LAMMPS on a 300-atom silica glass with the PMMCS potential (`eps_inf=2.1`; MD thermostatted to 300 K, then 4 NVE blocks of
10 ps after 5 ps of equilibration) compares the two routes on a real system:

| Quantity | Harmonic | MD |
|---|---|---|
| $\sum_m f_m$ against $\sum_i q_i^2/m_i$ | 38.5103 against 38.5103 $e^2$/amu | - |
| $\operatorname{Re} \varepsilon$ averaged over 1-2 THz | 3.242 | 3.241 |
| $\operatorname{Im} \varepsilon$ averaged over 18-20 THz | 5.67 | 5.41 ± 1.11 |
| $\operatorname{Im} \varepsilon$ averaged over 24-26 THz | 1.10 | 1.05 ± 0.17 |

In this rigid-ion model two thirds of the strength lies between 15 and 20 THz and only about a tenth in the
Si-O stretching band above 25 THz, although the modes there are the expected asymmetric stretches. Measured
spectra of silica are dominated by the stretching band, so the relative intensities of the bands should not be
compared with experiment without Born charges.

---

## Practical considerations

- **Size.** A finite cell has discrete modes; the harmonic spectrum of a few hundred atoms needs the damping to
  look continuous, and features narrower than $\gamma$ are not resolved.
- **Resolution of the MD route.** The lag window spans half a block, so features are smoothed over about
  $2 / $`block_ps`. `window=None` cuts the correlation function sharply and makes the spectrum ring.
- **Negative absorption.** A lag window with negative side lobes can make $\operatorname{Im} \varepsilon$
  slightly negative next to a sharp peak; the block standard error shows whether a feature is significant.
- **Volume.** Both routes use the volume of the cell, which is fixed in all their stages.
- **Potentials without charges.** A machine-learned potential sets no charges; pass `charges` explicitly.

---

## References

- R. Kubo, "Statistical-mechanical theory of irreversible processes. I", *J. Phys. Soc. Jpn.* **12**, 570 (1957).
  <https://doi.org/10.1143/JPSJ.12.570>
- R. H. Lyddane, R. G. Sachs and E. Teller, "On the polar vibrations of alkali halides", *Phys. Rev.* **59**, 673
  (1941). <https://doi.org/10.1103/PhysRev.59.673>
- X. Gonze and C. Lee, "Dynamical matrices, Born effective charges, dielectric permittivity tensors, and
  interatomic force constants from density-functional perturbation theory", *Phys. Rev. B* **55**, 10355 (1997).
  <https://doi.org/10.1103/PhysRevB.55.10355>
- M. Wilson, P. A. Madden, M. Hemmati and C. A. Angell, "Polarization effects, network dynamics, and the infrared
  spectrum of amorphous SiO$_2$", *Phys. Rev. Lett.* **77**, 4023 (1996).
  <https://doi.org/10.1103/PhysRevLett.77.4023>
- A. Pasquarello and R. Car, "Dynamical charge tensors and infrared spectrum of amorphous SiO$_2$", *Phys. Rev.
  Lett.* **79**, 1766 (1997). <https://doi.org/10.1103/PhysRevLett.79.1766>
