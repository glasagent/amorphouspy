# Structure Factor S(q)

The structure factor $S(q)$ is the reciprocal-space equivalent of the radial distribution function.
It is the quantity directly measured in neutron and X-ray total scattering experiments, making it
the primary bridge between MD simulations and experiment.

---

## Theory

### From real space to reciprocal space

Starting from the partial RDFs $g_{\alpha\beta}(r)$, the **Faber-Ziman partial structure factors**
are obtained via the isotropic sine transform:

$$
S_{\alpha\beta}(q) = 1 + \frac{4\pi\rho}{q} \int_0^{r_\mathrm{max}}
r\,\bigl[g_{\alpha\beta}(r) - 1\bigr]\,M(r)\,\sin(qr)\,\mathrm{d}r
$$

where $\rho = N/V$ is the total number density and $q = |\mathbf{q}|$ is the magnitude of the
momentum transfer.  This is the spherically-averaged (1D) form of the 3D Fourier transform,
valid for isotropic systems such as glasses.

### Lorch modification function

A finite simulation box imposes a hard cutoff at $r_\mathrm{max}$, which causes Gibbs-like
termination ripples in $S(q)$.  The **Lorch modification function**

$$
M(r) = \operatorname{sinc}\!\left(\frac{r}{r_\mathrm{max}}\right)
     = \frac{\sin(\pi r / r_\mathrm{max})}{\pi r / r_\mathrm{max}}
$$

smoothly damps the integrand to zero at $r_\mathrm{max}$, strongly suppressing these artefacts
at the cost of a slight broadening of sharp peaks.

---

## Neutron diffraction

For neutron scattering the total structure factor is:

$$
S(q) = 1 + \frac{1}{\langle b \rangle^2}
\sum_{\alpha \le \beta} (2 - \delta_{\alpha\beta})\,
c_\alpha c_\beta\, b_\alpha b_\beta\,
\bigl[S_{\alpha\beta}(q) - 1\bigr]
$$

where:

| Symbol | Meaning |
|---|---|
| $c_\alpha$ | atomic concentration of species $\alpha$ |
| $b_\alpha$ | coherent neutron scattering length (fm) — **q-independent constant** |
| $\langle b \rangle = \sum_\alpha c_\alpha b_\alpha$ | composition-averaged scattering length |
| $\delta_{\alpha\beta}$ | Kronecker delta (avoids double-counting like pairs) |

Because $b_\alpha$ is a nuclear property, it is **independent of $q$** and can be positive or
negative (e.g. $b_\mathrm{H} = -3.74$ fm).  This makes neutron weights fixed across the entire
$q$-range and provides unique sensitivity to light elements and isotope contrast.

### Tabulated scattering lengths

The implementation uses the Sears (1992) / NIST coherent scattering lengths:

| Element | Z | $b$ (fm) |
|---|---|---|
| H  |  1 | −3.739 |
| O  |  8 |  5.803 |
| Na | 11 |  3.630 |
| Si | 14 |  4.149 |
| Fe | 26 |  9.450 |
| Ni | 28 | 10.300 |

---

## X-ray diffraction

For X-ray scattering the same Faber-Ziman expression applies, but the constant scattering
lengths $b_\alpha$ are replaced by the **q-dependent atomic form factors** $f_\alpha(q)$:

$$
S(q) = 1 + \frac{1}{\langle f(q) \rangle^2}
\sum_{\alpha \le \beta} (2 - \delta_{\alpha\beta})\,
c_\alpha c_\beta\, f_\alpha(q)\, f_\beta(q)\,
\bigl[S_{\alpha\beta}(q) - 1\bigr]
$$

where $\langle f(q) \rangle = \sum_\alpha c_\alpha f_\alpha(q)$ is now itself $q$-dependent,
so both numerator and denominator vary across the diffractogram.

### Cromer–Mann parameterisation

The form factor is evaluated with the four-Gaussian Cromer–Mann fit:

$$
f_\alpha(q) = \sum_{i=1}^{4} a_i \exp\!\left(-b_i s^2\right) + c,
\qquad s = \frac{q}{4\pi}
$$

At $q = 0$, $f_\alpha(0) = Z_\alpha$ (total electron count).  The form factor decays monotonically
with $q$, so **heavy atoms dominate at low $q$ and contributions from all species fall off at
high $q$**.

---

## Neutron vs X-ray: key differences

| Property | Neutron | X-ray |
|---|---|---|
| Scatters from | Nucleus | Electron cloud |
| Weight $w_\alpha$ | Constant $b_\alpha$ (fm) | $q$-dependent $f_\alpha(q)$ (electrons) |
| $q$-dependence of weights | None | Decays to ~0 at high $q$ |
| Sensitivity to light elements | High (e.g. H, Li) | Low (scales with $Z$) |
| Isotope contrast | Yes (H vs D) | No |
| Normalization denominator | $\langle b \rangle^2$ — scalar | $\langle f(q) \rangle^2$ — vector |

For a Na₂O–SiO₂ glass the practical consequence is:

- **Neutron**: O ($b=5.80$ fm) and Si ($b=4.15$ fm) contribute comparably; the spectrum is
  sensitive to the Si–O–Si network.
- **X-ray**: Si ($Z=14$) dominates over O ($Z=8$) and Na ($Z=11$) at low $q$, but all
  contributions decay by $q \approx 15$ Å⁻¹.

---

## Usage

```python
from ase.io import read
from amorphouspy.analysis.structure_factor import compute_structure_factor

atoms = read("sodium_silicate.extxyz")

# Neutron S(q)
q, sq_n, sq_partials_n = compute_structure_factor(
    atoms,
    q_min=0.3,
    q_max=20.0,
    n_q=600,
    r_max=10.0,
    n_bins=2000,
    radiation="neutron",
    lorch_damping=True,
)

# X-ray S(q)
q, sq_x, sq_partials_x = compute_structure_factor(
    atoms,
    radiation="xray",
    lorch_damping=True,
)

# Access a partial: S_SiO(q)
sq_SiO = sq_partials_n[(8, 14)]
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `structure` | `Atoms` | — | ASE Atoms object with PBC |
| `q_min` | `float` | `0.5` | Minimum momentum transfer (Å⁻¹) |
| `q_max` | `float` | `20.0` | Maximum momentum transfer (Å⁻¹) |
| `n_q` | `int` | `500` | Number of $q$ grid points |
| `r_max` | `float` | `10.0` | Real-space RDF cutoff (Å) |
| `n_bins` | `int` | `2000` | Number of radial histogram bins |
| `radiation` | `str` | `"neutron"` | `"neutron"` or `"xray"` |
| `lorch_damping` | `bool` | `True` | Apply Lorch $M(r)$ to reduce termination ripples |
| `type_pairs` | `list[tuple[int,int]]` or `None` | `None` | Pairs to compute; `None` = all unique pairs |

### Returns

| Name | Shape | Description |
|---|---|---|
| `q` | `(n_q,)` | Momentum transfer grid (Å⁻¹) |
| `sq_total` | `(n_q,)` | Total $S(q)$ |
| `sq_partials` | `dict[(Z_a, Z_b) → (n_q,)]` | Faber-Ziman partial $S_{\alpha\beta}(q)$, keyed by `(min(Z), max(Z))` |

---

## References

1. Faber, T. E. & Ziman, J. M. (1965). *A theory of the electrical properties of liquid metals.* Phil. Mag. **11**, 153–173.
2. Lorch, E. (1969). *Neutron diffraction by germania, silica and radiation-damaged silica glasses.* J. Phys. C **2**, 229–237.
3. Sears, V. F. (1992). *Neutron scattering lengths and cross sections.* Neutron News **3**, 26–37.
4. Cromer, D. T. & Mann, J. B. (1968). *X-ray scattering factors computed from numerical Hartree–Fock wave functions.* Acta Cryst. A **24**, 321–324.
