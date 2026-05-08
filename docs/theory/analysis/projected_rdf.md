# Projected Radial Distribution Function

The projected RDF decomposes the standard pair correlation $g(r)$ into orientationally-resolved components using spherical harmonics. It answers the question: *do pairs at distance $r$ have a preferred direction, and in which direction?*

---

## Theory

### Expansion onto spherical harmonics

The full angular pair correlation can be written as:

$$
g(\mathbf{r}) = \sum_{l,m} g_{lm}(r) \, Y_l^m(\hat{\mathbf{r}})
$$

For amorphous materials the $l=0$ term recovers the standard isotropic $g(r)$. The $l=2$ terms carry the lowest-order anisotropy signal and are the ones accessible from glasses under deformation. The spherical harmonic decomposition of the pair correlation has been used to quantify shear- and deformation-induced structural anisotropy in model glass formers [1] and oxide glasses [2,3].

### $l=2$ components

Using real spherical harmonics and Cartesian unit-vector components $\hat{u}_x, \hat{u}_y, \hat{u}_z = \mathbf{r}_{ij}/r_{ij}$:

| Component | Cartesian form | Physical coupling |
|-----------|---------------|-------------------|
| $Y_2^0$ | $\sqrt{\tfrac{5}{4\pi}}\!\left(\tfrac{3}{2}\hat{u}_z^2 - \tfrac{1}{2}\right)$ | Uniaxial anisotropy along $z$ |
| $\text{Re}\,Y_2^1$ | $\sqrt{\tfrac{15}{8\pi}}\,\hat{u}_x\hat{u}_z$ | Shear coupling in $xz$ |
| $\text{Im}\,Y_2^1$ | $\sqrt{\tfrac{15}{8\pi}}\,\hat{u}_y\hat{u}_z$ | Shear coupling in $yz$ |
| $\text{Re}\,Y_2^2$ | $\sqrt{\tfrac{15}{32\pi}}\!\left(\hat{u}_x^2 - \hat{u}_y^2\right)$ | Uniaxial anisotropy in $xy$ |
| $\text{Im}\,Y_2^2$ | $\sqrt{\tfrac{15}{8\pi}}\,\hat{u}_x\hat{u}_y$ | Shear coupling in $xy$ |

### Uniaxial signal

$g_{20}(r)$ is accumulated with $z$ as the internal reference axis. For deformation along a different axis the signal is reconstructed analytically:

$$
g_\text{uniaxial}(r) = \begin{cases}
g_{20}(r) & \text{axis} = z \\
g_{20}(r) + \sqrt{3/5}\,\text{Re}\,g_{22}(r) & \text{axis} = x \\
g_{20}(r) - \sqrt{3/5}\,\text{Re}\,g_{22}(r) & \text{axis} = y
\end{cases}
$$

**Interpretation:**

- $g_\text{uniaxial}(r) > 0$ at the first peak: pairs at that bond length preferentially align *along* the deformation axis.
- $g_\text{uniaxial}(r) < 0$: pairs preferentially align *perpendicular* to the axis (transverse).
- $g_\text{uniaxial}(r) \approx 0$ everywhere: the structure is isotropic at that length scale.

### Shear signal

Each shear plane maps to the Y_lm component whose Cartesian form contains the off-diagonal product of that plane's two axes:

$$
g_\text{shear}(r) = \begin{cases}
\text{Im}\,g_{22}(r) & \text{plane} = xy \\
\text{Re}\,g_{21}(r) & \text{plane} = xz \\
\text{Im}\,g_{21}(r) & \text{plane} = yz
\end{cases}
$$

A nonzero value at the first-shell distance means pairs at that distance develop a net preference for the shear-plane direction.

---

## Computing the projected RDF

```python
from amorphouspy import compute_projected_rdf

# Uniaxial only (z-compression)
r, uniaxial, _ = compute_projected_rdf(
    structure,
    deformation_axis="z",
    r_max=8.0,
    n_bins=500,
)

# Shear only (xy plane)
r, _, shear = compute_projected_rdf(
    structure,
    shear_plane="xy",
    r_max=8.0,
    n_bins=500,
)

# Both at once
r, uniaxial, shear = compute_projected_rdf(
    structure,
    deformation_axis="z",
    shear_plane="xy",
    r_max=8.0,
    n_bins=500,
)
```

---

## Interactive tutorial

See the [Projected RDF Tutorial](../../notebooks/ProjectedRDFTutorial.ipynb) notebook for:

- Interactive 3D visualisations of the $Y_2^0$ and $Y_2^2$ spherical harmonic lobes with simulation-box overlays
- Worked example on a strained Si supercell showing a nonzero uniaxial signal
- Side-by-side comparison of isotropic vs deformed $g_\text{uniaxial}(r)$

---

## API reference

::: amorphouspy.analysis.projected_rdf.compute_projected_rdf

---

## References

1. J. Zausch and J. Horbach, "The build-up and relaxation of stresses in a glass-forming soft-sphere mixture under shear: A computer simulation study," *EPL* **88**, 60001 (2009). <https://doi.org/10.1209/0295-5075/88/60001>

2. S. Ganisetti, A. Atila, J. Guénolé, A. Prakash, J. Horbach, L. Wondraczek, and E. Bitzek, "The origin of deformation induced topological anisotropy in silica glass," *Acta Mater.* **257**, 119108 (2023). <https://doi.org/10.1016/j.actamat.2023.119108>

3. A. Atila and E. Bitzek, "Atomistic origins of deformation-induced structural anisotropy in metaphosphate glasses and its influence on mechanical properties," *J. Non-Cryst. Solids* **627**, 122822 (2024). <https://doi.org/10.1016/j.jnoncrysol.2024.122822>
