# Trajectory Averaging

Structural properties computed from a single quenched frame carry finite-size noise — periodic replicas are small, and every snapshot is one instantaneous configuration out of the equilibrium ensemble. Averaging over several decorrelated frames drawn from the same equilibrated trajectory reduces that noise and lets you report a meaningful uncertainty.

In amorphouspy, averaging is handled by the standalone `average_over_frames` wrapper. Individual analysis functions always operate on a single frame — averaging is an explicit, separate step.

---

## Concept

Given $N$ equilibrated frames, `average_over_frames` calls the analysis function on each frame independently, then aggregates across frames.

### Mean

For each output position $i$ (array element, dict key, or scalar), the mean is the simple element-wise average:

$$\bar{x}_i = \frac{1}{N} \sum_{k=1}^{N} x_i^{(k)}$$

### Standard error of the mean (SEM)

The SEM quantifies how precisely the mean estimates the true ensemble average. It is computed from the sample standard deviation $\sigma$ with Bessel's correction:

$$\text{SEM}_i = \frac{\sigma_i}{\sqrt{N}}, \quad \sigma_i = \sqrt{\frac{1}{N-1} \sum_{k=1}^{N} \left(x_i^{(k)} - \bar{x}_i\right)^2}$$

The $N-1$ denominator (Bessel's correction) gives an unbiased estimate of the population variance. When $N = 1$ the correction is dropped ($N-1 \to N$) to avoid division by zero, and the SEM is 0 — a single frame has no variance to estimate.

The SEM shrinks as $1/\sqrt{N}$: doubling the number of frames halves the uncertainty, quadrupling halves it again.

### What gets averaged

Not every output can be averaged across frames:

- **Arrays and histograms** (e.g. $g(r)$, $S(q)$, angle histograms) — averaged element-wise.
- **Scalar distributions** (e.g. $Q_n$ counts, coordination distributions) — averaged per key across the union of all keys seen in any frame; missing keys fill as 0.
- **Nested dicts** (e.g. partial $Q_n$ by former type) — the same scalar averaging applied recursively to each inner dict.
- **Per-atom labels** (e.g. oxygen classification BO/NBO/free) — not averaged; the last frame's result is returned as-is.

---

## How to use it

Import `average_over_frames` and pass it the analysis function, a `list[Atoms]`, and any keyword arguments the function accepts.

```python
from ase.io import read
from amorphouspy.properties.structural.averaging import average_over_frames
from amorphouspy import compute_rdf

# Load 10 decorrelated frames from an equilibrated trajectory
frames = read("production.extxyz", index=":")  # returns list[Atoms]

# Single-frame call (unchanged)
r, rdfs, cumcn = compute_rdf(frames[0], r_max=8.0, n_bins=500)

# Multi-frame call via average_over_frames
(r, rdfs_mean, cumcn_mean), (_, rdfs_sem, cumcn_sem) = average_over_frames(
    compute_rdf,
    frames,
    r_max=8.0,
    n_bins=500,
)
```

`average_over_frames` always returns two tuples: `(means, sems)`. Each position in the means tuple corresponds to the same position in the sems tuple.

---

## Function return signatures

Single-frame signatures (unchanged — `average_over_frames` calls these internally):

| Function | Return tuple |
|---|---|
| `compute_rdf` | `(r, rdfs, cumcn)` |
| `compute_coordination` | `dist` (plain dict, not a tuple) |
| `compute_angles` | `(bin_centers, hist)` |
| `compute_qn` | `(total_qn, partial_qn)` |
| `compute_qn_and_classify` | `(total_qn, partial_qn, o_classes)` |
| `compute_structure_factor` | `(q, sq, partials)` |
| `compute_guttmann_rings` | `(histogram, mean_size)` |

When wrapped with `average_over_frames`, the return is `(means_tuple, sems_tuple)` with the same positional layout.

> `o_classes` from `compute_qn_and_classify` is a per-atom classification dict — it cannot be averaged. The last frame's value is passed through as-is; the corresponding SEM position is `None`.

---

## Full example: averaged RDF with error bands

```python
import numpy as np
import plotly.graph_objects as go
from ase.io import read
from amorphouspy import compute_rdf
from amorphouspy.properties.structural.averaging import average_over_frames

frames = read("production.extxyz", index=":")

(r, rdfs_mean, cumcn_mean), (_, rdfs_sem, cumcn_sem) = average_over_frames(
    compute_rdf,
    frames,
    r_max=8.0,
    n_bins=500,
    type_pairs=[(14, 8), (11, 8)],
)

fig = go.Figure()
for pair, g_mean in rdfs_mean.items():
    g_sem = rdfs_sem[pair]
    label = f"{pair[0]}-{pair[1]}"
    fig.add_trace(go.Scatter(x=r, y=g_mean, name=label, mode="lines"))
    fig.add_trace(go.Scatter(
        x=np.concatenate([r, r[::-1]]),
        y=np.concatenate([g_mean + g_sem, (g_mean - g_sem)[::-1]]),
        fill="toself", opacity=0.2, showlegend=False,
    ))

fig.update_layout(xaxis_title="r (Å)", yaxis_title="g(r)")
fig.show()
```

---

## Full example: averaged S(q)

```python
from amorphouspy import compute_structure_factor
from amorphouspy.properties.structural.averaging import average_over_frames

(q, sq_mean, partials_mean), (_, sq_sem, partials_sem) = average_over_frames(
    compute_structure_factor,
    frames,
    q_min=0.5,
    q_max=15.0,
    n_q=500,
)
```

---

## Full example: averaged Qn distribution

```python
from amorphouspy import compute_qn
from amorphouspy.properties.structural.averaging import average_over_frames

(total_qn_mean, partial_qn_mean), (total_qn_sem, partial_qn_sem) = average_over_frames(
    compute_qn,
    frames,
    cutoff=2.0,
    former_types=[14],  # Si
    o_type=8,
)

for n, mean in total_qn_mean.items():
    sem = total_qn_sem[n]
    print(f"Q{n}: {mean:.1f} ± {sem:.1f}")
```

---

## How many frames?

There is no universal minimum, but a few practical rules:

- Frames should be **decorrelated** — separated by at least one structural relaxation time. Taking every 10th MD step from an NVT run at 300 K is rarely sufficient; taking every 50–100 ps typically is.
- **5–20 frames** is usually enough to reduce statistical noise by 50–80 % compared to a single frame.
- The standard error of the mean naturally tells you when you have enough: if it is already smaller than your measurement uncertainty (e.g. 1 % of the peak height), adding more frames gives diminishing returns.
- With only 1 frame the SEM is 0 — a single frame has no variance to estimate.

---

## Input validation

| Input | Behaviour |
|---|---|
| `average_over_frames(fn, frames, ...)` with `len(frames) > 0` | ✅ averages over all frames |
| `average_over_frames(fn, [single_frame], ...)` | ✅ single-frame result with SEM = 0 |
| `fn(atoms, ...)` single `Atoms` | ✅ single-frame result |
| `fn([atoms, ...], ...)` list passed directly | ✅ uses first frame silently |
| `average_over_frames(fn, [], ...)` empty list | ❌ raises `ValueError` |
