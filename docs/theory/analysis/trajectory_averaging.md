# Trajectory Averaging

Structural properties computed from a single quenched frame carry finite-size noise — periodic replicas are small, and every snapshot is one instantaneous configuration out of the equilibrium ensemble. Averaging over several decorrelated frames drawn from the same equilibrated trajectory reduces that noise and lets you report a meaningful uncertainty.

Every analysis function in amorphouspy supports this through the `frame_averaging` keyword argument.

---

## Concept

Given $N$ equilibrated frames, each analysis function computes the property independently per frame, then returns:

- **mean** — element-wise average across frames
- **SEM** — standard error of the mean, $\sigma / \sqrt{N}$, where $\sigma$ is the sample standard deviation (ddof = 1 for N > 1, ddof = 0 for N = 1)

The SEM quantifies the precision of the average: how much the mean would vary if you repeated the trajectory. It shrinks as $1/\sqrt{N}$, so doubling the number of frames halves the uncertainty.

---

## How to use it

Pass a `list[Atoms]` instead of a single `Atoms` object and set `frame_averaging=True`.

```python
from ase.io import read

# Load 10 decorrelated frames from an equilibrated trajectory
frames = read("production.extxyz", index=":")  # returns list[Atoms]

# Single-frame call (unchanged behaviour)
r, rdfs, cn = compute_rdf(frames[0], r_max=8.0, n_bins=500)

# Multi-frame call
r, rdfs_mean, cn_mean, rdfs_sem, cn_sem = compute_rdf(
    frames,
    r_max=8.0,
    n_bins=500,
    frame_averaging=True,
)
```

The return signature extends to a tuple of `(result_mean, ..., result_sem, ...)`. Each function documents the exact expanded tuple in its docstring.

---

## Function return signatures

| Function | Single-frame | Multi-frame (`frame_averaging=True`) |
|---|---|---|
| `compute_rdf` | `(r, rdfs, cumcn)` | `(r, rdfs_mean, cumcn_mean, rdfs_sem, cumcn_sem)` |
| `compute_coordination` | `(dist, per_atom)` | `(dist_mean, per_atom_last, dist_sem)` |
| `compute_angles` | `(bin_centers, hist)` | `(bin_centers, hist_mean, hist_sem)` |
| `compute_qn` | `(total_qn, partial_qn)` | `(total_qn_mean, partial_qn_mean, total_qn_sem, partial_qn_sem)` |
| `compute_qn_and_classify` | `(total_qn, partial_qn, o_classes)` | `(total_qn_mean, partial_qn_mean, o_classes_last, total_qn_sem, partial_qn_sem)` |
| `compute_structure_factor` | `(q, sq, partials)` | `(q, sq_mean, partials_mean, sq_sem, partials_sem)` |
| `compute_guttmann_rings` | `(histogram, mean_size)` | `(histogram_mean, mean_size_mean, histogram_sem, mean_size_sem)` |
| `analyze_structure` | `StructureData` | `(StructureData_mean, StructureData_sem)` |

> `per_atom_last` and `o_classes_last` are the per-atom classifications from the **last** frame — per-atom labels cannot be meaningfully averaged.

---

## Full example: averaged RDF with error bands

```python
import numpy as np
import plotly.graph_objects as go
from ase.io import read
from amorphouspy import compute_rdf

frames = read("production.extxyz", index=":")

r, rdfs_mean, cn_mean, rdfs_sem, cn_sem = compute_rdf(
    frames,
    r_max=8.0,
    n_bins=500,
    type_pairs=[(14, 8), (11, 8)],
    frame_averaging=True,
)

fig = go.Figure()
for pair, g_mean in rdfs_mean.items():
    g_sem = rdfs_sem[pair]
    label = f"{pair[0]}-{pair[1]}"
    fig.add_trace(go.Scatter(
        x=r, y=g_mean, name=label, mode="lines",
    ))
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

q, sq_mean, partials_mean, sq_sem, partials_sem = compute_structure_factor(
    frames,
    q_min=0.5,
    q_max=15.0,
    n_q=500,
    frame_averaging=True,
)
```

---

## Full example: averaged Qn distribution

```python
from amorphouspy import compute_qn

total_qn_mean, partial_qn_mean, total_qn_sem, partial_qn_sem = compute_qn(
    frames,
    cutoff=2.0,
    former_types=[14],  # Si
    o_type=8,
    frame_averaging=True,
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
- The SEM naturally tells you when you have enough: if it is already smaller than your measurement uncertainty (e.g. 1 % of the peak height), adding more frames gives diminishing returns.
- With only 1 frame and `frame_averaging=True`, SEM is 0 — a single frame has no variance to estimate.

---

## Input validation

| Input | Behaviour |
|---|---|
| `list[Atoms]` + `frame_averaging=True` | ✅ averages over all frames |
| `Atoms` + `frame_averaging=True` | ✅ single-frame result with SEM = 0 |
| `Atoms` + `frame_averaging=False` (default) | ✅ single-frame result |
| `list[Atoms]` + `frame_averaging=False` | ✅ uses first frame silently |
| empty `list` + `frame_averaging=True` | ❌ raises `ValueError` |
