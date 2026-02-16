# Simulation Workflows

All simulation workflows use LAMMPS as the MD engine via `lammpsparser`. Each workflow handles the complete lifecycle: setting up the simulation, running LAMMPS in a temporary directory, parsing outputs, and returning structured results.

---

## Overview

| Workflow | Function | Purpose |
|---|---|---|
| [**Melt-Quench**](melt_quench.md) | `melt_quench_simulation()` | Generate realistic glass structures from random initial configs |
| [**Molecular Dynamics**](md.md) | `md_simulation()` | Run NVT/NPT equilibration or production simulations |
| [**Elastic Moduli**](elastic.md) | `elastic_simulation()` | Calculate $C_{11}$, $C_{12}$, $C_{44}$ via stress-strain |
| [**Viscosity**](viscosity.md) | `viscosity_simulation()` | Compute viscosity via Green-Kubo (SACF) |
| [**CTE**](cte.md) | `cte_simulation()` | Coefficient of thermal expansion from NPT |

---

## Common Architecture

All workflows share a common execution pattern:

```mermaid
graph LR
    A[Structure + Potential] --> B[Build LAMMPS input]
    B --> C[Execute LAMMPS]
    C --> D[Parse output]
    D --> E[Return results]
```

1. **Input**: ASE `Atoms` object + potential `DataFrame`
2. **Build**: `lammpsparser` generates LAMMPS input scripts and data files
3. **Execute**: LAMMPS runs in a temporary directory via subprocess
4. **Parse**: Output files (dump, log, custom) are read back
5. **Return**: Structured dictionary with results + final structure

### LAMMPS Command

The LAMMPS executable path is determined by `get_lammps_command()`:

```python
from amorphouspy.workflows.shared import get_lammps_command

cmd = get_lammps_command()
# Returns: "lmp_mpi" (or custom path from environment)
```

By default, it looks for `lmp_mpi` on the system PATH. Set the `LAMMPS_COMMAND` environment variable to override.

---

## Detailed Guides

Each workflow has a dedicated page with full parameter documentation, implementation details, and examples:

- [**Melt-Quench Simulation**](melt_quench.md) — Heating, equilibration, and cooling protocols
- [**Molecular Dynamics**](md.md) — NVT/NPT single-point simulations
- [**Elastic Moduli**](elastic.md) — Stress-strain finite differences method
- [**Viscosity**](viscosity.md) — Green-Kubo stress autocorrelation
- [**CTE Simulation**](cte.md) — NPT fluctuation-based thermal expansion
