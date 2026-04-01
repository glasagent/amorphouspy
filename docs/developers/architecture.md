# Developers & Architecture

For more information on the internal workings, this section provides an overview of how `amorphouspy` is organized internally.

## Package Organization

```
amorphouspy/
├── structure.py          # Composition parsing, structure generation, density model
├── mass.py               # Atomic mass utilities (wraps ASE data)
├── neighbors.py          # Cell-list neighbor search with periodic boundary conditions
├── io_utils.py           # LAMMPS I/O, XYZ writer, ASE Atoms helpers
├── shared.py             # Element type mapping, distribution counting utilities
├── potentials/
│   ├── potential.py      # Unified potential generator interface
│   ├── pmmcs_potential.py  # Pedone (PMMCS) Morse + Coulomb
│   ├── bjp_potential.py    # Bouhadja Born-Mayer-Huggins + Coulomb
│   └── shik_potential.py   # SHIK Buckingham + r⁻²⁴ + Coulomb
├── analysis/
│   ├── radial_distribution_functions.py  # RDF g(r) and coordination n(r)
│   ├── qn_network_connectivity.py        # Qⁿ distribution and network connectivity
│   ├── bond_angle_distribution.py        # O-X-O and X-O-X bond angle histograms
│   ├── rings.py                          # Guttman ring statistics (via sovapy)
│   ├── cavities.py                       # Void/cavity volume analysis (via sovapy)
│   └── cte.py                            # CTE from NPT fluctuations
└── workflows/
    ├── meltquench.py           # Core melt-quench simulation logic
    ├── meltquench_protocols.py # Potential-specific multi-stage protocols
    ├── md.py                   # Single-point NVT/NPT molecular dynamics
    ├── elastic_mod.py          # Elastic moduli via stress-strain finite differences
    ├── viscosity.py            # Viscosity via Green-Kubo (SACF integration)
    ├── cte.py                  # CTE simulation with convergence checking
    ├── structural_analysis.py  # Comprehensive analysis pipeline + Plotly plotting
    └── shared.py               # LAMMPS command builder utility
```

## API Design Decisions

The API service (`amorphouspy_api`) follows a two-layer design:

- **Materials layer** (`/glasses`): Read-only, property-centric. "What do we know about this glass?"
- **Jobs layer** (`/jobs`): Simulation-centric. "Run this computation."

Both layers share the same underlying data store. The materials layer is a view over completed jobs.

Full endpoint documentation is available via the auto-generated OpenAPI docs at `/docs`.

### Composition Normalization

The server uses a `Composition` model that accepts a dict (e.g. `{"SiO2": 70, "Na2O": 15, "CaO": 15}`) and generates a canonical string internally for database storage and matching. This ensures that `{"SiO2": 70, "Na2O": 15, "CaO": 15}` and `{"Na2O": 15, "SiO2": 70, "CaO": 15}` resolve to the same material. The canonical form (alphabetical oxide ordering, rounded values) is an implementation detail — API consumers always work with dicts.

### DAG Resolution

The user never specifies intermediate steps. If they request `elastic`, the server knows it needs structure generation → melt-quench → elastic. The `progress` dict on the job status response exposes the resolved pipeline so the user can see what's happening.

### Error Handling

- Job-level status is `completed` even if some analyses failed. Only core pipeline failure (melt-quench crash) results in job status `failed`.
- Individual analysis failures appear in the `errors` dict on the job status and are omitted from results.
- The `missing` field on the `/glasses` endpoint tells the LLM what hasn't been computed yet.

### Google Custom Method Convention

Actions that don't map to CRUD use the colon convention: `/jobs:search`, `/jobs/{id}:cancel`. This avoids polluting the resource ID namespace (e.g., `search` being confused with a job ID) and clearly signals "this is a verb, not a noun."

### MCP Tool Mapping

The API is designed to map cleanly to MCP tools:

| MCP Tool | Endpoint |
|---|---|
| `get_glass_properties` | `POST /glasses:lookup` |
| `search_simulations` | `POST /jobs:search` |
| `submit_simulation` | `POST /jobs` |
| `check_simulation_status` | `GET /jobs/{id}` |
| `get_simulation_results` | `GET /jobs/{id}/results` |
| `cancel_simulation` | `POST /jobs/{id}:cancel` |

The LLM's typical workflow:
1. `get_glass_properties` — check what's already known
2. If missing properties → `search_simulations` — check for cached/similar jobs
3. If no good match → `submit_simulation` — run new computation (after confirming with user)
4. `check_simulation_status` — poll until done
5. `get_simulation_results` — retrieve and present results

### Data Lifecycle Classification

Simulation data falls into three tiers with different retention guarantees:

1. **Ephemeral simulation files** — Raw output files in the LAMMPS working
   directory (trajectories, log files, restart files).  These are *not* parsed
   or retained by the API.  If the simulation directory is purged, the data is
   gone.

2. **Cached intermediate results** — Large data returned by the Python
   analysis functions that is too voluminous to store in the database.  This
   includes, for example, the full melt-quench trajectory and the raw
   stress-autocorrelation arrays from the viscosity calculation.  These live in
   the **executorlib cache** and can be re-materialised by re-running the
   function with the same inputs.  However, if the cache is invalidated (e.g.
   after a Python version upgrade), the data is lost.

3. **Persistent database results** — Compact, presentation-ready data that
   enters the SQLite `result_data` column and is retained indefinitely.  This
   includes scalar properties (viscosity values, elastic moduli),
   per-composition metadata, and downsampled plot data (e.g. convergence curves
   reduced to ≤ 1 000 points via log-spaced sampling).  These results survive
   cache purges and are the authoritative record of a completed job.

When adding a new analysis, decide for each output field which tier it belongs
to.  The guiding rule: **only store in the database what is needed to reproduce
the plots and summary tables shown in the results page**.
