# Installation

## Using pixi (recommended)

This project uses [pixi](https://pixi.sh) for environment management.
Pixi handles all dependencies — including LAMMPS with OpenMPI — in a single step.

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# Clone the repository
git clone https://github.com/glasagent/amorphouspy.git
cd amorphouspy

# Install the environment (including dev dependencies and editable packages)
pixi install
```

> **Note:** The pixi environment installs LAMMPS with OpenMPI support (`lammps =2024.08.29=*_openmpi_*`), which provides the `lmp_mpi` executable required by all simulation workflows.

## Using pip only (analysis-only)

If you only need the analysis and structure generation tools (no LAMMPS simulations):

```bash
pip install amorphouspy
```

> **Warning:** Without a LAMMPS installation, the simulation workflows (`melt_quench_simulation`, `md_simulation`, `elastic_simulation`, etc.) will not work. Only structure generation and analysis functions will be available.

To also install the API server and its dependencies:

```bash
pip install amorphouspy[api]
```

## Dependencies

Core dependencies installed automatically:

| Package | Purpose |
|---|---|
| [ASE](https://wiki.fysik.dtu.dk/ase/) (≥3.25) | Atomic Simulation Environment — structure representation and I/O |
| [lammpsparser](https://github.com/pyiron/lammpsparser) (0.0.1) | LAMMPS file interface — reads/writes LAMMPS input and dump files |
| [numba](https://numba.pydata.org/) | JIT compilation for performance-critical neighbor search and RDF calculations |
| [pymatgen](https://pymatgen.org/) (2025.10) | Charge neutrality validation for oxide compositions |
| [sovapy](https://github.com/MotokiShiga/sova-cui) (0.8.3) | Ring analysis (Guttman algorithm) and cavity/void detection |
| [scipy](https://scipy.org/) (1.16) | Curve fitting (VFT model), signal processing (Savitzky-Golay smoothing) |
| [pandas](https://pandas.pydata.org/) (2.3) | Potential configuration DataFrames |
| [numpy](https://numpy.org/) (2.3) | Core numerical computing |

System requirements:

| Requirement | Details |
|---|---|
| **Python** | ≥ 3.9 (developed with 3.13) |
| **LAMMPS** | Available as `lmp_mpi` on PATH (for simulation workflows) |
| **MPI** | OpenMPI recommended (for parallel LAMMPS) |

## Developer setup

```bash
pixi run -- pre-commit install
```

## Running the API

Start the API server locally:

```bash
pixi run serve
```

This starts uvicorn on `0.0.0.0:8000`. The MCP endpoint is available at `/mcp`.

## Configuring the executor backend

The API uses [executorlib](https://executorlib.readthedocs.io/) to run simulation jobs.
The backend is selected via the `EXECUTOR_TYPE` environment variable.

| `EXECUTOR_TYPE` | Executor | Description |
|---|---|---|
| `single` | `SingleNodeExecutor` | Runs jobs locally (default for development) |
| `slurm` | `SlurmClusterExecutor` | Submits jobs to a SLURM scheduler |
| `flux` | `FluxClusterExecutor` | Submits jobs to a Flux scheduler |

### Local execution

No extra configuration needed — this is the default:

```bash
EXECUTOR_TYPE=single pixi run serve
```

### SLURM cluster

Set the following environment variables:

| Variable | Description | Example |
|---|---|---|
| `EXECUTOR_TYPE` | Must be `slurm` | `slurm` |
| `LAMMPS_CORES` | MPI cores for LAMMPS jobs (default: `4`) | `8` |
| `SLURM_PARTITION` | SLURM partition name | `batch` |
| `SLURM_RUN_TIME_MAX` | Max run time per job in seconds | `7200` |
| `SLURM_MEMORY_MAX` | Max memory per job in GB | `16` |

For most setups, `EXECUTOR_TYPE`, `LAMMPS_CORES`, and `SLURM_PARTITION` are sufficient.

#### Custom submission template

For advanced control (e.g. account, QOS, custom flags), place a Jinja2
[submission template](https://executorlib.readthedocs.io/en/latest/2-hpc-cluster.html#slurm)
at `<AMORPHOUSPY_PROJECTS>/submission_template.sh` (defaults to `amorphouspy_api/projects/submission_template.sh`).
If present, it is automatically picked up:

```bash
#!/bin/bash
#SBATCH --output=time.out
#SBATCH --job-name={{job_name}}
#SBATCH --chdir={{working_directory}}
#SBATCH --get-user-env=L
#SBATCH --partition={{partition}}
#SBATCH --account=myproject
{%- if run_time_max %}
#SBATCH --time={{ [1, run_time_max // 60]|max }}
{%- endif %}
{%- if dependency %}
#SBATCH --dependency=afterok:{{ dependency | join(',') }}
{%- endif %}
{%- if memory_max %}
#SBATCH --mem={{memory_max}}G
{%- endif %}
#SBATCH --cpus-per-task={{cores}}

{{command}}
```

Example:

```bash
EXECUTOR_TYPE=slurm \
SLURM_PARTITION=batch \
LAMMPS_CORES=8 \
SLURM_RUN_TIME_MAX=7200 \
pixi run serve
```

For systemd, add these to the unit file under `[Service]`:

```ini
Environment=EXECUTOR_TYPE=slurm
Environment=SLURM_PARTITION=batch
Environment=LAMMPS_CORES=8
Environment=SLURM_RUN_TIME_MAX=7200
```