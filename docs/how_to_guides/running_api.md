# Running the API

## Starting the server

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
