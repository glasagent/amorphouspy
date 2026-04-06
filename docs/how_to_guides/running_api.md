# Running the API

When using the `pixi` environment, the API dependencies will already be installed.

When using the pip/conda approach, install the API server and its dependencies via:

```bash
pip install amorphouspy[api]
```


## Starting the server

```bash
pixi run serve
```

The API will be available at:
- **REST API**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs`
- **MCP SSE**: `http://localhost:8000/mcp`


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

## Systemd service (production)

The repository includes systemd unit files in `amorphouspy_api/system-service/` for running the API as a persistent service that starts on boot and auto-restarts on failure.

### Installation

As `root` (or via `sudo`):

```bash
# Copy the unit files
cp amorphouspy_api/system-service/amorphouspy-api.service /etc/systemd/system/
cp amorphouspy_api/system-service/amorphouspy-api.path /etc/systemd/system/

# Reload, enable, and start
systemctl daemon-reload
systemctl enable --now amorphouspy-api.service
systemctl enable --now amorphouspy-api.path
```

The `.path` unit watches the source directory for changes and automatically restarts the service when code is updated.

### Configuration

Edit the `[Service]` section of the unit file to set environment variables:

```ini
Environment=EXECUTOR_TYPE=slurm
Environment=SLURM_PARTITION=main_queue
Environment=LAMMPS_CORES=8
Environment=AMORPHOUSPY_PROJECTS=/path/to/data
Environment=AMORPHOUSPY_VERSION_PROJECTS=0
```

| Variable | Default | Description |
|---|---|---|
| `AMORPHOUSPY_PROJECTS` | `<package>/projects` | Root directory for project data and the SQLite database |
| `AMORPHOUSPY_VERSION_PROJECTS` | `1` | Set to `0` to keep cached results accessible across version upgrades |

### Managing the service

```bash
# Check status
systemctl status amorphouspy-api

# Restart
systemctl restart amorphouspy-api

# View logs (follow mode)
journalctl -u amorphouspy-api -f
```
