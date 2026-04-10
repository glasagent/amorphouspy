# Use the Web API

## Installation

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

## Key features

- Each simulation request is hashed based on composition, potential, and simulation parameters.
- Automatic cache lookups prevent duplicate simulations.
- Results persist across server restarts.

- All job metadata stored in SQLite database (`jobs.db`).
- Tracks job states: `pending` → `running` → `completed`/`failed`/`cancelled`.
- Supports local execution (`TestClusterExecutor`) or SLURM cluster (`SlurmClusterExecutor`/`FluxClusterExecutor`).
- Built-in job caching at the executor level: Re-submitting same job returns cached result or running future.

- Exposes simulation capabilities as MCP tools at `/mcp` via `fastmcp`.
- Compatible with Claude, VS Code, and other MCP clients.

## Using the API

The API follows a two-layer design:

- **Jobs layer** (`/jobs`): Simulation-centric. "Run this computation."
- **Materials layer** (`/glasses`): Read-only, property-centric. "What do we know about this glass?"

Both layers share the same underlying data store. The materials layer is a view over completed jobs.

Full endpoint documentation is available via the auto-generated OpenAPI docs at `/docs`.


## Authentication

The API supports optional bearer-token authentication via the `API_TOKEN` environment variable.

**Without `API_TOKEN`** (default): The API is open — no credentials required. A warning is logged at startup.

**With `API_TOKEN` set**: All requests (except docs endpoints) must include the token:

```bash
# Set the token when starting the server
API_TOKEN=my-secret-token pixi run serve

# Include it in requests
curl -H "Authorization: Bearer my-secret-token" http://localhost:8000/jobs
```

The docs UI (`/docs`, `/redoc`) and OpenAPI schema (`/openapi.json`) remain accessible without a token.

## Configuring the executor backend

The API uses [executorlib](https://executorlib.readthedocs.io/) to run simulation jobs.
The backend is selected via the `EXECUTOR_TYPE` environment variable.

| `EXECUTOR_TYPE` | Executor | Description |
|---|---|---|
| `single` | `SingleNodeExecutor` | Runs jobs locally (default for development) |
| `slurm` | `SlurmClusterExecutor` | Submits jobs to a SLURM scheduler |
| `flux` | `FluxClusterExecutor` | Submits jobs to a Flux scheduler |

### Local execution

No extra configuration needed — this is the default.

### SLURM cluster

Set the following environment variables before running `pixi run serve`
(see the [Web API Reference](../api_service.md#environment-variables) for the full list):

- `EXECUTOR_TYPE=slurm`
- `LAMMPS_CORES` — MPI cores for LAMMPS jobs (default: `4`)
- `SLURM_PARTITION` — SLURM partition name
- `SLURM_RUN_TIME_MAX` — Max run time per job in seconds (optional)
- `SLURM_MEMORY_MAX` — Max memory per job in GB (optional)

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

The repository includes systemd unit files in `docs/system-service/` for running the API as a persistent service that starts on boot and auto-restarts on failure.

### Installation

As `root` (or via `sudo`):

```bash
# Copy the unit files
cp docs/system-service/amorphouspy-api.service /etc/systemd/system/
cp docs/system-service/amorphouspy-api.path /etc/systemd/system/

# Reload, enable, and start
systemctl daemon-reload
systemctl enable --now amorphouspy-api.service
systemctl enable --now amorphouspy-api.path
```

The `.path` unit watches the source directory for changes and automatically restarts the service when code is updated.

### Configuration

Edit the `[Service]` section of the unit file to set environment variables
(see the [Web API Reference](../api_service.md#environment-variables) for the full list):

```ini
Environment=EXECUTOR_TYPE=slurm
Environment=SLURM_PARTITION=main_queue
Environment=LAMMPS_CORES=8
Environment=AMORPHOUSPY_PROJECTS=/path/to/data
Environment=AMORPHOUSPY_VERSION_PROJECTS=0
Environment=API_TOKEN=<your-secret-token>
```

### Managing the service

```bash
# Check status
systemctl status amorphouspy-api

# Restart
systemctl restart amorphouspy-api

# View logs (follow mode)
journalctl -u amorphouspy-api -f
```
