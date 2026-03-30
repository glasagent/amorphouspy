# API Service (MCP)

The `amorphouspy-api` is a FastAPI-based service that provides a Model Context Protocol (MCP) interface for running oxide glass simulations with intelligent caching.

The API has two layers:

- **Materials layer** (`/glasses`): Read-only, property-centric. "What do we know about this glass?"
- **Jobs layer** (`/jobs`): Simulation-centric. "Run this computation."

Both layers share the same underlying data store. The materials layer is a view over completed jobs.

## Simulation Pipeline

The standard melt-quench workflow for oxide glasses follows this DAG:

```
composition dict
  → random structure generation
    → select interatomic potential
      → melt-quench MD simulation
        → structural analysis (RDF, coordination, bond angles)
        → elastic moduli (bulk, shear, Young's, Poisson)
        → viscosity (Green-Kubo at multiple temperatures)
        → CTE (coefficient of thermal expansion)
```

The API hides this DAG from the user. The user specifies a composition and which analyses they want; the server resolves dependencies automatically.

## How It Works

### Architecture Overview

``` mermaid
graph LR
    A[FastAPI App] --> B[SQLite Cache]
    B --> C[executorlib]
    
    subgraph FastAPI
    A1[Request hash]
    A2[Cache lookup]
    A3[Job creation]
    end
    
    subgraph SQLite
    B1[Job metadata]
    B2[Results]
    B3[Hash index]
    end
    
    subgraph executorlib
    C1[Local exec]
    C2[SLURM cluster]
    C3[Job caching]
    end
```

### Key Components

#### 1. Request Hashing & Caching
- Each simulation request is hashed based on composition, potential, and simulation parameters.
- Deterministic SHA256 hash for consistent, reproducible cache keys.
- Automatic cache lookups prevent duplicate simulations.
- Results persist across server restarts.

#### 2. Job Store (SQLite)
- All job metadata stored in SQLite database (`jobs.db`).
- Efficient indexed lookups by request hash and composition.
- Tracks job states: `pending` → `running` → `completed`/`failed`/`cancelled`.
- Survives server restarts and process crashes.

#### 3. Job Execution with executorlib
- Supports local execution (`TestClusterExecutor`) or SLURM cluster (`SlurmClusterExecutor`/`FluxClusterExecutor`).
- Executor type configured via environment variables.
- Built-in job caching at the executor level.
- Re-submitting same job returns cached result or running future.

#### 4. Model Context Protocol (MCP) Integration
- Exposes simulation capabilities as MCP tools via `fastapi-mcp`.
- Compatible with Claude, VS Code, and other MCP clients.
- Server-Sent Events (SSE) endpoint at `/mcp`.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EXECUTOR_TYPE` | Executor backend: `test`, `slurm`, `flux`, or `single` | `test` |
| `SLURM_PARTITION` | SLURM partition name (slurm only) | - |
| `SLURM_RUN_TIME_MAX` | Max run time per job in seconds (slurm only) | - |
| `SLURM_MEMORY_MAX` | Max memory per job in GB (slurm only) | - |
| `AMORPHOUSPY_PROJECTS` | Directory for project/cache files | `./projects` |
| `API_BASE_URL` | Base URL for visualization links | - |

## Installation

```bash
pip install amorphouspy[api]
```

Or for development (editable install):

```bash
pip install -e ".[api]"
```

## Launch API

```bash
python -m uvicorn amorphouspy_api.app:app
```

The API will be available at:
- **REST API**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs`
- **MCP SSE**: `http://localhost:8000/mcp`
