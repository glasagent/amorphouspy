# API Service (MCP)

The `amorphouspy-api` is a FastAPI-based service that provides a Model Context Protocol (MCP) interface for running long-running glass simulation tasks with intelligent caching and persistent task management.

## How It Works

### Architecture Overview

``` mermaid
graph LR
    A[FastAPI App] --> B[SQLite Cache]
    B --> C[executorlib]
    
    subgraph FastAPI
    A1[Request hash]
    A2[Cache lookup]
    A3[Task creation]
    end
    
    subgraph SQLite
    B1[Task metadata]
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
- Each simulation request is hashed based on composition and simulation parameters.
- Uses `cloudpickle` + SHA256 for consistent, reproducible hashes.
- Automatic cache lookups prevent duplicate simulations.
- Results persist across server restarts.

#### 2. Persistent Task Store (SQLite)
- All task metadata stored in SQLite database (`tasks.db`).
- Efficient indexed lookups by request hash.
- Tracks task states: `processing` → `complete`/`error`.
- Survives server restarts and process crashes.

#### 3. Job Execution with executorlib
- Supports local execution (`SingleNodeExecutor`) or SLURM cluster (`SlurmClusterExecutor`).
- Executor type configured via environment variables.
- Built-in job caching at the executor level.
- Re-submitting same job returns cached result or running future.

#### 4. Model Context Protocol (MCP) Integration
- Exposes simulation capabilities as MCP tools.
- Compatible with Claude, VS Code, and other MCP clients.
- Server-Sent Events (SSE) endpoint at `/mcp`.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EXECUTOR_TYPE` | Executor backend: `local` or `slurm` | `local` |
| `EXECUTOR_CORES` | Number of CPU cores per worker | `4` |
| `SLURM_PARTITION` | SLURM partition name (slurm only) | - |
| `SLURM_TIME` | SLURM job time limit (slurm only) | - |
| `AMORPHOUSPY_PROJECTS` | Directory for project/cache files | `./projects` |
| `API_BASE_URL` | Base URL for visualization links | - |

## Installation

```bash
# Install amorphouspy dependency
pip install -e ./amorphouspy/

# Install the API
pip install -e ./amorphouspy_api/
```

## Launch API

```bash
python -m uvicorn amorphouspy_api.app:app
```

The API will be available at:
- **REST API**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs`
- **MCP SSE**: `http://localhost:8000/mcp`
