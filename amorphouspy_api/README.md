# amorphouspy-api

API for atomistic modeling of oxide glasses using the amorphouspy workflows.

This FastAPI-based service provides a Model Context Protocol (MCP) interface for running long-running glass simulation tasks with intelligent caching and persistent task management.

## How It Works

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │ ── │  SQLite Cache   │ ── │   executorlib   │
│                 │    │                 │    │                 │
│ • Request hash  │    │ • Task metadata │    │ • Local exec    │
│ • Cache lookup  │    │ • Results       │    │ • SLURM cluster │
│ • Task creation │    │ • Hash index    │    │ • Job caching   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components

#### 1. **Request Hashing & Caching**
- Each simulation request is hashed based on composition and simulation parameters
- Uses `cloudpickle` + SHA256 for consistent, reproducible hashes
- Automatic cache lookups prevent duplicate simulations
- Results persist across server restarts

#### 2. **Persistent Task Store (SQLite)**
- All task metadata stored in SQLite database (`tasks.db`)
- Efficient indexed lookups by request hash
- Tracks task states: `processing` → `complete`/`error`
- Survives server restarts and process crashes

#### 3. **Job Execution with executorlib**
- Supports local execution (`SingleNodeExecutor`) or SLURM cluster (`SlurmClusterExecutor`)
- Executor type configured via environment variables
- Built-in job caching at the executor level
- Re-submitting same job returns cached result or running future

#### 4. **Model Context Protocol (MCP) Integration**
- Exposes simulation capabilities as MCP tools
- Compatible with Claude, VS Code, and other MCP clients
- Server-Sent Events (SSE) endpoint at `/mcp`

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
pip install -e ../amorphouspy/

# Install the API
pip install -e .
```

## Launch API (including MCP server)

```bash
python -m uvicorn amorphouspy_api.app:app
```

The API will be available at:
- REST API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- MCP SSE: http://localhost:8000/mcp

## Developer Setup

Install development dependencies:
```bash
pip install -r requirements-dev.txt
pre-commit install
```

## Run Tests

**Unit tests:**
```bash
pytest
```

**Integration tests**:
```bash
# Start API server
uvicorn amorphouspy_api.app:app --port 8002 --reload

# Run integration tests
pytest -m integration -s
```
