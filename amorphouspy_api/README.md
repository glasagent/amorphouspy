# amorphouspy-api

FastAPI service that exposes `amorphouspy` via REST and Model Context Protocol (MCP), with intelligent caching, persistent task management, and support for local and SLURM cluster execution.

**Documentation**: [glasagent.github.io/amorphouspy/api_service](https://glasagent.github.io/amorphouspy/api_service)

## Contents

- `src/amorphouspy_api/` — Application source (endpoints, models, executor)
- `src/tests/` — Unit and integration tests
- `system-service/` — Systemd unit files for production deployment
