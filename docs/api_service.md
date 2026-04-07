# Web API Reference

The `amorphouspy-api` is a FastAPI-based service that provides a Model Context Protocol (MCP) interface for running oxide glass simulations with intelligent caching.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EXECUTOR_TYPE` | Executor backend: `test`, `slurm`, `flux`, or `single` | `test` |
| `SLURM_PARTITION` | SLURM partition name (slurm only) | - |
| `SLURM_RUN_TIME_MAX` | Max run time per job in seconds (slurm only) | - |
| `SLURM_MEMORY_MAX` | Max memory per job in GB (slurm only) | - |
| `AMORPHOUSPY_PROJECTS` | Directory for project/cache files | `./projects` |
| `API_BASE_URL` | Base URL for visualization links | - |


## Endpoint Reference

Complete endpoint reference created from the OpenAPI schema, as found on `/docs` of the running API.

<swagger-ui src="openapi.json" defaultModelsExpandDepth="0"/>
