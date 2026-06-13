# Installation guide

## From conda-forge (recommended for stable releases)

For users wanting to install a stable release of `amorphouspy`, we recommend using `conda-forge`. The package is available on `conda-forge` via a multi-output recipe, allowing you to install either the core simulation workflow package, or the web API server with its advanced process orchestrators. Installing via `conda` automatically handles all complex dependencies including LAMMPS.

*(Note: Users are free to use whatever tool they prefer to manage their conda/mamba/pixi environments here).*

### Core Package
To install the core `amorphouspy` library with LAMMPS and all numerical/scientific dependencies:
```bash
conda install -c conda-forge amorphouspy
```

### Web API Server
To install the web API & Model Context Protocol (MCP) server integration, which automatically pulls in the core `amorphouspy` library, `executorlib`, `fastapi`, and associated packages:
```bash
conda install -c conda-forge amorphouspy-api
```

## From git / source (for development)

If you want to contribute to development or run the latest source code of this project from git, we recommend using [pixi](https://pixi.sh) for environment management as it handles all dependencies — including LAMMPS with OpenMPI and API dependencies — in a single step using the local clone of the repository.

```bash
# Clone the repository
git clone https://github.com/glasagent/amorphouspy.git
cd amorphouspy

# Install the environment (core package, LAMMPS, API dependencies, etc.)
pixi install
```

> **Note:** The pixi environment installs LAMMPS with OpenMPI support (`lammps =2024.08.29=*_openmpi_*`), which provides the `lmp_mpi` executable required by all simulation workflows.

## Using pip only (analysis-only)

If you only need the analysis and structure generation tools (no LAMMPS simulations):

```bash
pip install amorphouspy
```

> **Warning:** Without a LAMMPS installation, the simulation workflows (`melt_quench_simulation`, `md_simulation`, `elastic_simulation`, etc.) will not work. Only structure generation and analysis functions will be available.
