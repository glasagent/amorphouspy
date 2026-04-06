# Installation guide

## Using pixi (recommended)

This project uses [pixi](https://pixi.sh) for environment management.
Pixi handles all dependencies — including LAMMPS with OpenMPI and API dependencies — in a single step.

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

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

To also install the API server and its dependencies:

```bash
pip install amorphouspy[api]
```

For systemd, add these to the unit file under `[Service]`:

```ini
Environment=EXECUTOR_TYPE=slurm
Environment=SLURM_PARTITION=batch
Environment=LAMMPS_CORES=8
Environment=SLURM_RUN_TIME_MAX=7200
```
