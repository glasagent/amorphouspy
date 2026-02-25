# Installation

## Using conda (recommended)

The recommended approach uses conda to install LAMMPS and all dependencies, then pip to install the package itself:

```bash
# Clone the repository
git clone https://github.com/glasagent/amorphouspy.git
cd amorphouspy

# Create and activate the conda environment
conda env create -f environment.yml
conda activate amorphouspy

# Install the core library in editable mode
pip install -e amorphouspy
```

> **Note:** The conda environment installs LAMMPS with OpenMPI support (`lammps =2024.08.29=*_openmpi_*`), which provides the `lmp_mpi` executable required by all simulation workflows.

## Using pip only (analysis-only)

If you only need the analysis and structure generation tools (no LAMMPS simulations):

```bash
pip install amorphouspy
```

> **Warning:** Without a LAMMPS installation, the simulation workflows (`melt_quench_simulation`, `md_simulation`, `elastic_simulation`, etc.) will not work. Only structure generation and analysis functions will be available.

## Dependencies

Core dependencies installed automatically:

| Package | Purpose |
|---|---|
| [ASE](https://wiki.fysik.dtu.dk/ase/) (≥3.25) | Atomic Simulation Environment — structure representation and I/O |
| [lammpsparser](https://github.com/pyiron/lammpsparser) (0.0.1) | LAMMPS file interface — reads/writes LAMMPS input and dump files |
| [numba](https://numba.pydata.org/) | JIT compilation for performance-critical neighbor search and RDF calculations |
| [pymatgen](https://pymatgen.org/) (2025.10) | Charge neutrality validation for oxide compositions |
| [sovapy](https://github.com/MotokiShiga/sova-cui) (0.8.3) | Ring analysis (Guttman algorithm) and cavity/void detection |
| [scipy](https://scipy.org/) (1.16) | Curve fitting (VFT model), signal processing (Savitzky-Golay smoothing) |
| [pandas](https://pandas.pydata.org/) (2.3) | Potential configuration DataFrames |
| [numpy](https://numpy.org/) (2.3) | Core numerical computing |

System requirements:

| Requirement | Details |
|---|---|
| **Python** | ≥ 3.9 (developed with 3.13) |
| **LAMMPS** | Available as `lmp_mpi` on PATH (for simulation workflows) |
| **MPI** | OpenMPI recommended (for parallel LAMMPS) |

## Developer setup

```bash
pip install -r amorphouspy/requirements-dev.txt
pre-commit install
```
