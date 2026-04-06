# amorphouspy

A Python package for atomistic simulations of glasses.

It provides end-to-end workflows that span from generating initial structural models through running molecular dynamics simulations with LAMMPS, all the way to computing material properties and performing detailed structural analysis.

## Installation

```bash
pip install amorphouspy
```

For the full environment including LAMMPS (recommended):

```bash
# Install pixi (https://pixi.sh)
curl -fsSL https://pixi.sh/install.sh | bash

git clone https://github.com/glasagent/amorphouspy.git
cd amorphouspy
pixi install
```

See the [Installation guide](https://glasagent.github.io/amorphouspy/installation) for details.

## Quick Start

```python
from amorphouspy import get_structure_dict, get_ase_structure, generate_potential

# Generate a soda-lime silicate glass structure (~3000 atoms)
composition = {"SiO2": 75, "Na2O": 15, "CaO": 10}
structure_dict = get_structure_dict(composition, target_atoms=3000)
atoms = get_ase_structure(structure_dict)

# Set up an interatomic potential
potential = generate_potential(structure_dict, potential_type="pmmcs")
```

From here you can run melt-quench simulations, structural analysis, elastic moduli calculations, and more. See the [Tutorial](https://glasagent.github.io/amorphouspy/tutorial) and [How-To Guides](https://glasagent.github.io/amorphouspy/how_to_guides/) for complete examples.

## Documentation

**[glasagent.github.io/amorphouspy](https://glasagent.github.io/amorphouspy)**

## License

BSD 3-Clause. See [LICENSE](LICENSE).