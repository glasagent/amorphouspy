import pandas
from pyiron_base import job

from pyiron_glasagent.shared import get_element_types_dict


@job
def generate_potential(atoms_dict):
    types = get_element_types_dict(atoms_dict)
    return pandas.DataFrame(
        {
            "Name": ["Pedone"],
            "Filename": [[]],
            "Model": ["Morse"],
            "Species": [["O", "Si", "Ca", "Al"]],
            "Config": [
                [
                    "# A. Pedone et.al., JPCB (2006)",
                    "#\n",
                    "\n",
                    "units metal\n",
                    "dimension 3\n",
                    "atom_style charge\n",
                    "\n",
                    "# create groups ###\n",
                    f"group O type {types.get('O')}\n",
                    f"group Si type {types.get('Si')}\n",
                    f"group Al type {types.get('Al')}\n",
                    f"group Ca type {types.get('Ca')}\n",
                    "\n",
                    "## set charges - beside manually ###\n",
                    "set group O charge -1.2\n",
                    "set group Si charge 2.4\n",
                    "set group Al charge 1.8\n",
                    "set group Ca charge 1.2\n",
                    "\n",
                    "### Pedone Potential Parameters ###\n",
                    "pair_style hybrid/overlay coul/dsf 0.25 8.0 morse 5.5 lennard/mdf 0.2 3.0 \n",
                    "pair_coeff * * coul/dsf \n",
                    "pair_coeff 1 3 morse 0.361581 1.900442 2.164818\n",
                    "pair_coeff 2 3 morse 0.030211 2.241334 2.923245\n",
                    "pair_coeff 3 3 morse 0.042395 1.379316 3.618701\n",
                    "pair_coeff 4 3 morse 0.340554 2.006700 2.100000\n",
                    "pair_coeff 1 3 lennard/mdf  0.9 0.0\n",
                    "pair_coeff 2 3 lennard/mdf  5.0 0.0\n",
                    "pair_coeff 3 3 lennard/mdf 22.0 0.0\n",
                    "pair_coeff 4 3 lennard/mdf  1.0 0.0\n",
                    "\n",
                    "pair_modify shift yes\n",
                    "\n",
                ]
            ],
        }
    )
