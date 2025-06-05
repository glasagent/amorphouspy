"""
Author: Achraf Atila (achraf.atila@bam.de)

This script provides tests for structural analysis functions related to glassy systems.
The 0.2Na2O-0.8SiO2 glass is used. It calculates expected values for non-bridging oxygens (NBO),
bridging oxygens (BO), and network connectivity based on chemical composition. These
expected values are compared against values computed using the implemented analysis functions.

The structure used in this test is a realistic representation of a 20Na2O-80SiO2 glass system,
prepared using Pedone (2006) potential parameters.

Glass Composition and Expected Values:
--------------------------------------
x = 0.2  # Sodium content in the glass
N_Si = (1 - x) * N_mols  # Number of Si atoms
N_Na = 2 * x * N_mols  # Number of Na atoms
N_O = 2 * N_Si + N_Na / 2  # Number of O atoms

N_NBO = 2 * x * N_mols  # Number of non-bridging oxygen atoms (NBO)
N_BO = N_O - N_NBO  # Number of bridging oxygen atoms (BO)

Expected network connectivity:
expected_NC = (4 * (1 - x) - 2 * x) / (1 - x)
Reference: https://doi.org/10.1039/D4TB02414A
"""

from . import DATA_DIR

import pytest
from pyiron_glass import (
    read_lammps_dump,
    compute_coordination,
    compute_Qn,
    compute_network_connectivity,
)

# Glass composition parameters
x = 0.2  # Sodium molar fraction (20% Na2O, 80% SiO2)
N_mols = 100  # Number of oxide molecules (SiO2 + Na2O)

# Expected number of atoms based on stoichiometry
N_Si = int((1 - x) * N_mols)  # Should be integer (80)
N_Na = int(2 * x * N_mols)  # Should be integer (40)
N_O = int(2 * N_Si + N_Na / 2)  # Should be integer (180)

N_NBO = int(2 * x * N_mols)  # Number of non-bridging oxygens (40)
N_BO = N_O - N_NBO  # Number of bridging oxygens (140)

# Expected network connectivity
expected_NC = (4 * (1 - x) - 2 * x) / (1 - x)  # 3.5

# Cutoff distances for computing coordination numbers (in Ångström)
cutoff_map = {
    "O": 1.9,
    "Si": 1.9,
    "Na": 3.0,
}

# Mapping from atom type ID to element name
type_map = {
    1: "Na",
    2: "O",
    3: "Si",
}


network_formers = {"Si"}
modifiers = {"Na"}
O_type = [t for t, e in type_map.items() if e == "O"][0]
former_types = [t for t, e in type_map.items() if e in network_formers]
modifier_types = [t for t, e in type_map.items() if e in modifiers]


def test_compute_coordination_O():
    """Test the compute_coordination function for oxygens."""
    filename = DATA_DIR / "20Na2O-80SiO2.dump"
    ids, types, coords, box_size = read_lammps_dump(filename, unwrap=False)

    # compute_coordination returns (distribution_dict, per-atom coordination dict)
    O_coord_dist, _ = compute_coordination(ids, types, coords, box_size, [O_type], cutoff_map["O"], former_types)


    # Check types
    assert isinstance(O_coord_dist, dict), "O coordination should return a dictionary"
    assert all(isinstance(k, int) for k in O_coord_dist.keys()), "Keys of O coordination should be integers"
    assert all(isinstance(v, int) for v in O_coord_dist.values()), "Values of O coordination should be integers"

    # Two categories: NBO (coordination = 1) and BO (coordination = 2)
    assert O_coord_dist.get(1, 0) == N_NBO, f"NBO count mismatch. Expected {N_NBO}, got {O_coord_dist.get(1, 0)}"
    assert O_coord_dist.get(2, 0) == N_BO, f"BO count mismatch. Expected {N_BO}, got {O_coord_dist.get(2, 0)}"


def test_compute_network_connectivity():
    """Test the compute_network_connectivity function."""
    filename = DATA_DIR / "20Na2O-80SiO2.dump"
    ids, types, coords, box_size = read_lammps_dump(filename, unwrap=False)

    # compute_Qn returns a Qn distribution dict: {0: count, 1: count, ..., 6: count}
    Qn_dist = compute_Qn(ids, types, coords, box_size, cutoff_map["O"], former_types, [O_type])

    net_conn = compute_network_connectivity(Qn_dist)

    # Type checks
    assert isinstance(net_conn, float), "Network connectivity should return a float"
    assert net_conn >= 0, "Network connectivity should be non-negative"

    # Expected NC ≈ 3.5 for 20Na2O-80SiO2
    assert net_conn == pytest.approx(expected_NC), f"Network connectivity should be {expected_NC}, got {net_conn}"
