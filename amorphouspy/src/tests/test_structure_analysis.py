"""Tests for structural analysis functions related to glassy systems.

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

import numpy as np
import pytest
from amorphouspy.properties.structural.averaging import average_over_frames
from amorphouspy.properties.structural.qn import classify_oxygens, compute_qn_and_classify
from ase.io import read

from amorphouspy import (
    compute_coordination,
    compute_network_connectivity,
    compute_qn,
)

from . import DATA_DIR

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
expected_nc = (4 * (1 - x) - 2 * x) / (1 - x)  # 3.5


def test_compute_coordination_o() -> None:
    """Test the compute_coordination function for oxygens."""
    filename = DATA_DIR / "20Na2O-80SiO2.dump"
    atoms = read(filename, format="lammps-dump-text")
    type_id = atoms.get_atomic_numbers().copy()  # this currently holds IDs 1/2/3
    to_Z = np.array([0, 11, 8, 14], dtype=int)  # index 0 unused; 1->8, 2->14, 3->11
    Z = to_Z[type_id]  # vectorized mapping 1/2/3 -> 8/14/11
    atoms.set_atomic_numbers(Z)  # fixes both numbers and symbols

    # Preserve original LAMMPS IDs and expose a clean 'type' = atomic numbers
    atoms.new_array("lammps_type", type_id)  # keeps 1/2/3 if you still need them
    atoms.arrays["type"] = Z  # downstream code can index by atomic number

    cutoff_map = {"O": 1.9, "Si": 1.9, "Na": 3.0}

    O_type = [8]
    former_types = [14]

    o_coord_dist = compute_coordination(
        atoms,
        [O_type],
        cutoff_map["O"],
        former_types,
    )

    # Check types
    assert isinstance(o_coord_dist, dict), "O coordination should return a dictionary"
    assert all(isinstance(k, int) for k in o_coord_dist), "Keys of O coordination should be integers"
    assert all(isinstance(v, (int, float)) for v in o_coord_dist.values()), (
        "Values of O coordination should be integers or floats"
    )

    # Two categories: NBO (coordination = 1) and BO (coordination = 2)
    assert o_coord_dist.get(1, 0) == N_NBO, f"NBO count mismatch. Expected {N_NBO}, got {o_coord_dist.get(1, 0)}"
    assert o_coord_dist.get(2, 0) == N_BO, f"BO count mismatch. Expected {N_BO}, got {o_coord_dist.get(2, 0)}"


def test_compute_network_connectivity() -> None:
    """Test the compute_network_connectivity function."""
    filename = DATA_DIR / "20Na2O-80SiO2.dump"
    atoms = read(filename, format="lammps-dump-text")
    type_id = atoms.get_atomic_numbers().copy()  # this currently holds IDs 1/2/3
    to_Z = np.array([0, 11, 8, 14], dtype=int)  # index 0 unused; 1->8, 2->14, 3->11
    Z = to_Z[type_id]  # vectorized mapping 1/2/3 -> 8/14/11
    atoms.set_atomic_numbers(Z)  # fixes both numbers and symbols

    # Preserve original LAMMPS IDs and expose a clean 'type' = atomic numbers
    atoms.new_array("lammps_type", type_id)  # keeps 1/2/3 if you still need them
    atoms.arrays["type"] = Z  # downstream code can index by atomic number

    cutoff_map = {"O": 1.9, "Si": 1.9, "Na": 3.0}

    O_type = [8]
    former_types = [14]

    # compute_qn returns a qn distribution dict: {0: count, 1: count, ..., 6: count}
    qn_dist, _ = compute_qn(
        atoms,
        cutoff_map["O"],
        former_types,
        O_type[0],
    )

    net_conn = compute_network_connectivity(qn_dist)

    # Type checks
    assert isinstance(net_conn, float), "Network connectivity should return a float"
    assert net_conn >= 0, "Network connectivity should be non-negative"

    # Expected NC ≈ 3.5 for 20Na2O-80SiO2
    assert net_conn == pytest.approx(
        expected_nc,
    ), f"Network connectivity should be {expected_nc}, got {net_conn}"


def test_compute_network_connectivity_multi() -> None:
    """Test the compute_network_connectivity function."""
    filename = DATA_DIR / "20Na2O-10B2O3-70SiO2.dump"

    # Cutoff distances for computing coordination numbers (in Ångström)
    cutoff_map = {"O": 1.9, "Si": 1.9, "B": 1.9, "Na": 3.0}

    atoms = read(filename, format="lammps-dump-text")
    type_id = atoms.get_atomic_numbers().copy()  # this currently holds IDs 1/2/3
    to_Z = np.array([0, 8, 14, 5, 11], dtype=int)  # index 0 unused; 1->8, 2->14, 3->5, 4->11
    Z = to_Z[type_id]  # vectorized mapping 1/2/3 -> 8/14/5/11
    atoms.set_atomic_numbers(Z)  # fixes both numbers and symbols

    # Preserve original LAMMPS IDs and expose a clean 'type' = atomic numbers
    atoms.new_array("lammps_type", type_id)  # keeps 1/2/3 if you still need them
    atoms.arrays["type"] = Z  # downstream code can index by atomic number

    O_type_multi = [8]
    former_types = [5, 14]

    # compute_qn returns a qn distribution dict: {0: count, 1: count, ..., 6: count}
    qn_dist, _ = compute_qn(
        atoms,
        cutoff_map["O"],
        former_types,
        O_type_multi[0],
    )

    net_conn = compute_network_connectivity(qn_dist)

    # Type checks
    assert isinstance(net_conn, float), "Network connectivity should return a float"
    assert net_conn >= 0, "Network connectivity should be non-negative"

    # Expected NC ≈ pr_qn for 20Na2O-10B2O3-70SiO2.dump
    assert round(net_conn, 3) == pytest.approx(
        3.577,
    ), f"Network connectivity should be {3.577}, got {net_conn}"


# ---------------------------------------------------------------------------
# Shared setup helper for 20Na2O-80SiO2
# ---------------------------------------------------------------------------


def _load_na_si_o_structure():
    filename = DATA_DIR / "20Na2O-80SiO2.dump"
    atoms = read(filename, format="lammps-dump-text")
    type_id = atoms.get_atomic_numbers().copy()
    to_Z = np.array([0, 11, 8, 14], dtype=int)  # 1->Na, 2->O, 3->Si
    Z = to_Z[type_id]
    atoms.set_atomic_numbers(Z)
    atoms.new_array("lammps_type", type_id)
    atoms.arrays["type"] = Z
    return atoms


FORMER_TYPES = [14]  # Si
O_TYPE = 8
CUTOFF = 1.9


# ---------------------------------------------------------------------------
# classify_oxygens
# ---------------------------------------------------------------------------


def test_classify_oxygens_returns_dict():
    """classify_oxygens returns a dict."""
    atoms = _load_na_si_o_structure()
    result = classify_oxygens(atoms, CUTOFF, FORMER_TYPES, O_TYPE)
    assert isinstance(result, dict)


def test_classify_oxygens_valid_class_labels():
    """All oxygen classification values are one of BO, NBO, free, or tri."""
    atoms = _load_na_si_o_structure()
    result = classify_oxygens(atoms, CUTOFF, FORMER_TYPES, O_TYPE)
    valid = {"BO", "NBO", "free", "tri"}
    assert all(v in valid for v in result.values())


def test_classify_oxygens_keys_are_ints():
    """classify_oxygens keys (atom IDs) are integers."""
    atoms = _load_na_si_o_structure()
    result = classify_oxygens(atoms, CUTOFF, FORMER_TYPES, O_TYPE)
    assert all(isinstance(k, int) for k in result)


def test_classify_oxygens_nbo_bo_counts_match_stoichiometry():
    """NBO and BO counts from classify_oxygens match stoichiometric expectations."""
    atoms = _load_na_si_o_structure()
    result = classify_oxygens(atoms, CUTOFF, FORMER_TYPES, O_TYPE)
    n_nbo = sum(1 for v in result.values() if v == "NBO")
    n_bo = sum(1 for v in result.values() if v == "BO")
    assert n_nbo == N_NBO, f"Expected {N_NBO} NBOs, got {n_nbo}"
    assert n_bo == N_BO, f"Expected {N_BO} BOs, got {n_bo}"


# ---------------------------------------------------------------------------
# compute_qn_and_classify
# ---------------------------------------------------------------------------


def test_compute_qn_and_classify_returns_three_tuple():
    """compute_qn_and_classify returns a 3-tuple."""
    atoms = _load_na_si_o_structure()
    result = compute_qn_and_classify(atoms, CUTOFF, FORMER_TYPES, O_TYPE)
    assert len(result) == 3


def test_compute_qn_and_classify_total_qn_matches_compute_qn():
    """Total and partial Qn from compute_qn_and_classify agree with compute_qn."""
    atoms = _load_na_si_o_structure()
    total_qn, partial_qn, _ = compute_qn_and_classify(atoms, CUTOFF, FORMER_TYPES, O_TYPE)
    total_qn_ref, partial_qn_ref = compute_qn(atoms, CUTOFF, FORMER_TYPES, O_TYPE)
    assert total_qn == total_qn_ref
    assert partial_qn == partial_qn_ref


def test_compute_qn_and_classify_oxygen_classes_match_classify_oxygens():
    """Oxygen classes from compute_qn_and_classify match classify_oxygens wrapper."""
    atoms = _load_na_si_o_structure()
    _, _, o_classes_full = compute_qn_and_classify(atoms, CUTOFF, FORMER_TYPES, O_TYPE)
    o_classes_wrapper = classify_oxygens(atoms, CUTOFF, FORMER_TYPES, O_TYPE)
    assert o_classes_full == o_classes_wrapper


def test_compute_qn_and_classify_oxygen_classes_valid_labels():
    """All oxygen class labels from compute_qn_and_classify are valid strings."""
    atoms = _load_na_si_o_structure()
    _, _, o_classes = compute_qn_and_classify(atoms, CUTOFF, FORMER_TYPES, O_TYPE)
    valid = {"BO", "NBO", "free", "tri"}
    assert all(v in valid for v in o_classes.values())


# ---------------------------------------------------------------------------
# average_over_frames — compute_qn and compute_qn_and_classify
# ---------------------------------------------------------------------------


def test_average_over_frames_qn_identical_frames():
    """Three identical frames: mean equals single-frame result, SEM ≈ 0."""
    atoms = _load_na_si_o_structure()
    total_single, partial_single = compute_qn(atoms, CUTOFF, FORMER_TYPES, O_TYPE)
    (total_mean, partial_mean), (total_sem, partial_sem) = average_over_frames(
        compute_qn, [atoms, atoms, atoms], cutoff=CUTOFF, former_types=FORMER_TYPES, o_type=O_TYPE
    )
    for q in range(7):
        assert total_mean[q] == pytest.approx(total_single[q], abs=1e-10)
        assert total_sem[q] == pytest.approx(0.0, abs=1e-10)
    for ft in FORMER_TYPES:
        for q in range(7):
            assert partial_mean[ft][q] == pytest.approx(partial_single[ft][q], abs=1e-10)
            assert partial_sem[ft][q] == pytest.approx(0.0, abs=1e-10)


def test_average_over_frames_qn_empty_list_raises():
    """average_over_frames with empty list raises ValueError."""
    with pytest.raises(ValueError, match="requires a non-empty list"):
        average_over_frames(compute_qn, [], cutoff=CUTOFF, former_types=FORMER_TYPES, o_type=O_TYPE)


def test_compute_qn_list_uses_first_frame():
    """Passing a list without average_over_frames uses the first frame."""
    atoms = _load_na_si_o_structure()
    total_s, _partial_s = compute_qn(atoms, CUTOFF, FORMER_TYPES, O_TYPE)
    total_a, _partial_a = compute_qn([atoms, atoms], CUTOFF, FORMER_TYPES, O_TYPE)
    for q in range(7):
        assert total_a[q] == pytest.approx(total_s[q], abs=1e-10)


def test_average_over_frames_qn_and_classify_identical_frames():
    """Three identical frames: qn means equal single-frame, SEM ≈ 0, o_classes passes through."""
    atoms = _load_na_si_o_structure()
    total_s, _partial_s, o_cls_s = compute_qn_and_classify(atoms, CUTOFF, FORMER_TYPES, O_TYPE)
    (total_mean, _partial_mean, o_cls_last), (total_sem, _partial_sem, o_cls_sem) = average_over_frames(
        compute_qn_and_classify, [atoms, atoms, atoms], cutoff=CUTOFF, former_types=FORMER_TYPES, o_type=O_TYPE
    )
    for q in range(7):
        assert total_mean[q] == pytest.approx(total_s[q], abs=1e-10)
        assert total_sem[q] == pytest.approx(0.0, abs=1e-10)
    assert o_cls_last == o_cls_s
    assert o_cls_sem is None


def test_compute_network_connectivity_zero_formers_raises():
    """compute_network_connectivity raises ValueError when all counts are zero."""
    with pytest.raises(ValueError, match="total_formers is zero"):
        compute_network_connectivity({0: 0, 1: 0, 2: 0})


def test_per_pair_cutoff_eliminates_false_free_oxygens():
    """Per-pair cutoffs avoid misclassifying oxygens as 'free' in multi-former glasses.

    In 20Na2O-10B2O3-70SiO2, B-O bonds (~1.4 Å) are shorter than Si-O bonds
    (~1.6 Å).  Using only the B-O cutoff (1.5 Å) for *all* former-O pairs would
    miss many Si-O bonds and produce spurious "free" oxygens.  A per-pair cutoff
    dict should produce zero (or near-zero) free oxygens.
    """
    filename = DATA_DIR / "20Na2O-10B2O3-70SiO2.dump"
    atoms = read(filename, format="lammps-dump-text")
    type_id = atoms.get_atomic_numbers().copy()
    to_Z = np.array([0, 8, 14, 5, 11], dtype=int)
    Z = to_Z[type_id]
    atoms.set_atomic_numbers(Z)
    atoms.arrays["type"] = Z

    O_Z = 8
    B_Z = 5
    SI_Z = 14
    former_types = [B_Z, SI_Z]

    # Per-pair cutoff: each former uses its own appropriate cutoff
    pair_cutoff = {(B_Z, O_Z): 1.5, (SI_Z, O_Z): 1.9}

    _, _, o_classes_pair = compute_qn_and_classify(atoms, pair_cutoff, former_types, O_Z)
    n_free_pair = sum(1 for c in o_classes_pair.values() if c == "free")

    # A too-short scalar cutoff (B-O only) produces many false free oxygens
    _, _, o_classes_short = compute_qn_and_classify(atoms, 1.5, former_types, O_Z)
    n_free_short = sum(1 for c in o_classes_short.values() if c == "free")

    # The short scalar cutoff should produce significantly more free oxygens
    assert n_free_short > 10, f"Expected many false free oxygens with scalar B-O cutoff, got {n_free_short}"
    # Per-pair cutoffs should produce very few or no free oxygens
    assert n_free_pair < n_free_short, (
        f"Per-pair cutoffs should reduce free oxygens: pair={n_free_pair}, short={n_free_short}"
    )
