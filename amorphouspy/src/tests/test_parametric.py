"""Unit tests for parametric_melt_quench workflow."""

from unittest.mock import MagicMock, patch

import pytest
from amorphouspy.workflows.parametric import parametric_melt_quench
from ase import Atoms

COMPOSITION = {"SiO2": 100}
MOCK_STRUCTURE = Atoms("SiO2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])


@pytest.fixture(autouse=True)
def _patch_deps():
    """Patch all external dependencies of parametric_melt_quench for every test.

    Replaces get_structure_dict, get_ase_structure, generate_potential, and
    melt_quench_simulation with lightweight fakes so no LAMMPS installation is needed.
    Yields the get_ase_structure mock for tests that need to inspect call counts.
    """
    with (
        patch("amorphouspy.workflows.parametric.get_structure_dict", return_value={"Si": 1, "O": 2}),
        patch("amorphouspy.workflows.parametric.get_ase_structure", return_value=MOCK_STRUCTURE) as mock_structure,
        patch("amorphouspy.workflows.parametric.generate_potential", return_value=MagicMock()),
        patch(
            "amorphouspy.workflows.parametric.melt_quench_simulation",
            return_value={"structure": MOCK_STRUCTURE, "result": {}},
        ),
    ):
        yield mock_structure


@pytest.fixture
def mock_structure(_patch_deps):
    """Expose the get_ase_structure mock from _patch_deps for call-count assertions."""
    return _patch_deps


def test_structure_generated_once_per_size(mock_structure):
    """Structure generation is cached per unique size, not repeated per cooling rate.

    With 2 distinct sizes and 3 total (size, rate) pairs, get_ase_structure must be
    called exactly twice — once per size entry in the study dict.
    """
    study = {500: [1e11, 1e12], 1000: [1e13]}  # 2 sizes, 3 total runs

    parametric_melt_quench(COMPOSITION, "pmmcs", study)

    assert mock_structure.call_count == 2  # NOT 3


def test_results_length_matches_total_runs():
    """Returned list has one entry per (n_atoms, cooling_rate) pair, not per size."""
    study = {500: [1e11, 1e12], 1000: [1e13]}  # 3 (n_atoms, cooling_rate) pairs

    results = parametric_melt_quench(COMPOSITION, "pmmcs", study)

    assert len(results) == 3


def test_result_dict_has_correct_keys_and_metadata():
    """Each result dict contains all expected keys with values from the call arguments."""
    study = {500: [1e12]}

    results = parametric_melt_quench(COMPOSITION, "pmmcs", study, heating_rate=1e14)

    r = results[0]
    assert set(r.keys()) == {"n_atoms", "target_n_atoms", "cooling_rate", "heating_rate", "structure", "result"}
    assert r["target_n_atoms"] == 500
    assert r["cooling_rate"] == 1e12
    assert r["heating_rate"] == 1e14


def test_heating_rate_none_defaults_to_1e13():
    """Passing heating_rate=None substitutes the protocol default of 1e13 K/s."""
    study = {500: [1e12]}

    results = parametric_melt_quench(COMPOSITION, "pmmcs", study, heating_rate=None)

    assert results[0]["heating_rate"] == 1e13
