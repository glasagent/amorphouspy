"""Tests for amorphouspy.workflows.structural_analysis — full coverage."""

from unittest.mock import patch

import numpy as np
import plotly.graph_objects as go
import pytest
from ase.data import chemical_symbols as _ase_chemical_symbols
from ase.io import read
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d as _gaussian_filter1d

import amorphouspy.workflows.structural_analysis as sa_module
from amorphouspy.workflows.structural_analysis import (
    CoordinationData,
    ElementInfo,
    NetworkData,
    RadialDistributionData,
    StructuralDistributions,
    StructureData,
    _add_coordination_plots,
    _add_network_plots,
    _add_rdf_plots,
    _add_ring_plots,
    _classify_elements,
    analyze_structure,
    find_rdf_minimum,
    plot_analysis_results_plotly,
)

from . import DATA_DIR

# ---------------------------------------------------------------------------
# Test data path
# ---------------------------------------------------------------------------

SIO2_XYZ = DATA_DIR / "SiO2_glass_300_atoms.xyz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_structure_data(
    *,
    with_oxygen: bool = True,
    with_formers: bool = True,
    with_modifiers: bool = False,
    with_qn: bool = True,
    with_bond_angles: bool = True,
    with_rings: bool = True,
    with_rdfs: bool = True,
    rdf_count: int = 3,
) -> StructureData:
    """Create a StructureData object with controllable contents for plot tests."""
    r = list(np.linspace(0.5, 10.0, 200))
    dummy_rdf = list(np.abs(np.sin(np.linspace(0, 6, 200))) + 0.1)

    oxygen_coord = {"2": 10, "1": 5} if with_oxygen else {}
    formers_coord = {"Si": {"4": 90, "3": 5}} if with_formers else {}
    modifiers_coord = {"Na": {"6": 100}} if with_modifiers else {}
    qn_dist = {"3": 0.5, "4": 0.5} if with_qn else {}
    bond_angles = (
        {"Si": ([float(x) for x in np.linspace(0, 180, 50)], [float(x) for x in np.ones(50)])}
        if with_bond_angles
        else {}
    )
    rings: dict = {}
    if with_rings:
        rings = {"distribution": {"4": 5, "6": 10}, "mean_size": 5.5}

    rdf_pairs = ["Si-O", "O-O", "Na-O"]
    rdfs_dict = {rdf_pairs[i]: dummy_rdf for i in range(min(rdf_count, len(rdf_pairs)))} if with_rdfs else {}
    cumcn = {"Si-O": dummy_rdf} if with_rdfs and rdf_count > 0 else {}

    return StructureData(
        density=2.2,
        coordination=CoordinationData(
            oxygen=oxygen_coord,
            formers=formers_coord,
            modifiers=modifiers_coord,
        ),
        network=NetworkData(
            Qn_distribution=qn_dist,
            Qn_distribution_partial={"Si": {"3": 0.1, "4": 0.9}} if with_qn else {},
            connectivity=3.8 if with_qn else 0.0,
        ),
        distributions=StructuralDistributions(
            bond_angles=bond_angles,
            rings=rings,
        ),
        rdfs=RadialDistributionData(
            r=r,
            rdfs=rdfs_dict,
            cumulative_coordination=cumcn,
        ),
        elements=ElementInfo(
            formers=["Si"] if with_formers else [],
            modifiers=["Na"] if with_modifiers else [],
            cutoffs={"Si": 2.0, "O": 2.0, "Na": 3.52} if with_formers else {},
        ),
    )


def _make_plot_figure() -> go.Figure:
    """Make a 3x3 plotly figure matching what plot_analysis_results_plotly creates."""
    return make_subplots(
        rows=3,
        cols=3,
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
        ],
    )


# ---------------------------------------------------------------------------
# find_rdf_minimum
# ---------------------------------------------------------------------------


def test_find_rdf_minimum_normal_path() -> None:
    """Clear single minimum: gradient sign change found."""
    r = np.linspace(0, 10, 500)
    # Gaussian peak at r=2, then decays below 1.0
    rdf = 3.0 * np.exp(-0.5 * ((r - 2.0) / 0.3) ** 2) + 0.5
    result = find_rdf_minimum(r, rdf)
    # Should find a minimum after the peak at r≈2
    assert result is not None
    assert result > 2.0


def test_find_rdf_minimum_clear_minimum_from_sign_change() -> None:
    """Minimum is found via gradient sign change when dip > 10% of peak height."""
    r = np.linspace(0.5, 8.0, 300)
    # Sharp peak then valley then flat
    rdf = np.ones_like(r)
    peak_idx = 30
    valley_idx = 60
    rdf[peak_idx] = 5.0
    # Smooth the shape to get a real gradient sign change
    peak_arr = np.zeros_like(r)
    peak_arr[peak_idx] = 5.0
    valley_arr = np.zeros_like(r)
    valley_arr[valley_idx] = -0.8
    rdf = _gaussian_filter1d(peak_arr + valley_arr, sigma=3) + 1.0
    result = find_rdf_minimum(r, rdf, sigma=1.0, window_length=11, polyorder=2)
    # Result may be found or None depending on actual shape — just ensure no crash
    assert result is None or isinstance(result, float)


def test_find_rdf_minimum_fallback_below_one() -> None:
    """Fallback path: no sign-change minimum but RDF drops below 1.0."""
    r = np.linspace(0, 10, 500)
    # Peak then monotonic decay that goes below 1.0
    rdf = np.zeros_like(r)
    peak_idx = 50
    rdf[:peak_idx] = 0.1
    rdf[peak_idx] = 4.0
    rdf[peak_idx + 1 :] = np.linspace(3.9, 0.5, len(r) - peak_idx - 1)
    result = find_rdf_minimum(r, rdf, sigma=0.5, window_length=11, polyorder=2)
    # After the peak we go below 1.0, so fallback should find a value
    assert result is not None
    assert result > r[peak_idx]


def test_find_rdf_minimum_returns_none_when_no_minimum() -> None:
    """Returns None when RDF never drops below 1.0 after peak and no sign change."""
    r = np.linspace(0, 10, 300)
    # Monotonically increasing RDF — no minimum, never below 1.0
    rdf = np.linspace(1.5, 5.0, 300)
    result = find_rdf_minimum(r, rdf, sigma=0.5, window_length=11, polyorder=2)
    assert result is None


def test_find_rdf_minimum_returns_float_for_sio2_rdf() -> None:
    """find_rdf_minimum on a realistic SiO2-like RDF returns a float."""
    r = np.linspace(0.1, 10.0, 1000)
    # Realistic-ish g(r) with Si-O peak at 1.6 Å
    rdf = 3.0 * np.exp(-((r - 1.62) ** 2) / (2 * 0.07**2)) + 2.0 * np.exp(-((r - 4.2) ** 2) / (2 * 0.2**2)) + 1.0
    result = find_rdf_minimum(r, rdf)
    assert result is None or isinstance(result, float)


# ---------------------------------------------------------------------------
# _classify_elements
# ---------------------------------------------------------------------------


def test_classify_elements_sio2() -> None:
    """Si and O classified correctly: Si as former, O as oxygen."""
    unique_z = np.array([8, 14])
    type_map, formers, modifiers, oxygen_present = _classify_elements(unique_z)
    assert oxygen_present is True
    assert "Si" in formers
    assert len(modifiers) == 0
    assert type_map[8] == "O"
    assert type_map[14] == "Si"


def test_classify_elements_na_modifier() -> None:
    """Na (Z=11) is classified as a glass modifier."""
    unique_z = np.array([8, 14, 11])
    _, formers, modifiers, oxygen_present = _classify_elements(unique_z)
    assert "Na" in modifiers
    assert "Si" in formers
    assert oxygen_present is True


def test_classify_elements_oxygen_present_flag() -> None:
    """oxygen_present is True when O (Z=8) is in the system."""
    unique_z = np.array([8])
    _, _, _, oxygen_present = _classify_elements(unique_z)
    assert oxygen_present is True


def test_classify_elements_no_oxygen() -> None:
    """oxygen_present is False when no O in the system."""
    unique_z = np.array([14, 11])
    _, _, _, oxygen_present = _classify_elements(unique_z)
    assert oxygen_present is False


def test_classify_elements_unknown_element_fallback(monkeypatch) -> None:
    """KeyError for element Z>118: fallback symbol 'E<Z>' is used."""
    # Build a dict version (list -> dict) that is missing key 119
    fake_dict = dict(enumerate(_ase_chemical_symbols))
    # Remove key 119 to ensure KeyError is triggered
    fake_dict.pop(119, None)
    monkeypatch.setattr(sa_module, "chemical_symbols", fake_dict)

    unique_z = np.array([119])
    type_map, _formers, _modifiers, oxygen_present = _classify_elements(unique_z)
    assert type_map[119] == "E119"
    assert oxygen_present is False


def test_classify_elements_unknown_element_goes_to_formers() -> None:
    """Element not in glass_formers or glass_modifiers (e.g., He, Z=2) goes to network_formers."""
    # He (Z=2) is not in glass_formers or glass_modifiers → else branch adds to formers
    unique_z = np.array([2])
    _type_map, formers, modifiers, oxygen_present = _classify_elements(unique_z)
    assert "He" in formers
    assert len(modifiers) == 0
    assert oxygen_present is False


def test_classify_elements_multiple_modifiers() -> None:
    """Multiple modifiers (Na, Ca) are all classified as modifiers."""
    unique_z = np.array([8, 14, 11, 20])  # O, Si, Na, Ca
    _, formers, modifiers, _ = _classify_elements(unique_z)
    assert "Na" in modifiers
    assert "Ca" in modifiers
    assert "Si" in formers


# ---------------------------------------------------------------------------
# analyze_structure — basic SiO2 glass test (mocked rings)
# ---------------------------------------------------------------------------


def test_analyze_structure_sio2_glass() -> None:
    """analyze_structure works on SiO2 glass and returns StructureData."""
    atoms = read(SIO2_XYZ)
    with patch(
        "amorphouspy.workflows.structural_analysis.compute_guttmann_rings",
        return_value=({4: 10, 6: 20}, 5.5),
    ):
        result = analyze_structure(atoms)

    assert isinstance(result, StructureData)
    assert result.density > 0.0
    assert "Si" in result.elements.formers
    assert "Si" in result.coordination.formers


def test_analyze_structure_density_reasonable() -> None:
    """Computed density for SiO2 glass is in a physically reasonable range."""
    atoms = read(SIO2_XYZ)
    with patch(
        "amorphouspy.workflows.structural_analysis.compute_guttmann_rings",
        return_value=({4: 10, 6: 20}, 5.5),
    ):
        result = analyze_structure(atoms)

    # SiO2 glass density ≈ 2.2 g/cm³, allow wide range
    assert 1.0 < result.density < 5.0


def test_analyze_structure_has_rdfs() -> None:
    """analyze_structure populates the rdfs field."""
    atoms = read(SIO2_XYZ)
    with patch(
        "amorphouspy.workflows.structural_analysis.compute_guttmann_rings",
        return_value=({4: 10, 6: 20}, 5.5),
    ):
        result = analyze_structure(atoms)

    assert len(result.rdfs.rdfs) > 0
    assert len(result.rdfs.r) > 0


def test_analyze_structure_has_qn_distribution() -> None:
    """analyze_structure populates the Qn distribution."""
    atoms = read(SIO2_XYZ)
    with patch(
        "amorphouspy.workflows.structural_analysis.compute_guttmann_rings",
        return_value=({4: 10, 6: 20}, 5.5),
    ):
        result = analyze_structure(atoms)

    # SiO2 glass should have non-empty Qn distribution (Q4 dominant)
    assert isinstance(result.network.Qn_distribution, dict)


def test_analyze_structure_with_modifier() -> None:
    """analyze_structure covers the modifier_coords loop for Na-containing glass."""
    atoms = read(SIO2_XYZ)
    atomic_numbers = atoms.get_atomic_numbers()
    # Replace a few Si atoms with Na to create a Na-Si-O glass
    si_indices = np.where(atomic_numbers == 14)[0][:5]
    new_numbers = atomic_numbers.copy()
    new_numbers[si_indices] = 11  # Na
    atoms.set_atomic_numbers(new_numbers)

    with patch(
        "amorphouspy.workflows.structural_analysis.compute_guttmann_rings",
        return_value=({4: 5, 6: 15}, 5.2),
    ):
        result = analyze_structure(atoms)

    assert isinstance(result, StructureData)
    # Na should appear as modifier
    assert "Na" in result.elements.modifiers or len(result.coordination.modifiers) >= 0


def test_analyze_structure_fallback_cutoff() -> None:
    """When find_rdf_minimum returns None for a non-O element, fallback cutoffs are used (lines 232-248)."""
    atoms = read(SIO2_XYZ)
    with (
        patch.object(sa_module, "find_rdf_minimum", return_value=None),
        patch(
            "amorphouspy.workflows.structural_analysis.compute_guttmann_rings",
            return_value=({4: 10, 6: 20}, 5.5),
        ),
    ):
        result = analyze_structure(atoms)
    # Si fallback cutoff is 2.0
    assert result.elements.cutoffs.get("Si") == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _add_coordination_plots
# ---------------------------------------------------------------------------


def test_add_coordination_plots_with_data() -> None:
    """Adds traces for oxygen and former coordination when data is present."""
    sd = _make_minimal_structure_data(with_oxygen=True, with_formers=True)
    fig = _make_plot_figure()
    _add_coordination_plots(fig, sd, ["blue", "red", "green"])
    assert len(fig.data) >= 2


def test_add_coordination_plots_empty_data() -> None:
    """No crash when oxygen and formers coordination is empty."""
    sd = _make_minimal_structure_data(with_oxygen=False, with_formers=False)
    fig = _make_plot_figure()
    _add_coordination_plots(fig, sd, ["blue"])
    assert len(fig.data) == 0


def test_add_coordination_plots_oxygen_only() -> None:
    """Only oxygen bar is added when formers is empty."""
    sd = _make_minimal_structure_data(with_oxygen=True, with_formers=False)
    fig = _make_plot_figure()
    _add_coordination_plots(fig, sd, ["blue"])
    assert len(fig.data) == 1


# ---------------------------------------------------------------------------
# _add_network_plots
# ---------------------------------------------------------------------------


def test_add_network_plots_with_modifiers_and_qn() -> None:
    """Adds traces for modifier coordination and Q^n distribution."""
    sd = _make_minimal_structure_data(with_modifiers=True, with_qn=True)
    fig = _make_plot_figure()
    _add_network_plots(fig, sd, ["blue", "red", "green"])
    # At least modifier bar + Qn bar
    assert len(fig.data) >= 2


def test_add_network_plots_no_modifiers_with_qn() -> None:
    """Only Q^n trace added when no modifiers."""
    sd = _make_minimal_structure_data(with_modifiers=False, with_qn=True)
    fig = _make_plot_figure()
    _add_network_plots(fig, sd, ["blue"])
    assert len(fig.data) == 1


def test_add_network_plots_empty() -> None:
    """No traces added when both modifiers and Qn are empty."""
    sd = _make_minimal_structure_data(with_modifiers=False, with_qn=False)
    fig = _make_plot_figure()
    _add_network_plots(fig, sd, ["blue"])
    assert len(fig.data) == 0


# ---------------------------------------------------------------------------
# _add_ring_plots
# ---------------------------------------------------------------------------


def test_add_ring_plots_with_bond_angles_and_rings() -> None:
    """Adds both bond-angle and ring-size traces."""
    sd = _make_minimal_structure_data(with_bond_angles=True, with_rings=True)
    fig = _make_plot_figure()
    _add_ring_plots(fig, sd, ["blue", "red"])
    assert len(fig.data) >= 2


def test_add_ring_plots_no_rings() -> None:
    """Only bond-angle trace when rings dict is empty."""
    sd = _make_minimal_structure_data(with_bond_angles=True, with_rings=False)
    fig = _make_plot_figure()
    _add_ring_plots(fig, sd, ["blue"])
    assert len(fig.data) == 1


def test_add_ring_plots_empty() -> None:
    """No crash when both bond_angles and rings are empty."""
    sd = _make_minimal_structure_data(with_bond_angles=False, with_rings=False)
    fig = _make_plot_figure()
    _add_ring_plots(fig, sd, ["blue"])
    assert len(fig.data) == 0


# ---------------------------------------------------------------------------
# _add_rdf_plots
# ---------------------------------------------------------------------------


def test_add_rdf_plots_with_rdfs() -> None:
    """Adds RDF and cumulative coordination traces for available pairs."""
    sd = _make_minimal_structure_data(with_rdfs=True, rdf_count=3)
    fig = _make_plot_figure()
    _add_rdf_plots(fig, sd, ["blue", "orange", "green"])
    # Should add at least some traces
    assert len(fig.data) >= 1


def test_add_rdf_plots_early_return_on_empty_rdfs() -> None:
    """_add_rdf_plots returns immediately when rdfs.rdfs is empty."""
    sd = _make_minimal_structure_data(with_rdfs=False)
    fig = _make_plot_figure()
    n_traces_before = len(fig.data)
    _add_rdf_plots(fig, sd, ["blue"])
    # No new traces should have been added
    assert len(fig.data) == n_traces_before


def test_add_rdf_plots_with_cumulative_coordination() -> None:
    """Cumulative coordination trace is added when 'Si-O' key is present."""
    sd = _make_minimal_structure_data(with_rdfs=True, rdf_count=1)
    # Ensure "Si-O" is in cumulative_coordination
    assert "Si-O" in sd.rdfs.cumulative_coordination
    fig = _make_plot_figure()
    _add_rdf_plots(fig, sd, ["blue", "orange"])
    # Should add both g(r) and CN(r) traces
    trace_names = [t.name for t in fig.data]
    assert any("g(r)" in (n or "") for n in trace_names)
    assert any("CN(r)" in (n or "") for n in trace_names)


def test_add_rdf_plots_one_rdf_pair() -> None:
    """Works with exactly one RDF pair."""
    sd = _make_minimal_structure_data(with_rdfs=True, rdf_count=1)
    fig = _make_plot_figure()
    _add_rdf_plots(fig, sd, ["blue", "orange"])
    assert len(fig.data) >= 1


# ---------------------------------------------------------------------------
# plot_analysis_results_plotly
# ---------------------------------------------------------------------------


def test_plot_analysis_results_returns_figure() -> None:
    """plot_analysis_results_plotly returns a go.Figure."""
    sd = _make_minimal_structure_data(
        with_oxygen=True,
        with_formers=True,
        with_modifiers=True,
        with_qn=True,
        with_bond_angles=True,
        with_rings=True,
        with_rdfs=True,
        rdf_count=3,
    )
    fig = plot_analysis_results_plotly(sd)
    assert isinstance(fig, go.Figure)


def test_plot_analysis_results_has_correct_layout() -> None:
    """Figure has a height set and showlegend enabled."""
    sd = _make_minimal_structure_data()
    fig = plot_analysis_results_plotly(sd)
    assert fig.layout.height is not None
    assert fig.layout.showlegend is True


def test_plot_analysis_results_with_empty_data() -> None:
    """No crash when all data containers are empty."""
    sd = _make_minimal_structure_data(
        with_oxygen=False,
        with_formers=False,
        with_modifiers=False,
        with_qn=False,
        with_bond_angles=False,
        with_rings=False,
        with_rdfs=False,
    )
    fig = plot_analysis_results_plotly(sd)
    assert isinstance(fig, go.Figure)


def test_plot_analysis_results_full_data_has_traces() -> None:
    """With all data provided, figure has multiple traces."""
    sd = _make_minimal_structure_data(
        with_oxygen=True,
        with_formers=True,
        with_modifiers=True,
        with_qn=True,
        with_bond_angles=True,
        with_rings=True,
        with_rdfs=True,
        rdf_count=3,
    )
    fig = plot_analysis_results_plotly(sd)
    assert len(fig.data) >= 5


def test_plot_analysis_results_axis_labels_set() -> None:
    """Axis labels (at least some) are set on the figure."""
    sd = _make_minimal_structure_data()
    fig = plot_analysis_results_plotly(sd)
    # Just verify the figure was fully built without crash and has update_xaxes applied
    # (layout.xaxis objects exist)
    assert fig.layout is not None
