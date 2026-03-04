"""Tools to get results from all available analysis functions in this code.

Author: Achraf Atila (achraf.atila@bam.de)
"""

import numpy as np
import plotly.graph_objects as go
from ase.atoms import Atoms
from ase.data import chemical_symbols
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from amorphouspy.analysis.bond_angle_distribution import compute_angles
from amorphouspy.analysis.qn_network_connectivity import compute_network_connectivity, compute_qn
from amorphouspy.analysis.radial_distribution_functions import compute_coordination, compute_rdf
from amorphouspy.analysis.rings import compute_guttmann_rings, generate_bond_length_dict


class CoordinationData(BaseModel):
    """Coordination number distributions for different element types."""

    oxygen: dict[str, int] = Field(default_factory=dict, description="Oxygen coordination number distribution")
    formers: dict[str, dict[str, int]] = Field(
        default_factory=dict, description="Network former coordination distributions"
    )
    modifiers: dict[str, dict[str, int]] = Field(
        default_factory=dict, description="Modifier coordination distributions"
    )


class NetworkData(BaseModel):
    """Network connectivity and Q^n distribution data."""

    Qn_distribution: dict[str, float] = Field(..., description="Q^n distribution for network connectivity")
    Qn_distribution_partial: dict[str, dict[str, float]] = Field(
        ..., description="Partial Q^n distributions by former type"
    )
    connectivity: float = Field(default=0.0, description="Overall network connectivity (e.g. 0, 3.5, ...)")


class StructuralDistributions(BaseModel):
    """Structural distributions including bond angles and ring statistics."""

    bond_angles: dict[str, tuple[list[float], list[float]]] = Field(
        ..., description="Bond angle distributions for each former type"
    )
    rings: dict[str, dict[str, int] | float] = Field(
        default_factory=dict, description="Ring statistics with 'distribution' (dict[str, int]) and 'mean_size' (float)"
    )


class RadialDistributionData(BaseModel):
    """Radial distribution function and cumulative coordination data."""

    r: list[float] = Field(..., description="Radial distance array for RDFs (Å)")
    rdfs: dict[str, list[float]] = Field(..., description="Radial distribution functions for atom pairs")
    cumulative_coordination: dict[str, list[float]] = Field(..., description="Cumulative coordination numbers")


class ElementInfo(BaseModel):
    """Element classification and type information."""

    formers: list[str] = Field(default_factory=list, description="List of network forming elements")
    modifiers: list[str] = Field(default_factory=list, description="List of modifier elements")
    cutoffs: dict[str, float] = Field(
        default_factory=dict, description="Coordination cutoff distances for each element"
    )


class StructureData(BaseModel):
    """Structured results of structural analysis with all computed properties organized into logical groups."""

    density: float = Field(..., description="Calculated density of the structure (g/cm³)")
    coordination: CoordinationData = Field(
        default_factory=CoordinationData, description="Coordination number distributions"
    )
    network: NetworkData = Field(..., description="Network connectivity data")
    distributions: StructuralDistributions = Field(
        default_factory=StructuralDistributions, description="Structural distributions"
    )
    rdfs: RadialDistributionData = Field(..., description="Radial distribution function data")
    elements: ElementInfo = Field(..., description="Element classification and properties")


def find_rdf_minimum(
    r: np.ndarray, rdf_data: np.ndarray, sigma: float = 2.0, window_length: int = 21, polyorder: int = 3
) -> float | None:
    """Identify the first minimum in a radial distribution function (RDF) curve.

    The function applies Gaussian and Savitzky-Golay smoothing to reduce noise,
    detects the first peak, and searches for the first subsequent minimum using
    gradient sign changes. If no valid minimum is found, it falls back to
    detecting where the RDF drops below 1.0 after the peak.

    Args:
        r: Array of radial distances (Å), assumed to be uniformly spaced.
        rdf_data: RDF values corresponding to `r`.
        sigma: Standard deviation for Gaussian smoothing.
        window_length: Window length for Savitzky-Golay filter. Must be odd.
        polyorder: Polynomial order for Savitzky-Golay filter.

    Returns:
        The radial distance of the first minimum (Å) if found, otherwise `None`.

    Example:
        >>> r = np.linspace(0, 10, 100)
        >>> rdf = np.sin(r) + 1.0  # Dummy data
        >>> min_r = find_rdf_minimum(r, rdf)

    """
    rdf_smooth1 = gaussian_filter1d(rdf_data, sigma=sigma)
    rdf_smooth2 = savgol_filter(rdf_smooth1, window_length=window_length, polyorder=polyorder)

    first_peak_idx = np.argmax(rdf_smooth2)
    dr = r[1] - r[0]  # assuming uniform spacing
    gradient = np.gradient(rdf_smooth2, dr)

    # PERF401 fix: list comprehension instead of appending in a loop
    sign_changes = [i for i in range(first_peak_idx + 1, len(gradient) - 1) if gradient[i] < 0 and gradient[i + 1] > 0]

    if sign_changes:
        first_min_idx = sign_changes[0]
        peak_height = rdf_smooth2[first_peak_idx]
        min_height = rdf_smooth2[first_min_idx]
        if (peak_height - min_height) > 0.1 * peak_height:
            return r[first_min_idx]

    for i in range(first_peak_idx + 1, len(rdf_smooth2)):
        if rdf_smooth2[i] < 1.0:
            return r[i]

    return None


def _classify_elements(unique_z: np.ndarray) -> tuple[dict[int, str], set[str], set[str], bool]:
    """Classifies elements into network formers, modifiers, and oxygen.

    Args:
        unique_z: Unique atomic numbers in the system.

    Returns:
        A tuple containing:
            - type_map: Mapping atomic number -> element symbol.
            - network_formers: Set of network former symbols.
            - modifiers: Set of modifier symbols.
            - oxygen_present: Whether oxygen is present.

    Example:
        >>> unique_z = np.array([14, 8, 11])
        >>> type_map, formers, modifiers, has_O = _classify_elements(unique_z)

    """
    type_map: dict[int, str] = {}
    for z in unique_z:
        try:
            type_map[z] = chemical_symbols[z]
        except KeyError:
            type_map[z] = f"E{z}"

    glass_formers = {"Si", "B", "P", "Ge", "Al", "Ti", "Zr"}
    glass_modifiers = {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Zn", "Pb"}

    network_formers: set[str] = set()
    modifiers: set[str] = set()
    oxygen_present = False

    for z in unique_z:
        symbol = type_map[z]
        if symbol == "O":
            oxygen_present = True
        elif symbol in glass_formers:
            network_formers.add(symbol)
        elif symbol in glass_modifiers:
            modifiers.add(symbol)
        else:
            network_formers.add(symbol)

    return type_map, network_formers, modifiers, oxygen_present


def analyze_structure(atoms: Atoms) -> StructureData:  # noqa: C901, PLR0912, PLR0915
    """Perform a comprehensive structural analysis of an atomic configuration.

    Args:
        atoms: Atomic configuration to analyze.

    Returns:
        StructureData: Object containing structured analysis results.

    Example:
        >>> structure_data = analyze_structure(my_atoms)
        >>> print(structure_data.density)

    """
    atomic_numbers = atoms.get_atomic_numbers()
    unique_z = np.unique(atomic_numbers)

    # Calculate density
    volume_cm3 = atoms.get_volume() * 1e-24  # Convert Å³ to cm³
    avogadro_number = 6.022140857e23  # Avogadro's number (mol⁻¹)
    total_mass_g = atoms.get_masses().sum() / avogadro_number  # Convert amu to g
    density = total_mass_g / volume_cm3

    type_map, network_formers, modifiers, _oxygen_present = _classify_elements(unique_z)
    former_types = [z for z, sym in type_map.items() if sym in network_formers]
    modifier_types = [z for z, sym in type_map.items() if sym in modifiers]
    O_type = [z for z, sym in type_map.items() if sym == "O"]

    r, rdfs, cumcn = compute_rdf(atoms, r_max=10, n_bins=2000)

    def _canonical(a: int, b: int) -> tuple[int, int]:
        return (min(a, b), max(a, b))

    cutoff_map: dict[str, float] = {}
    for z in unique_z:
        symbol = type_map[z]
        if symbol == "O":
            if former_types:
                pair = _canonical(O_type[0], former_types[0])
                if pair in rdfs:
                    r_min = find_rdf_minimum(r, rdfs[pair])
                    cutoff_map[symbol] = r_min if r_min is not None else 2.0
        else:
            pair = _canonical(z, O_type[0]) if O_type else None
            if pair and pair in rdfs:
                r_min = find_rdf_minimum(r, rdfs[pair])
                if r_min is not None:
                    cutoff_map[symbol] = r_min
                else:
                    element_fallbacks = {
                        "Si": 2.0,
                        "B": 1.8,
                        "P": 1.8,
                        "Ge": 2.0,
                        "Al": 2.0,
                        "Ti": 2.0,
                        "Zr": 2.2,
                        "Na": 3.52,
                        "K": 3.8,
                        "Ca": 3.0,
                        "Mg": 2.8,
                        "Li": 2.5,
                        "Ba": 3.2,
                        "Zn": 2.5,
                    }
                    cutoff_map[symbol] = element_fallbacks.get(symbol, 3.0)

    # Coordination calculations
    O_coord = {}
    if O_type:
        O_cutoff = cutoff_map.get("O", 2.0)
        O_coord_raw, _ = compute_coordination(atoms, O_type[0], O_cutoff, former_types + modifier_types)
        # Convert integer keys to strings for HDF5 compatibility
        O_coord = {str(k): v for k, v in O_coord_raw.items()}

    former_coords = {}
    for former in network_formers:
        z = next(k for k, v in type_map.items() if v == former)
        if O_type:
            coord_raw, _ = compute_coordination(atoms, z, cutoff_map[former], O_type)
            # Convert integer keys to strings for HDF5 compatibility
            former_coords[former] = {str(k): v for k, v in coord_raw.items()}

    modifier_coords = {}
    for mod in modifiers:
        z = next(k for k, v in type_map.items() if v == mod)
        if O_type:
            coord_raw, _ = compute_coordination(atoms, z, cutoff_map[mod], O_type)
            # Convert integer keys to strings for HDF5 compatibility
            modifier_coords[mod] = {str(k): v for k, v in coord_raw.items()}

    Qn_dist = {}
    Qn_dist_partial = {}
    network_connectivity = 0.0
    if network_formers and O_type:
        Qn_dist_raw, Qn_dist_partial_raw = compute_qn(atoms, cutoff_map["O"], former_types, O_type[0])

        # Convert integer keys to strings for Pydantic compatibility
        Qn_dist = {str(k): v for k, v in Qn_dist_raw.items()} if Qn_dist_raw else {}
        Qn_dist_partial = {}
        if Qn_dist_partial_raw:
            for element_key, qn_dict in Qn_dist_partial_raw.items():
                # Convert both outer and inner keys to strings
                Qn_dist_partial[str(element_key)] = {str(k): v for k, v in qn_dict.items()}

        if Qn_dist_raw and sum(Qn_dist_raw.values()) > 0:
            network_connectivity = compute_network_connectivity(Qn_dist_raw)

    # Bond angles and rings
    bond_angle_distributions = {}
    for former in network_formers:
        z = next(k for k, v in type_map.items() if v == former)
        if O_type:
            bond_angle_distributions[former] = compute_angles(
                atoms, center_type=z, neighbor_type=O_type[0], cutoff=cutoff_map[former], bins=180
            )

    ring_statistics_data = {}
    if network_formers and O_type:
        specific_cutoffs = {(former, "O"): cutoff_map[former] for former in network_formers}
        bond_lengths = generate_bond_length_dict(atoms, specific_cutoffs=specific_cutoffs)
        rings_dist, mean_ring_size = compute_guttmann_rings(atoms, bond_lengths=bond_lengths, max_size=40, n_cpus=1)
        # Convert integer keys to strings for HDF5 compatibility
        rings_dist_str = {str(k): v for k, v in rings_dist.items()}
        ring_statistics_data = {"distribution": rings_dist_str, "mean_size": mean_ring_size}

    # Convert data for serialization
    def to_list(data: np.ndarray | list) -> list:
        return data.tolist() if hasattr(data, "tolist") else list(data)

    bond_angle_distributions_serializable = {
        former: (to_list(angles), to_list(counts)) for former, (angles, counts) in bond_angle_distributions.items()
    }

    rdfs_serializable = {}
    cumcn_serializable = {}
    for pair, rdf_data in rdfs.items():
        elem1, elem2 = type_map[pair[0]], type_map[pair[1]]
        key = f"{elem1}-{elem2}"
        rdfs_serializable[key] = to_list(rdf_data)
        if pair in cumcn:
            cumcn_serializable[key] = to_list(cumcn[pair])

    return StructureData(
        density=density,
        coordination=CoordinationData(oxygen=O_coord, formers=former_coords, modifiers=modifier_coords),
        network=NetworkData(
            Qn_distribution=Qn_dist, Qn_distribution_partial=Qn_dist_partial, connectivity=network_connectivity
        ),
        distributions=StructuralDistributions(
            bond_angles=bond_angle_distributions_serializable, rings=ring_statistics_data
        ),
        rdfs=RadialDistributionData(r=to_list(r), rdfs=rdfs_serializable, cumulative_coordination=cumcn_serializable),
        elements=ElementInfo(formers=list(network_formers), modifiers=list(modifiers), cutoffs=cutoff_map),
    )


def _add_coordination_plots(fig: go.Figure, structure_data: StructureData, colors: list[str]) -> None:
    """Add coordination distribution plots to the figure."""
    # Plot 1: Oxygen coordination distribution
    if structure_data.coordination.oxygen:
        keys = [int(k) for k in structure_data.coordination.oxygen]
        values = list(structure_data.coordination.oxygen.values())
        total = sum(values)
        percentages = [v * 100 / total for v in values] if total > 0 else values

        fig.add_trace(
            go.Bar(
                x=keys,
                y=percentages,
                name="O coordination",
                marker_color="steelblue",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Plot 2: Former coordination distributions
    if structure_data.coordination.formers:
        for i, (former, coord_data) in enumerate(structure_data.coordination.formers.items()):
            if coord_data:
                keys = [int(k) for k in coord_data]
                values = list(coord_data.values())
                total = sum(values)
                percentages = [v * 100 / total for v in values] if total > 0 else values

                fig.add_trace(
                    go.Bar(
                        x=keys,
                        y=percentages,
                        name=f"CN_{former}",
                        marker_color=colors[i % len(colors)],
                        offsetgroup=i,
                        showlegend=True,
                    ),
                    row=1,
                    col=2,
                )


def _add_network_plots(fig: go.Figure, structure_data: StructureData, colors: list[str]) -> None:
    """Add modifier coordination and Q^n distribution plots to the figure."""
    # Plot 3: Modifier coordination distributions
    if structure_data.coordination.modifiers:
        for i, (modifier, coord_data) in enumerate(structure_data.coordination.modifiers.items()):
            if coord_data:
                keys = [int(k) for k in coord_data]
                values = list(coord_data.values())
                total = sum(values)
                percentages = [v * 100 / total for v in values] if total > 0 else values

                fig.add_trace(
                    go.Bar(
                        x=keys,
                        y=percentages,
                        name=f"CN_{modifier}",
                        marker_color=colors[i % len(colors)],
                        offsetgroup=i,
                        showlegend=True,
                    ),
                    row=2,
                    col=1,
                )

    # Plot 4: Qn distribution
    if structure_data.network.Qn_distribution:
        keys = [int(k) for k in structure_data.network.Qn_distribution]
        values = list(structure_data.network.Qn_distribution.values())
        total = sum(values)
        percentages = [v * 100 / total for v in values] if total > 0 else values

        fig.add_trace(
            go.Bar(
                x=keys,
                y=percentages,
                name="Q^n",
                marker_color="purple",
                showlegend=True,
            ),
            row=2,
            col=2,
        )


def _add_ring_plots(fig: go.Figure, structure_data: StructureData, colors: list[str]) -> None:
    """Add ring statistics and bond angle plots to the figure."""
    # Constants
    EXPECTED_TUPLE_LENGTH = 2

    # Plot 5: Bond angle distribution
    if structure_data.distributions.bond_angles:
        for i, (angle_type, angle_data) in enumerate(structure_data.distributions.bond_angles.items()):
            if angle_data and len(angle_data) == EXPECTED_TUPLE_LENGTH:
                angles, distribution = angle_data
                fig.add_trace(
                    go.Scatter(
                        x=angles,
                        y=distribution,
                        mode="lines",
                        name=f"Angles {angle_type}",
                        line={"color": colors[i % len(colors)], "width": 2},
                        showlegend=True,
                    ),
                    row=3,
                    col=1,
                )

    # Plot 6: Ring size distribution
    if (
        structure_data.distributions.rings
        and "distribution" in structure_data.distributions.rings
        and structure_data.distributions.rings["distribution"]
    ):
        ring_dist = structure_data.distributions.rings["distribution"]
        ring_sizes = [int(k) for k in ring_dist]
        ring_counts = list(ring_dist.values())
        total_rings = sum(ring_counts)
        percentages = [count * 100 / total_rings for count in ring_counts] if total_rings > 0 else ring_counts

        fig.add_trace(
            go.Bar(
                x=ring_sizes,
                y=percentages,
                name="Ring sizes",
                marker_color="green",
                showlegend=True,
            ),
            row=3,
            col=2,
        )


def _add_rdf_plots(fig: go.Figure, structure_data: StructureData, colors: list[str]) -> None:
    """Add radial distribution function plots to the figure."""
    if not structure_data.rdfs.rdfs:
        return

    plot_positions = [(4, 1), (4, 2), (5, 1)]  # Skip (5, 2) as it's empty
    rdf_pairs = list(structure_data.rdfs.rdfs.items())

    for idx, (row, col) in enumerate(plot_positions):
        if idx < len(rdf_pairs):
            pair, rdf_data = rdf_pairs[idx]
            if rdf_data and structure_data.rdfs.r:
                # RDF trace
                fig.add_trace(
                    go.Scatter(
                        x=structure_data.rdfs.r,
                        y=rdf_data,
                        mode="lines",
                        name=f"g(r) {pair}",
                        line={"color": colors[0], "width": 2},
                        showlegend=True,
                        yaxis="y",
                    ),
                    row=row,
                    col=col,
                )

                # Coordination number trace (if available)
                if structure_data.rdfs.cumulative_coordination and pair in structure_data.rdfs.cumulative_coordination:
                    fig.add_trace(
                        go.Scatter(
                            x=structure_data.rdfs.r,
                            y=structure_data.rdfs.cumulative_coordination[pair],
                            mode="lines",
                            name=f"CN(r) {pair}",
                            line={"color": colors[1], "width": 2, "dash": "dash"},
                            showlegend=True,
                            yaxis="y",
                        ),
                        row=row,
                        col=col,
                    )


def plot_analysis_results_plotly(structure_data: StructureData) -> go.Figure:
    """Generate interactive Plotly plots for structural analysis results.

    Args:
        structure_data: Structural analysis results.

    Returns:
        Interactive Plotly figure object.

    Example:
        >>> fig = plot_analysis_results_plotly(structure_data)
        >>> fig.show()

    """
    # Constants for magic values
    LAST_ROW = 5
    LAST_COL = 2

    # Create subplot layout (5x2 grid)
    fig = make_subplots(
        rows=5,
        cols=2,
        subplot_titles=[
            "Oxygen Coordination",
            "Former Coordination",
            "Modifier Coordination",
            "Q^n Distribution",
            "Bond Angles",
            "Ring Statistics",
            "O-O RDF",
            "Former-O RDFs",
            "Modifier-O RDFs",
            "",
        ],
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, None],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.12,
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    _add_coordination_plots(fig, structure_data, colors)
    _add_network_plots(fig, structure_data, colors)
    _add_ring_plots(fig, structure_data, colors)
    _add_rdf_plots(fig, structure_data, colors)

    # Update layout and axes
    fig.update_layout(
        height=1500,  # Increased height for 5 rows
        width=1000,  # Reduced width for 2 columns
        showlegend=True,
        legend={"orientation": "v", "yanchor": "top", "y": 1, "xanchor": "left", "x": 1.02},
    )

    # Update individual subplot axes
    # Row 1: Coordination plots
    fig.update_xaxes(title_text="O_n", row=1, col=1)
    fig.update_yaxes(title_text="Percentage", row=1, col=1)
    fig.update_xaxes(title_text="Former_n", row=1, col=2)
    fig.update_yaxes(title_text="Percentage", row=1, col=2)

    # Row 2: Modifier and Q^n distributions
    fig.update_xaxes(title_text="Modifier_n", row=2, col=1)
    fig.update_yaxes(title_text="Percentage", row=2, col=1)
    fig.update_xaxes(title_text="Q^n", row=2, col=2)
    fig.update_yaxes(title_text="Percentage", row=2, col=2)

    # Row 3: Bond angles and rings
    fig.update_xaxes(title_text="Angle (degrees)", row=3, col=1)
    fig.update_yaxes(title_text="Frequency", row=3, col=1)
    fig.update_xaxes(title_text="Former atoms in ring", row=3, col=2)
    fig.update_yaxes(title_text="Normalized count", row=3, col=2)

    # Row 4-5: RDF plots
    for row in [4, 5]:
        for col in [1, 2]:
            if not (row == LAST_ROW and col == LAST_COL):  # Skip the empty subplot
                fig.update_xaxes(title_text="r (Å)", range=[0, 10], row=row, col=col)
                fig.update_yaxes(title_text="g(r), CN(r)", range=[0, 25], row=row, col=col)

    return fig
