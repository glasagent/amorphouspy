"""Tools to get results from all available analysis functions in this code.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from ase import units
from ase.atoms import Atoms
from ase.data import chemical_symbols
from pyiron_base import job
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

from pyiron_glass.analysis.bond_angle_distribution import compute_angles
from pyiron_glass.analysis.qn_network_connectivity import compute_network_connectivity, compute_qn
from pyiron_glass.analysis.radial_distribution_functions import compute_coordination, compute_rdf
from pyiron_glass.analysis.rings import compute_guttmann_rings, generate_bond_length_dict


@dataclass
class StructureData:
    """Structured results of structural analysis."""

    density: float
    O_coordination: dict
    former_coordination: dict[str, dict]
    modifier_coordination: dict[str, dict]
    Qn_distribution: dict[str, float]
    Qn_distribution_partial: dict[str, dict]
    network_connectivity: float
    bond_angle_distributions: dict[str, tuple[np.ndarray, np.ndarray]]
    ring_statistics: dict
    type_map: dict[int, str]
    network_formers: list[str]
    modifiers: list[str]
    cutoff_map: dict[str, float]


@dataclass
class PlottingData:
    """Pre-processed data suitable for visualization."""

    r: np.ndarray
    rdfs: dict[tuple[int, int], np.ndarray]
    cumcn: dict
    bond_angle_distributions: dict[str, tuple[np.ndarray, np.ndarray]]
    ring_statistics: dict
    network_formers: list[str]
    modifiers: list[str]
    type_map: dict[int, str]
    O_type: list[int]
    former_types: list[int]
    modifier_types: list[int]
    O_coordination: dict
    former_coordination: dict[str, dict]
    modifier_coordination: dict[str, dict]
    Qn_distribution: dict[str, float]
    Qn_distribution_partial: dict[str, dict]


@dataclass
class MeltQuenchResult:
    """Final return object containing analysis results and plotting data."""

    results: StructureData
    plotting_data: PlottingData


def find_rdf_minimum(
    r: np.ndarray, rdf_data: np.ndarray, sigma: float = 2.0, window_length: int = 21, polyorder: int = 3
) -> float | None:
    """Identify the first minimum in a radial distribution function (RDF) curve.

    The function applies Gaussian and Savitzky-Golay smoothing to reduce noise,
    detects the first peak, and searches for the first subsequent minimum using
    gradient sign changes. If no valid minimum is found, it falls back to
    detecting where the RDF drops below 1.0 after the peak.

    Args:
        r (np.ndarray): Array of radial distances (Å), assumed to be uniformly spaced.
        rdf_data (np.ndarray): RDF values corresponding to `r`.
        sigma (float, optional): Standard deviation for Gaussian smoothing. Default is 2.0.
        window_length (int, optional): Window length for Savitzky-Golay filter. Must be odd. Default is 21.
        polyorder (int, optional): Polynomial order for Savitzky-Golay filter. Default is 3.

    Returns:
        float | None: The radial distance of the first minimum (Å) if found, otherwise `None`.

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
    """Classify elements into network formers, modifiers, and oxygen.

    Args:
        unique_z (np.ndarray): Unique atomic numbers in the system.

    Returns:
        tuple:
            - type_map (dict[int, str]): Mapping atomic number -> element symbol.
            - network_formers (set[str]): Set of network former symbols.
            - modifiers (set[str]): Set of modifier symbols.
            - oxygen_present (bool): Whether oxygen is present.

    """
    type_map: dict[int, str] = {}
    for z in unique_z:
        try:
            type_map[z] = chemical_symbols[z]
        except KeyError:
            type_map[z] = f"E{z}"

    glass_formers = {"Si", "B", "P", "Ge", "Al", "Ti", "Zr"}
    glass_modifiers = {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Zn", "Pb"}
    oxygen_symbol = "O"

    network_formers: set[str] = set()
    modifiers: set[str] = set()
    oxygen_present = False

    for z in unique_z:
        symbol = type_map[z]
        if symbol == oxygen_symbol:
            oxygen_present = True
        elif symbol in glass_formers:
            network_formers.add(symbol)
        elif symbol in glass_modifiers:
            modifiers.add(symbol)
        else:
            network_formers.add(symbol)

    return type_map, network_formers, modifiers, oxygen_present


@job
def analyze_structure(atoms: Atoms) -> MeltQuenchResult:  # noqa: C901, PLR0912, PLR0915
    """Perform a comprehensive structural analysis of an atomic configuration.

    Args:
        atoms (ase.Atoms): Atomic configuration to analyze.

    Returns:
        MeltQuenchResult: Object containing structured analysis results and plotting data.

    """
    atomic_numbers = atoms.get_atomic_numbers()
    unique_z = np.unique(atomic_numbers)

    # Calculate density
    volume_angstrom3 = atoms.get_volume()  # Volume in Å³
    volume_cm3 = volume_angstrom3 * 1e-24  # Convert to cm³
    total_mass_amu = atoms.get_masses().sum()  # Total mass in amu
    total_mass_g = total_mass_amu / units._Nav  # Convert to grams using Avogadro's number
    density = total_mass_g / volume_cm3  # Density in g/cm³

    type_map, network_formers, modifiers, oxygen_present = _classify_elements(unique_z)
    oxygen_symbol = "O"
    former_types = [z for z, sym in type_map.items() if sym in network_formers]
    modifier_types = [z for z, sym in type_map.items() if sym in modifiers]
    O_type = [z for z, sym in type_map.items() if sym == oxygen_symbol]

    r, rdfs, cumcn = compute_rdf(atoms, r_max=10, n_bins=2000)

    cutoff_map: dict[str, float] = {}
    for z in unique_z:
        symbol = type_map[z]
        if symbol == oxygen_symbol:
            if former_types:
                pair = (O_type[0], former_types[0])
                if pair in rdfs:
                    r_min = find_rdf_minimum(r, rdfs[pair])
                    cutoff_map[symbol] = r_min if r_min is not None else 2.0
        else:
            pair = (z, O_type[0]) if O_type else None
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

    O_coord = {}
    if O_type:
        O_cutoff = cutoff_map.get("O", 2.0)
        O_coord, _ = compute_coordination(atoms, O_type, O_cutoff, former_types + modifier_types)

    former_coords: dict[str, dict] = {}
    for former in network_formers:
        z = next(k for k, v in type_map.items() if v == former)
        if O_type:
            coord, _ = compute_coordination(atoms, [z], cutoff_map[former], O_type)
            former_coords[former] = coord

    modifier_coords: dict[str, dict] = {}
    for mod in modifiers:
        z = next(k for k, v in type_map.items() if v == mod)
        if O_type:
            coord, _ = compute_coordination(atoms, [z], cutoff_map[mod], O_type)
            modifier_coords[mod] = coord

    Qn_dist = {}
    Qn_dist_partial = {}
    network_connectivity = 0.0
    if network_formers and O_type:
        Qn_dist, Qn_dist_partial = compute_qn(atoms, cutoff_map["O"], former_types, O_type)
        if Qn_dist and sum(Qn_dist.values()) > 0:
            network_connectivity = compute_network_connectivity(Qn_dist)

    bond_angle_distributions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for former in network_formers:
        z = next(k for k, v in type_map.items() if v == former)
        if O_type:
            bond_angle_distributions[former] = compute_angles(
                atoms, center_type=[z], neighbor_type=O_type, cutoff=cutoff_map[former], bins=180
            )

    ring_statistics_data: dict = {}
    if network_formers and O_type:
        specific_cutoffs = {(former, "O"): cutoff_map[former] for former in network_formers}
        bond_lengths = generate_bond_length_dict(atoms, specific_cutoffs=specific_cutoffs)
        rings_dist, mean_ring_size = compute_guttmann_rings(atoms, bond_lengths=bond_lengths, max_size=40, n_cpus=1)
        ring_statistics_data["distribution"] = rings_dist
        ring_statistics_data["mean_size"] = mean_ring_size

    results = StructureData(
        density=density,
        O_coordination=O_coord,
        former_coordination=former_coords,
        modifier_coordination=modifier_coords,
        Qn_distribution=Qn_dist,
        Qn_distribution_partial=Qn_dist_partial,
        network_connectivity=network_connectivity,
        bond_angle_distributions=bond_angle_distributions,
        ring_statistics=ring_statistics_data,
        type_map=type_map,
        network_formers=network_formers,
        modifiers=modifiers,
        cutoff_map=cutoff_map,
    )

    plotting_data = PlottingData(
        r=r,
        rdfs=rdfs,
        cumcn=cumcn,
        bond_angle_distributions=bond_angle_distributions,
        ring_statistics=ring_statistics_data,
        network_formers=network_formers,
        modifiers=modifiers,
        type_map=type_map,
        O_type=O_type,
        former_types=former_types,
        modifier_types=modifier_types,
        O_coordination=O_coord,
        former_coordination=former_coords,
        modifier_coordination=modifier_coords,
        Qn_distribution=Qn_dist,
        Qn_distribution_partial=Qn_dist_partial,
    )

    return MeltQuenchResult(results=results, plotting_data=plotting_data)


def plot_analysis_results(plotting_data: PlottingData) -> plt.Figure:  # noqa: C901, PLR0912, PLR0915
    """Generate a set of plots summarizing the structural analysis of an atomic system.

    The figure includes:
      - Oxygen coordination distribution.
      - Former and modifier coordination distributions.
      - Qn distribution and partial Qn contributions.
      - Bond angle distributions for network formers.
      - Ring size distribution.
      - RDF and cumulative coordination number curves
        for O-O, former-O, and modifier-O pairs.

    Args:
        plotting_data (PlottingData): Pre-processed plotting data
        returned by `analyze_structure`.

    Returns:
        matplotlib.figure.Figure: The generated figure object with subplots.

    """
    # Create a larger figure with multiple subplots
    fig, ax = plt.subplots(3, 3, figsize=(18, 15), dpi=300)
    colors = ["C1", "C2", "C3", "C4"]

    # Extract data from dataclass
    bond_angle_distributions = plotting_data.bond_angle_distributions
    r = plotting_data.r
    rdfs = plotting_data.rdfs
    cumcn = plotting_data.cumcn
    ring_statistics = plotting_data.ring_statistics
    network_formers = plotting_data.network_formers
    modifiers = plotting_data.modifiers
    type_map = plotting_data.type_map
    O_type = plotting_data.O_type
    O_coord = plotting_data.O_coordination
    former_coords = plotting_data.former_coordination
    modifier_coords = plotting_data.modifier_coordination
    Qn_dist = plotting_data.Qn_distribution
    Qn_dist_partial = plotting_data.Qn_distribution_partial

    offset = 0.25

    # Plot 1: Oxygen coordination distribution
    if O_coord:
        ax[0, 0].bar(
            np.array(list(O_coord.keys())),
            np.array(list(O_coord.values())) * 100 / (np.array(list(O_coord.values()), dtype=float).sum()),
            width=0.2,
            align="center",
            linewidth=0.5,
            edgecolor="black",
            facecolor="C0",
            label="Total",
        )
        ax[0, 0].set_xlabel("O_n")
        ax[0, 0].set_ylabel("Percentage")
        ax[0, 0].set_ylim(0, 100)
        ax[0, 0].legend(prop={"size": 10})
    else:
        ax[0, 0].text(0.5, 0.5, "No oxygen coordination data", ha="center", va="center", transform=ax[0, 0].transAxes)
        ax[0, 0].set_xlabel("O_n")
        ax[0, 0].set_ylabel("Percentage")

    # Plot 2: Former coordination distributions
    if former_coords:
        for i, (former, coord_data) in enumerate(former_coords.items()):
            if coord_data:  # Check if coordination data exists
                keys = np.array(list(coord_data.keys()))
                values = np.array(list(coord_data.values()))
                total = values.sum()
                if total > 0:
                    ax[0, 1].bar(
                        keys + offset * i,
                        values * 100 / total,
                        width=0.2,
                        align="center",
                        linewidth=0.5,
                        edgecolor="black",
                        facecolor=colors[i],
                        label=f"CN_{former}",
                    )

        ax[0, 1].set_xlabel("T_n")
        ax[0, 1].set_ylabel("Percentage")
        ax[0, 1].set_ylim(0, 100)
        ax[0, 1].legend(prop={"size": 10})
    else:
        ax[0, 1].text(0.5, 0.5, "No former coordination data", ha="center", va="center", transform=ax[0, 1].transAxes)
        ax[0, 1].set_xlabel("T_n")
        ax[0, 1].set_ylabel("Percentage")

    # Plot 3: Modifier coordination distributions
    if modifier_coords:
        for i, (modifier, coord_data) in enumerate(modifier_coords.items()):
            if coord_data:  # Check if coordination data exists
                keys = np.array(list(coord_data.keys()))
                values = np.array(list(coord_data.values()))
                total = values.sum()
                if total > 0:
                    ax[1, 0].bar(
                        keys + offset * i,
                        values * 100 / total,
                        width=0.2,
                        align="center",
                        linewidth=0.5,
                        edgecolor="black",
                        facecolor=colors[i],
                        label=f"CN_{modifier}",
                    )

        ax[1, 0].set_xlabel("Modifier_n")
        ax[1, 0].set_ylabel("Percentage")
        ax[1, 0].set_ylim(0, 100)
        ax[1, 0].legend(prop={"size": 10})
    else:
        ax[1, 0].text(0.5, 0.5, "No modifier coordination data", ha="center", va="center", transform=ax[1, 0].transAxes)
        ax[1, 0].set_xlabel("Modifier_n")
        ax[1, 0].set_ylabel("Percentage")

    # Plot 4: Qn distributions
    if Qn_dist and Qn_dist_partial:
        # Plot total Qn distribution
        qn_keys = np.array(list(Qn_dist.keys()))
        qn_values = np.array(list(Qn_dist.values()))
        total_qn = qn_values.sum()
        if total_qn > 0:
            ax[1, 1].bar(
                qn_keys + offset * 0,
                qn_values * 100 / total_qn,
                width=0.2,
                align="center",
                linewidth=0.5,
                edgecolor="black",
                facecolor="C0",
                label="Total",
            )

        # Plot partial Qn distributions for each former
        for i, former in enumerate(network_formers):
            former_z = next(k for k, v in type_map.items() if v == former)
            if former_z in Qn_dist_partial:
                qn_partial = Qn_dist_partial[former_z]
                if qn_partial:
                    qn_p_keys = np.array(list(qn_partial.keys()))
                    qn_p_values = np.array(list(qn_partial.values()))
                    total_qn_p = qn_p_values.sum()
                    if total_qn_p > 0:
                        ax[1, 1].bar(
                            qn_p_keys + offset * (i + 1),
                            qn_p_values * 100 / total_qn_p,
                            width=0.2,
                            align="center",
                            linewidth=0.5,
                            edgecolor="black",
                            facecolor=colors[i],
                            label=f"Qn_{former}",
                        )

        ax[1, 1].set_xlabel("Qn")
        ax[1, 1].set_ylabel("Percentage")
        ax[1, 1].set_ylim(0, 100)
        ax[1, 1].legend(prop={"size": 10})
    else:
        ax[1, 1].text(0.5, 0.5, "No Qn distribution data", ha="center", va="center", transform=ax[1, 1].transAxes)
        ax[1, 1].set_xlabel("Qn")
        ax[1, 1].set_ylabel("Percentage")

    # Plot 5: Bond angles
    if bond_angle_distributions:
        for i, (former, (angles, counts)) in enumerate(bond_angle_distributions.items()):
            ax[0, 2].plot(angles, counts, label=f"O-{former}-O angles", color=colors[i])
        ax[0, 2].set_xlabel("Angle (degrees)")
        ax[0, 2].set_ylabel("Frequency")
        ax[0, 2].legend()
    else:
        ax[0, 2].text(0.5, 0.5, "No bond angle data", ha="center", va="center", transform=ax[0, 2].transAxes)
        ax[0, 2].set_xlabel("Angle (degrees)")
        ax[0, 2].set_ylabel("Frequency")

    # Plot 6: Ring statistics
    if ring_statistics and "distribution" in ring_statistics and ring_statistics["distribution"]:
        rings_dist = ring_statistics["distribution"]
        sizes = np.array(sorted(rings_dist.keys()))
        counts = np.array([rings_dist[s] for s in sizes])

        ax[1, 2].bar(
            sizes / 2, counts / counts.sum(), width=0.4, align="center", linewidth=1, edgecolor="black", facecolor="C1"
        )
        ax[1, 2].set_xlim(2, (sizes / 2).max() + 1)
        ax[1, 2].set_xticks(np.arange(2, (sizes / 2).max() + 1, 1))
        ax[1, 2].set_ylim(0, 0.5)
        ax[1, 2].set_xlabel(r"Former atoms in ring ($\#$)")
        ax[1, 2].set_ylabel(r"Normalized count")
    else:
        ax[1, 2].text(0.5, 0.5, "No ring data", ha="center", va="center", transform=ax[1, 2].transAxes)
        ax[1, 2].set_xlabel(r"Former atoms in ring ($\#$)")
        ax[1, 2].set_ylabel(r"Normalized count")

    # Plot 7-9: RDF plots (spanning the bottom row)
    if rdfs:
        # Plot O-O RDF
        o_atomic_number = 8
        oo_pair = (8, 8) if o_atomic_number in list(type_map.keys()) else None
        if oo_pair and oo_pair in rdfs:
            ax[2, 0].plot(r[::4], rdfs[oo_pair][::4], "-", color="C0", label=r"$g_{O-O}(r)$")
            ax[2, 0].plot(r, cumcn[oo_pair], "--", color="C0", label=r"$CN_{O-O}(r)$")
            ax[2, 0].set_title("O-O RDF")

        # Plot former-oxygen RDFs
        for i, former in enumerate(network_formers):
            z = next(k for k, v in type_map.items() if v == former)
            pair = (z, O_type[0]) if O_type else None
            if pair and pair in rdfs:
                ax[2, 1].plot(r[::4], rdfs[pair][::4], "-", color=f"C{i + 1}", label=rf"$g_{{{former}-O}}(r)$")
                ax[2, 1].plot(r, cumcn[pair], "--", color=f"C{i + 1}", label=rf"$CN_{{{former}-O}}(r)$")
                ax[2, 1].set_title("Former-O RDFs")

        # Plot modifier-oxygen RDFs
        for i, mod in enumerate(modifiers):
            z = next(k for k, v in type_map.items() if v == mod)
            pair = (z, O_type[0]) if O_type else None
            if pair and pair in rdfs:
                ax[2, 2].plot(
                    r[::4], rdfs[pair][::4], "-", color=f"C{i + len(network_formers) + 1}", label=rf"$g_{{{mod}-O}}(r)$"
                )
                ax[2, 2].plot(
                    r, cumcn[pair], "--", color=f"C{i + len(network_formers) + 1}", label=rf"$CN_{{{mod}-O}}(r)$"
                )
                ax[2, 2].set_title("Modifier-O RDFs")

        # Set common properties for RDF plots
        for i in range(3):
            ax[2, i].set_xlim(0, 10)
            ax[2, i].set_ylim(0, 25)
            ax[2, i].set_xlabel(r"$r (\AA)$")
            ax[2, i].set_ylabel(r"$g(r), CN(r)$")
            ax[2, i].legend(prop={"size": 8})
    else:
        for i in range(3):
            ax[2, i].text(0.5, 0.5, "No RDF data", ha="center", va="center", transform=ax[2, i].transAxes)
            ax[2, i].set_xlabel(r"$r (\AA)$")
            ax[2, i].set_ylabel(r"$g(r), CN(r)$")
            ax[2, i].set_title(["O-O RDF", "Former-O RDFs", "Modifier-O RDFs"][i])

    plt.tight_layout()
    return fig
