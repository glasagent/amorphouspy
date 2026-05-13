"""Tools to get results from all available analysis functions in this code.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, cast

import numpy as np
import plotly.graph_objects as go
from ase.data import chemical_symbols
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

if TYPE_CHECKING:
    from ase.atoms import Atoms

from amorphouspy.analysis.bond_angle_distribution import compute_angles
from amorphouspy.analysis.qn_network_connectivity import compute_network_connectivity, compute_qn_and_classify
from amorphouspy.analysis.radial_distribution_functions import compute_coordination, compute_rdf
from amorphouspy.analysis.rings import compute_guttmann_rings, generate_bond_length_dict
from amorphouspy.analysis.structure_factor import compute_structure_factor

# Network Formers: Elements that can form a glass network on their own.
GLASS_FORMERS = {"Si", "B", "P", "Ge", "As", "Sb", "Te", "V"}

# Network Intermediates: Elements that strengthen or modify the network
# but cannot form glass alone.
GLASS_INTERMEDIATES = {"Al", "Ti", "Zr", "Be", "Zn", "Pb", "Bi", "Nb", "Ta", "W", "Mo", "Ga", "In", "Sn", "Fe", "Cr"}

# Network Modifiers: Elements that break the network to lower melting
# points and viscosity.
GLASS_MODIFIERS = {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "La", "Y", "Cd", "Tl"}


class CoordinationData(BaseModel):
    """Coordination number distributions for different element types."""

    oxygen: dict[str, float] = Field(default_factory=dict, description="Oxygen coordination number distribution")
    formers: dict[str, dict[str, float]] = Field(
        default_factory=dict, description="Network former coordination distributions"
    )
    modifiers: dict[str, dict[str, float]] = Field(
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
    rings: dict[str, dict[str, float] | float] = Field(
        default_factory=dict,
        description="Ring statistics with 'distribution' (dict[str, float]) and 'mean_size' (float)",
    )


class StructureFactorData(BaseModel):
    """Structure factor S(q) data for neutron and X-ray weighting."""

    q: list[float] = Field(..., description="Momentum transfer array (Å⁻¹)")
    sq_neutron: list[float] = Field(..., description="Neutron-weighted total structure factor S(q)")
    sq_xray: list[float] = Field(..., description="X-ray-weighted total structure factor S(q)")
    sq_partials: dict[str, list[float]] = Field(
        default_factory=dict, description="Neutron partial structure factors S_ab(q) keyed by 'El1-El2'"
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
    oxygen_classes: dict[str, int] = Field(
        default_factory=dict,
        description="Per-class oxygen counts: BO, NBO, free, tri",
    )
    oxygen_ids: dict[str, list[int]] = Field(
        default_factory=dict,
        description="Per-class lists of oxygen atom IDs (1-based)",
    )


class StructureData(BaseModel):
    """Structured results of structural analysis with all computed properties organized into logical groups."""

    density: float = Field(..., description="Calculated density of the structure (g/cm³)")
    coordination: CoordinationData = Field(
        default_factory=CoordinationData, description="Coordination number distributions"
    )
    network: NetworkData = Field(..., description="Network connectivity data")
    distributions: StructuralDistributions = Field(
        default_factory=lambda: StructuralDistributions(bond_angles={}, rings={}),
        description="Structural distributions",
    )
    rdfs: RadialDistributionData = Field(..., description="Radial distribution function data")
    structure_factor: StructureFactorData | None = Field(default=None, description="Structure factor S(q) data")
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
    rdf_smooth2 = np.asarray(
        savgol_filter(rdf_smooth1, window_length=window_length, polyorder=polyorder),
        dtype=np.float64,
    )

    first_peak_idx = int(np.argmax(rdf_smooth2))
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

    network_formers: set[str] = set()
    modifiers: set[str] = set()
    oxygen_present = False

    for z in unique_z:
        symbol = type_map[z]
        if symbol == "O":
            oxygen_present = True
        elif symbol in GLASS_FORMERS or symbol in GLASS_INTERMEDIATES:
            network_formers.add(symbol)
        elif symbol in GLASS_MODIFIERS:
            modifiers.add(symbol)
        else:
            warnings.warn(f"Unknown element {symbol!r}, treating as modifier", stacklevel=2)
            modifiers.add(symbol)

    return type_map, network_formers, modifiers, oxygen_present


def _sem(values: list[float]) -> float:
    """Standard error of the mean for a list of floats.

    Args:
        values: List of scalar measurements.

    Returns:
        SEM value. Returns the population std / sqrt(n) for single-element lists.

    """
    arr = np.array(values, dtype=float)
    ddof = 1 if len(arr) > 1 else 0
    return float(arr.std(ddof=ddof) / np.sqrt(len(arr)))


def _merge_dict_mean_sem(
    per_frame_dicts: list[dict],
) -> tuple[dict, dict]:
    """Average a flat ``dict[key, float]`` across frames and compute SEM per key.

    Missing keys in a given frame are treated as 0.0.

    Args:
        per_frame_dicts: One dict per frame, each mapping string keys to float values.

    Returns:
        Tuple of ``(mean_dict, sem_dict)`` with the union of all keys.

    """
    all_keys = sorted({k for d in per_frame_dicts for k in d})
    mean_d = {k: float(np.mean([d.get(k, 0.0) for d in per_frame_dicts])) for k in all_keys}
    sem_d = {k: _sem([d.get(k, 0.0) for d in per_frame_dicts]) for k in all_keys}
    return mean_d, sem_d


def _merge_nested_dict_mean_sem(
    per_frame_dicts: list[dict],
) -> tuple[dict, dict]:
    """Average a nested ``dict[outer, dict[inner, float]]`` across frames and compute SEM.

    Args:
        per_frame_dicts: One nested dict per frame.

    Returns:
        Tuple of ``(mean_dict, sem_dict)`` preserving the two-level structure.

    """
    outer_keys = sorted({k for d in per_frame_dicts for k in d})
    mean_d: dict = {}
    sem_d: dict = {}
    for ok in outer_keys:
        inner_dicts = [d.get(ok, {}) for d in per_frame_dicts]
        m, s = _merge_dict_mean_sem(inner_dicts)
        mean_d[ok] = m
        sem_d[ok] = s
    return mean_d, sem_d


def _avg_list_arrays(per_frame: list[list[float]]) -> tuple[list[float], list[float]]:
    """Element-wise average a list of equal-length arrays across frames and compute SEM.

    Args:
        per_frame: One array (as a list of floats) per frame; all must have the same length.

    Returns:
        Tuple of ``(mean, sem)`` as plain Python lists of the same length.

    """
    arr = np.array(per_frame, dtype=float)
    mean = arr.mean(axis=0).tolist()
    ddof = 1 if len(per_frame) > 1 else 0
    sem = (arr.std(axis=0, ddof=ddof) / np.sqrt(len(per_frame))).tolist()
    return mean, sem


def _build_bond_angles_mean_sem(
    frames: list[StructureData],
) -> tuple[dict[str, tuple[list[float], list[float]]], dict[str, tuple[list[float], list[float]]]]:
    """Average bond-angle distributions across frames and compute SEM.

    Args:
        frames: Per-frame StructureData results.

    Returns:
        Tuple of ``(ba_mean, ba_sem)`` each mapping former symbol to
        ``(bins, histogram)`` lists.

    """
    all_formers = sorted({k for f in frames for k in f.distributions.bond_angles})
    ba_mean: dict[str, tuple[list[float], list[float]]] = {}
    ba_sem: dict[str, tuple[list[float], list[float]]] = {}
    for former in all_formers:
        hists = [f.distributions.bond_angles[former][1] for f in frames if former in f.distributions.bond_angles]
        bins = frames[0].distributions.bond_angles[former][0]
        hist_mean, hist_sem_arr = _avg_list_arrays(hists)
        ba_mean[former] = (bins, hist_mean)
        ba_sem[former] = (bins, hist_sem_arr)
    return ba_mean, ba_sem


def _build_rings_mean_sem(frames: list[StructureData]) -> tuple[dict, dict]:
    """Average ring statistics across frames and compute SEM.

    Args:
        frames: Per-frame StructureData results.

    Returns:
        Tuple of ``(mean_rings, sem_rings)`` each with keys ``"distribution"``
        (``dict[str, float]``) and ``"mean_size"`` (``float``).

    """
    rings_dist_per_frame = []
    mean_sizes = []
    for frame_result in frames:
        dist = frame_result.distributions.rings.get("distribution", {})
        rings_dist_per_frame.append({k: float(v) for k, v in dist.items()} if isinstance(dist, dict) else {})
        mean_size_value = frame_result.distributions.rings.get("mean_size", 0.0)
        mean_sizes.append(float(mean_size_value) if isinstance(mean_size_value, (int, float)) else 0.0)
    rings_dist_mean, rings_dist_sem = _merge_dict_mean_sem(rings_dist_per_frame)
    return (
        {"distribution": rings_dist_mean, "mean_size": float(np.mean(mean_sizes))},
        {"distribution": rings_dist_sem, "mean_size": _sem(mean_sizes)},
    )


def _build_rdfs_mean_sem(frames: list[StructureData]) -> tuple[list[float], dict, dict, dict, dict]:
    """Average RDF and cumulative coordination data across frames and compute SEM.

    Args:
        frames: Per-frame StructureData results.

    Returns:
        Tuple of ``(r, rdfs_mean, rdfs_sem, cumcn_mean, cumcn_sem)`` where ``r`` is the
        shared radial grid and the remaining dicts map ``"El1-El2"`` keys to value lists.

    """
    r = frames[0].rdfs.r
    all_rdf_keys = sorted({k for f in frames for k in f.rdfs.rdfs})
    rdfs_m: dict[str, list[float]] = {}
    rdfs_s: dict[str, list[float]] = {}
    cumcn_m: dict[str, list[float]] = {}
    cumcn_s: dict[str, list[float]] = {}
    for key in all_rdf_keys:
        rdf_vals = [f.rdfs.rdfs[key] for f in frames if key in f.rdfs.rdfs]
        m, s = _avg_list_arrays(rdf_vals)
        rdfs_m[key] = m
        rdfs_s[key] = s
        cn_vals = [f.rdfs.cumulative_coordination[key] for f in frames if key in f.rdfs.cumulative_coordination]
        if cn_vals:
            cm, cs = _avg_list_arrays(cn_vals)
            cumcn_m[key] = cm
            cumcn_s[key] = cs
    return r, rdfs_m, rdfs_s, cumcn_m, cumcn_s


def _build_sf_mean_sem(
    frames: list[StructureData],
) -> tuple[StructureFactorData | None, StructureFactorData | None]:
    """Average structure factor data across frames and compute SEM.

    Args:
        frames: Per-frame StructureData results.

    Returns:
        Tuple of ``(sf_mean, sf_sem)`` as StructureFactorData instances, or
        ``(None, None)`` if no structure factor was computed.

    """
    if frames[0].structure_factor is None:
        return None, None
    q = frames[0].structure_factor.q
    sq_n_vals = [f.structure_factor.sq_neutron for f in frames if f.structure_factor]
    sq_x_vals = [f.structure_factor.sq_xray for f in frames if f.structure_factor]
    sq_n_m, sq_n_s = _avg_list_arrays(sq_n_vals)
    sq_x_m, sq_x_s = _avg_list_arrays(sq_x_vals)
    all_partial_keys = sorted({k for f in frames if f.structure_factor for k in f.structure_factor.sq_partials})
    partials_m: dict[str, list[float]] = {}
    partials_s: dict[str, list[float]] = {}
    for pk in all_partial_keys:
        pv = [
            f.structure_factor.sq_partials[pk]
            for f in frames
            if f.structure_factor and pk in f.structure_factor.sq_partials
        ]
        pm, ps = _avg_list_arrays(pv)
        partials_m[pk] = pm
        partials_s[pk] = ps
    return (
        StructureFactorData(q=q, sq_neutron=sq_n_m, sq_xray=sq_x_m, sq_partials=partials_m),
        StructureFactorData(q=q, sq_neutron=sq_n_s, sq_xray=sq_x_s, sq_partials=partials_s),
    )


def _build_sem_structure_data(frames: list[StructureData]) -> tuple[StructureData, StructureData]:
    """Aggregate per-frame StructureData results into mean and SEM StructureData objects.

    Args:
        frames: Non-empty list of per-frame StructureData results.

    Returns:
        Tuple of ``(mean_data, sem_data)`` as StructureData instances. The ``elements``
        field is copied from the first frame without averaging.

    """
    densities = [f.density for f in frames]
    density_mean = float(np.mean(densities))
    density_sem = _sem(densities)

    o_coord_mean, o_coord_sem = _merge_dict_mean_sem([f.coordination.oxygen for f in frames])
    formers_mean, formers_sem = _merge_nested_dict_mean_sem([f.coordination.formers for f in frames])
    modifiers_mean, modifiers_sem = _merge_nested_dict_mean_sem([f.coordination.modifiers for f in frames])

    qn_mean, qn_sem = _merge_dict_mean_sem([f.network.Qn_distribution for f in frames])
    qn_partial_mean, qn_partial_sem = _merge_nested_dict_mean_sem([f.network.Qn_distribution_partial for f in frames])
    conn_values = [f.network.connectivity for f in frames]
    conn_mean = float(np.mean(conn_values))
    conn_sem = _sem(conn_values)

    ba_mean, ba_sem = _build_bond_angles_mean_sem(frames)
    rings_mean, rings_sem = _build_rings_mean_sem(frames)
    r, rdfs_m, rdfs_s, cumcn_m, cumcn_s = _build_rdfs_mean_sem(frames)
    sf_mean_data, sf_sem_data = _build_sf_mean_sem(frames)

    elements_ref = frames[0].elements

    mean_data = StructureData(
        density=density_mean,
        coordination=CoordinationData(oxygen=o_coord_mean, formers=formers_mean, modifiers=modifiers_mean),
        network=NetworkData(Qn_distribution=qn_mean, Qn_distribution_partial=qn_partial_mean, connectivity=conn_mean),
        distributions=StructuralDistributions(bond_angles=ba_mean, rings=rings_mean),
        rdfs=RadialDistributionData(r=r, rdfs=rdfs_m, cumulative_coordination=cumcn_m),
        structure_factor=sf_mean_data,
        elements=elements_ref,
    )
    sem_data = StructureData(
        density=density_sem,
        coordination=CoordinationData(oxygen=o_coord_sem, formers=formers_sem, modifiers=modifiers_sem),
        network=NetworkData(Qn_distribution=qn_sem, Qn_distribution_partial=qn_partial_sem, connectivity=conn_sem),
        distributions=StructuralDistributions(bond_angles=ba_sem, rings=rings_sem),
        rdfs=RadialDistributionData(r=r, rdfs=rdfs_s, cumulative_coordination=cumcn_s),
        structure_factor=sf_sem_data,
        elements=elements_ref,
    )
    return mean_data, sem_data


_ELEMENT_CUTOFF_FALLBACKS: dict[str, float] = {
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


def _compute_density(atoms: Atoms) -> float:
    """Compute mass density of an ASE Atoms object in g/cm³.

    Args:
        atoms: ASE Atoms object with cell and masses set.

    Returns:
        Density in g/cm³.

    """
    volume_cm3 = atoms.get_volume() * 1e-24
    avogadro_number = 6.022140857e23
    total_mass_g = atoms.get_masses().sum() / avogadro_number
    return total_mass_g / volume_cm3


def _build_cutoff_map(
    unique_z: np.ndarray,
    type_map: dict[int, str],
    former_types: list[int],
    o_type: list[int],
    r: np.ndarray,
    rdfs: dict,
) -> dict[str, float]:
    """Build a coordination cutoff map by locating the first RDF minimum for each element.

    For each element, the cutoff is taken as the first minimum of the element-oxygen RDF.
    Falls back to ``_ELEMENT_CUTOFF_FALLBACKS`` when no minimum is detected.

    Args:
        unique_z: Unique atomic numbers present in the structure.
        type_map: Mapping from atomic number to element symbol.
        former_types: Atomic numbers of network-forming elements.
        o_type: Atomic numbers of oxygen atoms (empty list if no oxygen).
        r: Radial distance array used during RDF computation (Å).
        rdfs: Raw RDF dict keyed by ``(z1, z2)`` atomic-number tuples.

    Returns:
        Dict mapping element symbol to cutoff distance in Å.

    """

    def _canonical(a: int, b: int) -> tuple[int, int]:
        return (min(a, b), max(a, b))

    cutoff_map: dict[str, float] = {}
    for z in unique_z:
        symbol = type_map[z]
        if symbol == "O":
            if former_types:
                pair = _canonical(o_type[0], former_types[0])
                if pair in rdfs:
                    r_min = find_rdf_minimum(r, rdfs[pair])
                    cutoff_map[symbol] = r_min if r_min is not None else 2.0
        else:
            pair = _canonical(z, o_type[0]) if o_type else None
            if pair and pair in rdfs:
                r_min = find_rdf_minimum(r, rdfs[pair])
                cutoff_map[symbol] = r_min if r_min is not None else _ELEMENT_CUTOFF_FALLBACKS.get(symbol, 3.0)
    return cutoff_map


def _compute_coordination_data(
    atoms: Atoms,
    type_map: dict[int, str],
    network_formers: set[str],
    modifiers: set[str],
    former_types: list[int],
    modifier_types: list[int],
    o_type: list[int],
    cutoff_map: dict[str, float],
) -> tuple[dict, dict, dict]:
    """Compute coordination number distributions for oxygen, formers, and modifiers.

    Args:
        atoms: ASE Atoms object to analyse.
        type_map: Mapping from atomic number to element symbol.
        network_formers: Set of network-former element symbols.
        modifiers: Set of modifier element symbols.
        former_types: Atomic numbers of network formers.
        modifier_types: Atomic numbers of modifiers.
        o_type: Atomic numbers of oxygen (empty list if no oxygen).
        cutoff_map: Element symbol → coordination cutoff (Å).

    Returns:
        Tuple of ``(o_coord, former_coords, modifier_coords)`` where each is a
        ``dict[str, int]`` or ``dict[str, dict[str, int]]`` coordination distribution.

    """
    o_coord: dict[str, float] = {}
    if o_type:
        o_cutoff = cutoff_map.get("O", 2.0)
        o_coord_raw = compute_coordination(atoms, o_type[0], o_cutoff, former_types + modifier_types)
        o_coord = {str(k): v for k, v in o_coord_raw.items()}

    former_coords: dict[str, dict[str, float]] = {}
    for former in network_formers:
        z = next(k for k, v in type_map.items() if v == former)
        if o_type:
            coord_raw = compute_coordination(atoms, z, cutoff_map[former], o_type)
            former_coords[former] = {str(k): v for k, v in coord_raw.items()}

    modifier_coords: dict[str, dict[str, float]] = {}
    for mod in modifiers:
        z = next(k for k, v in type_map.items() if v == mod)
        if o_type:
            coord_raw = compute_coordination(atoms, z, cutoff_map[mod], o_type)
            modifier_coords[mod] = {str(k): v for k, v in coord_raw.items()}

    return o_coord, former_coords, modifier_coords


def _compute_network_data(
    atoms: Atoms,
    network_formers: set[str],
    former_types: list[int],
    o_type: list[int],
    cutoff_map: dict[str, float],
) -> tuple[dict, dict, float, dict[str, int], dict[str, list[int]]]:
    """Compute Q^n distribution, network connectivity, and oxygen classification.

    Args:
        atoms: ASE Atoms object to analyse.
        network_formers: Set of network-former element symbols.
        former_types: Atomic numbers of network formers.
        o_type: Atomic numbers of oxygen (empty list if no oxygen).
        cutoff_map: Element symbol → coordination cutoff (Å).

    Returns:
        Tuple of ``(qn_dist, qn_dist_partial, network_connectivity,
        oxygen_class_counts, oxygen_class_ids)`` where:
            - ``qn_dist``: Overall Q^n distribution ``{str(n): fraction}``.
            - ``qn_dist_partial``: Per-former Q^n distribution.
            - ``network_connectivity``: Scalar network connectivity value.
            - ``oxygen_class_counts``: Count per oxygen class (BO, NBO, free, tri).
            - ``oxygen_class_ids``: 1-based atom IDs per oxygen class.

    """
    qn_dist: dict = {}
    qn_dist_partial: dict = {}
    network_connectivity = 0.0
    oxygen_class_counts: dict[str, int] = {}
    oxygen_class_ids: dict[str, list[int]] = {}
    if network_formers and o_type:
        qn_dist_raw, qn_dist_partial_raw, o_classes = compute_qn_and_classify(
            atoms, cutoff_map["O"], former_types, o_type[0]
        )
        for aid, cls in o_classes.items():
            oxygen_class_counts[cls] = oxygen_class_counts.get(cls, 0) + 1
            oxygen_class_ids.setdefault(cls, []).append(aid)
        qn_dist = {str(k): v for k, v in qn_dist_raw.items()} if qn_dist_raw else {}
        if qn_dist_partial_raw:
            for element_key, qn_dict in qn_dist_partial_raw.items():
                qn_dist_partial[str(element_key)] = {str(k): v for k, v in qn_dict.items()}
        if qn_dist_raw and sum(qn_dist_raw.values()) > 0:
            network_connectivity = compute_network_connectivity(qn_dist_raw)
    return qn_dist, qn_dist_partial, network_connectivity, oxygen_class_counts, oxygen_class_ids


def _compute_distributions(
    atoms: Atoms,
    type_map: dict[int, str],
    network_formers: set[str],
    o_type: list[int],
    cutoff_map: dict[str, float],
) -> tuple[dict, dict]:
    """Compute bond-angle distributions and ring statistics.

    Args:
        atoms: ASE Atoms object to analyse.
        type_map: Mapping from atomic number to element symbol.
        network_formers: Set of network-former element symbols.
        o_type: Atomic numbers of oxygen (empty list if no oxygen).
        cutoff_map: Element symbol → coordination cutoff (Å).

    Returns:
        Tuple of ``(bond_angle_distributions, ring_statistics_data)`` where:
            - ``bond_angle_distributions``: ``{former: (bins, histogram)}`` per former.
            - ``ring_statistics_data``: ``{"distribution": {...}, "mean_size": float}``.

    """
    bond_angle_distributions: dict = {}
    for former in network_formers:
        z = next(k for k, v in type_map.items() if v == former)
        if o_type:
            _bins, _hist = compute_angles(
                atoms, center_type=z, neighbor_type=o_type[0], cutoff=cutoff_map[former], bins=180
            )
            bond_angle_distributions[former] = (_bins, _hist)

    ring_statistics_data: dict = {}
    if network_formers and o_type:
        specific_cutoffs = {(former, "O"): cutoff_map[former] for former in network_formers}
        bond_lengths = generate_bond_length_dict(atoms, specific_cutoffs=specific_cutoffs)
        rings_dist, mean_ring_size = compute_guttmann_rings(atoms, bond_lengths=bond_lengths, max_size=40, n_cpus=1)
        rings_dist_str = {str(k): v for k, v in rings_dist.items()}
        ring_statistics_data = {"distribution": rings_dist_str, "mean_size": mean_ring_size}

    return bond_angle_distributions, ring_statistics_data


def _serialize_rdfs(
    type_map: dict[int, str], r: np.ndarray, rdfs: dict, cumcn: dict
) -> tuple[list[float], dict[str, list[float]], dict[str, list[float]]]:
    """Convert raw RDF dicts (keyed by atomic-number pairs) to JSON-serialisable form.

    Args:
        type_map: Mapping from atomic number to element symbol.
        r: Radial distance array (Å).
        rdfs: RDF data keyed by ``(z1, z2)`` pairs.
        cumcn: Cumulative coordination number data keyed by ``(z1, z2)`` pairs.

    Returns:
        Tuple of ``(r_list, rdfs_serialisable, cumcn_serialisable)`` where keys are
        ``"El1-El2"`` strings and values are plain Python lists.

    """

    def to_list(data: np.ndarray | list[float]) -> list[float]:
        return data.tolist() if isinstance(data, np.ndarray) else list(data)

    rdfs_serializable: dict[str, list[float]] = {}
    cumcn_serializable: dict[str, list[float]] = {}
    for pair, rdf_data in rdfs.items():
        elem1, elem2 = type_map[pair[0]], type_map[pair[1]]
        key = f"{elem1}-{elem2}"
        rdfs_serializable[key] = to_list(rdf_data)
        if pair in cumcn:
            cumcn_serializable[key] = to_list(cumcn[pair])
    return to_list(r), rdfs_serializable, cumcn_serializable


def _compute_structure_factor_data(atoms: Atoms, type_map: dict[int, str]) -> StructureFactorData:
    """Compute neutron- and X-ray-weighted structure factors and pack them into a StructureFactorData.

    Args:
        atoms: ASE Atoms object to analyse.
        type_map: Mapping from atomic number to element symbol.

    Returns:
        StructureFactorData with q array, total S(q) for both radiations, and neutron partials.

    """

    def to_list(data: np.ndarray | list[float]) -> list[float]:
        return data.tolist() if isinstance(data, np.ndarray) else list(data)

    q_sf, sq_neutron, sq_partials_neutron = compute_structure_factor(atoms, radiation="neutron")
    _, sq_xray, _ = compute_structure_factor(atoms, radiation="xray")
    sq_partials_serializable = {
        f"{type_map[pair[0]]}-{type_map[pair[1]]}": to_list(sq_data) for pair, sq_data in sq_partials_neutron.items()
    }
    return StructureFactorData(
        q=to_list(q_sf),
        sq_neutron=to_list(sq_neutron),
        sq_xray=to_list(sq_xray),
        sq_partials=sq_partials_serializable,
    )


def analyze_structure(
    atoms: Atoms | list[Atoms],
    *,
    frame_averaging: bool = False,
) -> tuple[StructureData, StructureData]:
    """Perform a comprehensive structural analysis of an atomic configuration.

    Args:
        atoms: Atomic configuration to analyze, or a list of frames when frame_averaging=True.
        frame_averaging: If True, average results over all frames in atoms (list[Atoms]).
            Returns a tuple (mean_data, sem_data) where both are StructureData instances.

    Returns:
        StructureData: Object containing structured analysis results.
        When frame_averaging=True, returns tuple[StructureData, StructureData] — (mean, sem).

    Example:
        >>> structure_data = analyze_structure(my_atoms)
        >>> print(structure_data.density)

    """
    if frame_averaging:
        if isinstance(atoms, list) and len(atoms) == 0:
            msg = "frame_averaging=True requires a non-empty list[Atoms]"
            raise ValueError(msg)
        frames = cast("list[Atoms]", atoms if isinstance(atoms, list) else [atoms])
        per_frame_results = [analyze_structure(frame_atoms)[0] for frame_atoms in frames]
        return _build_sem_structure_data(per_frame_results)
    if isinstance(atoms, list):
        atoms = cast("Atoms", atoms[0])

    unique_z = np.unique(atoms.get_atomic_numbers())
    density = _compute_density(atoms)
    type_map, network_formers, modifiers, _oxygen_present = _classify_elements(unique_z)
    former_types = [z for z, sym in type_map.items() if sym in network_formers]
    modifier_types = [z for z, sym in type_map.items() if sym in modifiers]
    o_type = [z for z, sym in type_map.items() if sym == "O"]

    r, rdfs, cumcn = compute_rdf(atoms)
    cutoff_map = _build_cutoff_map(unique_z, type_map, former_types, o_type, r, rdfs)

    o_coord, former_coords, modifier_coords = _compute_coordination_data(
        atoms, type_map, network_formers, modifiers, former_types, modifier_types, o_type, cutoff_map
    )
    qn_dist, qn_dist_partial, network_connectivity, oxygen_class_counts, oxygen_class_ids = _compute_network_data(
        atoms, network_formers, former_types, o_type, cutoff_map
    )
    bond_angle_distributions, ring_statistics_data = _compute_distributions(
        atoms, type_map, network_formers, o_type, cutoff_map
    )

    def to_list(data: np.ndarray | list[float]) -> list[float]:
        return data.tolist() if isinstance(data, np.ndarray) else list(data)

    bond_angle_distributions_serializable = {
        former: (to_list(angles), to_list(counts)) for former, (angles, counts) in bond_angle_distributions.items()
    }
    r_list, rdfs_serializable, cumcn_serializable = _serialize_rdfs(type_map, r, rdfs, cumcn)
    structure_factor_data = _compute_structure_factor_data(atoms, type_map)

    mean_data = StructureData(
        density=density,
        coordination=CoordinationData(oxygen=o_coord, formers=former_coords, modifiers=modifier_coords),
        network=NetworkData(
            Qn_distribution={k: float(v) for k, v in qn_dist.items()},
            Qn_distribution_partial={
                outer: {inner: float(v) for inner, v in inner_d.items()} for outer, inner_d in qn_dist_partial.items()
            },
            connectivity=network_connectivity,
        ),
        distributions=StructuralDistributions(
            bond_angles=bond_angle_distributions_serializable, rings=ring_statistics_data
        ),
        rdfs=RadialDistributionData(r=r_list, rdfs=rdfs_serializable, cumulative_coordination=cumcn_serializable),
        structure_factor=structure_factor_data,
        elements=ElementInfo(
            formers=list(network_formers),
            modifiers=list(modifiers),
            cutoffs=cutoff_map,
            oxygen_classes=oxygen_class_counts,
            oxygen_ids=oxygen_class_ids,
        ),
    )
    _, sem_data = _build_sem_structure_data([mean_data])
    return mean_data, sem_data


def _add_coordination_plots(fig: go.Figure, structure_data: StructureData, colors: list[str]) -> None:
    """Add coordination distribution bar plots for oxygen and network formers.

    Populates subplots (row=1, col=1) for oxygen coordination and (row=1, col=2)
    for per-former coordination.

    Args:
        fig: Plotly figure with a 4x3 subplot grid.
        structure_data: Structural analysis results.
        colors: Colour cycle list used to distinguish traces.

    """
    # Plot 1: Oxygen coordination distribution
    if structure_data.coordination.oxygen:
        keys = [int(k) for k in structure_data.coordination.oxygen]
        values = list(structure_data.coordination.oxygen.values())
        total = sum(values)
        percentages = [v * 100 / total for v in values] if total > 0 else values

        rc = structure_data.elements.cutoffs.get("O")
        o_label = f"O coordination (r={rc:.2f} Å)" if rc else "O coordination"
        fig.add_trace(
            go.Bar(
                x=keys,
                y=percentages,
                name=o_label,
                marker_color=colors[0],
                opacity=0.85,
                marker_line_color="white",
                marker_line_width=0.8,
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

                rc = structure_data.elements.cutoffs.get(former)
                label = f"CN_{former} (r={rc:.2f} Å)" if rc else f"CN_{former}"
                fig.add_trace(
                    go.Bar(
                        x=keys,
                        y=percentages,
                        name=label,
                        marker_color=colors[i % len(colors)],
                        opacity=0.85,
                        marker_line_color="white",
                        marker_line_width=0.8,
                        offsetgroup=i,
                        showlegend=True,
                    ),
                    row=1,
                    col=2,
                )


def _add_network_plots(fig: go.Figure, structure_data: StructureData, colors: list[str]) -> None:
    """Add modifier coordination and Q^n distribution bar plots.

    Populates subplots (row=1, col=3) for modifier coordination and (row=2, col=1)
    for the Q^n distribution.

    Args:
        fig: Plotly figure with a 4x3 subplot grid.
        structure_data: Structural analysis results.
        colors: Colour cycle list used to distinguish traces.

    """
    # Plot 3: Modifier coordination distributions
    if structure_data.coordination.modifiers:
        for i, (modifier, coord_data) in enumerate(structure_data.coordination.modifiers.items()):
            if coord_data:
                keys = [int(k) for k in coord_data]
                values = list(coord_data.values())
                total = sum(values)
                percentages = [v * 100 / total for v in values] if total > 0 else values

                rc = structure_data.elements.cutoffs.get(modifier)
                label = f"CN_{modifier} (r={rc:.2f} Å)" if rc else f"CN_{modifier}"
                fig.add_trace(
                    go.Bar(
                        x=keys,
                        y=percentages,
                        name=label,
                        marker_color=colors[i % len(colors)],
                        opacity=0.85,
                        marker_line_color="white",
                        marker_line_width=0.8,
                        offsetgroup=i,
                        showlegend=True,
                    ),
                    row=1,
                    col=3,
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
                marker_color=colors[4 % len(colors)],
                opacity=0.85,
                marker_line_color="white",
                marker_line_width=0.8,
                showlegend=True,
            ),
            row=2,
            col=1,
        )


def _add_ring_plots(fig: go.Figure, structure_data: StructureData, colors: list[str]) -> None:
    """Add bond-angle distribution and ring-size distribution plots.

    Populates subplots (row=2, col=2) for bond angles and (row=2, col=3) for
    ring-size histograms.

    Args:
        fig: Plotly figure with a 4x3 subplot grid.
        structure_data: Structural analysis results.
        colors: Colour cycle list used to distinguish traces.

    """
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
                    row=2,
                    col=2,
                )

    # Plot 6: Ring size distribution
    if (
        structure_data.distributions.rings
        and "distribution" in structure_data.distributions.rings
        and structure_data.distributions.rings["distribution"]
    ):
        ring_dist = structure_data.distributions.rings["distribution"]
        if not isinstance(ring_dist, dict):
            return
        ring_sizes = [int(k) for k in ring_dist]
        ring_counts = list(ring_dist.values())
        total_rings = sum(ring_counts)
        percentages = [count * 100 / total_rings for count in ring_counts] if total_rings > 0 else ring_counts

        fig.add_trace(
            go.Bar(
                x=ring_sizes,
                y=percentages,
                name="Ring sizes",
                marker_color=colors[2 % len(colors)],
                opacity=0.85,
                marker_line_color="white",
                marker_line_width=0.8,
                showlegend=True,
            ),
            row=2,
            col=3,
        )


def _classify_rdf_pairs(
    structure_data: StructureData,
) -> dict[str, list[tuple[str, list[float]]]]:
    """Classify RDF pairs into O-O, former-O, and modifier-O categories.

    Args:
        structure_data: Structural analysis results containing RDF data and element info.

    Returns:
        Dict with keys ``"oo"``, ``"former"``, and ``"modifier"``, each mapping to a list
        of ``(pair_label, rdf_data)`` tuples.

    """
    formers = set(structure_data.elements.formers)
    modifiers = set(structure_data.elements.modifiers)

    classified: dict[str, list[tuple[str, list[float]]]] = {"oo": [], "former": [], "modifier": []}

    for pair, rdf_data in structure_data.rdfs.rdfs.items():
        e1, sep, e2 = pair.partition("-")
        if not sep:
            continue
        if e1 == "O" and e2 == "O":
            classified["oo"].append((pair, rdf_data))
        elif (e1 in formers and e2 == "O") or (e1 == "O" and e2 in formers):
            classified["former"].append((pair, rdf_data))
        elif (e1 in modifiers and e2 == "O") or (e1 == "O" and e2 in modifiers):
            classified["modifier"].append((pair, rdf_data))

    return classified


def _add_rdf_plots(fig: go.Figure, structure_data: StructureData, colors: list[str]) -> None:
    """Add radial distribution function and cumulative coordination number plots.

    Populates subplots (row=3, col=1) for O-O pairs, (row=3, col=2) for former-O pairs,
    and (row=3, col=3) for modifier-O pairs. Each RDF trace is accompanied by a dashed
    cumulative coordination number trace when available.

    Args:
        fig: Plotly figure with a 4x3 subplot grid.
        structure_data: Structural analysis results.
        colors: Colour cycle list used to distinguish traces.

    """
    if not structure_data.rdfs.rdfs:
        return

    classified = _classify_rdf_pairs(structure_data)

    # Map categories to subplot positions: (row, col)
    category_map = [
        ((3, 1), classified["oo"]),
        ((3, 2), classified["former"]),
        ((3, 3), classified["modifier"]),
    ]

    for (row, col), pairs in category_map:
        fig.add_hline(y=1.0, line_dash="dot", line_color="grey", line_width=1, opacity=0.6, row=row, col=col)
        for i, (pair, rdf_data) in enumerate(pairs):
            if rdf_data and structure_data.rdfs.r:
                # RDF trace
                fig.add_trace(
                    go.Scatter(
                        x=structure_data.rdfs.r,
                        y=rdf_data,
                        mode="lines",
                        name=f"g(r) {pair}",
                        line={"color": colors[i % len(colors)], "width": 2},
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
                            line={"color": colors[i % len(colors)], "width": 2, "dash": "dash"},
                            showlegend=True,
                            yaxis="y",
                        ),
                        row=row,
                        col=col,
                    )


def _add_structure_factor_plots(fig: go.Figure, structure_data: StructureData, colors: list[str]) -> None:
    """Add neutron S(q), X-ray S(q), and partial S_ab(q) plots.

    Populates subplots (row=4, col=1) for neutron S(q), (row=4, col=2) for X-ray S(q),
    and (row=4, col=3) for partial structure factors. Does nothing if no structure factor
    was computed.

    Args:
        fig: Plotly figure with a 4x3 subplot grid.
        structure_data: Structural analysis results.
        colors: Colour cycle list used to distinguish partial S(q) traces.

    """
    if structure_data.structure_factor is None:
        return

    sf = structure_data.structure_factor

    # Neutron S(q)
    fig.add_trace(
        go.Scatter(
            x=sf.q,
            y=sf.sq_neutron,
            mode="lines",
            name="S(q) neutron",
            line={"color": "black", "width": 2},
            showlegend=True,
        ),
        row=4,
        col=1,
    )

    # X-ray S(q)
    fig.add_trace(
        go.Scatter(
            x=sf.q,
            y=sf.sq_xray,
            mode="lines",
            name="S(q) X-ray",
            line={"color": "darkred", "width": 2},
            showlegend=True,
        ),
        row=4,
        col=2,
    )

    # Partial S_ab(q)
    for i, (pair, sq_data) in enumerate(sf.sq_partials.items()):
        fig.add_trace(
            go.Scatter(
                x=sf.q,
                y=sq_data,
                mode="lines",
                name=f"S(q) {pair}",
                line={"color": colors[i % len(colors)], "width": 2},
                showlegend=True,
            ),
            row=4,
            col=3,
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
    # Create subplot layout (4x3 grid)
    fig = make_subplots(
        rows=4,
        cols=3,
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
            "Total S(q) — Neutron",
            "Total S(q) — X-ray",
            "Partial S(q)",
        ],
        specs=[
            [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    fig.update_annotations(font_size=12, font_family="Arial, sans-serif")

    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]

    _add_coordination_plots(fig, structure_data, colors)
    _add_network_plots(fig, structure_data, colors)
    _add_ring_plots(fig, structure_data, colors)
    _add_rdf_plots(fig, structure_data, colors)
    _add_structure_factor_plots(fig, structure_data, colors)

    # Position legends inside each subplot (top-right corner)
    axis_to_legend: dict[str, str] = {}
    legend_layout: dict[str, dict] = {}

    for idx in range(1, 13):
        ax_suffix = str(idx) if idx > 1 else ""
        x_ref = f"x{ax_suffix}"
        legend_name = f"legend{idx}" if idx > 1 else "legend"

        x_domain = getattr(fig.layout, f"xaxis{ax_suffix}").domain
        y_domain = getattr(fig.layout, f"yaxis{ax_suffix}").domain

        axis_to_legend[x_ref] = legend_name
        legend_layout[legend_name] = {
            "x": x_domain[1],
            "y": y_domain[1],
            "xanchor": "right",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.7)",
            "font": {"size": 10},
        }

    for trace in fig.data:
        x_ref = trace.xaxis or "x"
        if x_ref in axis_to_legend:
            trace.update(legend=axis_to_legend[x_ref])

    # Update layout and axes
    fig.update_layout(
        height=1300,
        width=1300,
        showlegend=True,
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 12},
        title_text="Structural Analysis",
        title_font_size=16,
        **legend_layout,
    )

    # Row 1: Coordination plots
    fig.update_xaxes(title_text="O_n", row=1, col=1)
    fig.update_yaxes(title_text="Percentage", row=1, col=1)
    fig.update_xaxes(title_text="Former_n", row=1, col=2)
    fig.update_yaxes(title_text="Percentage", row=1, col=2)
    fig.update_xaxes(title_text="Modifier_n", row=1, col=3)
    fig.update_yaxes(title_text="Percentage", row=1, col=3)

    # Row 2: Q^n, Bond angles, Ring statistics
    fig.update_xaxes(title_text="Q^n", row=2, col=1)
    fig.update_yaxes(title_text="Percentage", row=2, col=1)
    fig.update_xaxes(title_text="Angle (degrees)", row=2, col=2)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    fig.update_xaxes(title_text="Former atoms in ring", row=2, col=3)
    fig.update_yaxes(title_text="Normalized count", row=2, col=3)

    # Row 3: RDF plots
    for col in [1, 2, 3]:
        fig.update_xaxes(title_text="r (Å)", range=[0, 10], row=3, col=col)
        fig.update_yaxes(title_text="g(r), CN(r)", range=[0, 25], row=3, col=col)

    # Row 4: Structure factor
    fig.update_xaxes(title_text="q (Å⁻¹)", row=4, col=1)
    fig.update_yaxes(title_text="S(q)", row=4, col=1)
    fig.update_xaxes(title_text="q (Å⁻¹)", row=4, col=2)
    fig.update_yaxes(title_text="S(q)", row=4, col=2)
    fig.update_xaxes(title_text="q (Å⁻¹)", row=4, col=3)
    fig.update_yaxes(title_text="S(q)", row=4, col=3)

    # Global grid and axis line styling (applied last so it doesn't conflict with per-axis overrides)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(180,180,180,0.3)", zeroline=False, showline=True, mirror=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(180,180,180,0.3)", zeroline=False, showline=True, mirror=False)

    return fig
