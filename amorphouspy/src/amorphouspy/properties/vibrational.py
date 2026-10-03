"""Vibrational properties of glass systems: vibrational density of states and related quantities.

Frequencies in this module are ordinary frequencies in THz (not angular frequencies) unless stated otherwise.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.constants import c, e, h

from amorphouspy.properties.structural.all import (
    _build_cutoff_map,
    _classify_elements,
    _former_oxygen_pair_cutoffs,
    _network_former_types,
)
from amorphouspy.properties.structural.qn import CutoffSpec, compute_qn_and_classify, compute_qn_per_atom
from amorphouspy.properties.structural.rdf import compute_rdf

if TYPE_CHECKING:
    from ase import Atoms

FrequencyUnit = Literal["THz", "cm-1", "meV", "rad/ps"]

# Value of 1 THz in each unit.
_THZ_TO_UNIT: dict[str, float] = {
    "THz": 1.0,
    "cm-1": 1e12 / (100 * c),
    "meV": (h * 1e12 / e) * 1e3,
    "rad/ps": 2 * np.pi,
}


def convert_frequency(
    values: float | NDArray, from_unit: FrequencyUnit, to_unit: FrequencyUnit
) -> float | NDArray[np.float64]:
    """Convert frequencies between THz, cm^-1, meV and rad/ps.

    The conversion goes through THz. Negative values (the convention for imaginary harmonic modes) keep their sign.

    Args:
        values: Frequency or array of frequencies in ``from_unit``.
        from_unit: Unit of ``values``: ``"THz"``, ``"cm-1"``, ``"meV"`` or ``"rad/ps"``.
        to_unit: Target unit, one of the same four.

    Returns:
        A float for scalar input, otherwise a new float64 array of the same shape as ``values``. The input is never
        modified; ``from_unit == to_unit`` returns a copy.

    Raises:
        ValueError: If ``from_unit`` or ``to_unit`` is not one of the allowed units.

    Example:
        >>> round(convert_frequency(1.0, "THz", "cm-1"), 5)
        33.35641
        >>> convert_frequency(np.array([-1.0, 2.0]), "THz", "rad/ps").round(4)
        array([-6.2832, 12.5664])
    """
    for unit in (from_unit, to_unit):
        if unit not in _THZ_TO_UNIT:
            msg = f"Unknown frequency unit {unit!r}; allowed units are {list(_THZ_TO_UNIT)}."
            raise ValueError(msg)
    converted = np.asarray(values, dtype=np.float64) * (_THZ_TO_UNIT[to_unit] / _THZ_TO_UNIT[from_unit])
    if converted.ndim == 0:
        return float(converted)
    return converted


def _positions_by_label(labels: dict[int, str], id_to_position: dict[int, int]) -> dict[str, np.ndarray]:
    """Group atom IDs by label and convert them to sorted int64 positions."""
    grouped: dict[str, list[int]] = {}
    for atom_id, label in labels.items():
        grouped.setdefault(label, []).append(id_to_position[atom_id])
    return {label: np.sort(np.array(positions, dtype=np.int64)) for label, positions in sorted(grouped.items())}


def classify_vibrational_groups(
    structure: "Atoms",
    cutoff: CutoffSpec | None = None,
    former_types: list[int] | None = None,
    o_type: int = 8,
) -> dict[str, dict[str, np.ndarray]]:
    """Label every atom by element, oxygen class and network-former Q^n for partial VDOS.

    Oxygen classes and Q^n come from :func:`compute_qn_and_classify` and :func:`compute_qn_per_atom`. Missing
    ``cutoff`` and ``former_types`` are derived as in :func:`analyze_structure`: formers are the glass formers and
    intermediates present, and each former-O cutoff is the first minimum of its RDF (``r_max=10.0``,
    ``n_bins=500``), with the same fallbacks.

    Args:
        structure: The atomic structure as ASE object. Atom IDs are read from the ``"id"`` array if present,
            else taken as 1-based indices; the returned indices are always 0-based positions in ``structure``.
        cutoff: Former-O cutoff in Å, either a scalar or a per-pair dict ``{(z_i, z_j): r_cut}``. Derived from
            the RDF if ``None``.
        former_types: Atomic numbers of the network formers. Derived from the composition if ``None``.
        o_type: Atomic number of oxygen.

    Returns:
        A dict with three groupings, each mapping a label to a sorted int64 array of 0-based positions:
            ``"element"``: element symbol, e.g. ``"Si"``, ``"O"``, ``"Na"``.
            ``"oxygen"``: ``"O_BO"``, ``"O_NBO"``, ``"O_free"``, ``"O_tri"``.
            ``"qn"``: former symbol and number of bridging oxygens, e.g. ``"Si_Q3"``, ``"B_Q4"``.
        Labels without atoms are omitted. Without oxygen, ``"oxygen"`` and ``"qn"`` are empty; with oxygen but
        no formers, ``"qn"`` is empty and every oxygen is ``"O_free"``.

    Raises:
        ValueError: If the ``"id"`` array contains duplicate IDs.

    Example:
        ```pycon
        >>> from ase import Atoms
        >>> d = 1.6
        >>> positions = [[10, 10, 10], [10 + 2 * d, 10, 10], [10 + d, 10, 10],
        ...              [10 - d, 10, 10], [10, 10 + d, 10], [10, 10, 10 + d],
        ...              [10 + 3 * d, 10, 10], [10 + 2 * d, 10 + d, 10], [10 + 2 * d, 10, 10 + d]]
        >>> dimer = Atoms("Si2O7", positions=positions, cell=[30, 30, 30], pbc=True)
        >>> groups = classify_vibrational_groups(dimer, cutoff=2.0, former_types=[14])
        >>> groups["element"]["Si"].tolist(), groups["oxygen"]["O_BO"].tolist(), groups["qn"]["Si_Q1"].tolist()
        ([0, 1], [2], [0, 1])

        ```
    """
    n_atoms = len(structure)
    if "id" in structure.arrays:
        atom_ids = structure.arrays["id"].astype(np.int64)
    else:
        atom_ids = np.arange(1, n_atoms + 1, dtype=np.int64)
    if len(np.unique(atom_ids)) != n_atoms:
        msg = "The 'id' array contains duplicate atom IDs."
        raise ValueError(msg)
    id_to_position = {int(atom_id): position for position, atom_id in enumerate(atom_ids)}

    symbols = np.array(structure.get_chemical_symbols())
    groups: dict[str, dict[str, np.ndarray]] = {
        "element": {str(sym): np.flatnonzero(symbols == sym).astype(np.int64) for sym in sorted(set(symbols))},
        "oxygen": {},
        "qn": {},
    }
    atomic_numbers = structure.get_atomic_numbers()
    if o_type not in atomic_numbers:
        return groups

    unique_z = np.unique(atomic_numbers)
    type_map, network_formers, _modifiers, _oxygen_present = _classify_elements(unique_z)
    if former_types is None:
        former_types = _network_former_types(type_map, network_formers)
    if not former_types:
        groups["oxygen"] = {"O_free": np.flatnonzero(atomic_numbers == o_type).astype(np.int64)}
        return groups
    if cutoff is None:
        r, rdfs, _cumcn = compute_rdf(structure, r_max=10.0, n_bins=500)
        cutoff_map = _build_cutoff_map(unique_z, type_map, former_types, [o_type], r, rdfs)
        cutoff = _former_oxygen_pair_cutoffs(former_types, o_type, type_map, cutoff_map)

    _total_qn, _partial_qn, oxygen_classes = compute_qn_and_classify(structure, cutoff, former_types, o_type)
    qn_per_atom = compute_qn_per_atom(structure, cutoff, former_types, o_type)
    id_to_symbol = {int(atom_id): str(sym) for atom_id, sym in zip(atom_ids, symbols, strict=True)}
    groups["oxygen"] = _positions_by_label({aid: f"O_{cls}" for aid, cls in oxygen_classes.items()}, id_to_position)
    groups["qn"] = _positions_by_label(
        {aid: f"{id_to_symbol[aid]}_Q{n}" for aid, n in qn_per_atom.items()}, id_to_position
    )
    return groups
