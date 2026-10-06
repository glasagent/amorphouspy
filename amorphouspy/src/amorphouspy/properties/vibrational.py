"""Vibrational properties of glass systems: vibrational density of states and related quantities.

Frequencies in this module are ordinary frequencies in THz (not angular frequencies) unless stated otherwise.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.fft
import scipy.signal
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


VdosNormalization = Literal["unit", "3N", "none"]
VdosWindow = Literal["hann", "hamming", "blackmanharris"] | tuple[Literal["gaussian"], float] | None

# Target size of one chunk of velocities; the complex spectrum of a chunk is about the same size.
_CHUNK_BYTES = 64 * 2**20
_VELOCITY_NDIM = 3  # (n_frames, n_atoms, 3)
_N_COMPONENTS = 3
_MIN_FRAMES = 2
# Internal (centre-of-mass-free) kinetic sum at or below this fraction of the uncentred one counts as no internal
# motion. It sits well above the rounding noise of removing the centre of mass from a uniform translation.
_ZERO_SPECTRUM_RTOL = 1e-10
_WINDOWS = ("hann", "hamming", "blackmanharris")
_GAUSSIAN_LEN = 2  # ("gaussian", std)


def _atoms_per_chunk(n_frames: int, itemsize: int) -> int:
    return max(1, _CHUNK_BYTES // (n_frames * _N_COMPONENTS * itemsize))


def _validate_vdos_inputs(
    velocities: NDArray, masses: NDArray, dt_fs: float, normalization: str, window: object
) -> tuple[NDArray, NDArray[np.float64]]:
    velocities = np.asarray(velocities)
    masses = np.asarray(masses, dtype=np.float64)
    if velocities.ndim != _VELOCITY_NDIM or velocities.shape[2] != _N_COMPONENTS or velocities.shape[1] == 0:
        msg = f"velocities must have shape (n_frames, n_atoms, 3) with n_atoms >= 1, got {velocities.shape}."
        raise ValueError(msg)
    n_frames, n_atoms, _ = velocities.shape
    if masses.shape != (n_atoms,):
        msg = f"masses must have shape ({n_atoms},) to match velocities, got {masses.shape}."
        raise ValueError(msg)
    if n_frames < _MIN_FRAMES:
        msg = f"At least 2 frames are needed, got {n_frames}."
        raise ValueError(msg)
    if not dt_fs > 0:
        msg = f"dt_fs must be positive, got {dt_fs}."
        raise ValueError(msg)
    if not np.all(np.isfinite(masses) & (masses > 0)):
        msg = "All masses must be positive and finite."
        raise ValueError(msg)
    if normalization not in ("unit", "3N", "none"):
        msg = f"Unknown normalization {normalization!r}; allowed values are 'unit', '3N' and 'none'."
        raise ValueError(msg)
    gaussian = (
        isinstance(window, tuple)
        and len(window) == _GAUSSIAN_LEN
        and window[0] == "gaussian"
        and isinstance(window[1], int | float | np.integer | np.floating)
        and not isinstance(window[1], bool)
        and np.isfinite(window[1])
        and window[1] > 0
    )
    if not (gaussian or window is None or (isinstance(window, str) and window in _WINDOWS)):
        msg = (
            f"Unknown window {window!r}; allowed values are 'hann', 'hamming', 'blackmanharris', "
            "('gaussian', std) with std > 0 in frames, and None."
        )
        raise ValueError(msg)
    return velocities, masses


def _validate_selection(indices: NDArray, n_atoms: int, name: str) -> NDArray[np.intp]:
    selection = np.asarray(indices)
    if selection.ndim != 1 or selection.size == 0 or not np.issubdtype(selection.dtype, np.integer):
        msg = f"{name} must be a non-empty 1-D array of integer atom positions."
        raise ValueError(msg)
    if selection.min() < 0 or selection.max() >= n_atoms:
        msg = f"{name} contains positions outside [0, {n_atoms})."
        raise ValueError(msg)
    if np.unique(selection).size != selection.size:
        msg = f"{name} contains duplicate positions."
        raise ValueError(msg)
    return selection.astype(np.intp)


def _trajectory_statistics(
    velocities: NDArray, masses: NDArray[np.float64], weights: NDArray[np.float64] | None, *, remove_com: bool
) -> tuple[NDArray[np.float64] | None, float]:
    """One read of all atoms: centre-of-mass velocity per frame and the trapezoid integral of the total spectrum.

    The integral of sum_i m_i |rfft(w v_i)|^2 over the rfft grid follows from Parseval's theorem without any FFT:
    (dnu / 2) * (n_frames * K - K_top), with K = sum_t w_t^2 sum_i m_i |v_i(t)|^2 and K_top the mass-weighted power
    of the last frequency bin of the windowed signal, which only enters for an odd number of frames. ``weights=None``
    means w = 1. With centre-of-mass removal both sums are reduced by the centre-of-mass term, since
    sum_i m_i (v_i - v_com) = 0. Non-finite velocities make the unwindowed sum non-finite, so the finiteness check
    costs nothing extra; the finiteness and zero-spectrum checks use the unwindowed sums.

    Returns:
        The centre-of-mass velocity ``(n_frames, 3)`` (``None`` without removal) and the integral divided by dnu / 2.
    """
    n_frames, n_atoms, _ = velocities.shape
    chunk = _atoms_per_chunk(n_frames, velocities.itemsize)
    odd = n_frames % 2 == 1
    phase = np.arange(n_frames) * (2 * np.pi * ((n_frames - 1) // 2) / n_frames)
    cos_t, sin_t = np.cos(phase), np.sin(phase)
    if weights is not None:
        cos_t, sin_t = cos_t * weights, sin_t * weights
        kinetic = np.zeros(n_frames)  # sum_i m_i |v_i(t)|^2 per frame
    momentum = np.zeros((n_frames, _N_COMPONENTS))
    top_momentum = np.zeros((2, _N_COMPONENTS))
    raw_sq = 0.0
    top_sq = 0.0
    for start in range(0, n_atoms, chunk):
        block = velocities[:, start : start + chunk]
        block_masses = masses[start : start + chunk]
        if weights is None:
            block_sq = float(np.einsum("tia,tia->i", block, block, dtype=np.float64) @ block_masses)
        else:
            frame_sq = np.einsum("tia,tia->ti", block, block, dtype=np.float64) @ block_masses
            block_sq = float(frame_sq.sum())
            kinetic += frame_sq
        if not np.isfinite(block_sq):
            msg = "velocities contain NaN or inf (or values too large to square)."
            raise ValueError(msg)
        raw_sq += block_sq
        momentum += np.einsum("tia,i->ta", block, block_masses, dtype=np.float64)
        if odd:
            top = np.stack(
                [
                    np.einsum("t,tia->ia", cos_t, block, dtype=np.float64),
                    np.einsum("t,tia->ia", sin_t, block, dtype=np.float64),
                ]
            )
            top_sq += float(np.einsum("cia,cia,i->", top, top, block_masses))
            top_momentum += np.einsum("cia,i->ca", top, block_masses)
    if not raw_sq > 0:
        msg = "The velocity spectrum is zero (all velocities are zero)."
        raise ValueError(msg)
    window_sq = None if weights is None else weights**2
    if not remove_com:
        weighted_sq = raw_sq if window_sq is None else float(window_sq @ kinetic)
        return None, n_frames * weighted_sq - top_sq
    total_mass = masses.sum()
    com_velocity = momentum / total_mass
    centred_sq = raw_sq - total_mass * float(np.sum(com_velocity**2))
    if centred_sq <= _ZERO_SPECTRUM_RTOL * raw_sq:
        msg = "The velocity spectrum is zero after removing the centre-of-mass velocity (no internal motion)."
        raise ValueError(msg)
    if window_sq is not None:
        centred_sq = float(window_sq @ kinetic) - total_mass * float(window_sq @ np.sum(com_velocity**2, axis=1))
    return com_velocity, n_frames * centred_sq - (top_sq - float(np.sum(top_momentum**2)) / total_mass)


def _accumulate_power(
    velocities: NDArray,
    masses: NDArray[np.float64],
    com_velocity: NDArray[np.float64] | None,
    membership: NDArray,
    weights: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Sum m_i * sum_a |rfft(w * (v_ia - v_com,a))|^2 over the atoms of each column of ``membership`` (n_atoms, n_cols).

    Only atoms that belong to at least one column are transformed. A contiguous range of atoms is sliced as a view;
    other selections are gathered chunk by chunk. float32 velocities are transformed in single precision (complex64)
    and float64 or other input in double precision, with the same forward, unnormalised DFT as ``np.fft.rfft``;
    per-atom powers are combined in float64. The window ``weights`` (``None`` for none) is applied per chunk in the
    working precision, never in place on a view of the input.
    """
    n_frames = velocities.shape[0]
    work = np.float32 if velocities.dtype == np.float32 else np.float64
    needed = np.flatnonzero(membership.any(axis=1))
    contiguous = needed[-1] - needed[0] + 1 == needed.size
    chunk = _atoms_per_chunk(n_frames, np.dtype(work).itemsize)
    com_work = None if com_velocity is None else com_velocity.astype(work)[:, None, :]
    window_work = None if weights is None else weights.astype(work)[:, None, None]
    power = np.zeros((n_frames // 2 + 1, membership.shape[1]))
    for start in range(0, needed.size, chunk):
        atoms = needed[start : start + chunk]
        is_view = False
        if contiguous:
            block = velocities[:, atoms[0] : atoms[-1] + 1]
            if com_work is not None:
                block = np.subtract(block, com_work, dtype=work)
            elif block.dtype != work:
                block = block.astype(work)
            else:
                is_view = True
        else:
            block = velocities[:, atoms].astype(work, copy=False)
            if com_work is not None:
                block -= com_work
        if window_work is not None:
            if is_view:
                block = block * window_work
            else:
                block *= window_work
        # scipy.fft transforms float32 natively; np.fft computes in double internally and needs ~3x the memory.
        spectrum = np.ascontiguousarray(scipy.fft.rfft(block, axis=0))
        del block
        # |z|^2 = Re(z)^2 + Im(z)^2 summed over x, y, z through a real view, without full-size temporaries.
        parts = spectrum.view(work).reshape(spectrum.shape[0], spectrum.shape[1], 2 * _N_COMPONENTS)
        atom_power = np.einsum("fik,fik->fi", parts, parts)
        del spectrum, parts
        power += atom_power @ (masses[atoms, None] * membership[atoms])
    return power


def _vdos_columns(
    velocities: NDArray,
    masses: NDArray,
    dt_fs: float,
    selections: dict[str, NDArray] | None,
    normalization: str,
    *,
    remove_com: bool,
    window: VdosWindow,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Shared implementation: VDOS of the whole system (``selections=None``) or of each named selection, as columns."""
    velocities, masses = _validate_vdos_inputs(velocities, masses, dt_fs, normalization, window)
    n_frames, n_atoms, _ = velocities.shape
    if selections is None:
        membership = np.ones((n_atoms, 1))
    else:
        membership = np.zeros((n_atoms, len(selections)))
        for column, (name, indices) in enumerate(selections.items()):
            membership[_validate_selection(indices, n_atoms, name), column] = 1.0
    # get_window builds the periodic (DFT-even) form of each window.
    weights = None if window is None else np.asarray(scipy.signal.get_window(window, n_frames), dtype=np.float64)
    com_velocity, parseval_sum = _trajectory_statistics(velocities, masses, weights, remove_com=remove_com)
    power = _accumulate_power(velocities, masses, com_velocity, membership, weights)

    dt_ps = dt_fs * 1e-3
    frequencies_thz = np.asarray(np.fft.rfftfreq(n_frames, d=dt_ps), dtype=np.float64)
    if normalization == "none":
        if weights is None:
            return frequencies_thz, power * (2 * dt_ps / n_frames)
        return frequencies_thz, power * (2 * dt_ps / float(np.sum(weights**2)))
    reference_integral = 0.5 * float(frequencies_thz[1]) * parseval_sum
    scale = 1.0 if normalization == "unit" else 3.0 * n_atoms
    return frequencies_thz, power * (scale / reference_integral)


def compute_vdos_from_velocities(
    velocities: NDArray,
    masses: NDArray,
    dt_fs: float,
    *,
    indices: NDArray | None = None,
    normalization: VdosNormalization = "unit",
    remove_com: bool = True,
    window: VdosWindow = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Compute the vibrational density of states from an MD velocity trajectory.

    The VDOS is the mass-weighted velocity power spectrum

    $$g(\nu_k) \propto \sum_{i} m_i \sum_{\alpha} |V_{i\alpha}(\nu_k)|^2,
    \quad V_{i\alpha}(\nu_k) = \sum_{t=0}^{N-1} v_{i\alpha}(t)\, e^{-2\pi \mathrm{i} k t / N},$$

    the unnormalised forward DFT along time (the ``np.fft.rfft`` convention; computed with ``scipy.fft.rfft``), on the
    one-sided grid
    $\nu_k = k / (N \Delta t)$, $k = 0, \dots, \lfloor N/2 \rfloor$, in THz. The frequency resolution is
    $1 / (N \Delta t)$ and the highest frequency is the Nyquist frequency $1 / (2 \Delta t)$. No zero padding or
    per-atom mean subtraction is applied, and no window unless ``window`` is set.

    With a window, the (centre-of-mass-free) velocities are multiplied by $w_t$, $t = 0, \dots, N-1$, before the
    transform. All windows are periodic (``scipy.signal.get_window(window, N)``):

    - ``"hann"``: $w_t = \tfrac{1}{2}\left(1 - \cos(2\pi t / N)\right)$.
    - ``"hamming"``: $w_t = 0.54 - 0.46 \cos(2\pi t / N)$. Lower first sidelobe than Hann, but the edges are not
      zero (0.08), so leakage far from a peak decays more slowly.
    - ``"blackmanharris"``: the 4-term Blackman-Harris window $w_t = \sum_{j=0}^{3} (-1)^j a_j \cos(2\pi j t / N)$,
      $a = (0.35875, 0.48829, 0.14128, 0.01168)$, with sidelobes below $-92$ dB.
    - ``("gaussian", std)``: $w_t = \exp\left(-\tfrac{1}{2}\left((t - N/2) / \sigma\right)^2\right)$ with
      $\sigma$ = ``std`` in frames, so its width relative to the block depends on $N$. It is truncated at the edges
      (edge value $e^{-8} \approx 3 \times 10^{-4}$ for $\sigma = N/8$). A narrow window ($\sigma \ll N/8$) widens
      every peak and can spread power further than no window at all.

    A window suppresses spectral leakage from the finite trajectory length at the cost of resolution. The effective
    resolution is $\mathrm{ENBW} / (N \Delta t)$ with the equivalent noise bandwidth
    $\mathrm{ENBW} = N \sum_t w_t^2 / (\sum_t w_t)^2$ bins: 1.5 (Hann), 1.36 (Hamming), 2.0 (Blackman-Harris) and
    about $N / (2 \sqrt{\pi} \sigma)$ (Gaussian, $\sigma \ll N$). For 10 ps blocks that is 0.15, 0.14 and 0.20 THz,
    and 0.23 THz for a Gaussian with $\sigma = N/8$. With ``window=None`` (default) the full $1 / (N \Delta t)$
    resolution is kept, and a peak between grid points leaks into distant bins.

    With ``remove_com=True`` (default), the instantaneous mass-weighted centre-of-mass velocity of the whole system,
    $\mathbf{v}_\mathrm{COM}(t) = \sum_i m_i \mathbf{v}_i(t) / \sum_i m_i$, is subtracted from every atom in every
    frame before the transform, also when ``indices`` selects a subset. With ``remove_com=False`` the velocities are
    transformed as given, and any centre-of-mass drift appears at and near zero frequency.

    Normalisation (all integrals are trapezoidal over the returned grid):

    - ``"unit"`` (default): the total VDOS integrates to 1 (units 1/THz).
    - ``"3N"``: the total VDOS integrates to ``3 * n_atoms`` (the number of degrees of freedom, including the 3
      removed with the centre of mass).
    - ``"none"``: the power spectral density $S(\nu_k) = (2 \Delta t / N) \sum_i m_i \sum_\alpha |V_{i\alpha}|^2$ in
      amu (velocity unit)$^2$ ps. For an even number of frames its integral equals the time average of
      $\sum_i m_i |\mathbf{v}_i|^2$ (twice the mean kinetic energy) exactly, by Parseval's theorem; for an odd number
      it is lower by half the last bin. With a window, $V_{i\alpha}$ is the transform of the windowed velocities and
      the prefactor is $2 \Delta t / \sum_t w_t^2$ instead of $2 \Delta t / N$; the integral is then the
      $w^2$-weighted time average $\sum_t w_t^2 \sum_i m_i |\mathbf{v}_i(t)|^2 / \sum_t w_t^2$ (exact for an even
      number of frames).

    With a window, ``"unit"`` and ``"3N"`` are applied to the windowed total VDOS, which still integrates to exactly 1
    and ``3 * n_atoms``.

    A partial VDOS (``indices`` given) is the same spectrum summed over the selected atoms only and is scaled by the
    total VDOS's normalisation, so partials over disjoint selections add up to the total. Its integral is the
    selection's share of the total kinetic energy, which is close to its share of atoms by equipartition: about
    ``3 * len(indices)`` for ``"3N"``. The integral of the total spectrum comes from Parseval's theorem in the time
    domain, so a partial only transforms the selected atoms. Use :func:`compute_partial_vdos` for many groups.

    Memory: the trajectory is read twice in chunks of atoms (about 64 MB each), once for the centre-of-mass velocity,
    the normalisation and the finiteness check, and once for the transforms. Besides the input, peak memory is about
    two chunks. float32 input is transformed in single precision (no float64 copy); the spectrum is accumulated in
    float64. The input is never modified, and it may be a read-only ``np.memmap`` (e.g. ``np.load(path,
    mmap_mode="r")``), in which case only the chunks being processed are read from disk.

    Args:
        velocities: Velocities of shape ``(n_frames, n_atoms, 3)`` in any unit; ``"unit"`` and ``"3N"`` remove the
            scale.
        masses: Atomic masses of shape ``(n_atoms,)`` in amu.
        dt_fs: Time between consecutive frames in fs (the dump interval, not the MD timestep).
        indices: 0-based positions of the atoms in a partial VDOS, e.g. one entry of
            :func:`classify_vibrational_groups`. ``None`` (default) gives the total VDOS.
        normalization: ``"unit"``, ``"3N"`` or ``"none"``, see above.
        remove_com: Subtract the centre-of-mass velocity of the whole system from every frame.
        window: ``"hann"``, ``"hamming"``, ``"blackmanharris"``, ``("gaussian", std)`` with ``std > 0`` in frames,
            or ``None`` (default) for no window, see above.

    Returns:
        A tuple ``(frequencies_thz, vdos)`` of float64 arrays of length ``n_frames // 2 + 1``: the frequencies in THz
        from ``np.fft.rfftfreq(n_frames, d=dt_fs * 1e-3)`` and the VDOS in 1/THz (``"unit"``, ``"3N"``) or the power
        spectral density (``"none"``).

    Raises:
        ValueError: If ``velocities`` is not ``(n_frames, n_atoms, 3)`` with at least one atom, ``masses`` does not
            have shape ``(n_atoms,)``, there are fewer than 2 frames, ``dt_fs <= 0``, any mass is not positive and
            finite, ``velocities`` contains NaN or inf, ``normalization`` or ``window`` is unknown, ``indices`` is
            empty, not 1-D integer, out of range or has duplicates, or the spectrum vanishes (all velocities zero, or
            no motion left after centre-of-mass removal, e.g. every atom moves with the same velocity).

    Example:
        ```pycon
        >>> from amorphouspy import get_atomic_mass
        >>> dt_fs, n_frames = 5.0, 2000
        >>> t_ps = np.arange(n_frames) * dt_fs * 1e-3
        >>> velocities = np.zeros((n_frames, 2, 3))
        >>> velocities[:, 0, 0] = np.cos(2 * np.pi * 10.0 * t_ps)
        >>> velocities[:, 1, 0] = -velocities[:, 0, 0]
        >>> masses = np.full(2, get_atomic_mass("Si"))
        >>> frequencies_thz, vdos = compute_vdos_from_velocities(velocities, masses, dt_fs)
        >>> float(frequencies_thz[np.argmax(vdos)])
        10.0
        >>> round(float(np.trapezoid(vdos, frequencies_thz)), 6)
        1.0
        >>> _, partial = compute_vdos_from_velocities(
        ...     velocities, masses, dt_fs, indices=np.array([0]), normalization="3N"
        ... )
        >>> round(float(np.trapezoid(partial, frequencies_thz)), 6)
        3.0
        >>> _, vdos_hann = compute_vdos_from_velocities(velocities, masses, dt_fs, window="hann")
        >>> float(frequencies_thz[np.argmax(vdos_hann)]), round(float(np.trapezoid(vdos_hann, frequencies_thz)), 6)
        (10.0, 1.0)

        ```
    """
    selections = None if indices is None else {"indices": indices}
    frequencies_thz, columns = _vdos_columns(
        velocities, masses, dt_fs, selections, normalization, remove_com=remove_com, window=window
    )
    return frequencies_thz, np.ascontiguousarray(columns[:, 0])


def compute_partial_vdos(
    velocities: NDArray,
    masses: NDArray,
    dt_fs: float,
    groups: dict[str, dict[str, NDArray]],
    *,
    normalization: VdosNormalization = "unit",
    remove_com: bool = True,
    window: VdosWindow = None,
) -> tuple[NDArray[np.float64], dict[str, dict[str, NDArray[np.float64]]]]:
    """Compute partial VDOS for many atom groups with one transform per atom.

    Each partial is exactly what :func:`compute_vdos_from_velocities` returns with ``indices`` set to the group
    (same centre-of-mass treatment, window, normalisation and units), but every atom is transformed once, however many
    groups it belongs to. ``groups`` has the layout returned by :func:`classify_vibrational_groups`: classify once, then
    reuse the indices for any number of trajectories or blocks. Within one grouping that covers every atom once (e.g.
    ``"element"``), the partials add up to the total VDOS computed with the same window and normalisation.

    Args:
        velocities: Velocities of shape ``(n_frames, n_atoms, 3)`` in any unit.
        masses: Atomic masses of shape ``(n_atoms,)`` in amu.
        dt_fs: Time between consecutive frames in fs (the dump interval, not the MD timestep).
        groups: ``{grouping: {label: indices}}`` with 0-based atom positions, e.g. the output of
            :func:`classify_vibrational_groups`. Atoms may appear in several groupings.
        normalization: ``"unit"``, ``"3N"`` or ``"none"``, as in :func:`compute_vdos_from_velocities`.
        remove_com: Subtract the centre-of-mass velocity of the whole system from every frame.
        window: ``"hann"``, ``"hamming"``, ``"blackmanharris"``, ``("gaussian", std)`` or ``None`` (default), as in
            :func:`compute_vdos_from_velocities`.

    Returns:
        A tuple ``(frequencies_thz, partial)``: the frequencies in THz and ``{grouping: {label: vdos}}`` with the
        same keys as ``groups``, each a float64 array of length ``n_frames // 2 + 1``.

    Raises:
        ValueError: As :func:`compute_vdos_from_velocities` (including an unknown ``window``), for any group's
            indices, or if ``groups`` has no labels.

    Example:
        ```pycon
        >>> rng = np.random.default_rng(0)
        >>> velocities = rng.normal(size=(100, 3, 3))
        >>> masses = np.array([28.085, 15.999, 15.999])
        >>> groups = {"element": {"Si": np.array([0]), "O": np.array([1, 2])}}
        >>> frequencies_thz, partial = compute_partial_vdos(velocities, masses, 5.0, groups)
        >>> _, total = compute_vdos_from_velocities(velocities, masses, 5.0)
        >>> bool(np.allclose(partial["element"]["Si"] + partial["element"]["O"], total))
        True

        ```
    """
    keys = [(grouping, label) for grouping, labels in groups.items() for label in labels]
    if not keys:
        msg = "groups contains no labels."
        raise ValueError(msg)
    selections = {f"groups[{g!r}][{label!r}]": groups[g][label] for g, label in keys}
    frequencies_thz, columns = _vdos_columns(
        velocities, masses, dt_fs, selections, normalization, remove_com=remove_com, window=window
    )
    partial: dict[str, dict[str, NDArray[np.float64]]] = {grouping: {} for grouping in groups}
    for column, (grouping, label) in enumerate(keys):
        partial[grouping][label] = np.ascontiguousarray(columns[:, column])
    return frequencies_thz, partial
