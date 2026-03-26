"""Structure factor computation for multicomponent glass systems.

Author: Achraf Atila (achraf.atila@bam.de)

"""

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from pymatgen.analysis.diffraction.xrd import ATOMIC_SCATTERING_PARAMS

from amorphouspy.analysis.radial_distribution_functions import compute_rdf

# ---------------------------------------------------------------------------
# Coherent neutron scattering lengths in fm (10^-15 m) for natural elements.
# Source: V.F. Sears, Neutron News 3 (1992) 26; NIST n-length tables
#         https://www.ncnr.nist.gov/resources/n-lengths/list.html
# Elements without a reliable natural-element value (all isotopes radioactive
# or no stable isotopes, e.g. Tc, Po, At, Rn, Fr, Ac) are omitted.
# ---------------------------------------------------------------------------
_NEUTRON_SCATTERING_LENGTHS: dict[int, float] = {
    1: -3.7390,  # H
    2: 3.26,  # He
    3: -1.90,  # Li
    4: 7.79,  # Be
    5: 5.30,  # B
    6: 6.6460,  # C
    7: 9.36,  # N
    8: 5.803,  # O
    9: 5.654,  # F
    10: 4.566,  # Ne
    11: 3.63,  # Na
    12: 5.375,  # Mg
    13: 3.449,  # Al
    14: 4.1491,  # Si
    15: 5.13,  # P
    16: 2.847,  # S
    17: 9.5770,  # Cl
    18: 1.909,  # Ar
    19: 3.67,  # K
    20: 4.70,  # Ca
    21: 12.29,  # Sc
    22: -3.438,  # Ti
    23: -0.3824,  # V
    24: 3.635,  # Cr
    25: -3.73,  # Mn
    26: 9.45,  # Fe
    27: 2.49,  # Co
    28: 10.3,  # Ni
    29: 7.718,  # Cu
    30: 5.680,  # Zn
    31: 7.288,  # Ga
    32: 8.185,  # Ge
    33: 6.58,  # As
    34: 7.970,  # Se
    35: 6.795,  # Br
    36: 7.81,  # Kr
    37: 7.09,  # Rb
    38: 7.02,  # Sr
    39: 7.75,  # Y
    40: 7.16,  # Zr
    41: 7.054,  # Nb
    42: 6.715,  # Mo
    44: 7.03,  # Ru  (Tc Z=43 omitted: no stable isotopes)
    45: 5.88,  # Rh
    46: 5.91,  # Pd
    47: 5.922,  # Ag
    48: 4.87,  # Cd
    49: 4.065,  # In
    50: 6.225,  # Sn
    51: 5.57,  # Sb
    52: 5.80,  # Te
    53: 5.28,  # I
    54: 4.92,  # Xe
    55: 5.42,  # Cs
    56: 5.07,  # Ba
    57: 8.24,  # La
    58: 4.84,  # Ce
    59: 4.58,  # Pr
    60: 7.69,  # Nd
    61: 12.6,  # Pm  (radioactive; value for longest-lived isotope)
    62: 0.80,  # Sm  (high absorption cross section)
    63: 7.22,  # Eu  (high absorption cross section)
    64: 6.5,  # Gd  (very high absorption cross section)
    65: 7.38,  # Tb
    66: 16.9,  # Dy  (high absorption cross section)
    67: 8.01,  # Ho
    68: 7.79,  # Er
    69: 7.07,  # Tm
    70: 12.43,  # Yb
    71: 7.21,  # Lu
    72: 7.77,  # Hf
    73: 6.91,  # Ta
    74: 4.86,  # W
    75: 9.2,  # Re
    76: 10.7,  # Os
    77: 10.6,  # Ir
    78: 9.60,  # Pt
    79: 7.63,  # Au
    80: 12.692,  # Hg
    81: 8.776,  # Tl
    82: 9.405,  # Pb
    83: 8.532,  # Bi
    88: 10.0,  # Ra  (radioactive; approximate)
    90: 10.31,  # Th
    91: 9.1,  # Pa
    92: 8.417,  # U
    93: 10.55,  # Np
}


def _neutron_scattering_length(atomic_number: int) -> float:
    """Return the coherent neutron scattering length for an element.

    Values are bound coherent scattering lengths for natural elements from
    the NIST table (Sears, Neutron News 3, 1992, 26).

    Args:
        atomic_number: Atomic number Z of the element.

    Returns:
        Coherent scattering length in fm.

    Raises:
        KeyError: If the atomic number is not in the database.

    Example:
        >>> b_Si = _neutron_scattering_length(14)

    """
    if atomic_number not in _NEUTRON_SCATTERING_LENGTHS:
        supported = sorted(_NEUTRON_SCATTERING_LENGTHS.keys())
        msg = f"Neutron scattering length not available for Z={atomic_number}. Supported atomic numbers: {supported}"
        raise KeyError(msg)
    return _NEUTRON_SCATTERING_LENGTHS[atomic_number]


def _xray_form_factor(atomic_number: int, q_values: np.ndarray) -> np.ndarray:
    """Compute the X-ray atomic form factor f(q) using the Doyle-Turner parameterisation.

    Evaluates the pymatgen form factor formula (Doyle & Turner, 1968, derived
    from electron scattering factors via Mott's formula):

        f(q) = Z - 41.78214 * s^2 * sum_{i=1}^{4} a_i * exp(-b_i * s^2),
        s = q / (4*pi)

    where q is in Angstroms^-1. Coefficients are loaded from pymatgen's
    ATOMIC_SCATTERING_PARAMS database. Coverage: all elements for which
    data is available (Z=1 to ~98).

    Args:
        atomic_number: Atomic number Z of the element.
        q_values: Momentum transfer values in Angstroms^-1, shape (n_q,).

    Returns:
        Form factor values in electrons, shape (n_q,).

    Raises:
        KeyError: If the atomic number is not in the database.

    Example:
        >>> q = np.linspace(0.5, 20.0, 500)
        >>> f_Si = _xray_form_factor(14, q)

    """
    element_symbol = chemical_symbols[atomic_number]
    if element_symbol not in ATOMIC_SCATTERING_PARAMS:
        available = sorted(ATOMIC_SCATTERING_PARAMS.keys())
        msg = (
            f"X-ray form factor not available for {element_symbol} (Z={atomic_number}). Available elements: {available}"
        )
        raise KeyError(msg)
    pairs = ATOMIC_SCATTERING_PARAMS[element_symbol]  # [[a1, b1], [a2, b2], ...]
    s_sq = (q_values / (4.0 * np.pi)) ** 2
    gaussian_sum = sum(a * np.exp(-b * s_sq) for a, b in pairs)
    return atomic_number - 41.78214 * s_sq * gaussian_sum


def _sine_transform_rdf(
    r: np.ndarray,
    gr: np.ndarray,
    q_values: np.ndarray,
    number_density: float,
    *,
    lorch_damping: bool,
) -> np.ndarray:
    """Compute a partial structure factor from an RDF via the sine transform.

    Evaluates the Faber-Ziman relation:

        S_ab(q) = 1 + (4*pi*rho / q) * integral r*(g_ab(r)-1)*M(r)*sin(q*r) dr

    integrated from 0 to r_max using the trapezoidal rule with uniform spacing.
    The optional Lorch modification function M(r) = sinc(r / r_max) suppresses
    termination ripples from the finite real-space cutoff.

    Args:
        r: Radial bin centres in Angstroms, shape (n_bins,). Must be uniformly spaced.
        gr: Radial distribution function g(r), shape (n_bins,).
        q_values: Momentum transfer values in Angstroms^-1, shape (n_q,).
        number_density: Total number density rho in Angstroms^-3.
        lorch_damping: Apply the Lorch modification function to reduce
            termination ripples from the finite r_max cutoff.

    Returns:
        Partial structure factor S_ab(q), shape (n_q,).

    """
    r_max = float(r[-1])
    dr = float(r[1] - r[0])

    integrand = r * (gr - 1.0)  # shape (n_bins,)

    if lorch_damping:
        # np.sinc(x) = sin(pi*x) / (pi*x), so sinc(r/r_max) is the Lorch function
        integrand = integrand * np.sinc(r / r_max)

    # Trapezoidal weights for uniform spacing
    trapz_weights = np.ones(len(r))
    trapz_weights[0] = 0.5
    trapz_weights[-1] = 0.5
    integrand_weighted = integrand * trapz_weights  # (n_bins,)

    # Vectorised sine matrix: shape (n_q, n_bins)
    sin_matrix = np.sin(np.outer(q_values, r))

    # Integrate along r for each q: (n_q,)
    integral = (sin_matrix * integrand_weighted[np.newaxis, :]).sum(axis=1) * dr

    return 1.0 + 4.0 * np.pi * number_density * integral / q_values


def compute_structure_factor(
    structure: Atoms,
    q_min: float = 0.5,
    q_max: float = 20.0,
    n_q: int = 500,
    r_max: float = 10.0,
    n_bins: int = 2000,
    radiation: str = "neutron",
    *,
    lorch_damping: bool = True,
    type_pairs: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], np.ndarray]]:
    """Compute the 1D isotropic structure factor S(q) and Faber-Ziman partials.

    Uses the Faber-Ziman formalism to compute partial structure factors S_ab(q)
    from partial radial distribution functions g_ab(r) via the sine transform,
    then combines them into the total S(q) weighted by coherent neutron scattering
    lengths or q-dependent X-ray form factors.

    Neutron diffraction (Faber-Ziman total structure factor):

        S(q) = 1 + sum_{a<=b} (2 - d_ab) * c_a * c_b * b_a * b_b * [S_ab(q) - 1] / <b>^2

    where c_a is the atomic concentration, b_a the coherent scattering length
    (from NIST), and <b> = sum_a c_a * b_a.

    X-ray diffraction: b_a is replaced by the q-dependent International Tables
    form factor f_a(q) (four-Gaussian fit via pymatgen), making the weights q-dependent.

    Args:
        structure: ASE Atoms object with periodic boundary conditions.
        q_min: Minimum momentum transfer in Angstroms^-1 (default 0.5).
        q_max: Maximum momentum transfer in Angstroms^-1 (default 20.0).
        n_q: Number of q-grid points (default 500).
        r_max: Real-space cutoff in Angstroms for the underlying RDF (default 10.0).
        n_bins: Number of radial bins for the RDF (default 2000).
        radiation: Scattering probe: ``"neutron"`` uses tabulated NIST coherent
            scattering lengths; ``"xray"`` uses the International Tables
            four-Gaussian form factors via pymatgen (default ``"neutron"``).
        lorch_damping: Apply the Lorch modification function M(r) = sinc(r/r_max)
            to suppress termination ripples from the finite r_max cutoff
            (default True).
        type_pairs: List of (Z_a, Z_b) pairs for which to compute partials.
            ``None`` computes all unique unordered pairs present in the structure.

    Returns:
        q: Momentum transfer grid in Angstroms^-1, shape (n_q,).
        sq_total: Total structure factor S(q), shape (n_q,).
        sq_partials: Faber-Ziman partial structure factors S_ab(q), keyed by
            the canonical pair (min(Z_a, Z_b), max(Z_a, Z_b)), each with
            shape (n_q,).

    Raises:
        ValueError: If ``radiation`` is not ``"neutron"`` or ``"xray"``.
        KeyError: If a scattering length or form factor is not available for
            an element present in the structure.

    Example:
        >>> from ase.io import read
        >>> structure = read("sodium_silicate.extxyz")
        >>> q, sq, sq_partials = compute_structure_factor(structure, q_max=20.0)
        >>> sq_SiO = sq_partials[(8, 14)]  # partial S_SiO(q)

    """
    if radiation not in ("neutron", "xray"):
        msg = f"radiation must be 'neutron' or 'xray', got {radiation!r}."
        raise ValueError(msg)

    types = structure.get_atomic_numbers()
    unique_types = sorted({int(t) for t in types})
    total_atoms = len(structure)
    cell = structure.get_cell().array
    volume = float(abs(np.linalg.det(cell)))
    number_density = total_atoms / volume  # Angstrom^-3

    concentrations = {t: float(np.sum(types == t)) / total_atoms for t in unique_types}

    # --- Radial distribution functions ----------------------------------------
    r, rdfs, _ = compute_rdf(structure, r_max=r_max, n_bins=n_bins, type_pairs=type_pairs)
    unordered_pairs = list(rdfs.keys())

    # --- Momentum transfer grid -----------------------------------------------
    q_values = np.linspace(q_min, q_max, n_q)

    # --- Partial structure factors --------------------------------------------
    sq_partials: dict[tuple[int, int], np.ndarray] = {
        pair: _sine_transform_rdf(r, rdfs[pair], q_values, number_density, lorch_damping=lorch_damping)
        for pair in unordered_pairs
    }

    # --- Scattering weights and total S(q) ------------------------------------
    if radiation == "neutron":
        scattering_lengths = {t: _neutron_scattering_length(t) for t in unique_types}
        mean_b = sum(concentrations[t] * scattering_lengths[t] for t in unique_types)
        mean_b_sq = mean_b**2

        sq_total = np.zeros(n_q)
        for t1, t2 in unordered_pairs:
            b1 = scattering_lengths[t1]
            b2 = scattering_lengths[t2]
            symmetry_factor = 1.0 if t1 == t2 else 2.0
            weight = symmetry_factor * concentrations[t1] * concentrations[t2] * b1 * b2
            sq_total += weight * (sq_partials[(t1, t2)] - 1.0)
        sq_total = 1.0 + sq_total / mean_b_sq

    else:  # xray
        form_factors = {t: _xray_form_factor(t, q_values) for t in unique_types}
        # mean_f(q) = sum_a c_a * f_a(q), shape (n_q,)
        mean_f = np.sum(
            [concentrations[t] * form_factors[t] for t in unique_types],
            axis=0,
        )
        mean_f_sq = mean_f**2

        sq_total = np.zeros(n_q)
        for t1, t2 in unordered_pairs:
            f1 = form_factors[t1]
            f2 = form_factors[t2]
            symmetry_factor = 1.0 if t1 == t2 else 2.0
            weight = symmetry_factor * concentrations[t1] * concentrations[t2] * f1 * f2
            sq_total += weight * (sq_partials[(t1, t2)] - 1.0)
        sq_total = 1.0 + sq_total / mean_f_sq

    return q_values, sq_total, sq_partials
