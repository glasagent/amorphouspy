"""LAMMPS potential generation for oxide glass simulations using Du/Teter parameters.

Author: Marcel Sadowski (marcel.sadowski@schott.com)


Note that the Teter potential has never been published by its original author.
Later, researchers only cited it under "private communication". For example, Du et al. have picked
up these parameters and used them in various publications (sometimes with slight differences in numbers).

We are using the following, more recent references in the definition:
[1] Jayani Kalahe, Xiaonan Lu, Kyle Miller, Miroslava Peterson, Brian J. Riley,
    John D. Vienna, James E. Saal, and Jincheng Du,
    "Revealing the Structure-Property Relationships of Nd2O3-Containing Alkali Iron
    Phosphate Glasses from Molecular Dynamics Simulations"
    Chem. Mater. 2025, 37, 8595-8613
    doi.org/10.1021/acs.chemmater.5c01505
    Note that this publications also defines three-body terms for P-O-P and O-P-O
    interactions, which we are not including in our current implementation.
[2] Lu Deng and Jincheng Du
    "Development of boron oxide potentials for computer simulations  of multicomponent
    oxide glasses"
    J Am Ceram Soc. 2019, 102, 2482-2505.
    doi.org/10.1111/jace.16082
    Also note the erratum, https://doi.org/10.1111/jace.16897, with a corrected formula for t3 parameter
Note: Reference [1] also has parameters for Fe2+ and F3+ (with effective charges of 1.2 and 1.8, respectively)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from amorphouspy.shared import get_element_types_dict

du_teter_potential_params = {
    "B": {"q": 1.8, "A": 0.0, "rho": 0.171271, "C": 28.500, "B": 0.0, "D": 0.0, "n": 0.0, "r0": 0.0},  # [2]
    "O": {
        "q": -1.2,
        "A": 2029.2204,
        "rho": 0.343640,
        "C": 192.580,
        "B": 46.462,
        "D": -0.32605,
        "n": 3.430,
        "r0": 1.906,
    },  # [1], almost identical to [2]
    "Si": {
        "q": 2.4,
        "A": 13702.9050,
        "rho": 0.193817,
        "C": 54.681,
        "B": 28.950,
        "D": -3.059,
        "n": 3.9320,
        "r0": 1.168,
    },  # [2]
    "Al": {
        "q": 1.8,
        "A": 12201.4170,
        "rho": 0.195628,
        "C": 31.997,
        "B": 51.728,
        "D": -10.197,
        "n": 3.1790,
        "r0": 1.024,
    },  # [2]
    "Li": {
        "q": 0.6,
        "A": 41051.9380,
        "rho": 0.151160,
        "C": 0.000,
        "B": 0.000,
        "D": 0.000,
        "n": 0.0,
        "r0": 0.000,
    },  # [2]
    "Na": {
        "q": 0.6,
        "A": 4383.7555,
        "rho": 0.243838,
        "C": 30.700,
        "B": 48.264,
        "D": -4.730,
        "n": 2.8930,
        "r0": 1.172,
    },  # [2]
    "K": {
        "q": 0.6,
        "A": 20526.9720,
        "rho": 0.233708,
        "C": 51.489,
        "B": 310.841,
        "D": -78.124,
        "n": 2.4680,
        "r0": 0.950,
    },  # [1], almost identical to [2]
    "Ca": {
        "q": 1.2,
        "A": 7747.1834,
        "rho": 0.252623,
        "C": 93.109,
        "B": 74.737,
        "D": -3.991,
        "n": 3.1660,
        "r0": 1.318,
    },  # [2]
    "P": {
        "q": 3.0,
        "A": 26655.4720,
        "rho": 0.181968,
        "C": 86.856,
        "B": 28.568,
        "D": -3.379,
        "n": 4.6540,
        "r0": 1.169,
    },  # [1], almost identical to [2]
    "Sr": {
        "q": 1.2,
        "A": 14566.6370,
        "rho": 0.245015,
        "C": 81.773,
        "B": 189.853,
        "D": -23.951,
        "n": 2.7310,
        "r0": 1.112,
    },  # [2]
    "Zr": {
        "q": 2.4,
        "A": 17943.3840,
        "rho": 0.226627,
        "C": 127.650,
        "B": 108.655,
        "D": -8.481,
        "n": 3.3050,
        "r0": 1.224,
    },  # [2]
    "Nd": {
        "q": 1.8,
        "A": 6207.7229,
        "rho": 0.257410,
        "C": 40.165,
        "B": 100.350,
        "D": -11.641,
        "n": 2.653,
        "r0": 0.970,
    },  # [1]
    # original Teter paramters:
    "Mg": {
        "q": 1.2,
        "A": 7060.0000,
        "rho": 0.211000,
        "C": 19.200,
        "B": 54.235218,
        "D": -11.69947,
        "n": 2.8397537,
        "r0": 0.99567,
    },
    "La": {
        "q": 1.8,
        "A": 4369.3900,
        "rho": 0.278600,
        "C": 60.278,
        "B": 75.669457,
        "D": -4.482141,
        "n": 2.810890,
        "r0": 1.301404,
    },
    "Ba": {
        "q": 1.2,
        "A": 8636.0000,
        "rho": 0.275100,
        "C": 122.900,
        "B": 135.948275,
        "D": -7.513441,
        "n": 2.883345,
        "r0": 1.317647,
    },
    "Ti": {
        "q": 2.4,
        "A": 23707.9090,
        "rho": 0.185580,
        "C": 14.513,
        "B": 207.407471,
        "D": -149.746093,
        "n": 2.446894,
        "r0": 0.749442,
    },
    "Y": {
        "q": 1.8,
        "A": 29526.9770,
        "rho": 0.211377,
        "C": 50.477,
        "B": 310.224807,
        "D": -103.137529,
        "n": 2.564633,
        "r0": 0.897969,
    },
}


def supported_elements() -> set[str]:
    """Return the set of elements supported by the Du/Teter potential."""
    return set(du_teter_potential_params)


def Buckingham(r: float, A: float, rho: float, C: float) -> float:
    """Calculate the Buckingham potential for a given distance r.

    Args:
        r: Distance between atoms.
        A: Pre-exponential factor.
        rho: Characteristic distance.
        C: Dispersion coefficient.

    Returns:
        Buckingham potential value.
    """
    return A * np.exp(-r / rho) - C / (r**6)


def dBuckingham(r: float, A: float, rho: float, C: float) -> float:
    """Calculate the first derivative of the Buckingham potential.

    Args:
        r: Distance between atoms.
        A: Pre-exponential factor.
        rho: Characteristic distance.
        C: Dispersion coefficient.

    Returns:
        First derivative of the Buckingham potential.
    """
    return -A / rho * np.exp(-r / rho) + 6 * C / (r**7)


def ddBuckingham(r: float, A: float, rho: float, C: float) -> float:
    """Calculate the second derivative of the Buckingham potential.

    Args:
        r: Distance between atoms.
        A: Pre-exponential factor.
        rho: Characteristic distance.
        C: Dispersion coefficient.

    Returns:
        Second derivative of the Buckingham potential.
    """
    return A / (rho**2) * np.exp(-r / rho) - 7 * 6 * C / (r**8)


def dddBuckingham(r: float, A: float, rho: float, C: float) -> float:
    """Calculate the third derivative of the Buckingham potential.

    Args:
        r: Distance between atoms.
        A: Pre-exponential factor.
        rho: Characteristic distance.
        C: Dispersion coefficient.

    Returns:
        Third derivative of the Buckingham potential.
    """
    return -A / (rho**3) * np.exp(-r / rho) + 8 * 7 * 6 * C / (r**9)


def V(r: float, B: float, n: float, D: float) -> float:
    """Calculate the short-range repulsion potential of the Du/Teter potential.

    Args:
        r: Distance between atoms.
        B: Pre-exponential factor.
        n: Exponent.
        D: Coefficient.

    Returns:
        Short-range repulsion potential of the Du/Teter potential.
    """
    return B / (r**n) + D * r**2


def dV(r: float, B: float, n: float, D: float) -> float:
    """Calculate the first derivative the short-range repulsion potential of the Du/Teter potential.

    Args:
        r: Distance between atoms.
        B: Pre-exponential factor.
        n: Exponent.
        D: Coefficient.

    Returns:
        First derivative of the short-range repulsion potential of the Du/Teter potential.
    """
    return -n * B / (r ** (n + 1)) + 2 * D * r


def ddV(r: float, B: float, n: float, D: float) -> float:
    """Calculate the second derivative of the short-range repulsion potential of the Du/Teter potential.

    Args:
        r: Distance between atoms.
        B: Pre-exponential factor.
        n: Exponent.
        D: Coefficient.

    Returns:
        Second derivative of the short-range repulsion potential of the Du/Teter potential.
    """
    return n * (n + 1) * B / (r ** (n + 2)) + 2 * D


def Du(r: float, A: float, rho: float, C: float, B: float, n: float, D: float, rc: float) -> float:
    """Calculate the Du/Teter potential for a given distance r.

    Args:
        r: Distance between atoms.
        A: Pre-exponential factor for Buckingham potential.
        rho: Characteristic distance for Buckingham potential.
        C: Dispersion coefficient for Buckingham potential.
        B: Pre-exponential factor for Du/Teter potential.
        n: Exponent for Du/Teter potential.
        D: Coefficient for Du/Teter potential.
        rc: Cutoff distance.

    Returns:
        Du/Teter potential value.
    """
    if r <= rc:
        return V(r, B, n, D)
    return Buckingham(r, A, rho, C)


def dDu(r: float, A: float, rho: float, C: float, B: float, n: float, D: float, rc: float) -> float:
    """Calculate the first derivative of the Du/Teter potential.

    Args:
        r: Distance between atoms.
        A: Pre-exponential factor for Buckingham potential.
        rho: Characteristic distance for Buckingham potential.
        C: Dispersion coefficient for Buckingham potential.
        B: Pre-exponential factor for Du/Teter potential.
        n: Exponent for Du/Teter potential.
        D: Coefficient for Du/Teter potential.
        rc: Cutoff distance.

    Returns:
        First derivative of the Du/Teter potential.
    """
    if r <= rc:
        return -dV(r, B, n, D)
    return -dBuckingham(r, A, rho, C)


def N4_dbx(R: float, K: float = 1) -> float:
    """Calculate the N4 value for the B-O interaction in the Du/Teter potential based on the composition.

    See:
    Dell WJ, Bray PJ, Xiao SZ,
    11B NMR studies and structural modeling of Na2O-B2O3-SiO2 glasses of high soda content.
    J Non Cryst Solids. 1983, 58(1), 1-16.

    Args:
        R: Ratio of alkali oxide to B2O3 molar fractions.
        K: Ratio of SiO2 to B2O3 molar fractions.

    Returns:
        N4 value for the B-O interaction in the Du/Teter potential.
    """
    R_MAX = K / 16 + 0.5
    R_D1 = K / 4 + 0.5
    R_D3 = 2 + K
    R_CUT = 0.5

    if R < R_CUT or R < R_MAX:
        N4 = R
    elif R < R_D1:
        N4 = R_MAX
    elif R < R_D3:
        if K == 0:
            N4 = 1 - R
            N4 = max(N4, 0)
            N4 = min(N4, 1)
        else:
            m1 = ((2 - K / 4) / (K + K / 4)) / (1 + (2 - K / 4) / (K + K / 4)) * (R - R_D1)
            m2 = R - R_D1 - m1
            N4 = (0.5 - K / 16 - 1 / 3 * m1) + (K / 8 - 2 / 15 * m2)
    elif K == 0:
        N4 = 0

    else:
        msg = f"N4 could not be calculated: R={R}, K={K}, R_MAX={R_MAX}"
        raise (ValueError(msg))
    return min(N4, 1)


def get_A_for_BO(K: float, R: float, N4: float) -> float:
    """Calculate the A parameter for the B-O interaction in the Du/Teter potential based on the composition.

    Args:
        K: Ratio of SiO2 to B2O3 molar fractions.
        R: Ratio of alkali oxide to B2O3 molar fractions.
        N4: N4 value.

    Returns:
        A parameter for the B-O interaction in the Du/Teter potential.

    """
    A1 = 11900
    A2 = 12525
    T1 = 4350
    T2 = 85
    R_MAX = K / 16 + 0.5
    T3 = (A1 + T2 * K**2 - A2) / R_MAX + T1

    A = A2 + T3 * N4 * R_MAX / R if R > R_MAX else A1 + T1 * (R_MAX - (R_MAX - N4) ** 2 / R_MAX) + T2 * K**2

    # Maybe we should cap A at some maximum value to avoid numerical issues in the Buckingham potential?
    return min(A, 25000)


def _equations(x: tuple, r: float, buck_params: dict) -> tuple:
    """System of equations for matching Buckingham and short-range potentials at the crossover.

    Args:
        x: Tuple of (B, n, D) parameters to solve for.
        r: Crossover distance.
        buck_params: Dictionary with Buckingham parameters (A, rho, C).

    Returns:
        Tuple of residuals for continuity of value, first, and second derivatives.
    """
    B_val, n, D = x
    if n < 0:
        return (100.0, 100.0, 100.0)
    eq1 = Buckingham(r, **buck_params) - V(r, B_val, n, D)
    eq2 = dBuckingham(r, **buck_params) - dV(r, B_val, n, D)
    eq3 = ddBuckingham(r, **buck_params) - ddV(r, B_val, n, D)
    return (eq1, eq2, eq3)


def fit_BO_params(K: float, R: float, N4: float | None = None) -> dict:
    """Compute composition-dependent B-O interaction parameters.

    The Buckingham A parameter for B-O is determined from the glass composition
    via the Dell-Bray-Xiao model.  The short-range repulsion parameters (B, n, D)
    and crossover distance (r0) are then obtained by enforcing continuity of the
    potential and its first two derivatives at the inflection point of the
    Buckingham third derivative.

    Args:
        K: Ratio of SiO2 to B2O3 molar fractions.
        R: Ratio of modifier oxide to B2O3 molar fractions.
        N4: Fraction of tetrahedral boron.  If ``None``, computed via :func:`N4_dbx`.

    Returns:
        Dictionary with Du/Teter parameters (A, rho, C, B, n, D, r0) for the B-O pair.
    """
    if N4 is None:
        N4 = N4_dbx(R, K)

    buck_params = {
        "A": get_A_for_BO(K, R, N4),
        "rho": du_teter_potential_params["B"]["rho"],
        "C": du_teter_potential_params["B"]["C"],
    }

    # Find crossover distance from inflection point of Buckingham
    rc_solution, _, ier, _ = fsolve(lambda r: dddBuckingham(r, **buck_params), 0.5, full_output=True)
    rc = rc_solution[0]
    if ier != 1:
        msg = "Could not find B-O crossover distance (r0)"
        raise RuntimeError(msg)

    # Find B, n, D by matching V, dV, ddV at rc
    MAX_ATTEMPTS = 100
    attempts = 0
    ier = 0
    start = [30.0, 4.0, -10.0]
    x = np.array(start)
    rng = np.random.default_rng()
    while ier != 1 and attempts < MAX_ATTEMPTS:
        x, _, ier, _ = fsolve(lambda x: _equations(x, rc, buck_params), start, full_output=True)
        attempts += 1
        start = [
            float(rng.integers(1, 100)),
            float(rng.integers(1, 5)),
            float(rng.integers(-20, -1)),
        ]

    if ier != 1:
        msg = "Could not find B-O short-range parameters (B, n, D)"
        raise RuntimeError(msg)

    B_val, n, D = x
    return {
        "A": buck_params["A"],
        "rho": buck_params["rho"],
        "C": buck_params["C"],
        "B": B_val,
        "n": n,
        "D": D,
        "r0": rc,
    }


def get_all_BO_params(structure_dict: dict) -> dict:
    """Compute all B-O interaction parameters for the given composition.

    Args:
        structure_dict: Dictionary containing the structure information,

    Returns:
        Dictionary with Du/Teter parameters for the B-O pair.
    """
    # Handle composition-dependent B-O parameters
    mol_fraction = structure_dict.get("mol_fraction", {})
    cB2O3 = mol_fraction.get("B2O3", 0)
    cSiO2 = mol_fraction.get("SiO2", 0)
    cLi2O = mol_fraction.get("Li2O", 0)
    cNa2O = mol_fraction.get("Na2O", 0)
    cK2O = mol_fraction.get("K2O", 0)
    cBeO = mol_fraction.get("BeO", 0)
    cMgO = mol_fraction.get("MgO", 0)
    cCaO = mol_fraction.get("CaO", 0)
    cSrO = mol_fraction.get("SrO", 0)
    cBaO = mol_fraction.get("BaO", 0)
    cAl2O3 = mol_fraction.get("Al2O3", 0)

    # -------------------------------------------------------------------------------------------
    # Original approach by:
    # Dell WJ, Bray PJ, Xiao SZ. 11B NMR studies and structural modeling of Na2O-B2O3-SiO2
    # glasses of high soda content.
    # J Non Cryst Solids. 1983;58(1):1-16.
    # Note: They only considered R = cNa2O / cB2O3.
    #
    # But there is no publication that unifies this approach to other modifiers.
    # Extending to all alkali makes sense. Probably also to earth-alkali?
    modifier_sum = cLi2O + cNa2O + cK2O + cBeO + cMgO + cCaO + cSrO + cBaO - cAl2O3
    K = cSiO2 / cB2O3
    R = max(modifier_sum / cB2O3, 0)

    bo = fit_BO_params(K, R)
    return {
        "A": bo["A"],
        "rho": bo["rho"],
        "C": bo["C"],
        "B": bo["B"],
        "n": bo["n"],
        "D": bo["D"],
        "rc": bo["r0"],
    }


def write_table_file(
    pair: str,
    params: dict,
    rmin: float = 0.001,
    rmax: float = 11.0,
    npoints: int = 11000,
    output_dir: str | Path = ".",
) -> Path:
    """Write a LAMMPS table file for a given atomic pair.

    Args:
        pair: Name of the atomic pair (e.g. ``"Si-O"``).
        params: Dictionary with keys ``A``, ``rho``, ``C``, ``B``, ``n``, ``D``, ``rc``.
        rmin: Minimum distance for table.
        rmax: Maximum distance for table.
        npoints: Number of points in table.
        output_dir: Directory to save the table file.

    Returns:
        Path to the generated table file.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    r_values = np.linspace(rmin, rmax, num=npoints)

    filename = out_dir / f"table_{pair.replace('-', '_')}.tbl"
    with filename.open("w") as f:
        f.write(f"# Du/Teter potential table for {pair}\n")
        f.write("DU_TETER\n")
        f.write(f"N {npoints}\n\n")
        for i, r in enumerate(r_values, 1):
            energy = Du(r, **params)
            force = dDu(r, **params)
            f.write(f"{i} {r:.10E} {energy:.10E} {force:.10E}\n")

    return filename


def _build_pair_params(elem: str) -> dict:
    """Build Du/Teter pair parameter dict for the X-O interaction of *elem*."""
    p = du_teter_potential_params[elem]
    return {
        "A": p["A"],
        "rho": p["rho"],
        "C": p["C"],
        "B": p["B"],
        "n": p["n"],
        "D": p["D"],
        "rc": p["r0"],
    }


def _build_all_pair_params(species: list[str], structure_dict: dict) -> dict[str, dict]:
    """Build per-pair Du/Teter parameter dicts for all X-O interactions."""
    pair_params: dict[str, dict] = {}
    if "B" in species:
        pair_params["B-O"] = get_all_BO_params(structure_dict)
    for elem in species:
        pair_name = "O-O" if elem == "O" else f"{elem}-O"
        if pair_name not in pair_params:
            pair_params[pair_name] = _build_pair_params(elem)
    return pair_params


def generate_du_teter_potential(structure_dict: dict, output_dir: str = ".", *, melt: bool = True) -> pd.DataFrame:
    """Generate a LAMMPS potential file for the Du/Teter potential.

    For compositions containing B2O3 the B-O interaction parameters are
    determined from the Dell-Bray-Xiao model; all other X-O pairs use the
    tabulated values from :data:`du_teter_potential_params`.

    Args:
        structure_dict: Dictionary containing the structure information,
            including ``"atoms"`` and optionally ``"mol_fraction"``.
        output_dir: Directory to save the generated table files.
        melt: If True, append a Langevin + NVE melt run block (10 000 steps).

    Returns:
        A DataFrame containing the potential configuration.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    types = get_element_types_dict(structure_dict["atoms"])
    species = list(types.keys())

    if "O" not in species:
        msg = "Oxygen must be present in the structure for the Du/Teter potential."
        raise ValueError(msg)

    # Validate that all elements are supported
    unsupported = [elem for elem in species if elem not in du_teter_potential_params]
    if unsupported:
        msg = f"Du/Teter potential does not include parameters for: {unsupported}"
        raise ValueError(msg)

    pair_params = _build_all_pair_params(species, structure_dict)

    # ------------------------------------------------------------------
    # Write table files
    # ------------------------------------------------------------------
    table_files: dict[str, Path] = {}
    for pair_name, params in pair_params.items():
        table_files[pair_name] = write_table_file(pair_name, params, output_dir=out_dir)

    # ------------------------------------------------------------------
    # Build LAMMPS configuration
    # ------------------------------------------------------------------
    rvdw = 11.0
    lines: list[str] = [
        "# Du/Teter potential\n",
        "# Deng & Du, J Am Ceram Soc. 2019, 102, 2482-2505\n",
        "# Kalahe et al., Chem. Mater. 2025, 37, 8595-8613\n",
        "\n",
        "units metal\n",
        "dimension 3\n",
        "atom_style charge\n",
        "\n",
        "### Group Definitions ###\n",
    ]
    lines.extend(f"group {elem} type {types[elem]}\n" for elem in species)

    lines.append("\n### Charges ###\n")
    for elem in species:
        q = du_teter_potential_params[elem]["q"]
        lines.append(f"set type {types[elem]} charge {q}\n")

    lines.append("\n### Du/Teter Potential ###\n")
    lines.append("pair_style hybrid/overlay coul/dsf 0.25 8.0 table spline 11000\n")
    lines.append("pair_coeff * * coul/dsf\n")

    o_type = types["O"]
    for pair_name, table_path in table_files.items():
        abs_path = Path(table_path).resolve()
        elems = pair_name.split("-")
        if pair_name == "O-O":
            lines.append(f'pair_coeff {o_type} {o_type} table "{abs_path}" DU_TETER {rvdw}\n')
        else:
            lines.append(f'pair_coeff {types[elems[0]]} {o_type} table "{abs_path}" DU_TETER {rvdw}\n')

    lines.append("\npair_modify shift yes\n\n")
    lines.append("thermo_style custom step temp pe etotal pxx pxy pxz pyy pyz pzz vol\n")
    lines.append("thermo_modify flush yes\n")
    lines.append("thermo 100\n")

    if melt:
        lines.append("\nfix langevinnve all langevin 5000 5000 0.01 48279\n")
        lines.append("\nfix ensemblenve all nve/limit 0.5\n")
        lines.append("\nrun 10000\n")
        lines.append("\nunfix langevinnve\n")
        lines.append("\nunfix ensemblenve\n")

    return pd.DataFrame(
        {
            "Name": ["Du/Teter"],
            "Filename": [""],
            "Model": ["Du/Teter"],
            "Species": [species],
            "Config": [lines],
        }
    )
