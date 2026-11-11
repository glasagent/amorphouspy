"""Harmonic route workflow: minimisation, dynamical matrix, VDOS and thermodynamics."""

import contextlib
import tempfile
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from amorphouspy.lammps.runner import _is_positive_int, _run_lammps_dynamical_matrix, _validate_minimize_arguments
from amorphouspy.properties.structural.qn import CutoffSpec
from amorphouspy.properties.vibrational.classification import classify_vibrational_groups
from amorphouspy.properties.vibrational.dynamical_matrix import (
    EigenDriver,
    _validate_eigh_options,
    diagonalize_dynamical_matrix,
    read_dynamical_matrix,
)
from amorphouspy.properties.vibrational.harmonic import (
    HarmonicNormalization,
    _validate_broadening_options,
    _validate_frequency_grid,
    compute_harmonic_vdos,
    compute_participation_ratio,
)
from amorphouspy.properties.vibrational.thermodynamics import _check_temperatures, compute_vibrational_thermodynamics

if TYPE_CHECKING:
    from ase import Atoms

# The matrix file inside the directory that harmonic_vdos_simulation makes for it.
_HARMONIC_MATRIX_FILE = "dynamical_matrix.bin"
# Peak of diagonalize_dynamical_matrix with eigenvectors and overwrite_matrix=True, in matrices.
_HARMONIC_PEAK_MATRICES = 2
# Defaults of _run_lammps_minimize that harmonic_vdos_simulation does not expose; used only to validate upfront.
_MINIMIZE_ENERGY_TOLERANCE = 0.0
_MINIMIZE_N_PRINT = 100


def _harmonic_simulation_settings(
    structure: "Atoms",
    potential: pd.DataFrame,
    max_atoms: int,
    displacement: float,
    force_tolerance: float,
    max_iter: int,
    broadening_thz: float,
    frequency_grid_thz: ArrayLike | None,
    normalization: str,
    temperatures_k: ArrayLike | None,
    driver: str,
    imaginary_threshold_thz: float,
    *,
    tmp_working_directory: str | Path | None,
    keep_dynamical_matrix: bool,
) -> None:
    """Validate the arguments of :func:`harmonic_vdos_simulation` before anything runs."""
    if not _is_positive_int(max_atoms):
        msg = f"max_atoms must be a positive integer, got {max_atoms!r}."
        raise ValueError(msg)
    n_atoms = len(structure)
    if n_atoms > max_atoms:
        matrix_gb = 9 * n_atoms**2 * 8 / 1e9
        msg = (
            f"The structure has {n_atoms} atoms, more than max_atoms={max_atoms}. Its dynamical matrix takes "
            f"9 N² x 8 bytes = {matrix_gb:.3g} GB, and the diagonalisation with eigenvectors peaks at "
            f"{_HARMONIC_PEAK_MATRICES} matrices, {_HARMONIC_PEAK_MATRICES * matrix_gb:.3g} GB. Raise max_atoms "
            "deliberately to run it anyway."
        )
        raise ValueError(msg)
    if potential.empty:
        msg = "No matching potential found for the given configuration."
        raise ValueError(msg)
    if not np.isfinite(displacement) or displacement <= 0:
        msg = f"displacement must be positive and finite (Å), got {displacement!r}."
        raise ValueError(msg)
    _validate_minimize_arguments(force_tolerance, _MINIMIZE_ENERGY_TOLERANCE, max_iter, "fire", None, _MINIMIZE_N_PRINT)
    _validate_broadening_options(broadening_thz, normalization)
    if frequency_grid_thz is not None:
        _validate_frequency_grid(frequency_grid_thz)
    if temperatures_k is not None:
        _check_temperatures(np.array(temperatures_k, dtype=np.float64, ndmin=1))
    _validate_eigh_options(driver, imaginary_threshold_thz)
    if keep_dynamical_matrix and tmp_working_directory is None:
        msg = "keep_dynamical_matrix=True needs a tmp_working_directory; without one every run removes its files."
        raise ValueError(msg)


def harmonic_vdos_simulation(
    structure: "Atoms",
    potential: pd.DataFrame,
    server_kwargs: dict[str, Any] | None = None,
    *,
    max_atoms: int = 10_000,
    displacement: float = 1e-5,
    force_tolerance: float = 1e-6,
    max_iter: int = 100_000,
    broadening_thz: float = 0.2,
    frequency_grid_thz: ArrayLike | None = None,
    normalization: HarmonicNormalization = "unit",
    temperatures_k: ArrayLike | None = None,
    cutoff: CutoffSpec | None = None,
    former_types: list[int] | None = None,
    o_type: int = 8,
    driver: EigenDriver = "evr",
    imaginary_threshold_thz: float = 0.01,
    tmp_working_directory: str | Path | None = None,
    keep_dynamical_matrix: bool = False,
    keep_eigenvectors: bool = False,
) -> dict[str, Any]:
    r"""Minimise a structure, compute its finite-difference dynamical matrix and return the harmonic VDOS and modes.

    The harmonic counterpart of :func:`vdos_simulation`: the VDOS of the 0 K inherent structure from its normal modes,
    with no MD and no statistical noise. Steps:

    1. Minimisation at fixed cell and the dynamical matrix, both in LAMMPS (``_run_lammps_dynamical_matrix``).
    2. The atom groups (element, oxygen class, $\mathrm{Q}^n$) with :func:`classify_vibrational_groups` and the
       masses, both from the minimised structure.
    3. The matrix is read with :func:`read_dynamical_matrix` and diagonalised in place with
       :func:`diagonalize_dynamical_matrix`, which drops the three translations.
    4. The broadened VDOS and its group partials with :func:`compute_harmonic_vdos`, the participation ratio of every
       mode with :func:`compute_participation_ratio` and, with ``temperatures_k``, the harmonic thermodynamics with
       :func:`compute_vibrational_thermodynamics`.

    The structure is used as given. Preparing it (melt-quench, NPT relaxation to the target pressure) is the caller's
    job; this function adds only the minimisation. The minimisation uses FIRE at fixed cell
    (``min_style="fire"``, no ``box/relax``) down to ``force_tolerance``, the LAMMPS ``ftol``: the 2-norm of the global
    force vector in eV/Å, not a per-atom maximum. A fixed cell leaves a residual pressure, reported in
    ``"minimization"`` (``"pressure_gpa"`` and ``"mean_pressure_gpa"``; for FIRE they also hold the kinetic term of
    the residual velocities), together with the iteration count, the stopping criterion and the final forces. If the
    tolerance is not reached, the minimiser warns and the result is still returned; the step-0 forces of the
    structure that was differentiated are in ``"dynamical_matrix"`` (``"max_force_ev_per_a"``).

    Finite differences: each entry of the matrix is a central difference of the forces with step ``displacement``.

    Imaginary modes are reported as negative frequencies in ``"mode_frequencies_thz"``. Those below
    ``-imaginary_threshold_thz`` are counted in ``"diagonalization"`` (``"n_imaginary"``, with a warning that names
    the asymmetry atoms); they point at an incomplete minimisation or a corrupted entry. Every negative frequency is
    left out of the VDOS (``"n_imaginary_excluded"`` and ``"imaginary_weight"``) and of the thermodynamics.

    ``max_atoms`` guards against an accidental large run: above it the function raises before anything runs, and the
    message gives the matrix size and the expected peak. It can be raised deliberately.

    The groups are classified on the minimised structure, the one whose modes are computed. Errors in ``cutoff``,
    ``former_types`` or ``o_type`` therefore surface only after the LAMMPS runs.

    The thermodynamics is computed from the broadened total VDOS on the frequency grid, not from a sum over the modes,
    with ``lowest_frequency_thz`` set to the smallest positive mode frequency, the lowest mode the box supports: the
    values below its temperature are flagged as unreliable. With ``normalization="3N"`` the total is divided by 3N
    first, as :func:`compute_vibrational_thermodynamics` needs a VDOS normalised to 1. On the 300-atom PMMCS glass at
    300 K with the default 0.2 THz broadening, the values differ from an exact sum over the modes by -1.3e-5 (C_V),
    8.5e-4 (S), 3.0e-5 (U) and -3.7e-4 (F), relative. The default grid starts at 0, so modes within a few
    ``broadening_thz`` of 0 lose part of their weight (``"excluded_weight"`` and ``"vdos_integral"`` report it).

    Disk and memory: the matrix file takes 9 N² x 8 bytes (2.6 GB at 6000 atoms, 7.2 GB at 10,000) and is deleted once
    it is read unless ``keep_dynamical_matrix``. The peak in memory is still 2 matrices, during the diagonalisation.
    Without the eigenvectors, the result holds only O(M x number of labels) floats (the mode frequencies, participation
    ratios and group weights) plus the VDOS curves on the frequency grid. An eigenvector-based analysis later needs
    ``keep_eigenvectors=True`` or a rerun.

    Args:
        structure: The structure, e.g. a quenched and NPT-relaxed glass, used as given. It is not modified.
        potential: LAMMPS potential as a pandas DataFrame.
        server_kwargs: Additional arguments for the LAMMPS server (e.g. ``{"cores": 4}``), used by both LAMMPS runs.
        max_atoms: Largest number of atoms accepted, a positive integer.
        displacement: Finite-difference displacement in Å.
        force_tolerance: FIRE stops when the 2-norm of the global force vector is below this, in eV/Å.
        max_iter: Maximum number of FIRE iterations.
        broadening_thz: Standard deviation of the Gaussian broadening of each mode in THz (not the FWHM).
        frequency_grid_thz: Frequencies in THz at which the VDOS is evaluated, 1-D, finite and strictly increasing.
            ``None``: from 0 in steps of ``broadening_thz / 5`` up to the highest mode plus 5 ``broadening_thz``.
        normalization: ``"unit"`` (the total VDOS integrates to 1, in 1/THz) or ``"3N"`` (to ``3 * n_atoms``), each
            short by the share of imaginary modes.
        temperatures_k: Temperatures in K for the harmonic thermodynamics, a scalar or a 1-D array of finite values
            >= 0. ``None`` skips it.
        cutoff: Former-O cutoff in Å for the groups, as in :func:`classify_vibrational_groups`. Derived from the RDF
            of the minimised structure if ``None``.
        former_types: Atomic numbers of the network formers. Derived from the composition if ``None``.
        o_type: Atomic number of oxygen.
        driver: LAPACK driver of ``scipy.linalg.eigh``: ``"evr"`` (default), ``"evd"``, ``"evx"`` or ``"ev"``; see
            :func:`diagonalize_dynamical_matrix` for their memory and time.
        imaginary_threshold_thz: A mode below ``-imaginary_threshold_thz`` counts as imaginary in
            ``"diagonalization"``.
        tmp_working_directory: Directory for the LAMMPS runs and the matrix file. The run directories, with their
            logs, input and data files, are kept. The matrix file goes to its own ``harmonic_vdos_*`` subdirectory,
            which is removed once the matrix is read unless ``keep_dynamical_matrix``; nothing else in the directory is
            removed. If None, every run and the matrix file are removed.
        keep_dynamical_matrix: Keep the matrix file, in ``harmonic_vdos_*/dynamical_matrix.bin`` under
            ``tmp_working_directory``, instead of deleting it once read. Needs a ``tmp_working_directory``.
        keep_eigenvectors: Return the eigenvectors in ``"eigenvectors"``, one matrix in memory, instead of dropping
            them once the VDOS, the group partials and the participation ratios are computed.

    Returns:
        A dict with:
            ``"frequencies_thz"``: the frequency grid in THz.
            ``"vdos"``: ``{"total": array, "element": {...}, "oxygen": {...}, "qn": {...}}`` with the labels of
            ``"groups"``, in 1/THz, the layout of :func:`vdos_simulation`.
            ``"groups"``: the output of :func:`classify_vibrational_groups` on the minimised structure.
            ``"structure"``: the minimised structure.
            ``"normalization"``: the VDOS normalisation.
            ``"broadening_thz"``: the broadening in THz.
            ``"mode_frequencies_thz"``: the $M = 3N - 3$ mode frequencies in THz, ascending, negative for imaginary
            modes.
            ``"participation_ratio"``: the participation ratio of every mode, shape $(M,)$.
            ``"mode_weights"``: ``{grouping: {label: w}}``, each group's $(M,)$ share of every mode, as in
            :func:`compute_harmonic_vdos`.
            ``"n_imaginary_excluded"``: the number of modes with a negative frequency, left out of the VDOS.
            ``"imaginary_weight"``: their share of the normalisation.
            ``"grid_integral"``: the trapezoid integral of the total VDOS on the grid.
            ``"eigenvectors"``: the $(3N, M)$ mass-weighted eigenvectors, one mode per column; None unless
            ``keep_eigenvectors``.
            ``"dynamical_matrix"``: the summary of the LAMMPS dynamical-matrix run (``"matrix_path"``,
            ``"matrix_bytes"``, ``"n_atoms"``, ``"displacement_a"``, ``"style"``, the step-0 ``"max_force_ev_per_a"``
            and ``"force_norm_ev_per_a"``, ``"minimization"`` and ``"wall_time_s"``). ``"matrix_path"`` is the
            absolute path of the kept file; None unless ``keep_dynamical_matrix``.
            ``"minimization"``: the FIRE summary: ``"converged"``, ``"stopping_criterion"``, ``"n_iterations"``,
            ``"max_force_ev_per_a"``, ``"force_norm_ev_per_a"``, ``"pressure_gpa"``, ``"mean_pressure_gpa"``, the
            energies and volumes.
            ``"diagonalization"``: ``"asymmetry"``, ``"asymmetry_atoms"``, ``"translational_frequencies_thz"``,
            ``"translational_overlap"``, ``"n_imaginary"``, ``"min_frequency_thz"`` and ``"driver"``.
            ``"thermodynamics"``: the output of :func:`compute_vibrational_thermodynamics`, or None without
            ``temperatures_k`` or when no mode has a positive frequency (with a warning).
            ``"displacement_a"``: the finite-difference displacement in Å.
            ``"wall_time_s"``: ``{"lammps", "diagonalization", "analysis"}`` in s: both LAMMPS runs; reading and
            diagonalising the matrix; the groups, VDOS, participation ratios and thermodynamics.

    Raises:
        ValueError: Before any LAMMPS run, if ``max_atoms`` is not a positive integer, the structure has more than
            ``max_atoms`` atoms, the potential is empty, ``displacement`` is not positive and finite,
            ``force_tolerance`` is negative, zero or not finite, ``max_iter`` is not a positive integer,
            ``broadening_thz`` is not positive and finite, ``normalization`` is unknown, ``frequency_grid_thz`` is not
            1-D, finite and strictly increasing with at least 2 points, ``temperatures_k`` is not finite and >= 0,
            ``driver`` is unknown, ``imaginary_threshold_thz`` is not >= 0, or ``keep_dynamical_matrix`` is True
            without a ``tmp_working_directory``. After the LAMMPS runs, if ``cutoff``, ``former_types`` or ``o_type``
            is invalid for :func:`classify_vibrational_groups`.
        RuntimeError: If a LAMMPS run fails or the matrix file is missing or has the wrong size.

    Warns:
        UserWarning: If the minimisation stops on a criterion other than the force or energy tolerance, the
            diagonalisation finds imaginary modes or dropped modes that are not translations, a caller's
            ``frequency_grid_thz`` is coarser than ``broadening_thz / 2``, or ``temperatures_k`` is given but no
            mode has a positive frequency.

    Example:
        ```pycon
        >>> out = harmonic_vdos_simulation(glass, potential, temperatures_k=[300.0])  # doctest: +SKIP
        >>> out["minimization"]["converged"], out["diagonalization"]["asymmetry"]  # doctest: +SKIP
        >>> out["diagonalization"]["n_imaginary"], out["grid_integral"]  # doctest: +SKIP
        >>> out["vdos"]["oxygen"]["O_BO"]  # doctest: +SKIP
        >>> out["thermodynamics"]["heat_capacity_kb_per_atom"]  # doctest: +SKIP

        ```
    """
    _harmonic_simulation_settings(
        structure,
        potential,
        max_atoms,
        displacement,
        force_tolerance,
        max_iter,
        broadening_thz,
        frequency_grid_thz,
        normalization,
        temperatures_k,
        driver,
        imaginary_threshold_thz,
        tmp_working_directory=tmp_working_directory,
        keep_dynamical_matrix=keep_dynamical_matrix,
    )
    with (
        contextlib.nullcontext(tempfile.mkdtemp(prefix="harmonic_vdos_", dir=tmp_working_directory))
        if keep_dynamical_matrix
        else tempfile.TemporaryDirectory(prefix="harmonic_vdos_", dir=tmp_working_directory)
    ) as matrix_dir:
        start = time.perf_counter()
        minimised, summary = _run_lammps_dynamical_matrix(
            structure,
            potential,
            Path(matrix_dir) / _HARMONIC_MATRIX_FILE,
            displacement=displacement,
            minimize=True,
            minimize_kwargs={
                "min_style": "fire",
                "pressure": None,
                "force_tolerance": force_tolerance,
                "max_iter": max_iter,
            },
            server_kwargs=server_kwargs,
            tmp_working_directory=tmp_working_directory,
        )
        lammps_s = time.perf_counter() - start
        start = time.perf_counter()
        groups = classify_vibrational_groups(minimised, cutoff, former_types, o_type)
        masses = minimised.get_masses()
        groups_s = time.perf_counter() - start
        start = time.perf_counter()
        matrix = read_dynamical_matrix(summary["matrix_path"], summary["n_atoms"])
    if not keep_dynamical_matrix:
        summary = {**summary, "matrix_path": None}
    # Diagonalised in place; dropping the only reference afterwards frees the overwritten matrix, so the peak is the
    # matrix plus its eigenvectors and only the eigenvectors stay.
    modes = diagonalize_dynamical_matrix(
        matrix,
        masses=masses,
        overwrite_matrix=True,
        eigenvectors=True,
        driver=driver,
        imaginary_threshold_thz=imaginary_threshold_thz,
    )
    del matrix
    diagonalization_s = time.perf_counter() - start

    start = time.perf_counter()
    mode_frequencies, eigenvectors = modes["frequencies_thz"], modes.pop("eigenvectors")
    vdos = compute_harmonic_vdos(
        mode_frequencies,
        eigenvectors,
        groups=groups,
        frequency_grid_thz=frequency_grid_thz,
        broadening_thz=broadening_thz,
        normalization=normalization,
    )
    participation_ratio = compute_participation_ratio(eigenvectors, masses)
    if not keep_eigenvectors:
        # The last reference to the eigenvector buffer: dropping it frees one matrix before the thermodynamics.
        eigenvectors = None
    thermodynamics = None
    positive_modes = mode_frequencies[mode_frequencies > 0]
    if temperatures_k is not None and positive_modes.size == 0:
        warnings.warn(
            "No mode has a positive frequency, so the thermodynamics are not computed ('thermodynamics' is None); "
            f"the lowest mode is {modes['min_frequency_thz']:.4g} THz. Check the minimisation and the asymmetry.",
            UserWarning,
            stacklevel=2,
        )
    elif temperatures_k is not None:
        total = vdos["vdos"]["total"] if normalization == "unit" else vdos["vdos"]["total"] / (3 * len(minimised))
        thermodynamics = compute_vibrational_thermodynamics(
            vdos["frequencies_thz"],
            total,
            temperatures_k,
            lowest_frequency_thz=float(positive_modes.min()),
        )
    analysis_s = groups_s + time.perf_counter() - start

    return {
        "frequencies_thz": vdos["frequencies_thz"],
        "vdos": vdos["vdos"],
        "groups": groups,
        "structure": minimised,
        "normalization": normalization,
        "broadening_thz": vdos["broadening_thz"],
        "mode_frequencies_thz": mode_frequencies,
        "participation_ratio": participation_ratio,
        "mode_weights": vdos["mode_weights"],
        "n_imaginary_excluded": vdos["n_imaginary_excluded"],
        "imaginary_weight": vdos["imaginary_weight"],
        "grid_integral": vdos["grid_integral"],
        "eigenvectors": eigenvectors,
        "dynamical_matrix": summary,
        "minimization": summary["minimization"],
        "diagonalization": {
            key: modes[key]
            for key in (
                "asymmetry",
                "asymmetry_atoms",
                "translational_frequencies_thz",
                "translational_overlap",
                "n_imaginary",
                "min_frequency_thz",
                "driver",
            )
        },
        "thermodynamics": thermodynamics,
        "displacement_a": summary["displacement_a"],
        "wall_time_s": {"lammps": lammps_s, "diagonalization": diagonalization_s, "analysis": analysis_s},
    }
