"""Harmonic route: reading and diagonalising the LAMMPS dynamical matrix."""

import math
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray
from scipy.constants import atomic_mass, e

from amorphouspy.properties.vibrational import _common
from amorphouspy.properties.vibrational._common import _MATRIX_NDIM, _N_TRANSLATIONS

DynamicalMatrixStyle = Literal["regular", "eskm"]

# LAMMPS writes ``eskm`` as ``regular`` times this literal (measured: ratio 9648.5 to 2e-16 on LAMMPS 29 Aug 2024).
# The CODATA value e N_A / 10 = 9648.533 is 3.4e-6 higher; dividing by it would not recover the ``regular`` matrix.
_ESKM_PER_REGULAR = 9648.5
_DYNAMICAL_MATRIX_STYLES = ("regular", "eskm")


def _check_value_count(count: int, n_atoms: int, source: str) -> None:
    """Raise if ``count`` values cannot form the (3 n_atoms, 3 n_atoms) matrix; name the n_atoms they imply."""
    expected = 9 * n_atoms**2
    if count == expected:
        return
    msg = f"{source} holds {count} values, expected 9 N² = {expected} for n_atoms={n_atoms}"
    side = math.isqrt(count // 9)
    if count and count % 9 == 0 and side**2 == count // 9:
        msg += f"; {count} values imply n_atoms={side}"
    raise ValueError(msg + ".")


def read_dynamical_matrix(
    path: str | Path,
    n_atoms: int,
    *,
    binary: bool = True,
    style: DynamicalMatrixStyle = "regular",
) -> NDArray[np.float64]:
    r"""Read a dynamical-matrix file written by the LAMMPS ``dynamical_matrix`` command.

    The dynamical matrix is the mass-weighted Hessian

    $$D_{i\alpha, j\beta} = \frac{\Phi_{i\alpha, j\beta}}{\sqrt{m_i m_j}}, \quad
    \Phi_{i\alpha, j\beta} = \frac{\partial^2 E}{\partial r_{i\alpha}\, \partial r_{j\beta}},$$

    with $E$ the potential energy, $r_{i\alpha}$ the coordinates and $m_i$ the masses. It is returned in eV/Å²/amu
    (LAMMPS ``regular`` in metal units). An eigenvalue $\lambda$ of $D$ gives the angular frequency
    $\omega = \sqrt{\lambda}$ and the frequency $\nu = \sqrt{\lambda} / (2\pi)$, where
    $\sqrt{1\ \mathrm{eV/Å^2/amu}}$ corresponds to $\nu = 15.633$ THz. Row and column $3i + \alpha$ belong to atom
    $i$ (0-based, the LAMMPS atom id minus one) and $\alpha \in \{x, y, z\}$ ($0, 1, 2$), so the atom order is the
    LAMMPS id order. For files written by ``_run_lammps_dynamical_matrix`` that is the order of the atoms in the
    structure.

    The matrix is returned as written, NOT symmetrised. A finite-difference matrix is symmetric only to the
    difference error, and a larger asymmetry $\max|D - D^\mathsf{T}| / \max|D|$ flags corrupted entries, for example
    a pair that crosses a cutoff with a force jump during a displacement. The asymmetry is therefore left for the
    caller to measure before symmetrising.

    File formats (verified on LAMMPS 29 Aug 2024):

    - ``binary=True`` (``binary yes``): raw float64 in native byte order, no header, exactly $9 N^2 \times 8$ bytes.
      It is read with ``np.fromfile`` and reshaped without a copy.
    - ``binary=False`` (``binary no``): three values per line in ``%.8f``, one line per atom of a row, so each matrix
      row spans $N$ lines. Eight fixed decimals limit the precision to $5 \times 10^{-9}$ eV/Å²/amu absolute. The
      text is parsed with ``np.loadtxt``, which is about six times faster than ``np.fromfile(sep=" ")`` here.

    With ``style="eskm"`` the file holds the matrix in 10 J/mol/Å²/(g/mol); it is divided in place by 9648.5, the
    factor LAMMPS uses (the CODATA value $e N_\mathrm{A} / 10 = 9648.533$ differs by $3.4 \times 10^{-6}$), and
    returned in eV/Å²/amu.

    Memory: the result is one array of $9 N^2 \times 8$ bytes, 2.6 GB at 6000 atoms and 7.2 GB at 10,000. A binary
    read needs nothing more. Parsing text peaks at about 2.2 times the matrix (numpy allocations measured at 1000 and
    2000 atoms), about 16 GB at 10,000 atoms. Non-finite values are checked in blocks of rows, so the check adds at
    most ``_CHUNK_BYTES / 8`` bytes.

    Args:
        path: The dynamical-matrix file.
        n_atoms: Number of atoms $N$ in the structure the matrix was computed on.
        binary: True for a file written with ``binary yes``, False for ``binary no``.
        style: The LAMMPS style the file was written with, ``"regular"`` or ``"eskm"``.

    Returns:
        The $(3N, 3N)$ dynamical matrix in eV/Å²/amu, float64, C-contiguous and writable.

    Raises:
        ValueError: If ``n_atoms`` is not a positive integer, ``style`` is unknown, the file does not hold $9 N^2$
            values (the size is checked before a binary file is read), or the matrix contains NaN or inf.
        FileNotFoundError: If ``path`` does not exist.

    Example:
        ```pycon
        >>> minimised, summary = _run_lammps_dynamical_matrix(glass, potential, "dynmat.bin")  # doctest: +SKIP
        >>> matrix = read_dynamical_matrix(summary["matrix_path"], summary["n_atoms"])  # doctest: +SKIP
        >>> asymmetry = np.abs(matrix - matrix.T).max() / np.abs(matrix).max()  # doctest: +SKIP
        ```
    """
    if isinstance(n_atoms, bool) or not isinstance(n_atoms, int | np.integer) or n_atoms < 1:
        msg = f"n_atoms must be a positive integer, got {n_atoms!r}."
        raise ValueError(msg)
    if style not in _DYNAMICAL_MATRIX_STYLES:
        msg = f"Unknown style {style!r}; allowed styles are {list(_DYNAMICAL_MATRIX_STYLES)}."
        raise ValueError(msg)
    n_dof = 3 * int(n_atoms)
    if binary:
        n_bytes = Path(path).stat().st_size
        itemsize = np.dtype(np.float64).itemsize
        if n_bytes % itemsize:
            msg = f"The binary file {path} has {n_bytes} bytes, not a whole number of float64 values."
            raise ValueError(msg)
        _check_value_count(n_bytes // itemsize, n_atoms, f"The binary file {path}")
        values = np.fromfile(path, dtype=np.float64)
    else:
        values = np.loadtxt(path, dtype=np.float64).ravel()
        _check_value_count(values.size, n_atoms, f"The text file {path}")
    matrix = values.reshape(n_dof, n_dof)
    rows_per_chunk = max(1, _common._CHUNK_BYTES // (n_dof * matrix.itemsize))  # noqa: SLF001
    for start in range(0, n_dof, rows_per_chunk):
        if not np.isfinite(matrix[start : start + rows_per_chunk]).all():
            msg = (
                f"The dynamical matrix in {path} contains NaN or inf "
                f"in rows {start} to {min(start + rows_per_chunk, n_dof) - 1}."
            )
            raise ValueError(msg)
    if style == "eskm":
        matrix /= _ESKM_PER_REGULAR
    return matrix


# Frequency in THz of an eigenvalue of 1 eV/Å²/amu: sqrt(e / (1e-20 atomic_mass)) / (2 pi) / 1e12 = 15.633.
_THZ_PER_SQRT_EV_PER_A2_AMU = math.sqrt(e / (1e-20 * atomic_mass)) / (2 * math.pi) / 1e12
EigenDriver = Literal["ev", "evd", "evr", "evx"]
_EIGH_DRIVERS = ("ev", "evd", "evr", "evx")
# Smallest squared projection of a dropped mode on the rigid translations that counts as a translation.
_MIN_TRANSLATIONAL_OVERLAP = 0.9


def _eigenvalues_to_thz(eigenvalues: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * _THZ_PER_SQRT_EV_PER_A2_AMU


def _validate_dynamical_matrix_input(
    matrix: NDArray,
    masses: ArrayLike | None,
    *,
    overwrite_matrix: bool,
    eigenvectors: bool,
    driver: str,
    imaginary_threshold_thz: float,
) -> NDArray[np.float64] | None:
    """Check everything except finiteness; return the masses as float64 (or None)."""
    if matrix.ndim != _MATRIX_NDIM or matrix.shape[0] != matrix.shape[1]:
        msg = f"matrix must be a square 2-D array, got shape {matrix.shape}."
        raise ValueError(msg)
    n_dof = matrix.shape[0]
    if n_dof % 3 or n_dof < 2 * 3:
        msg = f"matrix must be (3N, 3N) with N >= 2 atoms, got size {n_dof}."
        raise ValueError(msg)
    if matrix.dtype != np.float64:
        msg = f"matrix must be float64 (a cast would copy it), got {matrix.dtype}."
        raise ValueError(msg)
    if overwrite_matrix and not (matrix.flags.writeable and (matrix.flags.c_contiguous or matrix.flags.f_contiguous)):
        msg = "overwrite_matrix=True needs a writable C- or F-contiguous matrix; otherwise it would be copied."
        raise ValueError(msg)
    _validate_eigh_options(driver, imaginary_threshold_thz)
    if masses is None:
        return None
    if not eigenvectors:
        msg = "masses are only used for the translational overlap, which needs eigenvectors=True."
        raise ValueError(msg)
    masses = np.asarray(masses, dtype=np.float64)
    if masses.shape != (n_dof // 3,) or not np.all(np.isfinite(masses) & (masses > 0)):
        msg = f"masses must be {n_dof // 3} positive finite values, one per atom, got shape {masses.shape}."
        raise ValueError(msg)
    return masses


def _validate_eigh_options(driver: str, imaginary_threshold_thz: float) -> None:
    """Raise for an unknown LAPACK driver or a threshold that is not >= 0."""
    if driver not in _EIGH_DRIVERS:
        msg = f"Unknown driver {driver!r}; allowed drivers are {list(_EIGH_DRIVERS)}."
        raise ValueError(msg)
    if not imaginary_threshold_thz >= 0:
        msg = f"imaginary_threshold_thz must be >= 0, got {imaginary_threshold_thz!r}."
        raise ValueError(msg)


def _row_blocks(n_dof: int) -> range:
    return range(0, n_dof, max(1, _common._CHUNK_BYTES // (n_dof * np.dtype(np.float64).itemsize)))  # noqa: SLF001


def _asymmetry(matrix: NDArray[np.float64]) -> tuple[float, tuple[int, int] | None]:
    """Relative asymmetry max|D - D^T| / max|D| and the atom pair of the worst entry; raise on NaN or inf.

    Each block of rows [s, e) is compared over the columns from s on with the matching columns, so every entry and
    every (r, c) pair is seen (the diagonal block twice) and no temporary exceeds one block.
    """
    n_dof = matrix.shape[0]
    rows = _row_blocks(n_dof)
    max_abs, max_diff, worst = 0.0, 0.0, (0, 0)
    for start in rows:
        stop = min(start + rows.step, n_dof)
        top, left = matrix[start:stop, start:], matrix[start:, start:stop].T
        block_max = (float(np.abs(top).max()), float(np.abs(left).max()))  # NaN and inf propagate through max
        if not all(math.isfinite(value) for value in block_max):
            msg = f"The dynamical matrix contains NaN or inf in rows or columns {start} to {stop - 1}."
            raise ValueError(msg)
        max_abs = max(max_abs, *block_max)
        diff = top - left
        np.abs(diff, out=diff)
        flat = int(diff.argmax())
        if diff.flat[flat] > max_diff:
            max_diff = float(diff.flat[flat])
            row, column = divmod(flat, diff.shape[1])
            worst = (start + row, start + column)
    if not max_diff:
        return 0.0, None
    first, second = sorted((worst[0] // 3, worst[1] // 3))
    return max_diff / max_abs, (first, second)


def _symmetrise_in_place(work: NDArray[np.float64]) -> None:
    """Replace ``work`` by (work + work^T) / 2 block by block, without a full-size temporary.

    The block of rows [s, e) writes rows [s, e) and columns [s, e) from s on; later blocks only touch rows and
    columns from e on, so no entry is read after it was written. The diagonal block of the average is symmetric, so
    its two writes agree.
    """
    n_dof = work.shape[0]
    rows = _row_blocks(n_dof)
    for start in rows:
        stop = min(start + rows.step, n_dof)
        average = work[start:stop, start:] + work[start:, start:stop].T
        average *= 0.5
        work[start:stop, start:] = average
        work[start:, start:stop] = average.T


def diagonalize_dynamical_matrix(
    matrix: NDArray[np.float64],
    *,
    masses: ArrayLike | None = None,
    overwrite_matrix: bool = False,
    eigenvectors: bool = True,
    driver: EigenDriver = "evr",
    imaginary_threshold_thz: float = 0.01,
) -> dict[str, Any]:
    r"""Symmetrise and diagonalise a mass-weighted dynamical matrix and drop the three translational modes.

    The input is the dynamical matrix of :func:`read_dynamical_matrix`,

    $$D_{i\alpha, j\beta} = \frac{\Phi_{i\alpha, j\beta}}{\sqrt{m_i m_j}}$$

    in eV/Å²/amu, $(3N, 3N)$, row and column $3i + \alpha$ for atom $i$ and direction $\alpha$. An eigenvalue
    $\lambda$ gives the frequency $\nu = \sqrt{\lambda} / (2\pi)$, which is
    $\nu = 15.633\ \mathrm{THz} \times \sqrt{\lambda / (\mathrm{eV/Å^2/amu})}$ (the factor is
    $\sqrt{e / (10^{-20}\, m_\mathrm{u})} / (2\pi) / 10^{12}$ from ``scipy.constants``). An imaginary mode,
    $\lambda < 0$, is reported as a negative frequency $\nu = -\sqrt{|\lambda|} / (2\pi)$.

    Steps:

    1. Asymmetry of the raw matrix, $\max|D - D^\mathsf{T}| / \max|D|$, and the atom pair $(i, j)$, $i \le j$, of the
       worst entry. A finite-difference matrix is symmetric only to the difference error. A pair that crosses a cutoff
       where the force jumps during a displacement $h$ (an energy-shifted pair style such as the PMMCS ``pedone`` term
       at 5.5 Å) gets a spurious term of about jump / (2h) in one entry but not in its transpose, so a large asymmetry
       points at such corrupted entries, and the atom pair tells where. It is measured before symmetrising, in blocks
       of rows, together with the check for NaN and inf.
    2. Symmetrisation to $(D + D^\mathsf{T}) / 2$ in place, in the same blocks. With ``overwrite_matrix=False`` this
       works on one copy of the input; with True on the input itself.
    3. Diagonalisation with ``scipy.linalg.eigh(..., overwrite_a=True, driver=driver)``. This deliberately replaces
       the ``numpy.linalg.eigh`` of the original plan: numpy always copies the matrix and uses the divide-and-conquer
       driver, which with eigenvectors needs four matrices besides the input, against one for ``"evr"`` (see Memory).
    4. The three translational modes are dropped: the three eigenvalues of smallest $|\lambda|$, which are three
       consecutive ones of the ascending spectrum. With ``masses``, each dropped mode's squared projection on the span
       of the mass-weighted rigid translations $t_\alpha = \sqrt{m_i}\, e_\alpha / \sqrt{\sum_i m_i}$ is reported; it
       is 1 for a pure translation. A value below 0.9 warns: a soft mode is mixed in, or the translations are not the
       smallest $|\lambda|$ (a sum-rule violation that lifts them, or an isolated cluster, whose rotations are zero
       modes too).
    5. Kept frequencies below ``-imaginary_threshold_thz`` count as imaginary and warn, with the asymmetry atoms.

    Memory (benchmark: a 3000-atom spring network, n = 9000, OpenBLAS 0.3.34 on 12 Apple M-series cores; peak
    allocations, confirmed by the peak process RSS; scaled as $n^2$ to $N$ = 6000 and 10,000 atoms, where one matrix
    is 2.6 GB and 7.2 GB). Totals include the input matrix; ``overwrite_matrix=False`` adds one copy:

    - eigenvalues only, any driver: 1 matrix, 2.6 / 7.2 GB (2 matrices, 5.2 / 14.4 GB without overwrite).
    - ``"evr"`` with eigenvectors: 2 matrices (the eigenvectors are a new array), 5.2 / 14.4 GB (7.8 / 21.6 GB).
    - ``"evx"`` with eigenvectors: the same as ``"evr"``.
    - ``"evd"`` with eigenvectors: 3 matrices (the eigenvectors overwrite the input, plus two of workspace),
      7.8 / 21.6 GB (10.4 / 28.8 GB).
    - ``"ev"`` with eigenvectors: 1 matrix (the eigenvectors overwrite the input), 2.6 / 7.2 GB (5.2 / 14.4 GB).
    - For comparison, ``numpy.linalg.eigh``: 5 matrices, 13 / 36 GB.

    The asymmetry and symmetrisation passes add up to about 0.2 GB of row blocks (``_CHUNK_BYTES`` each), independent
    of $N$, and LAPACK $O(n)$ workspace.

    Time at n = 9000, with eigenvectors (scaled as $n^3$, by 8 and 37, to 6000 and 10,000 atoms): ``"evr"`` 33 s
    (4.4 min, 20 min), ``"evd"`` 27 s (3.6 min, 16 min; 66 s in a first, disturbed run), ``"evx"`` 316 s (42 min,
    3.2 h), ``"ev"`` 339 s (45 min, 3.5 h). Eigenvalues only: about 10 s with every driver (1.3 min, 6 min).
    ``"evr"`` is the default: within about 20 % of the fastest, ``"evd"``, at one matrix less, and memory is the limit
    at 10,000 atoms.

    With ``overwrite_matrix=True`` the input's contents are undefined afterwards (LAPACK overwrites it); with
    ``"ev"`` or ``"evd"`` the returned eigenvectors share its memory. Only a writable C- or F-contiguous array can
    be used without a copy, which is checked.

    Args:
        matrix: The $(3N, 3N)$ mass-weighted dynamical matrix in eV/Å²/amu, float64 (no cast is made, since a cast
            would copy it). Not symmetrised.
        masses: Atomic masses in amu, shape $(N,)$, in the order of the matrix rows. Only used for the
            translational overlap, so it needs ``eigenvectors=True``.
        overwrite_matrix: Symmetrise and diagonalise ``matrix`` itself instead of a copy, saving one matrix of memory.
            Its contents are undefined afterwards.
        eigenvectors: Also return the eigenvectors of the kept modes.
        driver: The LAPACK driver of ``scipy.linalg.eigh``: ``"evr"`` (MRRR), ``"evd"`` (divide and conquer),
            ``"ev"`` (QR iteration) or ``"evx"`` (bisection and inverse iteration).
        imaginary_threshold_thz: Kept frequencies below minus this value, in THz, count as imaginary.

    Returns:
        A dictionary with:
            ``"frequencies_thz"``: the $3N - 3$ kept frequencies in THz, ascending, negative for imaginary modes.
            ``"eigenvectors"``: $(3N, 3N - 3)$ array whose orthonormal columns are the kept modes, in the order of
            ``"frequencies_thz"`` (a view of the LAPACK output), or None with ``eigenvectors=False``.
            ``"translational_frequencies_thz"``: the three dropped frequencies in THz, ascending.
            ``"translational_overlap"``: the dropped modes' squared projections on the rigid translations, shape
            (3,), or None without ``masses``.
            ``"asymmetry"``: $\max|D - D^\mathsf{T}| / \max|D|$ of the raw matrix (0.0 if exactly symmetric).
            ``"asymmetry_atoms"``: the atom pair $(i, j)$, $i \le j$, of the largest $|D - D^\mathsf{T}|$, or None if
            the matrix is exactly symmetric. $i = j$ when the worst entry lies in an atom's own $3 \times 3$ block;
            after a cutoff crossing that atom is one of the crossing pair.
            ``"n_imaginary"``: the number of kept frequencies below ``-imaginary_threshold_thz``.
            ``"min_frequency_thz"``: the lowest kept frequency in THz.
            ``"n_atoms"``: $N$.
            ``"driver"``: the LAPACK driver used.

    Raises:
        ValueError: If ``matrix`` is not square and 2-D, its size is not $3N$ with $N \ge 2$, it is not float64, it
            contains NaN or inf, or with ``overwrite_matrix=True`` it is read-only or not C- or F-contiguous; if
            ``masses`` does not hold $N$ positive finite values or is given with ``eigenvectors=False``; if
            ``driver`` is unknown or ``imaginary_threshold_thz`` is negative.

    Warns:
        UserWarning: If a dropped mode overlaps the rigid translations by less than 0.9, or if any kept frequency is
            imaginary.

    Example:
        ```pycon
        >>> minimised, summary = _run_lammps_dynamical_matrix(glass, potential, "dynmat.bin")  # doctest: +SKIP
        >>> matrix = read_dynamical_matrix(summary["matrix_path"], summary["n_atoms"])  # doctest: +SKIP
        >>> modes = diagonalize_dynamical_matrix(  # doctest: +SKIP
        ...     matrix, masses=minimised.get_masses(), overwrite_matrix=True
        ... )
        >>> modes["asymmetry"], modes["n_imaginary"], modes["translational_overlap"]  # doctest: +SKIP

        ```
    """
    matrix = np.asarray(matrix)
    masses = _validate_dynamical_matrix_input(
        matrix,
        masses,
        overwrite_matrix=overwrite_matrix,
        eigenvectors=eigenvectors,
        driver=driver,
        imaginary_threshold_thz=imaginary_threshold_thz,
    )
    n_dof = matrix.shape[0]
    asymmetry, asymmetry_atoms = _asymmetry(matrix)
    work = matrix if overwrite_matrix else np.array(matrix)
    _symmetrise_in_place(work)
    # LAPACK works on Fortran-ordered arrays; f2py copies anything else. The transpose of a C-ordered symmetric
    # matrix is the same matrix in Fortran order, so passing it avoids the copy.
    lapack_input = work.T if work.flags.c_contiguous else work
    del work
    result = scipy.linalg.eigh(
        lapack_input, overwrite_a=True, check_finite=False, driver=driver, eigvals_only=not eigenvectors
    )
    del lapack_input
    eigenvalues, vectors = result if eigenvectors else (result, None)

    # The eigenvalues are ascending, so the three of smallest |λ| are three consecutive ones: the window
    # [first, first + 3) with the smallest max |λ| (the first such window on an exact tie).
    magnitudes = np.abs(eigenvalues)
    first = int(np.argmin(np.maximum(np.maximum(magnitudes[:-2], magnitudes[1:-1]), magnitudes[2:])))
    dropped = slice(first, first + _N_TRANSLATIONS)
    translational_frequencies = _eigenvalues_to_thz(eigenvalues[dropped])
    frequencies = _eigenvalues_to_thz(np.delete(eigenvalues, dropped))

    translational_overlap = None
    if vectors is not None:
        if masses is not None:
            weights = np.sqrt(masses) / math.sqrt(masses.sum())
            projections = np.einsum("i,iam->am", weights, vectors[:, dropped].reshape(-1, 3, _N_TRANSLATIONS))
            translational_overlap = (projections**2).sum(axis=0)
            if translational_overlap.min() < _MIN_TRANSLATIONAL_OVERLAP:
                warnings.warn(
                    f"The dropped modes at {np.array2string(translational_frequencies, precision=4)} THz overlap the "
                    f"rigid translations by only {np.array2string(translational_overlap, precision=3)} (< "
                    f"{_MIN_TRANSLATIONAL_OVERLAP}): a soft mode is mixed in, or the translations are not the three "
                    "smallest |eigenvalues|.",
                    UserWarning,
                    stacklevel=2,
                )
        # Instead of copying the (3N, 3N - 3) kept columns, shift the `first` columns before the translations (the
        # modes below them, usually the few imaginary ones) three to the right, over the translations. Every column
        # after the translations is already in place, so the kept modes are then columns 3 onwards, in their original
        # order: a view. numpy buffers the overlapping assignment, which costs `first` columns, not the matrix.
        vectors[:, _N_TRANSLATIONS : first + _N_TRANSLATIONS] = vectors[:, :first]
        vectors = vectors[:, _N_TRANSLATIONS:]

    n_imaginary = int(np.count_nonzero(frequencies < -imaginary_threshold_thz))
    min_frequency = float(frequencies[0])
    if n_imaginary:
        warnings.warn(
            f"{n_imaginary} imaginary mode(s) below -{imaginary_threshold_thz} THz; the most negative is "
            f"{min_frequency:.4g} THz. Raw-matrix asymmetry {asymmetry:.3g}, largest between atoms {asymmetry_atoms}.",
            UserWarning,
            stacklevel=2,
        )
    return {
        "frequencies_thz": frequencies,
        "eigenvectors": vectors,
        "translational_frequencies_thz": translational_frequencies,
        "translational_overlap": translational_overlap,
        "asymmetry": asymmetry,
        "asymmetry_atoms": asymmetry_atoms,
        "n_imaginary": n_imaginary,
        "min_frequency_thz": min_frequency,
        "n_atoms": n_dof // 3,
        "driver": driver,
    }
