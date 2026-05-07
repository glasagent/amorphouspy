"""Spherical harmonic projected radial distribution functions for deformed glasses.

Computes the Y20 (uniaxial) and Y22/Y21 (shear) components of the projected
pair correlation function. The caller explicitly provides the deformation axis
and shear plane so that the correct signal component is always extracted.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from __future__ import annotations

import math
import warnings
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING, Literal

import numpy as np
from ase import Atoms

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from numba import jit

except ImportError:

    def jit(*args: object, **kwargs: object) -> Callable:  # noqa: ARG001
        """No-op decorator replacing numba.jit when numba is unavailable."""

        def decorator(func: Callable) -> Callable:
            return func

        return decorator


from amorphouspy.neighbors import (
    cell_perpendicular_heights,
    compute_cell_list_orthogonal,
    compute_cell_list_triclinic,
)

# sqrt(4pi) prefactor that converts raw Y*_lm-weighted counts to g_lm(r)
_SQRT_4PI: float = float(np.sqrt(4.0 * np.pi))
_MIN_SAME_TYPE_COUNT: int = 2


@jit(nopython=True, cache=True, fastmath=True)
def _accumulate_projected_rdf_orthogonal(  # pragma: no cover
    coords: np.ndarray,
    atomic_types: np.ndarray,
    box_size: np.ndarray,
    atom_cells: np.ndarray,
    n_cells: np.ndarray,
    cell_start: np.ndarray,
    cell_atoms: np.ndarray,
    r_max_sq: float,
    bin_width: float,
    n_bins: int,
    pair_lut: np.ndarray,
    histogram_Y20: np.ndarray,
    histogram_Y21_real: np.ndarray,
    histogram_Y21_imag: np.ndarray,
    histogram_Y22_real: np.ndarray,
    histogram_Y22_imag: np.ndarray,
) -> None:
    """Accumulate Y20, Y21, Y22 weighted pair counts into radial bins, orthogonal box.

    Iterates over i<j pairs within r_max and accumulates the Cartesian forms of
    the five Y_lm values needed for deformation analysis into pre-allocated
    histogram arrays. The sqrt(4pi) normalisation and density normalisation are
    applied by the calling Python function.

    Cartesian forms used (unit vector components unit_x, unit_y, unit_z):

    - Y20:      norm_Y20 * (1.5*unit_z^2 - 0.5)
    - Re Y21:  +norm_Y21 * unit_x*unit_z
    - Im Y21:  +norm_Y21 * unit_y*unit_z
    - Re Y22:  norm_Y22 * (unit_x^2 - unit_y^2)
    - Im Y22:  +norm_Y21 * unit_x*unit_y  (shares norm_Y21; Im Y22 coupling is xy/r^2 like Im Y21 coupling is yz/r^2)

    Args:
        coords: Wrapped Cartesian coordinates, shape (n_atoms, 3).
        atomic_types: Integer atomic numbers, shape (n_atoms,).
        box_size: Box edge lengths [Lx, Ly, Lz], shape (3,).
        atom_cells: Cell index (ix, iy, iz) per atom, shape (n_atoms, 3).
        n_cells: Number of cells along each axis, shape (3,).
        cell_start: CSR row-pointer for cell list, shape (n_cells_total + 1,).
        cell_atoms: CSR atom-index array for cell list.
        r_max_sq: Squared cutoff distance in Angstrom squared.
        bin_width: Radial bin width in Angstrom.
        n_bins: Total number of radial bins.
        pair_lut: 2-D lookup table (atomic_number_a, atomic_number_b) -> pair index;
            -1 for pairs not in the requested set, shape (max_Z+1, max_Z+1).
        histogram_Y20: Accumulation array for Y20, shape (n_pairs, n_bins). Modified in-place.
        histogram_Y21_real: Accumulation array for Re Y21, same shape. Modified in-place.
        histogram_Y21_imag: Accumulation array for Im Y21, same shape. Modified in-place.
        histogram_Y22_real: Accumulation array for Re Y22, same shape. Modified in-place.
        histogram_Y22_imag: Accumulation array for Im Y22, same shape. Modified in-place.

    """
    n_atoms = len(coords)
    n_cells_y = n_cells[1]
    n_cells_z = n_cells[2]
    inverse_bin_width = 1.0 / bin_width

    norm_Y20 = np.sqrt(5.0 / (4.0 * np.pi))
    norm_Y21 = np.sqrt(15.0 / (8.0 * np.pi))
    norm_Y22 = np.sqrt(15.0 / (32.0 * np.pi))

    for atom_i in range(n_atoms):
        atomic_type_i = atomic_types[atom_i]
        cell_index_i = atom_cells[atom_i]
        for delta_x in range(-1, 2):
            neighbor_cell_x = (cell_index_i[0] + delta_x) % n_cells[0]
            for delta_y in range(-1, 2):
                neighbor_cell_y = (cell_index_i[1] + delta_y) % n_cells[1]
                for delta_z in range(-1, 2):
                    neighbor_cell_z = (cell_index_i[2] + delta_z) % n_cells[2]
                    flat_cell_id = (
                        neighbor_cell_x * n_cells_y * n_cells_z + neighbor_cell_y * n_cells_z + neighbor_cell_z
                    )
                    for cell_slot in range(cell_start[flat_cell_id], cell_start[flat_cell_id + 1]):
                        atom_j = cell_atoms[cell_slot]
                        if atom_j <= atom_i:
                            continue

                        diff_x = coords[atom_i, 0] - coords[atom_j, 0]
                        diff_y = coords[atom_i, 1] - coords[atom_j, 1]
                        diff_z = coords[atom_i, 2] - coords[atom_j, 2]
                        diff_x -= box_size[0] * round(diff_x / box_size[0])
                        diff_y -= box_size[1] * round(diff_y / box_size[1])
                        diff_z -= box_size[2] * round(diff_z / box_size[2])
                        dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
                        if dist_sq > r_max_sq:
                            continue

                        atomic_type_j = atomic_types[atom_j]
                        pair_idx = pair_lut[atomic_type_i, atomic_type_j]
                        if pair_idx < 0:
                            continue

                        distance = np.sqrt(dist_sq)
                        bin_idx = int(distance * inverse_bin_width)
                        if bin_idx >= n_bins:
                            bin_idx = n_bins - 1

                        inverse_distance = 1.0 / distance
                        unit_x = diff_x * inverse_distance
                        unit_y = diff_y * inverse_distance
                        unit_z = diff_z * inverse_distance

                        histogram_Y20[pair_idx, bin_idx] += norm_Y20 * (1.5 * unit_z * unit_z - 0.5)
                        histogram_Y21_real[pair_idx, bin_idx] += norm_Y21 * unit_x * unit_z
                        histogram_Y21_imag[pair_idx, bin_idx] += norm_Y21 * unit_y * unit_z
                        histogram_Y22_real[pair_idx, bin_idx] += norm_Y22 * (unit_x * unit_x - unit_y * unit_y)
                        histogram_Y22_imag[pair_idx, bin_idx] += norm_Y21 * unit_x * unit_y


@jit(nopython=True, cache=True, fastmath=True)
def _accumulate_projected_rdf_triclinic(  # pragma: no cover
    coords_frac: np.ndarray,
    atomic_types: np.ndarray,
    cell: np.ndarray,
    atom_cells: np.ndarray,
    n_cells: np.ndarray,
    cell_start: np.ndarray,
    cell_atoms: np.ndarray,
    r_max_sq: float,
    bin_width: float,
    n_bins: int,
    pair_lut: np.ndarray,
    histogram_Y20: np.ndarray,
    histogram_Y21_real: np.ndarray,
    histogram_Y21_imag: np.ndarray,
    histogram_Y22_real: np.ndarray,
    histogram_Y22_imag: np.ndarray,
) -> None:
    """Accumulate Y20, Y21, Y22 weighted pair counts into radial bins, triclinic box.

    Triclinic variant of :func:`_accumulate_projected_rdf_orthogonal`. Uses
    fractional coordinates and the full 3x3 cell matrix for correct
    minimum-image convention in non-orthogonal cells.

    Args:
        coords_frac: Wrapped fractional coordinates, shape (n_atoms, 3).
        atomic_types: Integer atomic numbers, shape (n_atoms,).
        cell: Lattice vector matrix; cell[i] is the i-th lattice vector, shape (3, 3).
        atom_cells: Cell index (ix, iy, iz) per atom, shape (n_atoms, 3).
        n_cells: Number of cells along each axis, shape (3,).
        cell_start: CSR row-pointer for cell list, shape (n_cells_total + 1,).
        cell_atoms: CSR atom-index array for cell list.
        r_max_sq: Squared cutoff distance in Angstrom squared.
        bin_width: Radial bin width in Angstrom.
        n_bins: Total number of radial bins.
        pair_lut: 2-D lookup table (atomic_number_a, atomic_number_b) -> pair index;
            -1 for pairs not in the requested set, shape (max_Z+1, max_Z+1).
        histogram_Y20: Accumulation array for Y20, shape (n_pairs, n_bins). Modified in-place.
        histogram_Y21_real: Accumulation array for Re Y21, same shape. Modified in-place.
        histogram_Y21_imag: Accumulation array for Im Y21, same shape. Modified in-place.
        histogram_Y22_real: Accumulation array for Re Y22, same shape. Modified in-place.
        histogram_Y22_imag: Accumulation array for Im Y22, same shape. Modified in-place.

    """
    n_atoms = len(coords_frac)
    n_cells_y = n_cells[1]
    n_cells_z = n_cells[2]
    inverse_bin_width = 1.0 / bin_width

    norm_Y20 = np.sqrt(5.0 / (4.0 * np.pi))
    norm_Y21 = np.sqrt(15.0 / (8.0 * np.pi))
    norm_Y22 = np.sqrt(15.0 / (32.0 * np.pi))

    for atom_i in range(n_atoms):
        atomic_type_i = atomic_types[atom_i]
        cell_index_i = atom_cells[atom_i]
        for delta_x in range(-1, 2):
            neighbor_cell_x = (cell_index_i[0] + delta_x) % n_cells[0]
            for delta_y in range(-1, 2):
                neighbor_cell_y = (cell_index_i[1] + delta_y) % n_cells[1]
                for delta_z in range(-1, 2):
                    neighbor_cell_z = (cell_index_i[2] + delta_z) % n_cells[2]
                    flat_cell_id = (
                        neighbor_cell_x * n_cells_y * n_cells_z + neighbor_cell_y * n_cells_z + neighbor_cell_z
                    )
                    for cell_slot in range(cell_start[flat_cell_id], cell_start[flat_cell_id + 1]):
                        atom_j = cell_atoms[cell_slot]
                        if atom_j <= atom_i:
                            continue

                        diff_frac_x = coords_frac[atom_i, 0] - coords_frac[atom_j, 0]
                        diff_frac_x -= round(diff_frac_x)
                        diff_frac_y = coords_frac[atom_i, 1] - coords_frac[atom_j, 1]
                        diff_frac_y -= round(diff_frac_y)
                        diff_frac_z = coords_frac[atom_i, 2] - coords_frac[atom_j, 2]
                        diff_frac_z -= round(diff_frac_z)
                        diff_x = diff_frac_x * cell[0, 0] + diff_frac_y * cell[1, 0] + diff_frac_z * cell[2, 0]
                        diff_y = diff_frac_x * cell[0, 1] + diff_frac_y * cell[1, 1] + diff_frac_z * cell[2, 1]
                        diff_z = diff_frac_x * cell[0, 2] + diff_frac_y * cell[1, 2] + diff_frac_z * cell[2, 2]
                        dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
                        if dist_sq > r_max_sq:
                            continue

                        atomic_type_j = atomic_types[atom_j]
                        pair_idx = pair_lut[atomic_type_i, atomic_type_j]
                        if pair_idx < 0:
                            continue

                        distance = np.sqrt(dist_sq)
                        bin_idx = int(distance * inverse_bin_width)
                        if bin_idx >= n_bins:
                            bin_idx = n_bins - 1

                        inverse_distance = 1.0 / distance
                        unit_x = diff_x * inverse_distance
                        unit_y = diff_y * inverse_distance
                        unit_z = diff_z * inverse_distance

                        histogram_Y20[pair_idx, bin_idx] += norm_Y20 * (1.5 * unit_z * unit_z - 0.5)
                        histogram_Y21_real[pair_idx, bin_idx] += norm_Y21 * unit_x * unit_z
                        histogram_Y21_imag[pair_idx, bin_idx] += norm_Y21 * unit_y * unit_z
                        histogram_Y22_real[pair_idx, bin_idx] += norm_Y22 * (unit_x * unit_x - unit_y * unit_y)
                        histogram_Y22_imag[pair_idx, bin_idx] += norm_Y21 * unit_x * unit_y


def _extract_uniaxial_signal(
    g20: np.ndarray,
    g22_real: np.ndarray,
    deformation_axis: Literal["x", "y", "z"],
) -> np.ndarray:
    """Return the dominant uniaxial anisotropy signal for the given deformation axis.

    Y20 is accumulated with z as the reference axis inside the kernel. For
    deformation along x or y the physical signal is a linear combination of
    g20 and Re g22, derived by requiring that a unit vector along the
    deformation axis gives a positive maximum. The kernel computes
    Y20(z) = N20*(1.5*uz^2-0.5) and Re Y22 = N22*(ux^2-uy^2). Solving for the
    combination a*Y20(z) + b*Re Y22 = Y20(deformation_axis) gives:

    - z-axis:  g20               (a=1, b=0)
    - x-axis:  -0.5*g20 + axis_combination_coefficient*g22_real   (a=-0.5, b=+3/4*sqrt(8/3))
    - y-axis:  -0.5*g20 - axis_combination_coefficient*g22_real   (a=-0.5, b=-3/4*sqrt(8/3))

    where axis_combination_coefficient = (3/4)*sqrt(8/3) = sqrt(3/2) * norm_Y20/norm_Y22.

    Args:
        g20: Normalised Y20 component per pair, shape (n_bins,).
        g22_real: Normalised Re Y22 component per pair, shape (n_bins,).
        deformation_axis: Axis along which the uniaxial deformation is applied.

    Returns:
        Real-valued anisotropy signal, shape (n_bins,).

    """
    # axis_combination_coefficient = 0.75 * sqrt(8/3) = sqrt(3/2)
    axis_combination_coefficient = float(0.75 * np.sqrt(8.0 / 3.0))
    if deformation_axis == "z":
        return g20.copy()
    if deformation_axis == "x":
        return -0.5 * g20 + axis_combination_coefficient * g22_real
    return -0.5 * g20 - axis_combination_coefficient * g22_real


def _extract_shear_signal(
    g21_real: np.ndarray,
    g21_imag: np.ndarray,
    g22_imag: np.ndarray,
    shear_plane: Literal["xy", "xz", "yz"],
) -> np.ndarray:
    """Return the dominant shear anisotropy signal for the given shear plane.

    Mapping from shear plane to dominant Y_lm component (Cartesian coupling):

    - xy plane -> Im Y22  (Im Y*22 proportional to xy/r^2)
    - xz plane -> Re Y21  (Re Y*21 proportional to xz/r^2)
    - yz plane -> Im Y21  (Im Y*21 proportional to yz/r^2)

    Args:
        g21_real: Normalised Re Y21 component per pair, shape (n_bins,).
        g21_imag: Normalised Im Y21 component per pair, shape (n_bins,).
        g22_imag: Normalised Im Y22 component per pair, shape (n_bins,).
        shear_plane: Plane in which the shear deformation is applied.

    Returns:
        Real-valued shear signal, shape (n_bins,).

    """
    if shear_plane == "xy":
        return g22_imag.copy()
    if shear_plane == "xz":
        return g21_real.copy()
    return g21_imag.copy()


def _resolve_r_max(first_frame: Atoms, r_max: float) -> float:
    """Clamp r_max to half the smallest perpendicular cell height, warning if adjusted."""
    perpendicular_heights = cell_perpendicular_heights(first_frame.get_cell().array)
    r_max_allowed = float(np.min(perpendicular_heights)) / 2.0
    if r_max <= r_max_allowed:
        return r_max
    r_max_adjusted = float(math.floor(r_max_allowed))
    if r_max_adjusted <= 0.0:
        msg = (
            f"r_max_allowed={r_max_allowed:.4f} A is less than 1 A; "
            "no valid integer cutoff exists. Use a larger simulation box."
        )
        raise ValueError(msg)
    warnings.warn(
        f"r_max={r_max:.4f} A exceeds half the smallest perpendicular cell "
        f"height ({r_max_allowed:.4f} A). Adjusted to {r_max_adjusted:.1f} A.",
        UserWarning,
        stacklevel=3,
    )
    return r_max_adjusted


def _build_pair_lut(
    first_frame: Atoms,
    type_pairs: list[tuple[int, int]] | None,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Return canonical pair list and atomic-number lookup table mapping pair -> index."""
    first_frame_types = np.unique(first_frame.get_atomic_numbers())
    if type_pairs is None:
        canonical_pairs = [(int(a), int(b)) for a, b in combinations_with_replacement(first_frame_types, 2)]
    else:
        canonical_pairs = list({(min(a, b), max(a, b)) for a, b in type_pairs})
    max_atomic_number = int(max(atomic_number for pair in canonical_pairs for atomic_number in pair))
    pair_lut = np.full((max_atomic_number + 1, max_atomic_number + 1), -1, dtype=np.int32)
    for pair_idx, (type1, type2) in enumerate(canonical_pairs):
        pair_lut[type1, type2] = pair_idx
        pair_lut[type2, type1] = pair_idx
    return canonical_pairs, pair_lut


def _accumulate_frame(
    atoms: Atoms,
    canonical_pairs: list[tuple[int, int]],
    pair_lut: np.ndarray,
    r_max: float,
    bin_width: float,
    n_bins: int,
    shell_volumes: np.ndarray,
    accumulated_Y20: np.ndarray,
    accumulated_Y21_real: np.ndarray,
    accumulated_Y21_imag: np.ndarray,
    accumulated_Y22_real: np.ndarray,
    accumulated_Y22_imag: np.ndarray,
) -> None:
    """Accumulate one frame's Y_lm-weighted histogram counts into the running totals."""
    n_pairs = len(canonical_pairs)
    frame_cell = atoms.get_cell().array
    frame_volume = abs(np.linalg.det(frame_cell))
    frame_types = atoms.get_atomic_numbers().astype(np.int32)
    frame_type_counts = {int(t): int(np.sum(frame_types == t)) for t in np.unique(frame_types)}

    frame_wrapped = atoms.copy()
    frame_wrapped.wrap()
    frame_coords = frame_wrapped.get_positions()

    hist_Y20 = np.zeros((n_pairs, n_bins), dtype=np.float64)
    hist_Y21_real = np.zeros((n_pairs, n_bins), dtype=np.float64)
    hist_Y21_imag = np.zeros((n_pairs, n_bins), dtype=np.float64)
    hist_Y22_real = np.zeros((n_pairs, n_bins), dtype=np.float64)
    hist_Y22_imag = np.zeros((n_pairs, n_bins), dtype=np.float64)

    is_orthogonal = np.allclose(frame_cell - np.diag(np.diag(frame_cell)), 0.0, atol=1e-10)
    if is_orthogonal:
        box_size = np.diag(frame_cell)
        cell_build_cutoff = min(r_max, float(np.min(box_size)) / 3.0 - 1e-10)
        atom_cells, n_cells, cell_start, cell_atoms = compute_cell_list_orthogonal(
            frame_coords, box_size, cell_build_cutoff
        )
        _accumulate_projected_rdf_orthogonal(
            frame_coords,
            frame_types,
            box_size,
            atom_cells,
            n_cells,
            cell_start,
            cell_atoms,
            r_max * r_max,
            bin_width,
            n_bins,
            pair_lut,
            hist_Y20,
            hist_Y21_real,
            hist_Y21_imag,
            hist_Y22_real,
            hist_Y22_imag,
        )
    else:
        perpendicular_heights = cell_perpendicular_heights(frame_cell)
        cell_build_cutoff = min(r_max, float(np.min(perpendicular_heights)) / 3.0 - 1e-10)
        coords_frac, atom_cells, n_cells, cell_start, cell_atoms = compute_cell_list_triclinic(
            frame_coords, frame_cell, cell_build_cutoff
        )
        _accumulate_projected_rdf_triclinic(
            coords_frac,
            frame_types,
            frame_cell,
            atom_cells,
            n_cells,
            cell_start,
            cell_atoms,
            r_max * r_max,
            bin_width,
            n_bins,
            pair_lut,
            hist_Y20,
            hist_Y21_real,
            hist_Y21_imag,
            hist_Y22_real,
            hist_Y22_imag,
        )

    for pair_idx, (type1, type2) in enumerate(canonical_pairs):
        n_type1 = frame_type_counts.get(type1, 0)
        n_type2 = frame_type_counts.get(type2, 0)
        if type1 == type2:
            if n_type1 < _MIN_SAME_TYPE_COUNT:
                continue
            excluded_number_density = (n_type1 - 1) / frame_volume
            normalisation = n_type1 * excluded_number_density * shell_volumes + 1e-30
            scale = _SQRT_4PI * 2.0 / normalisation
        else:
            normalisation = n_type1 * (n_type2 / frame_volume) * shell_volumes + 1e-30
            scale = _SQRT_4PI / normalisation

        accumulated_Y20[pair_idx] += hist_Y20[pair_idx] * scale
        accumulated_Y21_real[pair_idx] += hist_Y21_real[pair_idx] * scale
        accumulated_Y21_imag[pair_idx] += hist_Y21_imag[pair_idx] * scale
        accumulated_Y22_real[pair_idx] += hist_Y22_real[pair_idx] * scale
        accumulated_Y22_imag[pair_idx] += hist_Y22_imag[pair_idx] * scale


def compute_projected_rdf(
    frames: Atoms | list[Atoms],
    deformation_axis: Literal["x", "y", "z"] | None = None,
    shear_plane: Literal["xy", "xz", "yz"] | None = None,
    r_max: float = 10.0,
    n_bins: int = 500,
    type_pairs: list[tuple[int, int]] | None = None,
) -> tuple[
    np.ndarray,
    dict[tuple[int, int], np.ndarray] | None,
    dict[tuple[int, int], np.ndarray] | None,
]:
    r"""Compute the uniaxial and shear projected radial distribution functions.

    Expands the pair correlation function onto the l=2 spherical harmonic
    components that carry the deformation signal:

    - **uniaxial_rdf**: the Y20-based component along ``deformation_axis``.
      Vanishes for isotropic structures; grows with uniaxial strain.
    - **shear_rdf**: the Y21 or Y22 component for ``shear_plane``.
      Vanishes for isotropic structures; grows with shear.

    At least one of ``deformation_axis`` or ``shear_plane`` must be provided.
    Omit the one you do not need, its slot in the return tuple will be ``None``.

    For multiple frames the per-frame normalised values are averaged, so NPT
    trajectories with varying cell geometries are handled correctly.

    Performance: Numba-JIT kernels accumulate all five needed Y*_lm components
    in a single i<j pass. On first call Numba compiles the kernels (~5 s);
    subsequent calls use cached bytecode.

    Args:
        frames: Single ASE Atoms object or list of frames for trajectory averaging.
        deformation_axis: Axis along which uniaxial deformation was applied
            ('x', 'y', or 'z'). Determines which linear combination of Y20
            and Re Y22 is returned as the uniaxial signal. ``None`` skips
            uniaxial extraction and returns ``None`` for ``uniaxial_rdf``.
        shear_plane: Plane in which shear deformation was applied
            ('xy', 'xz', or 'yz'). Determines which Y21/Y22 component is
            returned as the shear signal. ``None`` skips shear extraction and
            returns ``None`` for ``shear_rdf``.
        r_max: Maximum distance in Angstrom (default 10.0). Automatically clamped to
            floor(min_perpendicular_height / 2) with a warning if exceeded.
        n_bins: Number of radial bins (default 500).
        type_pairs: List of (atomic_number_a, atomic_number_b) pairs to compute.
            None computes all unique unordered combinations present in the first frame.

    Returns:
        r: Radial bin centres in Angstrom, shape (n_bins,).
        uniaxial_rdf: Dict keyed by (type1, type2) -> real array of shape (n_bins,),
            or ``None`` if ``deformation_axis`` was not provided.
        shear_rdf: Dict keyed by (type1, type2) -> real array of shape (n_bins,),
            or ``None`` if ``shear_plane`` was not provided.

    Raises:
        ValueError: If both ``deformation_axis`` and ``shear_plane`` are ``None``.
        ValueError: If r_max after clamping would be <= 0.

    Example:
        >>> from ase.build import bulk
        >>> atoms = bulk("Si", "diamond", a=5.43) * (4, 4, 4)
        >>> # Uniaxial only -- no shear plane needed
        >>> r, uniaxial, _ = compute_projected_rdf(
        ...     atoms, deformation_axis="z", r_max=6.0
        ... )
        >>> signal_SiSi = uniaxial[(14, 14)]   # near zero for undeformed bulk

    """
    if deformation_axis is None and shear_plane is None:
        msg = "At least one of deformation_axis or shear_plane must be provided."
        raise ValueError(msg)

    if isinstance(frames, Atoms):
        frames = [frames]

    r_max = _resolve_r_max(frames[0], r_max)
    canonical_pairs, pair_lut = _build_pair_lut(frames[0], type_pairs)

    n_pairs = len(canonical_pairs)
    bin_edges = np.linspace(0.0, r_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_width = float(bin_edges[1] - bin_edges[0])
    shell_volumes = 4.0 * np.pi * bin_centers**2 * bin_width

    accumulated_Y20 = np.zeros((n_pairs, n_bins), dtype=np.float64)
    accumulated_Y21_real = np.zeros((n_pairs, n_bins), dtype=np.float64)
    accumulated_Y21_imag = np.zeros((n_pairs, n_bins), dtype=np.float64)
    accumulated_Y22_real = np.zeros((n_pairs, n_bins), dtype=np.float64)
    accumulated_Y22_imag = np.zeros((n_pairs, n_bins), dtype=np.float64)

    for atoms in frames:
        _accumulate_frame(
            atoms,
            canonical_pairs,
            pair_lut,
            r_max,
            bin_width,
            n_bins,
            shell_volumes,
            accumulated_Y20,
            accumulated_Y21_real,
            accumulated_Y21_imag,
            accumulated_Y22_real,
            accumulated_Y22_imag,
        )

    n_frames = len(frames)
    uniaxial_rdf: dict[tuple[int, int], np.ndarray] | None = {} if deformation_axis is not None else None
    shear_rdf: dict[tuple[int, int], np.ndarray] | None = {} if shear_plane is not None else None

    for pair_idx, (type1, type2) in enumerate(canonical_pairs):
        g20 = accumulated_Y20[pair_idx] / n_frames
        g21_real = accumulated_Y21_real[pair_idx] / n_frames
        g21_imag = accumulated_Y21_imag[pair_idx] / n_frames
        g22_real = accumulated_Y22_real[pair_idx] / n_frames
        g22_imag = accumulated_Y22_imag[pair_idx] / n_frames

        if deformation_axis is not None and uniaxial_rdf is not None:
            uniaxial_rdf[(type1, type2)] = _extract_uniaxial_signal(g20, g22_real, deformation_axis)
        if shear_plane is not None and shear_rdf is not None:
            shear_rdf[(type1, type2)] = _extract_shear_signal(g21_real, g21_imag, g22_imag, shear_plane)

    return bin_centers, uniaxial_rdf, shear_rdf
