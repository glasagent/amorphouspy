"""Velocity-trajectory validation, chunking and power accumulation for the VDOS and VACF."""

import numpy as np
import scipy.fft
from numpy.typing import NDArray

from amorphouspy.properties.vibrational import _common
from amorphouspy.properties.vibrational._common import _N_COMPONENTS

_VELOCITY_NDIM = 3  # (n_frames, n_atoms, 3)
_MIN_FRAMES = 2
# Internal (centre-of-mass-free) kinetic sum at or below this fraction of the uncentred one counts as no internal
# motion. It sits well above the rounding noise of removing the centre of mass from a uniform translation.
_ZERO_SPECTRUM_RTOL = 1e-10


def _atoms_per_chunk(n_frames: int, itemsize: int) -> int:
    return max(1, _common._CHUNK_BYTES // (n_frames * _N_COMPONENTS * itemsize))  # noqa: SLF001


def _validate_trajectory(velocities: NDArray, masses: NDArray, dt_fs: float) -> tuple[NDArray, NDArray[np.float64]]:
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
    is_number = isinstance(dt_fs, int | float | np.integer | np.floating) and not isinstance(dt_fs, bool)
    if not is_number or not np.isfinite(dt_fs) or dt_fs <= 0:
        msg = f"dt_fs must be positive and finite, got {dt_fs}."
        raise ValueError(msg)
    if not np.all(np.isfinite(masses) & (masses > 0)):
        msg = "All masses must be positive and finite."
        raise ValueError(msg)
    return velocities, masses


def _trajectory_statistics(
    velocities: NDArray, masses: NDArray[np.float64], weights: NDArray[np.float64] | None, *, remove_com: bool
) -> tuple[NDArray[np.float64] | None, float, float]:
    """One read of all atoms: centre-of-mass velocity per frame and the trapezoid integral of the total spectrum.

    The integral of sum_i m_i |rfft(w v_i)|^2 over the rfft grid follows from Parseval's theorem without any FFT:
    (dnu / 2) * (n_frames * K - K_top), with K = sum_t w_t^2 sum_i m_i |v_i(t)|^2 and K_top the mass-weighted power
    of the last frequency bin of the windowed signal, which only enters for an odd number of frames. ``weights=None``
    means w = 1. With centre-of-mass removal both sums are reduced by the centre-of-mass term, since
    sum_i m_i (v_i - v_com) = 0. Non-finite velocities make the unwindowed sum non-finite, so the finiteness check
    costs nothing extra; the finiteness and zero-spectrum checks use the unwindowed sums.

    Returns:
        The centre-of-mass velocity ``(n_frames, 3)`` (``None`` without removal), the integral divided by dnu / 2,
        and the unwindowed sum_t sum_i m_i |v_i(t) - v_com(t)|^2 (``v_com = 0`` without removal).
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
        return None, n_frames * weighted_sq - top_sq, raw_sq
    total_mass = masses.sum()
    com_velocity = momentum / total_mass
    centred_sq = raw_sq - total_mass * float(np.sum(com_velocity**2))
    if centred_sq <= _ZERO_SPECTRUM_RTOL * raw_sq:
        msg = "The velocity spectrum is zero after removing the centre-of-mass velocity (no internal motion)."
        raise ValueError(msg)
    weighted_sq = centred_sq
    if window_sq is not None:
        weighted_sq = float(window_sq @ kinetic) - total_mass * float(window_sq @ np.sum(com_velocity**2, axis=1))
    return com_velocity, n_frames * weighted_sq - (top_sq - float(np.sum(top_momentum**2)) / total_mass), centred_sq


def _accumulate_power(
    velocities: NDArray,
    masses: NDArray[np.float64],
    com_velocity: NDArray[np.float64] | None,
    membership: NDArray,
    weights: NDArray[np.float64] | None,
    n_fft: int | None = None,
) -> NDArray[np.float64]:
    """Sum m_i * sum_a |rfft(w * (v_ia - v_com,a))|^2 over the atoms of each column of ``membership`` (n_atoms, n_cols).

    Only atoms that belong to at least one column are transformed. A contiguous range of atoms is sliced as a view;
    other selections are gathered chunk by chunk. float32 velocities are transformed in single precision (complex64)
    and float64 or other input in double precision, with the same forward, unnormalised DFT as ``np.fft.rfft``;
    per-atom powers are combined in float64. The window ``weights`` (``None`` for none) is applied per chunk in the
    working precision, never in place on a view of the input. ``n_fft`` zero-pads each signal to that length before
    the transform (``None``: no padding), and chunks are sized for the padded length.
    """
    n_frames = velocities.shape[0]
    n_transform = n_frames if n_fft is None else n_fft
    work = np.float32 if velocities.dtype == np.float32 else np.float64
    needed = np.flatnonzero(membership.any(axis=1))
    contiguous = needed[-1] - needed[0] + 1 == needed.size
    chunk = _atoms_per_chunk(n_transform, np.dtype(work).itemsize)
    com_work = None if com_velocity is None else com_velocity.astype(work)[:, None, :]
    window_work = None if weights is None else weights.astype(work)[:, None, None]
    power = np.zeros((n_transform // 2 + 1, membership.shape[1]))
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
        spectrum = np.ascontiguousarray(scipy.fft.rfft(block, n=n_fft, axis=0))
        del block
        # |z|^2 = Re(z)^2 + Im(z)^2 summed over x, y, z through a real view, without full-size temporaries.
        parts = spectrum.view(work).reshape(spectrum.shape[0], spectrum.shape[1], 2 * _N_COMPONENTS)
        atom_power = np.einsum("fik,fik->fi", parts, parts)
        del spectrum, parts
        power += atom_power @ (masses[atoms, None] * membership[atoms])
    return power
