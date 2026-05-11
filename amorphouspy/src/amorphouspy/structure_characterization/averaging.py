"""Generic frame-averaging utility for analysis functions.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from ase import Atoms


def _avg_sem(per_frame_values: list[Any], ddof: int, n_frames: int) -> tuple[Any, Any]:
    """Compute mean and SEM for a list of values from per-frame results.

    Dispatches on the type of per_frame_values[0]:
        np.ndarray  → element-wise stack/mean/sem
        dict        → union keys, fill 0, per-key or per-key-array mean/sem
        float/int   → scalar mean/sem
    """
    first = per_frame_values[0]

    if isinstance(first, np.ndarray):
        stacked = np.stack(per_frame_values)
        mean = stacked.mean(axis=0)
        sem = stacked.std(axis=0, ddof=ddof) / np.sqrt(n_frames)
        return mean, sem

    if isinstance(first, dict):
        all_keys = sorted({key for frame_dict in per_frame_values for key in frame_dict})
        first_value = next(iter(first.values()), None) if first else None
        if isinstance(first_value, str):
            # per-atom label dict (e.g. oxygen_classes) — not averageable; pass last frame through
            return per_frame_values[-1], None
        if isinstance(first_value, np.ndarray):
            mean_by_key: dict = {}
            sem_by_key: dict = {}
            for key in all_keys:
                arrays = [frame_dict.get(key, np.zeros_like(first_value)) for frame_dict in per_frame_values]
                stacked = np.stack(arrays)
                mean_by_key[key] = stacked.mean(axis=0)
                sem_by_key[key] = stacked.std(axis=0, ddof=ddof) / np.sqrt(n_frames)
            return mean_by_key, sem_by_key
        if isinstance(first_value, dict):
            mean_by_key = {}
            sem_by_key = {}
            for key in all_keys:
                inner_values = [frame_dict.get(key, {}) for frame_dict in per_frame_values]
                mean_by_key[key], sem_by_key[key] = _avg_sem(inner_values, ddof, n_frames)
            return mean_by_key, sem_by_key
        mean_by_key = {}
        sem_by_key = {}
        for key in all_keys:
            scalars = np.array([float(frame_dict.get(key, 0.0)) for frame_dict in per_frame_values], dtype=float)
            mean_by_key[key] = float(scalars.mean())
            sem_by_key[key] = float(scalars.std(ddof=ddof) / np.sqrt(n_frames))
        return mean_by_key, sem_by_key

    # scalar fallback
    scalars = np.array([float(v) for v in per_frame_values], dtype=float)
    return float(scalars.mean()), float(scalars.std(ddof=ddof) / np.sqrt(n_frames))


def average_over_frames(
    func: Callable[..., tuple],
    frames: list[Atoms],
    **kwargs: object,
) -> tuple[tuple, tuple]:
    """Call a per-frame analysis function on each frame, return mean and SEM for every output.

    Args:
        func: Analysis function taking a single Atoms frame as first argument and returning a tuple.
        frames: List of Atoms frames to average over.
        **kwargs: Keyword arguments forwarded to func.

    Returns:
        (means, sems) — both tuples have the same length as func's return tuple.

    Example:
        >>> means, sems = average_over_frames(compute_rdf, frames, r_max=8.0, n_bins=500)
        >>> r, rdf_mean, cn_mean = means
        >>> _, rdf_sem, cn_sem = sems
    """
    if not frames:
        msg = "average_over_frames requires a non-empty list of frames"
        raise ValueError(msg)
    raw = [func(frame, **kwargs) for frame in frames]
    # Normalize: a bare non-tuple result (e.g. a plain dict) is treated as a 1-element tuple
    results = [r if isinstance(r, tuple) else (r,) for r in raw]
    n_frames = len(results)
    ddof = 1 if n_frames > 1 else 0

    means: list[Any] = list(results[-1])
    sems: list[Any] = [None] * len(means)

    for idx in range(len(means)):
        values = [r[idx] for r in results]
        means[idx], sems[idx] = _avg_sem(values, ddof, n_frames)

    return tuple(means), tuple(sems)
