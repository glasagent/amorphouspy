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


def frame_average(
    func: Callable[..., tuple],
    frames: list[Atoms],
    avg_indices: list[int],
    *args: object,
    **kwargs: object,
) -> tuple[tuple, tuple]:
    """Average a tuple-returning analysis function over a list of frames.

    Args:
        func: Analysis function taking a single Atoms frame and returning a tuple.
        frames: List of Atoms frames to average over.
        avg_indices: Indices in the return tuple to average; all other positions
            are passed through unchanged from the last frame.
        *args: Positional arguments forwarded to func.
        **kwargs: Keyword arguments forwarded to func.

    Returns:
        (means, sems) — both tuples have the same length as func's return tuple.
        Non-averaged positions in sems are None.
    """
    per_frame_results = [func(frame, *args, **kwargs) for frame in frames]
    n_frames = len(per_frame_results)
    ddof = 1 if n_frames > 1 else 0  # Bessel's correction: use n-1 for unbiased std when more than one frame

    means = list(per_frame_results[-1])
    sems: list[Any] = [None] * len(means)

    for idx in avg_indices:
        values = [result[idx] for result in per_frame_results]
        means[idx], sems[idx] = _avg_sem(values, ddof, n_frames)

    return tuple(means), tuple(sems)
