"""Constants and group helpers shared by the vibrational modules."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

VdosNormalization = Literal["unit", "3N", "none"]
# Target size of one chunk of velocities; the complex spectrum of a chunk is about the same size.
_CHUNK_BYTES = 64 * 2**20
_N_COMPONENTS = 3


def _is_real_number(value: object) -> bool:
    """True for a Python or numpy real scalar; False for bool, None, strings and arrays."""
    return isinstance(value, int | float | np.integer | np.floating) and not isinstance(value, bool)


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


def _membership(selections: dict[str, NDArray] | None, n_atoms: int) -> NDArray[np.float64]:
    """Membership matrix (n_atoms, n_cols): one column of ones for ``None``, else one 0/1 column per selection."""
    if selections is None:
        return np.ones((n_atoms, 1))
    membership = np.zeros((n_atoms, len(selections)))
    for column, (name, indices) in enumerate(selections.items()):
        membership[_validate_selection(indices, n_atoms, name), column] = 1.0
    return membership


def _group_selections(groups: dict[str, dict[str, NDArray]]) -> tuple[list[tuple[str, str]], dict[str, NDArray]]:
    """Flatten ``{grouping: {label: indices}}`` into ordered keys and selections named after each group."""
    keys = [(grouping, label) for grouping, labels in groups.items() for label in labels]
    if not keys:
        msg = "groups contains no labels."
        raise ValueError(msg)
    return keys, {f"groups[{g!r}][{label!r}]": groups[g][label] for g, label in keys}


def _columns_by_group(
    groups: dict[str, dict[str, NDArray]], keys: list[tuple[str, str]], columns: NDArray[np.float64]
) -> dict[str, dict[str, NDArray[np.float64]]]:
    """Split the result columns back into the ``{grouping: {label: array}}`` layout of ``groups``."""
    partial: dict[str, dict[str, NDArray[np.float64]]] = {grouping: {} for grouping in groups}
    for column, (grouping, label) in enumerate(keys):
        partial[grouping][label] = np.ascontiguousarray(columns[:, column])
    return partial


_MIN_GRID_POINTS = 2
_MATRIX_NDIM = 2
_N_TRANSLATIONS = 3
