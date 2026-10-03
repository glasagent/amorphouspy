"""Vibrational properties of glass systems: vibrational density of states and related quantities.

Frequencies in this module are ordinary frequencies in THz (not angular frequencies) unless stated otherwise.

Author: Achraf Atila (achraf.atila@bam.de)
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.constants import c, e, h

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
