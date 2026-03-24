"""Composition string normalization.

Provides a canonical form for oxide glass compositions so that
``SiO2 70 - Na2O 15 - CaO 15`` and ``Na2O 15 - SiO2 70 - CaO 15``
resolve to the same material key.
"""

from __future__ import annotations

import re


def normalize_composition(raw: str) -> str:
    """Return a canonical composition string.

    Rules:
    * Split on ``-`` or ``,`` separators.
    * Each component is ``<Oxide> <value>`` (whitespace-flexible).
    * Components are sorted alphabetically by oxide name.
    * Values are rounded to 2 decimal places; trailing zeros stripped.
    * Canonical separator is `` - `` (space-dash-space).

    Examples
    --------
    >>> normalize_composition("Na2O 15 - SiO2 70 - CaO 15")
    'CaO 15 - Na2O 15 - SiO2 70'
    >>> normalize_composition("SiO2 70, Na2O 15, CaO 15")
    'CaO 15 - Na2O 15 - SiO2 70'
    """
    parts = re.split(r"\s*[-,]\s*", raw.strip())
    components: list[tuple[str, float]] = []
    for part in parts:
        token = part.strip()
        if not token:
            continue
        match = re.match(r"([A-Za-z0-9]+)\s+([\d.]+)", token)
        if not match:
            msg = f"Cannot parse composition component: {token!r}"
            raise ValueError(msg)
        oxide = match.group(1)
        value = float(match.group(2))
        components.append((oxide, value))

    components.sort(key=lambda c: c[0])

    def _fmt_value(v: float) -> str:
        rounded = round(v, 2)
        if rounded == int(rounded):
            return str(int(rounded))
        return f"{rounded:g}"

    return " - ".join(f"{oxide} {_fmt_value(val)}" for oxide, val in components)


def parse_components(composition: str) -> tuple[list[str], list[float]]:
    """Parse a composition string into (components, values) lists.

    Works on both raw and normalized forms.

    Returns
    -------
    components : list[str]
        Oxide names, e.g. ``["CaO", "Na2O", "SiO2"]``.
    values : list[float]
        Corresponding values.
    """
    parts = re.split(r"\s*[-,]\s*", composition.strip())
    components: list[str] = []
    values: list[float] = []
    for part in parts:
        token = part.strip()
        if not token:
            continue
        match = re.match(r"([A-Za-z0-9]+)\s+([\d.]+)", token)
        if not match:
            msg = f"Cannot parse composition component: {token!r}"
            raise ValueError(msg)
        components.append(match.group(1))
        values.append(float(match.group(2)))
    return components, values
