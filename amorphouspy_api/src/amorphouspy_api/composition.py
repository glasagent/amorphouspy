"""Composition model for oxide glass compositions.

Provides a :class:`Composition` Pydantic model that serialises to / from a
plain dict (e.g. ``{"SiO2": 70, "Na2O": 15, "CaO": 15}``) and bundles
canonical-string generation for database storage.
"""

from __future__ import annotations

from pydantic import RootModel


def _fmt_value(v: float) -> str:
    rounded = round(v, 2)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:g}"


class Composition(RootModel[dict[str, float]]):
    """Oxide glass composition (mol%).

    Accepts and serialises as a plain ``dict[str, float]``.
    Values represent mol% and will be rescaled to sum to 100% where needed.

    Examples
    --------
    >>> c = Composition({"Na2O": 15, "SiO2": 70, "CaO": 15})
    >>> c.canonical
    'CaO 15 - Na2O 15 - SiO2 70'
    """

    @property
    def canonical(self) -> str:
        """Canonical string for DB storage and exact-match comparison.

        Components sorted alphabetically; values rounded to 2 dp,
        trailing zeros stripped.
        """
        components = sorted(self.root.items())
        return " - ".join(f"{oxide} {_fmt_value(val)}" for oxide, val in components)

    @classmethod
    def from_canonical(cls, canonical: str) -> Composition:
        """Construct from a canonical DB string.

        >>> Composition.from_canonical("CaO 15 - Na2O 15 - SiO2 70")
        Composition({'CaO': 15.0, 'Na2O': 15.0, 'SiO2': 70.0})
        """
        result: dict[str, float] = {}
        for part in canonical.split(" - "):
            token = part.strip()
            if not token:
                continue
            oxide, value_str = token.rsplit(" ", 1)
            result[oxide] = float(value_str)
        return cls(result)
