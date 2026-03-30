"""Tests for the Composition model."""

import math

from amorphouspy_api.models import Composition
from amorphouspy_api.routers.jobs_helpers import composition_distance


def test_canonical_sorts_alphabetically() -> None:
    c = Composition({"Na2O": 15, "SiO2": 70, "CaO": 15})
    assert c.canonical == "CaO 15 - Na2O 15 - SiO2 70"


def test_canonical_decimal_values() -> None:
    c = Composition({"SiO2": 70.5, "Na2O": 29.5})
    assert c.canonical == "Na2O 29.5 - SiO2 70.5"


def test_roundtrip_canonical() -> None:
    c = Composition({"CaO": 15, "Na2O": 15, "SiO2": 70})
    roundtripped = Composition.from_canonical(c.canonical)
    assert roundtripped.canonical == c.canonical


def test_from_canonical() -> None:
    c = Composition.from_canonical("CaO 15 - Na2O 15 - SiO2 70")
    assert c.root == {"CaO": 15.0, "Na2O": 15.0, "SiO2": 70.0}


def test_serialises_as_dict() -> None:
    c = Composition({"SiO2": 70, "Na2O": 30})
    dumped = c.model_dump()
    assert isinstance(dumped, dict)
    assert dumped == {"SiO2": 70.0, "Na2O": 30.0}


# ---------------------------------------------------------------------------
# composition_distance
# ---------------------------------------------------------------------------


def test_distance_identical() -> None:
    a = {"SiO2": 70, "Na2O": 30}
    assert composition_distance(a, a) == 0.0


def test_distance_symmetric() -> None:
    a = {"SiO2": 70, "Na2O": 30}
    b = {"SiO2": 65, "Na2O": 25, "CaO": 10}
    assert composition_distance(a, b) == composition_distance(b, a)


def test_distance_known_value() -> None:
    a = {"SiO2": 60, "CaO": 25, "Al2O3": 15}
    b = {"SiO2": 62, "CaO": 23, "Al2O3": 15}
    # diff: SiO2 -2, CaO +2 → sqrt(4+4) = sqrt(8)
    assert math.isclose(composition_distance(a, b), math.sqrt(8), rel_tol=1e-9)


def test_distance_disjoint_components() -> None:
    a = {"SiO2": 100}
    b = {"B2O3": 100}
    # diff: SiO2 100, B2O3 -100 → sqrt(10000+10000)
    assert math.isclose(composition_distance(a, b), math.sqrt(20000), rel_tol=1e-9)
