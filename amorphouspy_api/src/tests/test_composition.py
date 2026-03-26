"""Tests for the Composition model."""

from amorphouspy_api.models import Composition


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
