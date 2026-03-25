"""Tests for composition normalization."""

import pytest

from amorphouspy_api.composition import normalize_composition, parse_components


def test_normalize_sorts_alphabetically() -> None:
    assert normalize_composition("Na2O 15 - SiO2 70 - CaO 15") == "CaO 15 - Na2O 15 - SiO2 70"


def test_normalize_comma_separator() -> None:
    assert normalize_composition("SiO2 70, Na2O 15, CaO 15") == "CaO 15 - Na2O 15 - SiO2 70"


def test_normalize_strips_whitespace() -> None:
    assert normalize_composition("  SiO2  70  -  Na2O  30  ") == "Na2O 30 - SiO2 70"


def test_normalize_decimal_values() -> None:
    assert normalize_composition("SiO2 70.5 - Na2O 29.5") == "Na2O 29.5 - SiO2 70.5"


def test_normalize_idempotent() -> None:
    canon = "CaO 15 - Na2O 15 - SiO2 70"
    assert normalize_composition(canon) == canon


def test_normalize_bad_input() -> None:
    with pytest.raises(ValueError, match="Cannot parse"):
        normalize_composition("not a composition")


def test_parse_components() -> None:
    comps, vals = parse_components("CaO 15 - Na2O 15 - SiO2 70")
    assert comps == ["CaO", "Na2O", "SiO2"]
    assert vals == [15.0, 15.0, 70.0]
