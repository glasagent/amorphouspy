"""Tests for the Composition model and elemental-space helpers."""

import math

from amorphouspy_api.models import Composition
from amorphouspy_api.routers.jobs_helpers import (
    composition_distance,
    elemental_fractions_from_job,
    oxide_to_elemental_fractions,
)


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
# composition_distance  (operates in elemental atom-fraction space)
# ---------------------------------------------------------------------------


def test_distance_identical() -> None:
    a = {"Si": 0.33, "O": 0.67}
    assert composition_distance(a, a) == 0.0


def test_distance_symmetric() -> None:
    a = {"Si": 0.4, "O": 0.6}
    b = {"Si": 0.3, "O": 0.5, "Ca": 0.2}
    assert composition_distance(a, b) == composition_distance(b, a)


def test_distance_known_value() -> None:
    # diff: Si +0.1, O -0.1  →  sqrt(0.01 + 0.01) = sqrt(0.02)
    a = {"Si": 0.4, "O": 0.6}
    b = {"Si": 0.3, "O": 0.7}
    assert math.isclose(composition_distance(a, b), math.sqrt(0.02), rel_tol=1e-9)


def test_distance_disjoint_elements() -> None:
    a = {"Si": 1.0}
    b = {"B": 1.0}
    # sqrt(1 + 1) = sqrt(2)
    assert math.isclose(composition_distance(a, b), math.sqrt(2), rel_tol=1e-9)


# ---------------------------------------------------------------------------
# oxide_to_elemental_fractions
# ---------------------------------------------------------------------------


def test_oxide_to_elemental_pure_sio2() -> None:
    fracs = oxide_to_elemental_fractions({"SiO2": 100})
    # SiO2 → 1 Si + 2 O → 1/3 Si, 2/3 O
    assert math.isclose(fracs["Si"], 1 / 3, rel_tol=1e-9)
    assert math.isclose(fracs["O"], 2 / 3, rel_tol=1e-9)


def test_oxide_to_elemental_binary() -> None:
    fracs = oxide_to_elemental_fractions({"SiO2": 50, "Na2O": 50})
    # 0.5 SiO2 → 0.5 Si + 1.0 O;  0.5 Na2O → 1.0 Na + 0.5 O
    # totals: Si 0.5, Na 1.0, O 1.5 → sum 3.0
    assert math.isclose(fracs["Si"], 0.5 / 3.0, rel_tol=1e-9)
    assert math.isclose(fracs["Na"], 1.0 / 3.0, rel_tol=1e-9)
    assert math.isclose(fracs["O"], 1.5 / 3.0, rel_tol=1e-9)


def test_oxide_to_elemental_sums_to_one() -> None:
    fracs = oxide_to_elemental_fractions({"SiO2": 60, "CaO": 25, "Al2O3": 15})
    assert math.isclose(sum(fracs.values()), 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# elemental_fractions_from_job
# ---------------------------------------------------------------------------


def test_elemental_from_job_uses_structure() -> None:
    """When final_structure has atomic numbers, use those directly."""
    from unittest.mock import MagicMock

    job = MagicMock()
    job.result_data = {
        "melt_quench": {
            "final_structure": {
                "numbers": [14, 14, 8, 8, 8, 8],  # 2 Si + 4 O
                "positions": [[0, 0, 0]] * 6,
                "cell": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
                "pbc": [True, True, True],
            }
        }
    }
    fracs = elemental_fractions_from_job(job)
    assert math.isclose(fracs["Si"], 2 / 6, rel_tol=1e-9)
    assert math.isclose(fracs["O"], 4 / 6, rel_tol=1e-9)


def test_elemental_from_job_falls_back_to_composition() -> None:
    """Without final_structure, fall back to oxide composition."""
    from unittest.mock import MagicMock

    job = MagicMock()
    job.result_data = {
        "melt_quench": {
            "composition": {"SiO2": 100},
        }
    }
    fracs = elemental_fractions_from_job(job)
    assert math.isclose(fracs["Si"], 1 / 3, rel_tol=1e-9)
    assert math.isclose(fracs["O"], 2 / 3, rel_tol=1e-9)
