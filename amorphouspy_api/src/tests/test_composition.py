"""Tests for the Composition model and elemental-space helpers."""

import json
import math
from unittest.mock import MagicMock, patch

import pytest
from amorphouspy_api.models import (
    Composition,
    serialize_atoms,
    validate_atoms,
    validate_potential,
)
from amorphouspy_api.routers.jobs_helpers import (
    composition_distance,
    elemental_fractions_from_job,
    oxide_to_elemental_fractions,
)
from ase import Atoms


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


def test_from_canonical_with_extra_spaces() -> None:
    """from_canonical handles extra whitespace gracefully (covers line 102)."""
    # Double spaces around separator create empty tokens that should be skipped
    c = Composition.from_canonical("CaO 15  -  Na2O 15  -  SiO2 70")
    assert c.root == {"CaO": 15.0, "Na2O": 15.0, "SiO2": 70.0}


def test_from_canonical_with_trailing_separator() -> None:
    """from_canonical handles trailing separator gracefully."""
    # Trailing separator could create an empty token
    c = Composition.from_canonical("SiO2 70 - Na2O 30 - ")
    assert c.root == {"Na2O": 30.0, "SiO2": 70.0}


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
    job = MagicMock()
    job.result_data = {
        "melt_quench": {
            "composition": {"SiO2": 100},
        }
    }
    fracs = elemental_fractions_from_job(job)
    assert math.isclose(fracs["Si"], 1 / 3, rel_tol=1e-9)
    assert math.isclose(fracs["O"], 2 / 3, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# validate_atoms edge cases
# ---------------------------------------------------------------------------


def test_validate_atoms_none_returns_none() -> None:
    """validate_atoms returns None for None input."""
    assert validate_atoms(None) is None


def test_validate_atoms_atoms_object_returned_as_is() -> None:
    """validate_atoms returns Atoms object unchanged."""
    atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    result = validate_atoms(atoms)
    assert result is atoms


def test_validate_atoms_invalid_dict_raises_valueerror() -> None:
    """validate_atoms raises ValueError for invalid dict input (covers line 122-142)."""
    # Dict with invalid Atoms kwargs should raise ValueError
    invalid_dict = {"invalid_key": "value"}
    with pytest.raises(ValueError, match="Could not reconstruct Atoms from dict"):
        validate_atoms(invalid_dict)


def test_validate_atoms_invalid_string_raises_valueerror() -> None:
    """validate_atoms raises ValueError for invalid string input (covers line 137-142)."""
    # Invalid JSON string should raise ValueError
    invalid_string = "not valid json {{"
    with pytest.raises(ValueError, match="Could not parse Atoms from string"):
        validate_atoms(invalid_string)


def test_validate_atoms_invalid_type_raises_typeerror() -> None:
    """validate_atoms raises TypeError for invalid type input (covers line 160-161)."""
    # Invalid type (int) should raise TypeError
    with pytest.raises(TypeError, match="Expected ASE Atoms, dict, str, or None"):
        validate_atoms(12345)  # type: ignore[arg-type]


def test_validate_atoms_json_string_with_list_takes_last() -> None:
    """validate_atoms handles JSON string containing list (trajectory) by taking last frame (covers line 115-117)."""
    atoms1 = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms2 = Atoms("H2", positions=[[0, 0, 1], [1, 0, 1]])

    # Mock read to return a list of atoms (simulating trajectory file)
    with patch("amorphouspy_api.models.read", return_value=[atoms1, atoms2]):
        result = validate_atoms("{}")  # Any JSON string will trigger read
        # Should return the last frame
        assert result is atoms2


def test_validate_atoms_json_string_non_list_returns_result() -> None:
    """validate_atoms with JSON string returning single Atoms (not list) returns it directly."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])

    # Mock read to return a single Atoms object (not a list)
    with patch("amorphouspy_api.models.read", return_value=atoms):
        result = validate_atoms("{}")
        assert result is atoms


def test_serialize_atoms() -> None:
    """serialize_atoms converts Atoms to JSON string."""
    atoms = Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    serialized = serialize_atoms(atoms)

    # Should be valid JSON
    data = json.loads(serialized)
    assert isinstance(data, dict)
    # ASE JSON format has nested structure, check for expected top-level keys
    assert "ids" in data
    assert "nextid" in data


def test_validate_potential_invalid() -> None:
    """validate_potential raises ValueError for unsupported potential."""
    with pytest.raises(ValueError, match="Unsupported potential"):
        validate_potential("invalid_potential_xyz")
