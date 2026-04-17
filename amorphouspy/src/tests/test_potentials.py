"""Tests for potential generation utilities.

Author: Achraf Atila (achraf.atila@bam.de)
"""

import numpy as np
import pandas as pd
import pytest
from amorphouspy.potentials.bjp_potential import (
    generate_bjp_potential,
)
from amorphouspy.potentials.bjp_potential import (
    supported_elements as bjp_supported_elements,
)
from amorphouspy.potentials.pmmcs_potential import (
    generate_pmmcs_potential,
)
from amorphouspy.potentials.pmmcs_potential import (
    supported_elements as pmmcs_supported_elements,
)
from amorphouspy.potentials.potential import (
    compatible_potentials,
    get_supported_elements,
    select_potential,
)
from amorphouspy.potentials.shik_potential import (
    compute_oxygen_charge,
    generate_shik_potential,
    potential_and_force,
    shik_charges,
    shik_params,
    write_table_file,
)
from amorphouspy.potentials.shik_potential import (
    supported_elements as shik_supported_elements,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sio2_atoms_dict() -> dict:
    """Minimal SiO2 atoms_dict for potential generation."""
    return {"atoms": [{"element": "Si"}, {"element": "O"}, {"element": "O"}]}


def _cas_atoms_dict() -> dict:
    """CaAlSiO atoms_dict for BJP potential (needs all four BJP elements)."""
    return {"atoms": [{"element": "Ca"}, {"element": "Al"}, {"element": "Si"}, {"element": "O"}]}


# ---------------------------------------------------------------------------
# get_supported_elements
# ---------------------------------------------------------------------------


def test_get_supported_elements_pmmcs():
    """get_supported_elements returns a set containing Si and O for pmmcs."""
    result = get_supported_elements("pmmcs")
    assert isinstance(result, set)
    assert "Si" in result
    assert "O" in result


def test_get_supported_elements_shik():
    """get_supported_elements returns a set containing Si and O for shik."""
    result = get_supported_elements("shik")
    assert isinstance(result, set)
    assert "Si" in result
    assert "O" in result


def test_get_supported_elements_bjp():
    """get_supported_elements returns a set containing Si and O for bjp."""
    result = get_supported_elements("bjp")
    assert isinstance(result, set)
    assert "Si" in result
    assert "O" in result


def test_get_supported_elements_unknown():
    """get_supported_elements raises ValueError for an unknown potential type."""
    with pytest.raises(ValueError, match="Unsupported potential type"):
        get_supported_elements("unknown")


def test_get_supported_elements_case_insensitive():
    """get_supported_elements treats potential names case-insensitively."""
    assert get_supported_elements("PMMCS") == get_supported_elements("pmmcs")


# ---------------------------------------------------------------------------
# compatible_potentials
# ---------------------------------------------------------------------------


def test_compatible_potentials_sio2_contains_pmmcs_and_shik():
    """SiO2 composition is compatible with at least pmmcs and shik."""
    result = compatible_potentials({"Si", "O"})
    assert "pmmcs" in result
    assert "shik" in result


def test_compatible_potentials_unsupported_elements_returns_empty():
    """No potential supports noble gas elements."""
    assert compatible_potentials({"Xe", "Kr"}) == []


def test_compatible_potentials_preserves_preference_order():
    """Results are returned in pmmcs → shik → bjp preference order."""
    result = compatible_potentials({"Si", "O"})
    preference = ("pmmcs", "shik", "bjp")
    indices = [preference.index(p) for p in result if p in preference]
    assert indices == sorted(indices)


def test_compatible_potentials_all_subset_of_known():
    """Results contain only recognised potential names."""
    result = compatible_potentials({"Si", "O"})
    assert all(p in ("pmmcs", "shik", "bjp", "du_teter") for p in result)


# ---------------------------------------------------------------------------
# select_potential
# ---------------------------------------------------------------------------


def test_select_potential_prefers_pmmcs_for_sio2():
    """Pmmcs is selected for SiO2 as the highest-preference option."""
    assert select_potential({"Si", "O"}) == "pmmcs"


def test_select_potential_prefers_pmmcs_over_bjp_for_cas():
    """Pmmcs wins over bjp for CaAlSiO because pmmcs has higher preference."""
    assert select_potential({"Ca", "Al", "Si", "O"}) == "pmmcs"


def test_select_potential_returns_none_for_unsupported():
    """select_potential returns None when no potential covers all elements."""
    assert select_potential({"Xe", "Kr"}) is None


# ---------------------------------------------------------------------------
# bjp_potential.supported_elements
# ---------------------------------------------------------------------------


def test_bjp_supported_elements_correct_set():
    """BJP supports exactly {Ca, Al, Si, O}."""
    assert bjp_supported_elements() == {"Ca", "Al", "Si", "O"}


# ---------------------------------------------------------------------------
# generate_bjp_potential
# ---------------------------------------------------------------------------


def test_generate_bjp_potential_returns_dataframe():
    """generate_bjp_potential returns a pandas DataFrame."""
    result = generate_bjp_potential(_cas_atoms_dict())
    assert isinstance(result, pd.DataFrame)


def test_generate_bjp_potential_non_empty():
    """BJP DataFrame has at least one row."""
    result = generate_bjp_potential(_cas_atoms_dict())
    assert len(result) > 0


def test_generate_bjp_potential_contains_pair_style():
    """BJP config lines include a pair_style entry."""
    result = generate_bjp_potential(_cas_atoms_dict())
    config = result["Config"].iloc[0]
    assert any("pair_style" in line for line in config)


def test_generate_bjp_potential_name_column():
    """BJP DataFrame Name column equals 'BJP'."""
    result = generate_bjp_potential(_cas_atoms_dict())
    assert result["Name"].iloc[0] == "BJP"


# ---------------------------------------------------------------------------
# pmmcs_potential.supported_elements
# ---------------------------------------------------------------------------


def test_pmmcs_supported_elements_is_set():
    """Pmmcs supported_elements returns a set."""
    assert isinstance(pmmcs_supported_elements(), set)


def test_pmmcs_supported_elements_contains_si_and_o():
    """PMMCS supported elements include Si and O."""
    elems = pmmcs_supported_elements()
    assert "Si" in elems
    assert "O" in elems


# ---------------------------------------------------------------------------
# generate_pmmcs_potential
# ---------------------------------------------------------------------------


def test_generate_pmmcs_potential_returns_dataframe():
    """generate_pmmcs_potential returns a pandas DataFrame."""
    result = generate_pmmcs_potential(_sio2_atoms_dict())
    assert isinstance(result, pd.DataFrame)


def test_generate_pmmcs_potential_non_empty():
    """PMMCS DataFrame has at least one row."""
    result = generate_pmmcs_potential(_sio2_atoms_dict())
    assert len(result) > 0


def test_generate_pmmcs_potential_contains_pair_coeff():
    """PMMCS config lines include at least one pair_coeff entry."""
    result = generate_pmmcs_potential(_sio2_atoms_dict())
    config = result["Config"].iloc[0]
    assert any("pair_coeff" in line for line in config)


# ---------------------------------------------------------------------------
# shik_potential.supported_elements
# ---------------------------------------------------------------------------


def test_shik_supported_elements_is_set():
    """Shik supported_elements returns a set."""
    assert isinstance(shik_supported_elements(), set)


def test_shik_supported_elements_contains_si_and_o():
    """SHIK supported elements include Si and O."""
    elems = shik_supported_elements()
    assert "Si" in elems
    assert "O" in elems


# ---------------------------------------------------------------------------
# potential_and_force
# ---------------------------------------------------------------------------


def test_potential_and_force_output_shape():
    """potential_and_force returns three arrays with the same shape as input r."""
    r = np.linspace(1.5, 5.0, 30)
    A, B, C, D = shik_params[("O", "Si")]
    r_out, V, F = potential_and_force(r, A, B, C, D)
    assert r_out.shape == r.shape
    assert V.shape == r.shape
    assert F.shape == r.shape


def test_potential_and_force_returns_numpy_arrays():
    """potential_and_force V and F outputs are numpy arrays."""
    r = np.array([2.0, 3.0])
    A, B, C, D = shik_params[("O", "O")]
    _r_out, V, F = potential_and_force(r, A, B, C, D)
    assert isinstance(V, np.ndarray)
    assert isinstance(F, np.ndarray)


def test_potential_and_force_same_r_returned():
    """potential_and_force returns the input r array unchanged."""
    r = np.linspace(1.0, 5.0, 10)
    A, B, C, D = shik_params[("O", "Si")]
    r_out, _, _ = potential_and_force(r, A, B, C, D)
    np.testing.assert_array_equal(r_out, r)


# ---------------------------------------------------------------------------
# compute_oxygen_charge
# ---------------------------------------------------------------------------


def test_compute_oxygen_charge_returns_negative():
    """Oxygen charge is negative for a typical SiO2 composition."""
    q_O = compute_oxygen_charge(_sio2_atoms_dict(), shik_charges)
    assert q_O < 0


def test_compute_oxygen_charge_neutrality_sio2():
    """Oxygen charge satisfies charge neutrality for SiO2 (1 Si, 2 O)."""
    q_O = compute_oxygen_charge(_sio2_atoms_dict(), shik_charges)
    total_charge = shik_charges["Si"] * 1 + q_O * 2
    assert total_charge == pytest.approx(0.0, abs=1e-6)


def test_compute_oxygen_charge_raises_without_oxygen():
    """compute_oxygen_charge raises ValueError when no oxygen atoms are present."""
    atoms_dict = {"atoms": [{"element": "Si"}, {"element": "Si"}]}
    with pytest.raises(ValueError, match="No oxygen atoms"):
        compute_oxygen_charge(atoms_dict, shik_charges)


# ---------------------------------------------------------------------------
# write_table_file
# ---------------------------------------------------------------------------


def test_write_table_file_creates_file(tmp_path):
    """write_table_file creates a .tbl file in the output directory."""
    pair = "Si-O"
    params = shik_params[("O", "Si")]
    outfile = write_table_file(pair, params, npoints=100, output_dir=tmp_path)
    assert outfile.exists()


def test_write_table_file_returns_correct_path(tmp_path):
    """write_table_file returns a path named table_Si_O.tbl."""
    pair = "Si-O"
    params = shik_params[("O", "Si")]
    outfile = write_table_file(pair, params, npoints=100, output_dir=tmp_path)
    assert outfile.name == "table_Si_O.tbl"


def test_write_table_file_contains_header(tmp_path):
    """Table file contains SHIK_Buck_r24 keyword and correct N count."""
    pair = "Si-O"
    params = shik_params[("O", "Si")]
    outfile = write_table_file(pair, params, npoints=100, output_dir=tmp_path)
    content = outfile.read_text()
    assert "SHIK_Buck_r24" in content
    assert "N 100" in content


def test_write_table_file_correct_number_of_data_lines(tmp_path):
    """Table file contains exactly npoints numbered data lines."""
    pair = "O-O"
    params = shik_params[("O", "O")]
    npoints = 50
    outfile = write_table_file(pair, params, npoints=npoints, output_dir=tmp_path)
    lines = outfile.read_text().splitlines()
    data_lines = [line for line in lines if line and line[0].isdigit()]
    assert len(data_lines) == npoints


# ---------------------------------------------------------------------------
# generate_shik_potential
# ---------------------------------------------------------------------------


def test_generate_shik_potential_returns_dataframe(tmp_path):
    """generate_shik_potential returns a pandas DataFrame."""
    result = generate_shik_potential(_sio2_atoms_dict(), output_dir=tmp_path)
    assert isinstance(result, pd.DataFrame)


def test_generate_shik_potential_non_empty(tmp_path):
    """SHIK DataFrame has at least one row."""
    result = generate_shik_potential(_sio2_atoms_dict(), output_dir=tmp_path)
    assert len(result) > 0


def test_generate_shik_potential_writes_table_files(tmp_path):
    """generate_shik_potential writes at least one .tbl file."""
    generate_shik_potential(_sio2_atoms_dict(), output_dir=tmp_path)
    tbl_files = list(tmp_path.glob("*.tbl"))
    assert len(tbl_files) > 0


def test_generate_shik_potential_contains_pair_style(tmp_path):
    """SHIK config lines include a pair_style entry."""
    result = generate_shik_potential(_sio2_atoms_dict(), output_dir=tmp_path)
    config = result["Config"].iloc[0]
    assert any("pair_style" in line for line in config)


def test_generate_shik_potential_melt_false_omits_run(tmp_path):
    """melt=False omits the run 10000 line from SHIK config."""
    result = generate_shik_potential(_sio2_atoms_dict(), output_dir=tmp_path, melt=False)
    config = result["Config"].iloc[0]
    assert not any("run 10000" in line for line in config)


def test_generate_shik_potential_melt_true_includes_run(tmp_path):
    """melt=True includes the run 10000 line in SHIK config."""
    result = generate_shik_potential(_sio2_atoms_dict(), output_dir=tmp_path, melt=True)
    config = result["Config"].iloc[0]
    assert any("run 10000" in line for line in config)
