"""Tests for potential generation utilities.

Author: Achraf Atila (achraf.atila@bam.de)
"""

import numpy as np
import pandas as pd
import pytest
from amorphouspy.lammps.potentials.bjp_potential import generate_bjp_potential
from amorphouspy.lammps.potentials.bjp_potential import supported_elements as bjp_supported_elements
from amorphouspy.lammps.potentials.bmp_potential import generate_bmp_potential
from amorphouspy.lammps.potentials.du_teter_potential import (
    Buckingham,
    Du,
    N4_dbx,
    V,
    _build_all_pair_params,
    _build_pair_params,
    _equations,
    _validate_du_teter_inputs,
    dBuckingham,
    dddBuckingham,
    dDu,
    ddV,
    du_teter_potential_params,
    dV,
    fit_BO_params,
    generate_du_teter_potential,
    get_A_for_BO,
    get_all_BO_params,
    stillinger_weber_params,
    write_sw_file,
)
from amorphouspy.lammps.potentials.du_teter_potential import (
    write_table_file as du_teter_write_table_file,
)
from amorphouspy.lammps.potentials.pmmcs_potential import generate_pmmcs_potential
from amorphouspy.lammps.potentials.pmmcs_potential import supported_elements as pmmcs_supported_elements
from amorphouspy.lammps.potentials.potential import (
    DsfConfig,
    EwaldConfig,
    PppmConfig,
    WolfConfig,
    compatible_potentials,
    generate_potential,
    get_supported_elements,
    select_potential,
)
from amorphouspy.lammps.potentials.shik_potential import (
    compute_oxygen_charge,
    generate_shik_potential,
    potential_and_force,
    shik_charges,
    shik_params,
    write_table_file,
)
from amorphouspy.lammps.potentials.shik_potential import (
    supported_elements as shik_supported_elements,
)
from amorphouspy.lammps.potentials.yang_potential import (
    generate_yang2026_potential,
    yang2026_charges,
    yang2026_params,
)
from amorphouspy.lammps.potentials.yang_potential import (
    supported_elements as yang_supported_elements,
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


# ---------------------------------------------------------------------------
# Du/Teter SW three-body
# ---------------------------------------------------------------------------


def _po_atoms_dict() -> dict:
    """P-O2 atoms_dict for testing SW file generation with a three-body term."""
    return {"atoms": [{"element": "P"}, {"element": "O"}, {"element": "O"}]}


def test_sw_file_is_created(tmp_path):
    """write_sw_file creates a file at the expected location."""
    sw_path = write_sw_file(["O", "P"], output_dir=tmp_path)
    assert sw_path.exists()


def test_sw_file_active_triplets_have_nonzero_lambda(tmp_path):
    """Active triplets in the SW file have the correct nonzero lambda values."""
    sw_path = write_sw_file(["O", "P"], output_dir=tmp_path)
    active = {(i, j, k): lam for (i, j, k), (lam, *_) in stillinger_weber_params.items()}
    for line in sw_path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split()
        key = (cols[0], cols[1], cols[2])
        lam = float(cols[6])
        if key in active:
            assert lam == active[key]
        else:
            assert lam == 0.0


def test_sw_file_twobody_always_zero(tmp_path):
    """Two-body entries in the SW file have zero lambda values."""
    sw_path = write_sw_file(["O", "P"], output_dir=tmp_path)
    for line in sw_path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split()
        assert float(cols[9]) == 0.0  # A column
        assert float(cols[10]) == 0.0  # B column


def test_generate_du_teter_no_three_body(tmp_path):
    """generate_du_teter_potential with use_three_body=False omits SW entries and pair_coeff sw."""
    df = generate_du_teter_potential(_po_atoms_dict(), output_dir=tmp_path, use_three_body=False)
    config = "".join(df["Config"].iloc[0])
    assert " sw" not in config


def test_generate_du_teter_three_body_adds_sw(tmp_path):
    """generate_du_teter_potential with use_three_body=True includes SW entries and pair_coeff sw."""
    df = generate_du_teter_potential(_po_atoms_dict(), output_dir=tmp_path, use_three_body=True)
    config = "".join(df["Config"].iloc[0])
    assert "table spline 11000 sw" in config
    assert "pair_coeff * * sw" in config


def test_generate_du_teter_three_body_requires_P(tmp_path):
    """generate_du_teter_potential with use_three_body=True raises ValueError if P is missing."""
    atoms = {"atoms": [{"element": "O"}, {"element": "K"}]}
    with pytest.raises(ValueError, match="phosphorus"):
        generate_du_teter_potential(atoms, output_dir=tmp_path, use_three_body=True)


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


# ---------------------------------------------------------------------------
# InteractionConfig — PMMCS
# ---------------------------------------------------------------------------


_METHOD_TO_CONFIG = {
    "dsf": DsfConfig,
    "wolf": WolfConfig,
    "pppm": PppmConfig,
    "ewald": EwaldConfig,
}


@pytest.mark.parametrize("method", ["dsf", "wolf"])
def test_pmmcs_dsf_wolf_pair_style_contains_alpha(method):
    """pair_style includes alpha for DSF/Wolf and no kspace_style is emitted."""
    cfg = _METHOD_TO_CONFIG[method](alpha=0.3, long_range_cutoff=9.0)
    result = generate_pmmcs_potential(_sio2_atoms_dict(), electrostatics=cfg)
    config = result["Config"].iloc[0]
    pair_style_lines = [line for line in config if "pair_style" in line]
    assert any("0.3" in line for line in pair_style_lines)
    assert not any("kspace_style" in line for line in config)


@pytest.mark.parametrize("method", ["pppm", "ewald"])
def test_pmmcs_pppm_ewald_uses_coul_long_and_kspace(method):
    """pair_style contains coul/long, kspace_style is emitted, alpha is absent."""
    cfg = _METHOD_TO_CONFIG[method]()
    result = generate_pmmcs_potential(_sio2_atoms_dict(), electrostatics=cfg)
    config = result["Config"].iloc[0]
    assert any("coul/long" in line for line in config)
    assert any(f"kspace_style {method}" in line for line in config)
    pair_style_lines = [line for line in config if "pair_style" in line]
    assert not any("alpha" in line or "0.25" in line for line in pair_style_lines)


def test_pmmcs_custom_long_range_cutoff_appears_in_config():
    """Custom long_range_cutoff appears in generated lines."""
    cfg = DsfConfig(long_range_cutoff=9.5)
    result = generate_pmmcs_potential(_sio2_atoms_dict(), electrostatics=cfg)
    config_text = "".join(result["Config"].iloc[0])
    assert "9.5" in config_text


# ---------------------------------------------------------------------------
# InteractionConfig — BJP
# ---------------------------------------------------------------------------


def test_bjp_pppm_uses_born_coul_long():
    """born/coul/long is used when method is 'pppm'."""
    cfg = PppmConfig()
    result = generate_bjp_potential(_cas_atoms_dict(), electrostatics=cfg)
    config = result["Config"].iloc[0]
    assert any("born/coul/long" in line for line in config)
    assert any("kspace_style pppm" in line for line in config)


# ---------------------------------------------------------------------------
# InteractionConfig — SHIK
# ---------------------------------------------------------------------------


def test_shik_custom_lr_cutoff_used_in_table_pair_coeff(tmp_path):
    """Custom long_range_cutoff propagates to the table pair_coeff line."""
    cfg = DsfConfig(long_range_cutoff=8.5)
    result = generate_shik_potential(_sio2_atoms_dict(), output_dir=tmp_path, electrostatics=cfg)
    config_text = "".join(result["Config"].iloc[0])
    assert "8.5" in config_text


@pytest.mark.parametrize("cfg", [WolfConfig(), PppmConfig(), EwaldConfig()])
def test_shik_rejects_non_dsf_methods(tmp_path, cfg):
    """TypeError is raised with a descriptive message for non-DSF methods."""
    with pytest.raises(TypeError, match="only supports 'dsf'"):
        generate_shik_potential(_sio2_atoms_dict(), output_dir=tmp_path, electrostatics=cfg)


# ---------------------------------------------------------------------------
# Defaults regression — all three potentials with electrostatics=None
# ---------------------------------------------------------------------------


def test_defaults_unchanged_pmmcs():
    """PMMCS with electrostatics=None produces DSF pair_style with default cutoffs."""
    result = generate_pmmcs_potential(_sio2_atoms_dict())
    config_text = "".join(result["Config"].iloc[0])
    assert "coul/dsf 0.25 8.0" in config_text
    assert "pedone 5.5" in config_text


def test_defaults_unchanged_bjp():
    """BJP with electrostatics=None produces born/coul/dsf pair_style with default cutoffs."""
    result = generate_bjp_potential(_cas_atoms_dict())
    config_text = "".join(result["Config"].iloc[0])
    assert "born/coul/dsf 0.25 8.0" in config_text


def test_defaults_unchanged_shik(tmp_path):
    """SHIK with electrostatics=None produces coul/dsf 0.2 10.0 pair_style."""
    result = generate_shik_potential(_sio2_atoms_dict(), output_dir=tmp_path)
    config_text = "".join(result["Config"].iloc[0])
    assert "coul/dsf 0.2 10.0" in config_text


# ---------------------------------------------------------------------------
# Potential configs are pure force-field descriptions (no MD run commands)
# ---------------------------------------------------------------------------


def test_generated_configs_contain_no_run_commands(tmp_path):
    """Generators emit no run/fix-dynamics lines; pre-equilibration lives in the protocols."""
    potentials = [
        generate_pmmcs_potential(_sio2_atoms_dict()),
        generate_bjp_potential(_cas_atoms_dict()),
        generate_shik_potential(_sio2_atoms_dict(), output_dir=tmp_path),
    ]
    for result in potentials:
        config = result["Config"].iloc[0]
        assert not any(line.strip().startswith("run ") for line in config)
        assert not any("langevinnve" in line for line in config)


# ---------------------------------------------------------------------------
# generate_bmp_potential — boron composition guard
# ---------------------------------------------------------------------------


def _nabs_atoms_dict() -> dict:
    """Na-B-Si-O: valid alkali borosilicate."""
    return {
        "atoms": [
            {"element": "Na"},
            {"element": "B"},
            {"element": "Si"},
            {"element": "O"},
            {"element": "O"},
            {"element": "O"},
        ]
    }


def _b_al_si_o_atoms_dict() -> dict:
    """B-Al-Si-O: invalid (Al not allowed when B is present)."""
    return {"atoms": [{"element": "B"}, {"element": "Al"}, {"element": "Si"}, {"element": "O"}]}


def _al_si_o_atoms_dict() -> dict:
    """Al-Si-O without boron: BMP has parameters and no D-model restriction applies."""
    return {"atoms": [{"element": "Al"}, {"element": "Si"}, {"element": "O"}, {"element": "O"}]}


def test_generate_bmp_harmonic_raises_for_al_with_boron(tmp_path):
    """Al is rejected when B is present (Dell-Bray model not valid for aluminoborosilicates)."""
    with pytest.raises(ValueError, match="Unsupported elements"):
        generate_bmp_potential(_b_al_si_o_atoms_dict(), output_dir=tmp_path, variant="harmonic")


def test_generate_bmp_screened_harmonic_raises_for_al_with_boron(tmp_path):
    """Same guard applies for the screened-harmonic variant."""
    with pytest.raises(ValueError, match="Unsupported elements"):
        generate_bmp_potential(_b_al_si_o_atoms_dict(), output_dir=tmp_path, variant="screened-harmonic")


def test_generate_bmp_harmonic_al_without_boron_is_allowed(tmp_path):
    """Al-Si-O without boron does not trigger the composition guard."""
    result = generate_bmp_potential(_al_si_o_atoms_dict(), output_dir=tmp_path, variant="harmonic")
    assert isinstance(result, pd.DataFrame)


def test_generate_bmp_harmonic_valid_nabs(tmp_path):
    """Na-B-Si-O returns a DataFrame for the harmonic variant."""
    result = generate_bmp_potential(_nabs_atoms_dict(), output_dir=tmp_path, variant="harmonic")
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_generate_bmp_screened_harmonic_valid_nabs(tmp_path):
    """Na-B-Si-O returns a DataFrame for the screened-harmonic variant."""
    result = generate_bmp_potential(_nabs_atoms_dict(), output_dir=tmp_path, variant="screened-harmonic")
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# generate_potential dispatcher
# ---------------------------------------------------------------------------


def test_generate_potential_pmmcs():
    """generate_potential dispatches to pmmcs generator."""
    result = generate_potential(_sio2_atoms_dict(), potential_type="pmmcs")
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_generate_potential_bjp():
    """generate_potential dispatches to bjp generator."""
    result = generate_potential(_cas_atoms_dict(), potential_type="bjp")
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_generate_potential_shik(tmp_path, monkeypatch):
    """generate_potential dispatches to shik generator."""
    monkeypatch.chdir(tmp_path)
    result = generate_potential(_sio2_atoms_dict(), potential_type="shik")
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_generate_potential_du_teter(tmp_path, monkeypatch):
    """generate_potential dispatches to du_teter generator."""
    monkeypatch.chdir(tmp_path)
    result = generate_potential(_sio2_atoms_dict(), potential_type="du_teter")
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_generate_potential_bmp_harmonic(tmp_path, monkeypatch):
    """generate_potential dispatches to bmp-harmonic generator."""
    monkeypatch.chdir(tmp_path)
    result = generate_potential(_nabs_atoms_dict(), potential_type="bmp-harmonic")
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_generate_potential_bmp_screened_harmonic(tmp_path, monkeypatch):
    """generate_potential dispatches to bmp-screened-harmonic generator."""
    monkeypatch.chdir(tmp_path)
    result = generate_potential(_nabs_atoms_dict(), potential_type="bmp-screened-harmonic")
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_generate_potential_unsupported_raises():
    """generate_potential raises ValueError for unknown potential type."""
    with pytest.raises(ValueError, match="Unsupported potential type"):
        generate_potential(_sio2_atoms_dict(), potential_type="unknown_potential")


@pytest.mark.parametrize(
    ("potential_type", "atoms_dict", "should_work"),
    [
        ("du_teter", _po_atoms_dict(), True),
        ("du_teter_dbx_generalized", _po_atoms_dict(), True),
        ("pmmcs", _sio2_atoms_dict(), False),
        ("bjp", _cas_atoms_dict(), False),
        ("shik", _sio2_atoms_dict(), False),
        ("bmp-harmonic", _nabs_atoms_dict(), False),
    ],
)
def test_generate_potential_three_body_option(tmp_path, monkeypatch, potential_type, atoms_dict, should_work):
    """use_three_body option is only valid for du_teter potentials; other potentials raise ValueError."""
    monkeypatch.chdir(tmp_path)
    if should_work:
        result = generate_potential(atoms_dict, potential_type=potential_type, use_three_body=True)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # Verify that SW configuration is included for du_teter
        if potential_type.startswith("du_teter"):
            config = "".join(result["Config"].iloc[0])
            assert "pair_coeff * * sw" in config
    else:
        with pytest.raises(ValueError, match="use_three_body is only supported for du_teter"):
            generate_potential(atoms_dict, potential_type=potential_type, use_three_body=True)


# ---------------------------------------------------------------------------
# dddBuckingham
# ---------------------------------------------------------------------------


def test_dddBuckingham_returns_float():
    """Function dddBuckingham returns a finite float."""
    result = dddBuckingham(2.0, A=13702.905, rho=0.193817, C=54.681)
    assert np.isfinite(result)


def test_dddBuckingham_changes_sign():
    """Function dddBuckingham crosses zero (used to find inflection point for crossover)."""
    vals = [dddBuckingham(r, A=13702.905, rho=0.193817, C=54.681) for r in [0.5, 2.0]]
    assert vals[0] * vals[1] < 0 or any(v == 0 for v in vals)


# ---------------------------------------------------------------------------
# V, dV, ddV (short-range repulsion)
# ---------------------------------------------------------------------------


def test_V_basic():
    """V returns expected value for simple inputs."""
    assert V(1.0, B=10.0, n=2.0, D=-1.0) == pytest.approx(10.0 - 1.0)


def test_dV_basic():
    """Function dV returns expected derivative value."""
    assert dV(1.0, B=10.0, n=2.0, D=-1.0) == pytest.approx(-20.0 - 2.0)


def test_ddV_basic():
    """Function ddV returns expected second derivative value."""
    assert ddV(1.0, B=10.0, n=2.0, D=-1.0) == pytest.approx(60.0 - 2.0)


# ---------------------------------------------------------------------------
# Du and dDu
# ---------------------------------------------------------------------------


def test_Du_short_range_branch():
    """Function Du returns V(r) when r <= rc."""
    params = {"A": 100.0, "rho": 0.2, "C": 50.0, "B": 30.0, "n": 3.0, "D": -5.0, "rc": 2.0}
    r = 1.5
    assert Du(r, **params) == V(r, B=30.0, n=3.0, D=-5.0)


def test_Du_long_range_branch():
    """Function Du returns Buckingham(r) when r > rc."""
    params = {"A": 100.0, "rho": 0.2, "C": 50.0, "B": 30.0, "n": 3.0, "D": -5.0, "rc": 1.0}
    r = 1.5
    assert Du(r, **params) == Buckingham(r, A=100.0, rho=0.2, C=50.0)


def test_dDu_short_range_branch():
    """Function dDu returns -dV(r) when r <= rc."""
    params = {"A": 100.0, "rho": 0.2, "C": 50.0, "B": 30.0, "n": 3.0, "D": -5.0, "rc": 2.0}
    r = 1.5
    assert dDu(r, **params) == -dV(r, B=30.0, n=3.0, D=-5.0)


def test_dDu_long_range_branch():
    """Function dDu returns -dBuckingham(r) when r > rc."""
    params = {"A": 100.0, "rho": 0.2, "C": 50.0, "B": 30.0, "n": 3.0, "D": -5.0, "rc": 1.0}
    r = 1.5
    assert dDu(r, **params) == -dBuckingham(r, A=100.0, rho=0.2, C=50.0)


# ---------------------------------------------------------------------------
# N4_dbx - Dell-Bray-Xiao model branches
# ---------------------------------------------------------------------------


def test_N4_dbx_negative_R_raises():
    """Raises ValueError when R is negative."""
    with pytest.raises(ValueError, match="R must be non-negative"):
        N4_dbx(-0.1, K=1)


def test_N4_dbx_negative_K_raises():
    """Raises ValueError when K is negative."""
    with pytest.raises(ValueError, match="K must be non-negative"):
        N4_dbx(0.5, K=-1)


def test_N4_dbx_K_above_model_limit_raises():
    """Raises ValueError when K exceeds the model limit of 8."""
    with pytest.raises(ValueError, match="only valid for K"):
        N4_dbx(1.0, K=9)


def test_N4_dbx_linear_ramp_below_R_CUT():
    """N4 = R when R < 0.5 (R_CUT), regardless of K."""
    assert N4_dbx(0.3, K=1) == pytest.approx(0.3)
    assert N4_dbx(0.0, K=2) == pytest.approx(0.0)


def test_N4_dbx_linear_ramp_below_R_MAX():
    """N4 = R when R < R_MAX (K/16 + 0.5) — initial slope-1 region."""
    # K=4: R_MAX = 4/16 + 0.5 = 0.75, so R=0.6 is in the ramp region
    assert N4_dbx(0.6, K=4) == pytest.approx(0.6)


def test_N4_dbx_plateau_between_R_MAX_and_R_D1():
    """N4 = R_MAX when R_MAX <= R < R_D1 (plateau region)."""
    # K=1: R_MAX = 0.5625, R_D1 = 0.75 — R=0.65 is on the plateau
    R_MAX = 1 / 16 + 0.5
    assert N4_dbx(0.65, K=1) == pytest.approx(R_MAX)


def test_N4_dbx_plateau_K_equals_zero():
    """With K=0 all breakpoints coincide at 0.5; below 0.5 N4 = R."""
    # K=0: R_MAX = R_D1 = 0.5, R_D3 = 2 — plateau has zero width
    assert N4_dbx(0.3, K=0) == pytest.approx(0.3)


def test_N4_dbx_linear_decrease_between_R_D1_and_R_D3():
    """N4 decreases linearly from R_MAX to 0 between R_D1 and R_D3."""
    K = 1
    R_MAX = K / 16 + 0.5  # 0.5625
    R_D1 = K / 4 + 0.5  # 0.75
    R_D3 = 2 + K  # 3.0
    R = 1.5
    expected = R_MAX - (R - R_D1) * R_MAX / (R_D3 - R_D1)
    assert N4_dbx(R, K=K) == pytest.approx(expected)


def test_N4_dbx_linear_decrease_is_monotone():
    """N4 is monotonically decreasing between R_D1 and R_D3."""
    K = 2
    R_D1 = K / 4 + 0.5
    R_D3 = 2 + K
    r_values = [R_D1 + i * (R_D3 - R_D1) / 10 for i in range(11)]
    n4_values = [N4_dbx(r, K=K) for r in r_values]
    assert all(n4_values[i] >= n4_values[i + 1] for i in range(len(n4_values) - 1))


def test_N4_dbx_zero_at_R_D3():
    """N4 = 0 exactly at R_D3."""
    K = 1
    R_D3 = 2 + K
    assert N4_dbx(R_D3, K=K) == pytest.approx(0.0)


def test_N4_dbx_zero_beyond_R_D3():
    """N4 = 0 for R > R_D3 (K != 0)."""
    assert N4_dbx(5.0, K=1) == pytest.approx(0.0)
    assert N4_dbx(3.5, K=1) == pytest.approx(0.0)


def test_N4_dbx_zero_beyond_R_D3_K_zero():
    """N4 = 0 for R >= R_D3 when K=0 (R_D3 = 2)."""
    assert N4_dbx(2.5, K=0) == pytest.approx(0.0)


def test_N4_dbx_output_in_unit_interval():
    """N4 is always in [0, 1] across the full valid parameter space."""
    for K in [0, 0.5, 1, 2, 4, 8]:
        for R in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
            result = N4_dbx(R, K=K)
            assert 0.0 <= result <= 1.0, f"N4={result} out of [0,1] for R={R}, K={K}"


def test_N4_dbx_K_at_model_limit():
    """K=8 is accepted (boundary of valid range)."""
    result = N4_dbx(1.0, K=8)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# get_A_for_BO
# ---------------------------------------------------------------------------


def test_get_A_for_BO_R_greater_than_RMAX():
    """get_A_for_BO uses the formula for R > R_MAX."""
    A = get_A_for_BO(K=1, R=1.0, N4=0.5)
    assert A > 0
    assert A <= 25000


def test_get_A_for_BO_R_leq_RMAX():
    """get_A_for_BO uses the formula for R <= R_MAX."""
    A = get_A_for_BO(K=1, R=0.5, N4=0.4)
    assert A > 0
    assert A <= 25000


# ---------------------------------------------------------------------------
# _equations
# ---------------------------------------------------------------------------


def test_equations_returns_tuple_of_three():
    """_equations returns a 3-tuple of residuals."""
    p = du_teter_potential_params["Si"]
    buck_params = {"A": p["A"], "rho": p["rho"], "C": p["C"]}
    rc = p["r0"]
    x = (p["B"], p["n"], p["D"])
    residuals = _equations(x, rc, buck_params)
    assert len(residuals) == 3
    assert all(np.isfinite(r) for r in residuals)


def test_equations_penalty_for_negative_n():
    """_equations returns large penalty values when n < 0."""
    buck_params = {"A": 100.0, "rho": 0.2, "C": 50.0}
    result = _equations((-1.0, -2.0, 0.0), r=1.0, buck_params=buck_params)
    assert result == (100.0, 100.0, 100.0)


# ---------------------------------------------------------------------------
# fit_BO_params
# ---------------------------------------------------------------------------


def test_fit_BO_params_returns_all_keys():
    """fit_BO_params returns a dict with keys A, rho, C, B, n, D, r0."""
    result = fit_BO_params(K=2.0, R=0.5)
    expected_keys = {"A", "rho", "C", "B", "n", "D", "r0"}
    assert set(result.keys()) == expected_keys


def test_fit_BO_params_continuity_at_crossover():
    """Potential is continuous at crossover point."""
    result = fit_BO_params(K=2.0, R=0.5)
    rc = result["r0"]
    buck_val = Buckingham(rc, A=result["A"], rho=result["rho"], C=result["C"])
    v_val = V(rc, B=result["B"], n=result["n"], D=result["D"])
    assert buck_val == pytest.approx(v_val, rel=1e-4)


def test_fit_BO_params_with_explicit_N4():
    """fit_BO_params accepts an explicit N4 value."""
    result = fit_BO_params(K=2.0, R=0.5, N4=0.4)
    assert "r0" in result
    assert result["r0"] > 0


# ---------------------------------------------------------------------------
# get_all_BO_params
# ---------------------------------------------------------------------------

_NABS_STRUCTURE = {
    "atoms": [{"element": "B"}, {"element": "O"}, {"element": "O"}, {"element": "Na"}, {"element": "Si"}],
    "mol_fraction": {"B2O3": 0.2, "SiO2": 0.6, "Na2O": 0.2},
}


def test_get_all_BO_params_returns_dict():
    """get_all_BO_params returns a dict with all expected keys."""
    result = get_all_BO_params(_NABS_STRUCTURE)
    expected_keys = {"A", "rho", "C", "B", "n", "D", "rc"}
    assert expected_keys == set(result.keys())


def test_get_all_BO_params_rc_positive():
    """get_all_BO_params returns a positive crossover distance."""
    result = get_all_BO_params(_NABS_STRUCTURE)
    assert result["rc"] > 0


def test_get_all_BO_params_default_mol_fractions():
    """get_all_BO_params works when only B2O3 is given (all other fractions default to 0)."""
    structure_dict = {
        "atoms": [{"element": "B"}, {"element": "O"}],
        "mol_fraction": {"B2O3": 1.0},
    }
    result = get_all_BO_params(structure_dict)
    assert result["rc"] > 0


def test_get_all_BO_params_original_approach_uses_only_Na2O():
    """n4_model='dbx' uses R = cNa2O / cB2O3 (ignores other modifiers)."""
    # Same composition but different modifier: Ca instead of Na
    struct_na = {
        "atoms": [{"element": "B"}, {"element": "O"}, {"element": "Na"}],
        "mol_fraction": {"B2O3": 0.3, "Na2O": 0.2, "CaO": 0.0},
    }
    struct_ca = {
        "atoms": [{"element": "B"}, {"element": "O"}, {"element": "Ca"}],
        "mol_fraction": {"B2O3": 0.3, "Na2O": 0.0, "CaO": 0.2},
    }
    result_na = get_all_BO_params(struct_na, n4_model="dbx")
    result_ca = get_all_BO_params(struct_ca, n4_model="dbx")
    # Ca contributes nothing to R in the original approach (only Na2O counted)
    # → result_ca should have R=0, result_na has R=0.2/0.3 > 0, giving different A
    assert result_na["A"] != result_ca["A"]


def test_get_all_BO_params_modified_approach_includes_all_modifiers():
    """n4_model='dbx_generalized' sums all modifiers (Li2O, Na2O, K2O, MgO, CaO, SrO, BaO, BeO minus Al2O3)."""
    struct_na = {
        "atoms": [{"element": "B"}, {"element": "O"}, {"element": "Na"}],
        "mol_fraction": {"B2O3": 0.3, "Na2O": 0.2, "CaO": 0.0},
    }
    struct_ca = {
        "atoms": [{"element": "B"}, {"element": "O"}, {"element": "Ca"}],
        "mol_fraction": {"B2O3": 0.3, "Na2O": 0.0, "CaO": 0.2},
    }
    result_na = get_all_BO_params(struct_na, n4_model="dbx_generalized")
    result_ca = get_all_BO_params(struct_ca, n4_model="dbx_generalized")
    # Both have the same total modifier / B2O3 ratio → identical parameters
    assert result_na["A"] == pytest.approx(result_ca["A"], rel=1e-6)
    assert result_na["rc"] == pytest.approx(result_ca["rc"], rel=1e-4)


def test_get_all_BO_params_original_is_default():
    """n4_model defaults to 'dbx'."""
    result_default = get_all_BO_params(_NABS_STRUCTURE)
    result_explicit = get_all_BO_params(_NABS_STRUCTURE, n4_model="dbx")
    assert result_default["A"] == pytest.approx(result_explicit["A"])
    assert result_default["rc"] == pytest.approx(result_explicit["rc"])


# ---------------------------------------------------------------------------
# _build_all_pair_params with boron
# ---------------------------------------------------------------------------


def test_build_all_pair_params_with_boron():
    """_build_all_pair_params includes B-O when 'B' is in species."""
    structure_dict = {
        "atoms": [{"element": "B"}, {"element": "O"}, {"element": "Na"}, {"element": "Si"}],
        "mol_fraction": {"B2O3": 0.3, "SiO2": 0.5, "Na2O": 0.2},
    }
    result = _build_all_pair_params(["O", "B", "Na", "Si"], structure_dict)
    assert "B-O" in result
    assert "O-O" in result


# ---------------------------------------------------------------------------
# _validate_du_teter_inputs
# ---------------------------------------------------------------------------


def test_validate_du_teter_inputs_no_oxygen():
    """Raises ValueError when oxygen is not in species."""
    with pytest.raises(ValueError, match="Oxygen must be present"):
        _validate_du_teter_inputs(["Si", "Na"], use_three_body=False)


def test_validate_du_teter_inputs_unsupported_element():
    """Raises ValueError for unsupported elements."""
    with pytest.raises(ValueError, match="does not include parameters"):
        _validate_du_teter_inputs(["O", "Xe"], use_three_body=False)


# ---------------------------------------------------------------------------
# generate_du_teter_potential — pure force-field config
# ---------------------------------------------------------------------------


def test_generate_du_teter_omits_langevin(tmp_path):
    """The generated config contains no pre-equilibration block."""
    atoms = {"atoms": [{"element": "Si"}, {"element": "O"}, {"element": "O"}]}
    df = generate_du_teter_potential(atoms, output_dir=str(tmp_path))
    config = "".join(df["Config"].iloc[0])
    assert "langevin" not in config
    assert "run 10000" not in config


# ---------------------------------------------------------------------------
# write_table_file (Du/Teter specific)
# ---------------------------------------------------------------------------


def test_du_teter_write_table_file_creates_table(tmp_path):
    """write_table_file creates a valid DU_TETER table."""
    params = _build_pair_params("Si")
    path = du_teter_write_table_file("Si-O", params, npoints=100, output_dir=str(tmp_path))
    assert path.exists()
    content = path.read_text()
    assert "DU_TETER" in content
    assert "N 100" in content


# ---------------------------------------------------------------------------
# generate_du_teter_potential with boron composition
# ---------------------------------------------------------------------------


def test_generate_du_teter_with_boron(tmp_path):
    """generate_du_teter_potential handles compositions with boron."""
    atoms = {
        "atoms": [
            {"element": "B"},
            {"element": "O"},
            {"element": "O"},
            {"element": "Na"},
            {"element": "Si"},
            {"element": "O"},
        ],
        "mol_fraction": {"B2O3": 0.3, "SiO2": 0.5, "Na2O": 0.2},
    }
    df = generate_du_teter_potential(atoms, output_dir=str(tmp_path))
    config = "".join(df["Config"].iloc[0])
    assert "pair_coeff" in config
    tbl_files = list(tmp_path.glob("table_B_O*"))
    assert len(tbl_files) == 1


def test_generate_du_teter_original_dbx_approach_default(tmp_path):
    """n4_model defaults to 'dbx' (Na2O-only R)."""
    atoms = {
        "atoms": [{"element": "B"}, {"element": "O"}, {"element": "Na"}],
        "mol_fraction": {"B2O3": 0.3, "Na2O": 0.2},
    }
    generate_du_teter_potential(atoms, output_dir=str(tmp_path / "default"))
    generate_du_teter_potential(atoms, output_dir=str(tmp_path / "explicit"), n4_model="dbx")
    # Both should produce a table file for B-O
    assert list((tmp_path / "default").glob("table_B_O*"))
    assert list((tmp_path / "explicit").glob("table_B_O*"))


def test_generate_du_teter_original_vs_modified_dbx_approach_differ(tmp_path):
    """n4_model='dbx_generalized' (all modifiers) changes B-O parameters vs 'dbx' (Na only)."""
    # Na-B-O: original approach counts Na2O; modified approach also counts Na2O — same here
    # Use CaO only so original (Na-only) gives R=0, modified gives R=cCaO/cB2O3 > 0 → different A
    atoms = {
        "atoms": [{"element": "B"}, {"element": "O"}, {"element": "Ca"}],
        "mol_fraction": {"B2O3": 0.3, "CaO": 0.2},
    }

    result_orig = get_all_BO_params(atoms, n4_model="dbx")  # R = cNa2O / cB2O3 = 0
    result_mod = get_all_BO_params(atoms, n4_model="dbx_generalized")  # R = cCaO / cB2O3 > 0
    assert result_orig["A"] != result_mod["A"]

    # generate_du_teter_potential should accept and pass through the flag
    df = generate_du_teter_potential(atoms, output_dir=str(tmp_path), n4_model="dbx_generalized")
    assert "pair_coeff" in "".join(df["Config"].iloc[0])


# yang_potential.supported_elements
# ---------------------------------------------------------------------------


def test_yang_supported_elements_exact_set():
    """Yang2026 supports exactly {Ca, Na, B, Si, O} — no more, no less."""
    assert yang_supported_elements() == {"Ca", "Na", "B", "Si", "O"}


# ---------------------------------------------------------------------------
# yang2026_charges — physics constraints
# ---------------------------------------------------------------------------


def test_yang2026_charges_neutral_for_nabsio():
    """Na+B+Si+4O unit is charge-neutral with the published Yang2026 charges."""
    total = yang2026_charges["Na"] + yang2026_charges["B"] + yang2026_charges["Si"] + 4 * yang2026_charges["O"]
    assert total == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# generate_yang2026_potential — defaults regression
# ---------------------------------------------------------------------------


def _yang_nabo_atoms_dict() -> dict:
    """Na-B-Si-O atoms_dict for Yang2026 tests."""
    return {
        "atoms": [
            {"element": "Na"},
            {"element": "B"},
            {"element": "Si"},
            {"element": "O"},
            {"element": "O"},
            {"element": "O"},
        ]
    }


def test_yang2026_defaults_dsf_alpha_and_cutoff():
    """Default electrostatics is DSF with alpha=0.182 and cutoff=11.0 Å."""
    result = generate_yang2026_potential(_yang_nabo_atoms_dict())
    config_text = "".join(result["Config"].iloc[0])
    assert "coul/dsf 0.182 11.0" in config_text
    assert "hybrid/overlay" in config_text
    assert "kspace_style" not in config_text


# ---------------------------------------------------------------------------
# generate_yang2026_potential — electrostatics variants
# ---------------------------------------------------------------------------


def test_yang2026_pppm_emits_kspace_and_no_alpha():
    """PPPM path emits kspace_style and buck/coul/long; alpha must not appear."""
    result = generate_yang2026_potential(_yang_nabo_atoms_dict(), electrostatics=PppmConfig())
    config_text = "".join(result["Config"].iloc[0])
    assert "kspace_style pppm" in config_text
    assert "buck/coul/long" in config_text
    assert "0.182" not in config_text


def test_yang2026_wolf_emits_no_kspace():
    """Wolf path uses hybrid/overlay coul/wolf and emits no kspace_style."""
    result = generate_yang2026_potential(_yang_nabo_atoms_dict(), electrostatics=WolfConfig())
    config_text = "".join(result["Config"].iloc[0])
    assert "coul/wolf" in config_text
    assert "kspace_style" not in config_text


def test_yang2026_custom_long_range_cutoff_propagates():
    """A custom long_range_cutoff value appears in the generated config."""
    result = generate_yang2026_potential(_yang_nabo_atoms_dict(), electrostatics=DsfConfig(long_range_cutoff=14.0))
    config_text = "".join(result["Config"].iloc[0])
    assert "14.0" in config_text


@pytest.mark.parametrize(
    ("cfg", "expected_label"),
    [
        (None, "DSF"),
        (WolfConfig(), "WOLF"),
        (PppmConfig(), "PPPM"),
        (EwaldConfig(), "EWALD"),
    ],
)
def test_yang2026_model_column_reflects_electrostatics(cfg, expected_label):
    """Model column encodes the electrostatics method name in upper case."""
    result = generate_yang2026_potential(_yang_nabo_atoms_dict(), electrostatics=cfg)
    assert expected_label in result["Model"].iloc[0]


# ---------------------------------------------------------------------------
# generate_yang2026_potential — pair coefficients
# ---------------------------------------------------------------------------


def test_yang2026_pair_coeff_values_match_params():
    """O-O Buckingham A, rho, C values in config text match yang2026_params exactly."""
    result = generate_yang2026_potential({"atoms": [{"element": "Si"}, {"element": "O"}]})
    config_text = "".join(result["Config"].iloc[0])
    A, rho, C = yang2026_params[("O", "O")]
    assert f"{A:.6f}" in config_text
    assert f"{rho:.6f}" in config_text
    assert f"{C:.6f}" in config_text


def test_yang2026_pair_coeff_no_duplicates():
    """Each interacting pair appears exactly once in pair_coeff lines."""
    result = generate_yang2026_potential(_yang_nabo_atoms_dict())
    coeff_lines = [
        line for line in result["Config"].iloc[0] if line.strip().startswith("pair_coeff") and "*" not in line
    ]
    pairs = [tuple(line.split()[1:3]) for line in coeff_lines]
    seen: set[tuple[str, str]] = set()
    for i, j in pairs:
        key = (min(i, j), max(i, j))
        assert key not in seen, f"Duplicate pair_coeff for types {i} {j}"
        seen.add(key)


def test_yang2026_pair_coeff_count_for_sio2():
    """SiO2 emits exactly 2 pair_coeffs (O-O and Si-O); Si-Si has no params."""
    result = generate_yang2026_potential({"atoms": [{"element": "Si"}, {"element": "O"}]})
    coeff_lines = [
        line for line in result["Config"].iloc[0] if line.strip().startswith("pair_coeff") and "*" not in line
    ]
    assert len(coeff_lines) == 2


# ---------------------------------------------------------------------------
# generate_yang2026_potential — composition guard
# ---------------------------------------------------------------------------


def test_yang2026_raises_for_unsupported_element():
    """Al is not in Yang2026; passing it raises ValueError with a clear message."""
    atoms_dict = {"atoms": [{"element": "Al"}, {"element": "Si"}, {"element": "O"}]}
    with pytest.raises(ValueError, match="Yang2026 potential does not include parameters for elements"):
        generate_yang2026_potential(atoms_dict)


def test_yang2026_species_column_matches_input():
    """Species column contains exactly the elements from the input atoms_dict."""
    atoms_dict = {"atoms": [{"element": "Na"}, {"element": "Si"}, {"element": "O"}]}
    result = generate_yang2026_potential(atoms_dict)
    assert set(result["Species"].iloc[0]) == {"Na", "Si", "O"}


# ---------------------------------------------------------------------------
# generate_yang2026_potential — melt block
# ---------------------------------------------------------------------------


def test_yang2026_omits_melt_block():
    """The generated config contains no pre-equilibration block."""
    result = generate_yang2026_potential(_yang_nabo_atoms_dict())
    config = "".join(result["Config"].iloc[0])
    assert "run 10000" not in config
    assert "langevinnve" not in config
