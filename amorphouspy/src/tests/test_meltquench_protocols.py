"""Unit tests for meltquench protocols with dataclass parameters."""

from unittest.mock import MagicMock

import amorphouspy.fabrication.meltquench as mq_module
import pandas as pd
import pytest
from amorphouspy.fabrication.meltquench import melt_quench_simulation
from amorphouspy.fabrication.meltquench_protocols import (
    MeltQuenchParams,
    bjp_protocol,
    bmp_protocol,
    du_teter_protocol,
    pmmcs_protocol,
    shik_protocol,
    yang2026_protocol,
)
from ase import Atoms


@pytest.fixture
def mock_structure():
    """Create a mock Atoms structure."""
    return Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])


@pytest.fixture
def mock_potential():
    """Create a mock potential dataframe."""
    return pd.DataFrame(
        {
            "Name": ["pmmcs"],
            "Config": [["line1", "line2"]],
        }
    )


@pytest.fixture
def mock_runner():
    """Create a mock runner function."""
    mock = MagicMock()
    # Return a structure and parsed output
    mock.return_value = (Atoms("H2O", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]), {"generic": {"data": []}})
    return mock


def test_meltquench_params_creation(mock_structure, mock_potential):
    """Test that MeltQuenchParams dataclass can be created successfully."""
    params = MeltQuenchParams(
        structure=mock_structure,
        potential=mock_potential,
        temperature_high=5000.0,
        temperature_low=300.0,
        cooling_steps=100_000,
        timestep=1.0,
        n_dump=1000,
        langevin=False,
        seed=12345,
    )

    assert params.structure == mock_structure
    assert params.potential.equals(mock_potential)
    assert params.temperature_high == 5000.0
    assert params.temperature_low == 300.0
    assert params.cooling_steps == 100_000
    assert params.timestep == 1.0
    assert params.n_dump == 1000
    assert params.langevin is False
    assert params.seed == 12345
    assert params.server_kwargs is None
    assert params.tmp_working_directory is None


def test_meltquench_params_with_optional_values(mock_structure, mock_potential, tmp_path):
    """Test that MeltQuenchParams dataclass handles optional parameters."""
    params = MeltQuenchParams(
        structure=mock_structure,
        potential=mock_potential,
        temperature_high=5000.0,
        temperature_low=300.0,
        cooling_steps=100_000,
        timestep=1.0,
        n_dump=1000,
        langevin=True,
        seed=12345,
        server_kwargs={"cores": 4},
        tmp_working_directory=str(tmp_path),
    )

    assert params.server_kwargs == {"cores": 4}
    assert params.tmp_working_directory == str(tmp_path)


def test_pmmcs_protocol_accepts_dataclass(mock_runner, mock_structure, mock_potential):
    """Test that pmmcs_protocol accepts MeltQuenchParams dataclass."""
    params = MeltQuenchParams(
        structure=mock_structure,
        potential=mock_potential,
        temperature_high=5000.0,
        temperature_low=300.0,
        cooling_steps=100_000,
        timestep=1.0,
        n_dump=1000,
        langevin=False,
        seed=12345,
    )

    structure, output = pmmcs_protocol(mock_runner, params)

    assert mock_runner.called
    assert structure is not None
    assert output is not None


def test_bjp_protocol_accepts_dataclass(mock_runner, mock_structure, mock_potential):
    """Test that bjp_protocol accepts MeltQuenchParams dataclass."""
    params = MeltQuenchParams(
        structure=mock_structure,
        potential=mock_potential,
        temperature_high=5000.0,
        temperature_low=300.0,
        cooling_steps=100_000,
        timestep=1.0,
        n_dump=1000,
        langevin=False,
        seed=12345,
    )

    structure, output = bjp_protocol(mock_runner, params)

    assert mock_runner.called
    assert structure is not None
    assert output is not None


def test_shik_protocol_accepts_dataclass(mock_runner, mock_structure):
    """Test that shik_protocol accepts MeltQuenchParams dataclass."""
    potential = pd.DataFrame(
        {
            "Name": ["shik"],
            "Config": [["fix langevin all langevin 5000 5000 0.01 48279", "other line"]],
        }
    )

    params = MeltQuenchParams(
        structure=mock_structure,
        potential=potential,
        temperature_high=5000.0,
        temperature_low=300.0,
        cooling_steps=100_000,
        timestep=1.0,
        n_dump=1000,
        langevin=False,
        seed=12345,
    )

    structure, output = shik_protocol(mock_runner, params)

    assert mock_runner.called
    assert structure is not None
    assert output is not None


def test_pmmcs_protocol_calls_runner_correctly(mock_runner, mock_structure, mock_potential):
    """Test that pmmcs_protocol calls the runner with correct parameters."""
    params = MeltQuenchParams(
        structure=mock_structure,
        potential=mock_potential,
        temperature_high=5000.0,
        temperature_low=300.0,
        cooling_steps=200_000,
        timestep=1.0,
        n_dump=1000,
        langevin=True,
        seed=12345,
    )

    pmmcs_protocol(mock_runner, params)

    # 4 runner calls: stage 0 pre-equilibration + equilibrate, cool, release
    assert mock_runner.call_count == 4


def test_bjp_protocol_calls_runner_correctly(mock_runner, mock_structure, mock_potential):
    """Test that bjp_protocol calls the runner with correct parameters."""
    params = MeltQuenchParams(
        structure=mock_structure,
        potential=mock_potential,
        temperature_high=5000.0,
        temperature_low=300.0,
        cooling_steps=200_000,
        timestep=1.0,
        n_dump=1000,
        langevin=True,
        seed=12345,
    )

    bjp_protocol(mock_runner, params)

    # 4 runner calls: stage 0 pre-equilibration + equilibrate, cool, release
    assert mock_runner.call_count == 4


def test_shik_protocol_calls_runner_correctly(mock_runner, mock_structure):
    """Test that shik_protocol calls the runner with correct parameters."""
    potential = pd.DataFrame(
        {
            "Name": ["shik"],
            "Config": [["fix langevin all langevin 5000 5000 0.01 48279", "other line"]],
        }
    )

    params = MeltQuenchParams(
        structure=mock_structure,
        potential=potential,
        temperature_high=5000.0,
        temperature_low=300.0,
        cooling_steps=200_000,
        timestep=1.0,
        n_dump=1000,
        langevin=True,
        seed=12345,
    )

    shik_protocol(mock_runner, params)

    # 5 stages: pre-equilibration, NVT melt equil., NPT equil., NPT quench, NPT anneal
    assert mock_runner.call_count == 5


def _make_params(structure, potential, **kwargs):
    defaults = {
        "structure": structure,
        "potential": potential,
        "temperature_high": 4000.0,
        "temperature_low": 300.0,
        "cooling_steps": 100_000,
        "timestep": 1.0,
        "n_dump": 1000,
        "langevin": False,
        "seed": 12345,
    }
    defaults.update(kwargs)
    return MeltQuenchParams(**defaults)


# ---------------------------------------------------------------------------
# bmp_protocol
# ---------------------------------------------------------------------------


def test_bmp_protocol_accepts_dataclass(mock_runner, mock_structure, mock_potential):
    """bmp_protocol accepts MeltQuenchParams and returns structure + history."""
    params = _make_params(mock_structure, mock_potential)
    structure, history = bmp_protocol(mock_runner, params)
    assert structure is not None
    assert history is not None


def test_bmp_protocol_calls_runner_4_times(mock_runner, mock_structure, mock_potential):
    """bmp_protocol calls the runner exactly 4 times (stage 0 + 3 stages)."""
    params = _make_params(mock_structure, mock_potential)
    bmp_protocol(mock_runner, params)
    assert mock_runner.call_count == 4


def test_bmp_protocol_returns_4_history_entries(mock_runner, mock_structure, mock_potential):
    """bmp_protocol returns a list of 4 history entries."""
    params = _make_params(mock_structure, mock_potential)
    _, history = bmp_protocol(mock_runner, params)
    assert len(history) == 4


def test_bmp_protocol_with_equilibration_steps(mock_runner, mock_structure, mock_potential):
    """bmp_protocol respects custom equilibration_steps."""
    params = _make_params(mock_structure, mock_potential, equilibration_steps=50_000)
    bmp_protocol(mock_runner, params)
    assert mock_runner.call_count == 4


# ---------------------------------------------------------------------------
# du_teter_protocol
# ---------------------------------------------------------------------------


def test_du_teter_protocol_accepts_dataclass(mock_runner, mock_structure, mock_potential):
    """du_teter_protocol accepts MeltQuenchParams and returns structure + history."""
    params = _make_params(mock_structure, mock_potential, temperature_high=5000.0)
    structure, history = du_teter_protocol(mock_runner, params)
    assert structure is not None
    assert history is not None


def test_du_teter_protocol_calls_runner_4_times(mock_runner, mock_structure, mock_potential):
    """du_teter_protocol calls the runner exactly 4 times (stage 0 + 3 stages)."""
    params = _make_params(mock_structure, mock_potential, temperature_high=5000.0)
    du_teter_protocol(mock_runner, params)
    assert mock_runner.call_count == 4


def test_du_teter_protocol_returns_4_history_entries(mock_runner, mock_structure, mock_potential):
    """du_teter_protocol returns a list of 4 history entries."""
    params = _make_params(mock_structure, mock_potential, temperature_high=5000.0)
    _, history = du_teter_protocol(mock_runner, params)
    assert len(history) == 4


def test_du_teter_protocol_with_equilibration_steps(mock_runner, mock_structure, mock_potential):
    """du_teter_protocol respects custom equilibration_steps."""
    params = _make_params(mock_structure, mock_potential, temperature_high=5000.0, equilibration_steps=50_000)
    du_teter_protocol(mock_runner, params)
    assert mock_runner.call_count == 4


_ALL_PROTOCOLS = {
    "pmmcs": pmmcs_protocol,
    "bmp": bmp_protocol,
    "bjp": bjp_protocol,
    "shik": shik_protocol,
    "du_teter": du_teter_protocol,
    "yang2026": yang2026_protocol,
}

# Runner calls per protocol with pre_equilibrate=True (stage 0 included).
_EXPECTED_STAGE_COUNTS = {
    "pmmcs": 4,
    "bmp": 4,
    "bjp": 4,
    "shik": 5,
    "du_teter": 4,
    "yang2026": 7,
}

# Protocols whose stage 1 equilibrates directly at temperature_high with
# freshly created velocities (shik and yang2026 have their own stage-1 shapes).
_DIRECT_MELT_PROTOCOLS = ("pmmcs", "bmp", "bjp", "du_teter")


@pytest.mark.parametrize("name", list(_ALL_PROTOCOLS))
def test_protocol_stage0_uses_fix_override(name, mock_runner, mock_structure, mock_potential):
    """Every protocol runs the pre-equilibration as its own stage 0 via the fix override; Configs stay clean."""
    params = _make_params(mock_structure, mock_potential, temperature_high=4321.0)
    _, history = _ALL_PROTOCOLS[name](mock_runner, params)

    assert mock_runner.call_count == _EXPECTED_STAGE_COUNTS[name]
    assert len(history) == _EXPECTED_STAGE_COUNTS[name]
    stage0 = mock_runner.call_args_list[0].kwargs
    assert stage0["n_ionic_steps"] == 10_000
    assert stage0["initial_temperature"] == 0
    assert stage0["langevin"] is False
    override = stage0["input_control_file"]["fix"]
    assert "langevin 4321 4321 0.01 48279" in override
    assert "nve/limit 0.5" in override
    for c in mock_runner.call_args_list:
        assert c.kwargs["potential"]["Config"].iloc[0] == ["line1", "line2"]


@pytest.mark.parametrize("name", list(_ALL_PROTOCOLS))
def test_protocol_pre_equilibrate_false_skips_stage0(name, mock_runner, mock_structure, mock_potential):
    """pre_equilibrate=False skips stage 0 but keeps its None history slot for stable indices."""
    params = _make_params(mock_structure, mock_potential, pre_equilibrate=False)
    _, history = _ALL_PROTOCOLS[name](mock_runner, params)

    assert mock_runner.call_count == _EXPECTED_STAGE_COUNTS[name] - 1
    assert len(history) == _EXPECTED_STAGE_COUNTS[name]
    assert history[0] is None
    assert all("input_control_file" not in c.kwargs for c in mock_runner.call_args_list)
    for c in mock_runner.call_args_list:
        assert c.kwargs["potential"]["Config"].iloc[0] == ["line1", "line2"]


@pytest.mark.parametrize("name", _DIRECT_MELT_PROTOCOLS)
def test_protocol_stage1_equilibrates_directly_at_temperature_high(name, mock_runner, mock_structure, mock_potential):
    """Stage 1 creates velocities at temperature_high and holds it -- there is no heating ramp."""
    params = _make_params(mock_structure, mock_potential, temperature_high=4321.0)
    _ALL_PROTOCOLS[name](mock_runner, params)

    stage1 = mock_runner.call_args_list[1].kwargs
    assert stage1["temperature"] == 4321.0
    assert stage1["initial_temperature"] == 4321.0
    assert stage1["seed"] == 12345
    assert "temperature_end" not in stage1


def test_bmp_protocol_equilibration_steps_override(mock_runner, mock_structure, mock_potential):
    """equilibration_steps overrides the hardcoded defaults in bmp_protocol."""
    params = MeltQuenchParams(
        structure=mock_structure,
        potential=mock_potential,
        temperature_high=4000.0,
        temperature_low=300.0,
        cooling_steps=50,
        timestep=1.0,
        n_dump=1000,
        langevin=False,
        seed=12345,
        equilibration_steps=999,
    )
    bmp_protocol(mock_runner, params)
    # Stages 2 and 4 (indices 1 and 3) are equilibration stages — must use override
    for idx in (1, 3):
        n_steps = mock_runner.call_args_list[idx].kwargs["n_ionic_steps"]
        assert n_steps == 999, f"Stage {idx + 1}: expected 999 steps, got {n_steps}"


# ---------------------------------------------------------------------------
# yang2026_protocol
# ---------------------------------------------------------------------------


def test_yang2026_protocol_high_pressure_melt_stage(mock_runner, mock_structure, mock_potential):
    """Stage 3 (high-pressure melt) passes a non-zero pressure."""
    params = MeltQuenchParams(
        structure=mock_structure,
        potential=mock_potential,
        temperature_high=4000.0,
        temperature_low=300.0,
        cooling_steps=100,
        timestep=1.0,
        n_dump=1000,
        langevin=False,
        seed=12345,
        equilibration_steps=10,
    )
    yang2026_protocol(mock_runner, params)
    # Stage 3 is call index 3 (0-indexed): NPT at T_high with positive pressure
    stage3_kwargs = mock_runner.call_args_list[3].kwargs
    assert stage3_kwargs["pressure"] > 0
    assert stage3_kwargs["temperature"] == 4000.0


def test_yang2026_protocol_equilibration_steps_override(mock_runner, mock_structure, mock_potential):
    """equilibration_steps overrides hardcoded step counts in yang2026_protocol."""
    params = MeltQuenchParams(
        structure=mock_structure,
        potential=mock_potential,
        temperature_high=4000.0,
        temperature_low=300.0,
        cooling_steps=50,
        timestep=1.0,
        n_dump=1000,
        langevin=False,
        seed=12345,
        equilibration_steps=77,
    )
    yang2026_protocol(mock_runner, params)
    # Stages 1-4 and 6 use eq_steps; stage 0 is the pre-equilibration, stage 5 the cooling ramp
    for idx in (1, 2, 3, 4, 6):
        n_steps = mock_runner.call_args_list[idx].kwargs["n_ionic_steps"]
        assert n_steps == 77, f"Stage {idx}: expected 77 steps, got {n_steps}"


# ---------------------------------------------------------------------------
# melt_quench_simulation — step-count math and routing
# ---------------------------------------------------------------------------


def test_melt_quench_simulation_cooling_step_count(mock_structure):
    """The cooling step count is computed correctly from the rate and timestep."""
    # With T_high=4300, T_low=300, rate=1e12 K/s, timestep=1 fs:
    # steps = ((4300-300) / (1.0 * 1e12)) * 1e15 = 4000 * 1e3 = 4_000_000
    potential = pd.DataFrame({"Name": ["pmmcs"], "Config": [["line"]]})
    captured = {}

    def fake_protocol(_runner, params):
        captured["cooling_steps"] = params.cooling_steps
        return mock_structure, []

    original = mq_module.PROTOCOL_MAP.copy()
    mq_module.PROTOCOL_MAP["pmmcs"] = fake_protocol
    try:
        melt_quench_simulation(
            mock_structure,
            potential,
            temperature_high=4300.0,
            temperature_low=300.0,
            cooling_rate=1e12,
            timestep=1.0,
        )
    finally:
        mq_module.PROTOCOL_MAP.update(original)

    assert captured["cooling_steps"] == 4_000_000


def test_melt_quench_simulation_rejects_equal_temperatures(mock_structure):
    """temperature_high == temperature_low raises before any protocol stage runs (zero cooling steps)."""
    potential = pd.DataFrame({"Name": ["pmmcs"], "Config": [["line"]]})
    with pytest.raises(ValueError, match="must differ"):
        melt_quench_simulation(mock_structure, potential, temperature_high=3000.0, temperature_low=3000.0)


def test_melt_quench_simulation_rejects_default_high_equal_to_low(mock_structure):
    """temperature_high=None resolving to the potential default still triggers the guard if equal to temperature_low."""
    potential = pd.DataFrame({"Name": ["shik"], "Config": [["line"]]})
    with pytest.raises(ValueError, match="must differ"):
        melt_quench_simulation(mock_structure, potential, temperature_high=None, temperature_low=4000.0)


@pytest.mark.parametrize("variant", ["bmp-screened-harmonic", "bmp-harmonic"])
def test_melt_quench_simulation_bmp_variant_routing(mock_structure, variant):
    """bmp-screened-harmonic and bmp-harmonic are both routed to the 'bmp' protocol."""
    potential = pd.DataFrame({"Name": [variant], "Config": [["line"]]})
    called = []

    def fake_protocol(_runner, _params):
        called.append(True)
        return mock_structure, []

    original = mq_module.PROTOCOL_MAP.copy()
    mq_module.PROTOCOL_MAP["bmp"] = fake_protocol
    try:
        melt_quench_simulation(mock_structure, potential, temperature_high=4000.0)
    finally:
        mq_module.PROTOCOL_MAP.update(original)

    assert called, f"{variant} was not routed to bmp protocol"
