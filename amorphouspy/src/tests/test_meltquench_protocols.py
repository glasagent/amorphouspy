"""Unit tests for meltquench protocols with dataclass parameters."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from amorphouspy.workflows.meltquench_protocols import (
    MeltQuenchParams,
    bjp_protocol,
    bmp_protocol,
    du_teter_protocol,
    pmmcs_protocol,
    shik_protocol,
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
        heating_steps=100_000,
        cooling_steps=100_000,
        timestep=1.0,
        n_print=1000,
        langevin=False,
        seed=12345,
    )

    assert params.structure == mock_structure
    assert params.potential.equals(mock_potential)
    assert params.temperature_high == 5000.0
    assert params.temperature_low == 300.0
    assert params.heating_steps == 100_000
    assert params.cooling_steps == 100_000
    assert params.timestep == 1.0
    assert params.n_print == 1000
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
        heating_steps=100_000,
        cooling_steps=100_000,
        timestep=1.0,
        n_print=1000,
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
        heating_steps=100_000,
        cooling_steps=100_000,
        timestep=1.0,
        n_print=1000,
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
        heating_steps=100_000,
        cooling_steps=100_000,
        timestep=1.0,
        n_print=1000,
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
        heating_steps=100_000,
        cooling_steps=100_000,
        timestep=1.0,
        n_print=1000,
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
        heating_steps=100_000,
        cooling_steps=200_000,
        timestep=1.0,
        n_print=1000,
        langevin=True,
        seed=12345,
    )

    pmmcs_protocol(mock_runner, params)

    # 5 stages in the protocol
    assert mock_runner.call_count == 5


def test_bjp_protocol_calls_runner_correctly(mock_runner, mock_structure, mock_potential):
    """Test that bjp_protocol calls the runner with correct parameters."""
    params = MeltQuenchParams(
        structure=mock_structure,
        potential=mock_potential,
        temperature_high=5000.0,
        temperature_low=300.0,
        heating_steps=100_000,
        cooling_steps=200_000,
        timestep=1.0,
        n_print=1000,
        langevin=True,
        seed=12345,
    )

    bjp_protocol(mock_runner, params)

    # 5 stages in the protocol
    assert mock_runner.call_count == 5


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
        heating_steps=100_000,
        cooling_steps=200_000,
        timestep=1.0,
        n_print=1000,
        langevin=True,
        seed=12345,
    )

    shik_protocol(mock_runner, params)

    # 5 stages in the protocol
    assert mock_runner.call_count == 5


def _make_params(structure, potential, **kwargs):
    defaults = {
        "structure": structure,
        "potential": potential,
        "temperature_high": 4000.0,
        "temperature_low": 300.0,
        "heating_steps": 100_000,
        "cooling_steps": 100_000,
        "timestep": 1.0,
        "n_print": 1000,
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


def test_bmp_protocol_calls_runner_5_times(mock_runner, mock_structure, mock_potential):
    """bmp_protocol calls the runner exactly 5 times (5 stages)."""
    params = _make_params(mock_structure, mock_potential)
    bmp_protocol(mock_runner, params)
    assert mock_runner.call_count == 5


def test_bmp_protocol_returns_5_history_entries(mock_runner, mock_structure, mock_potential):
    """bmp_protocol returns a list of 5 history entries."""
    params = _make_params(mock_structure, mock_potential)
    _, history = bmp_protocol(mock_runner, params)
    assert len(history) == 5


def test_bmp_protocol_strips_exclude_patterns(mock_runner, mock_structure):
    """bmp_protocol strips langevin/nve patterns from potential config for stages 2+."""
    potential = pd.DataFrame(
        {
            "Name": ["bmp"],
            "Config": [
                [
                    "fix langevinnve all langevin 4000 4000 0.01 48279",
                    "fix ensemblenve all nve/limit 0.5",
                    "run 10000",
                    "unfix langevinnve",
                    "unfix ensemblenve",
                    "other line",
                ]
            ],
        }
    )
    params = _make_params(mock_structure, potential)
    bmp_protocol(mock_runner, params)
    assert mock_runner.call_count == 5


def test_bmp_protocol_with_equilibration_steps(mock_runner, mock_structure, mock_potential):
    """bmp_protocol respects custom equilibration_steps."""
    params = _make_params(mock_structure, mock_potential, equilibration_steps=50_000)
    bmp_protocol(mock_runner, params)
    assert mock_runner.call_count == 5


# ---------------------------------------------------------------------------
# du_teter_protocol
# ---------------------------------------------------------------------------


def test_du_teter_protocol_accepts_dataclass(mock_runner, mock_structure, mock_potential):
    """du_teter_protocol accepts MeltQuenchParams and returns structure + history."""
    params = _make_params(mock_structure, mock_potential, temperature_high=5000.0)
    structure, history = du_teter_protocol(mock_runner, params)
    assert structure is not None
    assert history is not None


def test_du_teter_protocol_calls_runner_5_times(mock_runner, mock_structure, mock_potential):
    """du_teter_protocol calls the runner exactly 5 times (5 stages)."""
    params = _make_params(mock_structure, mock_potential, temperature_high=5000.0)
    du_teter_protocol(mock_runner, params)
    assert mock_runner.call_count == 5


def test_du_teter_protocol_returns_5_history_entries(mock_runner, mock_structure, mock_potential):
    """du_teter_protocol returns a list of 5 history entries."""
    params = _make_params(mock_structure, mock_potential, temperature_high=5000.0)
    _, history = du_teter_protocol(mock_runner, params)
    assert len(history) == 5


def test_du_teter_protocol_strips_5000_langevin(mock_runner, mock_structure):
    """du_teter_protocol strips the 5000 K langevin pattern from stages 2+."""
    potential = pd.DataFrame(
        {
            "Name": ["du_teter"],
            "Config": [
                [
                    "fix langevinnve all langevin 5000 5000 0.01 48279",
                    "fix ensemblenve all nve/limit 0.5",
                    "run 10000",
                    "unfix langevinnve",
                    "unfix ensemblenve",
                    "pair_style hybrid/overlay coul/dsf 0.25 8.0 table spline 11000",
                ]
            ],
        }
    )
    params = _make_params(mock_structure, potential, temperature_high=5000.0)
    du_teter_protocol(mock_runner, params)
    assert mock_runner.call_count == 5


def test_du_teter_protocol_with_equilibration_steps(mock_runner, mock_structure, mock_potential):
    """du_teter_protocol respects custom equilibration_steps."""
    params = _make_params(mock_structure, mock_potential, temperature_high=5000.0, equilibration_steps=50_000)
    du_teter_protocol(mock_runner, params)
    assert mock_runner.call_count == 5
