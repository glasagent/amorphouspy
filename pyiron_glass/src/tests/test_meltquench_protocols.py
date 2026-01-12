"""Unit tests for meltquench protocols with dataclass parameters."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from ase import Atoms

# Import the module directly to avoid dependency issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "meltquench_protocols",
    Path(__file__).parent.parent / "pyiron_glass" / "workflows" / "meltquench_protocols.py"
)
meltquench_protocols = importlib.util.module_from_spec(spec)
spec.loader.exec_module(meltquench_protocols)

MeltQuenchParams = meltquench_protocols.MeltQuenchParams
pmmcs_protocol = meltquench_protocols.pmmcs_protocol
bjp_protocol = meltquench_protocols.bjp_protocol
shik_protocol = meltquench_protocols.shik_protocol


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


def test_meltquench_params_creation(mock_runner, mock_structure, mock_potential):
    """Test that MeltQuenchParams dataclass can be created successfully."""
    params = MeltQuenchParams(
        runner=mock_runner,
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

    assert params.runner == mock_runner
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


def test_meltquench_params_with_optional_values(mock_runner, mock_structure, mock_potential):
    """Test that MeltQuenchParams dataclass handles optional parameters."""
    params = MeltQuenchParams(
        runner=mock_runner,
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
        tmp_working_directory="/tmp/test",
    )

    assert params.server_kwargs == {"cores": 4}
    assert params.tmp_working_directory == "/tmp/test"


def test_pmmcs_protocol_accepts_dataclass(mock_runner, mock_structure, mock_potential):
    """Test that pmmcs_protocol accepts MeltQuenchParams dataclass."""
    params = MeltQuenchParams(
        runner=mock_runner,
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

    structure, output = pmmcs_protocol(params)

    # Verify the protocol called the runner
    assert mock_runner.called
    # Verify we got back results
    assert structure is not None
    assert output is not None


def test_bjp_protocol_accepts_dataclass(mock_runner, mock_structure, mock_potential):
    """Test that bjp_protocol accepts MeltQuenchParams dataclass."""
    params = MeltQuenchParams(
        runner=mock_runner,
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

    structure, output = bjp_protocol(params)

    # Verify the protocol called the runner
    assert mock_runner.called
    # Verify we got back results
    assert structure is not None
    assert output is not None


def test_shik_protocol_accepts_dataclass(mock_runner, mock_structure):
    """Test that shik_protocol accepts MeltQuenchParams dataclass."""
    # Create a proper potential dataframe with Config column
    potential = pd.DataFrame(
        {
            "Name": ["shik"],
            "Config": [["fix langevin all langevin 5000 5000 0.01 48279", "other line"]],
        }
    )

    params = MeltQuenchParams(
        runner=mock_runner,
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

    structure, output = shik_protocol(params)

    # Verify the protocol called the runner
    assert mock_runner.called
    # Verify we got back results
    assert structure is not None
    assert output is not None


def test_pmmcs_protocol_calls_runner_correctly(mock_runner, mock_structure, mock_potential):
    """Test that pmmcs_protocol calls the runner with correct parameters."""
    params = MeltQuenchParams(
        runner=mock_runner,
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

    pmmcs_protocol(params)

    # Verify the runner was called (it should be called 5 times for the 5 stages)
    assert mock_runner.call_count == 5


def test_bjp_protocol_calls_runner_correctly(mock_runner, mock_structure, mock_potential):
    """Test that bjp_protocol calls the runner with correct parameters."""
    params = MeltQuenchParams(
        runner=mock_runner,
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

    bjp_protocol(params)

    # Verify the runner was called (it should be called 5 times for the 5 stages)
    assert mock_runner.call_count == 5


def test_shik_protocol_calls_runner_correctly(mock_runner, mock_structure):
    """Test that shik_protocol calls the runner with correct parameters."""
    # Create a proper potential dataframe with Config column
    potential = pd.DataFrame(
        {
            "Name": ["shik"],
            "Config": [["fix langevin all langevin 5000 5000 0.01 48279", "other line"]],
        }
    )

    params = MeltQuenchParams(
        runner=mock_runner,
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

    shik_protocol(params)

    # Verify the runner was called (it should be called 5 times for the 5 stages)
    assert mock_runner.call_count == 5
