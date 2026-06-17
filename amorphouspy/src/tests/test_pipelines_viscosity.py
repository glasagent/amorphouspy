"""Tests for helper functions in amorphouspy.pipelines.viscosity."""

from __future__ import annotations

import itertools
from unittest.mock import MagicMock, patch

import pytest
from amorphouspy.atoms.shared import _logspace, downsample_log
from amorphouspy.pipelines.viscosity import run_viscosity_workflow
from ase import Atoms

# ---------------------------------------------------------------------------
# _logspace
# ---------------------------------------------------------------------------


class TestLogspace:
    """Tests for the _logspace helper."""

    def test_empty_when_num_zero(self) -> None:
        """Return empty list when num is zero."""
        assert _logspace(0, 100, 0) == []

    def test_empty_when_num_negative(self) -> None:
        """Return empty list when num is negative."""
        assert _logspace(0, 100, -1) == []

    def test_single_returns_stop(self) -> None:
        """Single-element result equals stop value."""
        result = _logspace(0, 99, 1)
        assert len(result) == 1
        assert result[0] == pytest.approx(99.0)

    def test_endpoints_included(self) -> None:
        """First and last values match start and stop."""
        result = _logspace(0, 999, 50)
        assert result[0] == pytest.approx(0.0, abs=1e-9)
        assert result[-1] == pytest.approx(999.0, abs=1e-6)

    def test_length_matches_num(self) -> None:
        """Output length equals requested num."""
        result = _logspace(0, 500, 20)
        assert len(result) == 20

    def test_values_monotonically_increase(self) -> None:
        """Values are strictly increasing."""
        result = _logspace(0, 1000, 100)
        for a, b in itertools.pairwise(result):
            assert b > a


# ---------------------------------------------------------------------------
# downsample_log
# ---------------------------------------------------------------------------


class TestDownsampleLog:
    """Tests for the downsample_log helper."""

    def test_passthrough_when_short(self) -> None:
        """Arrays shorter than max_points are returned unchanged."""
        arr = list(range(10))
        assert downsample_log(arr, max_points=20) == arr

    def test_passthrough_at_exact_limit(self) -> None:
        """Arrays exactly at max_points are returned unchanged."""
        arr = list(range(100))
        assert downsample_log(arr, max_points=100) == arr

    def test_reduces_length(self) -> None:
        """Long arrays are reduced to at most max_points."""
        arr = list(range(5000))
        result = downsample_log(arr, max_points=200)
        assert len(result) <= 200

    def test_preserves_first_and_last(self) -> None:
        """First and last elements of the original are preserved."""
        arr = list(range(5000))
        result = downsample_log(arr, max_points=100)
        assert result[0] == arr[0]
        assert result[-1] == arr[-1]

    def test_empty_list(self) -> None:
        """Empty input returns empty output."""
        assert downsample_log([], max_points=10) == []

    def test_single_element(self) -> None:
        """Single-element input is returned unchanged."""
        assert downsample_log([42.0], max_points=10) == [42.0]

    def test_values_are_subset_of_original(self) -> None:
        """All downsampled values exist in the original array."""
        arr = [float(i**2) for i in range(3000)]
        result = downsample_log(arr, max_points=50)
        for v in result:
            assert v in arr


# ---------------------------------------------------------------------------
# run_viscosity_workflow
# ---------------------------------------------------------------------------


@patch("amorphouspy.pipelines.viscosity.downsample_log", side_effect=lambda a: a)
@patch("amorphouspy.pipelines.viscosity.get_viscosity")
@patch("amorphouspy.pipelines.viscosity.viscosity_simulation")
@patch("amorphouspy.pipelines.viscosity.melt_quench_simulation")
def test_run_viscosity_workflow_sorts_temperatures_high_to_low(
    mock_mq: MagicMock,
    mock_visc_sim: MagicMock,
    mock_get_visc: MagicMock,
    mock_downsample: MagicMock,
) -> None:
    """Input temperatures are always processed from highest to lowest."""
    structure = Atoms("SiO2", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]], cell=[5, 5, 5], pbc=True)
    potential = MagicMock()

    mock_mq.side_effect = [
        {"structure": structure},
        {"structure": structure},
        {"structure": structure},
    ]
    mock_visc_sim.return_value = {"result": {}}
    mock_get_visc.side_effect = [
        {"viscosity": 1.0, "max_lag": 10.0, "lag_time_ps": [0.0, 1.0], "viscosity_integral": [0.0, 1.0]},
        {"viscosity": 2.0, "max_lag": 11.0, "lag_time_ps": [0.0, 1.0], "viscosity_integral": [0.0, 2.0]},
        {"viscosity": 3.0, "max_lag": 12.0, "lag_time_ps": [0.0, 1.0], "viscosity_integral": [0.0, 3.0]},
    ]

    out = run_viscosity_workflow(
        structure=structure,
        potential=potential,
        temperatures=[2000.0, 1500.0, 2500.0],
        heating_rate=1e12,
        cooling_rate=1e12,
    )

    assert mock_downsample is not None

    assert out["temperatures"] == [2500.0, 2000.0, 1500.0]
    assert out["viscosities"] == [1.0, 2.0, 3.0]


@patch("amorphouspy.pipelines.viscosity.downsample_log", side_effect=lambda a: a)
@patch(
    "amorphouspy.pipelines.viscosity.get_viscosity",
    return_value={"viscosity": 1.0, "max_lag": 10.0, "lag_time_ps": [], "viscosity_integral": []},
)
@patch("amorphouspy.pipelines.viscosity.viscosity_simulation", return_value={"result": {}})
@patch("amorphouspy.pipelines.viscosity.melt_quench_simulation")
def test_run_viscosity_workflow_uses_5000K_for_first_stage(
    mock_mq: MagicMock,
    mock_visc_sim: MagicMock,
    mock_get_visc: MagicMock,
    mock_downsample: MagicMock,
) -> None:
    """First cooling stage starts from 5000 K."""
    structure = Atoms("Si", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    mock_mq.return_value = {"structure": structure}

    run_viscosity_workflow(
        structure=structure,
        potential=MagicMock(),
        temperatures=[1800.0],
        heating_rate=1e12,
        cooling_rate=1e12,
    )

    assert mock_visc_sim is not None
    assert mock_get_visc is not None
    assert mock_downsample is not None

    _, kwargs = mock_mq.call_args
    assert kwargs["temperature_high"] == 5000.0
    assert kwargs["temperature_low"] == 1800.0


@patch("amorphouspy.pipelines.viscosity.downsample_log", side_effect=lambda a: a)
@patch(
    "amorphouspy.pipelines.viscosity.get_viscosity",
    return_value={"viscosity": 1.0, "max_lag": 10.0, "lag_time_ps": [], "viscosity_integral": []},
)
@patch("amorphouspy.pipelines.viscosity.viscosity_simulation", return_value={"result": {}})
@patch("amorphouspy.pipelines.viscosity.melt_quench_simulation")
def test_run_viscosity_workflow_uses_previous_temperature_for_next_stage(
    mock_mq: MagicMock,
    mock_visc_sim: MagicMock,
    mock_get_visc: MagicMock,
    mock_downsample: MagicMock,
) -> None:
    """Subsequent cooling stage starts from the previous target temperature."""
    structure = Atoms("Si", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    mock_mq.side_effect = [{"structure": structure}, {"structure": structure}]

    run_viscosity_workflow(
        structure=structure,
        potential=MagicMock(),
        temperatures=[2500.0, 2000.0],
        heating_rate=1e12,
        cooling_rate=1e12,
    )

    assert mock_visc_sim is not None
    assert mock_get_visc is not None
    assert mock_downsample is not None

    first_kwargs = mock_mq.call_args_list[0].kwargs
    second_kwargs = mock_mq.call_args_list[1].kwargs
    assert first_kwargs["temperature_high"] == 5000.0
    assert first_kwargs["temperature_low"] == 2500.0
    assert second_kwargs["temperature_high"] == 2500.0
    assert second_kwargs["temperature_low"] == 2000.0


@patch("amorphouspy.pipelines.viscosity.downsample_log", side_effect=lambda a: a)
@patch(
    "amorphouspy.pipelines.viscosity.get_viscosity",
    return_value={"viscosity": 1.0, "max_lag": 10.0, "lag_time_ps": [], "viscosity_integral": []},
)
@patch("amorphouspy.pipelines.viscosity.viscosity_simulation", return_value={"result": {}})
@patch("amorphouspy.pipelines.viscosity.melt_quench_simulation")
def test_run_viscosity_workflow_defaults_server_kwargs_to_empty_dict(
    mock_mq: MagicMock,
    mock_visc_sim: MagicMock,
    mock_get_visc: MagicMock,
    mock_downsample: MagicMock,
) -> None:
    """When server_kwargs is omitted, both stages receive {}."""
    structure = Atoms("Si", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    mock_mq.return_value = {"structure": structure}

    run_viscosity_workflow(
        structure=structure,
        potential=MagicMock(),
        temperatures=[2000.0],
        heating_rate=1e12,
        cooling_rate=1e12,
    )

    assert mock_get_visc is not None
    assert mock_downsample is not None

    assert mock_mq.call_args.kwargs["server_kwargs"] == {}
    assert mock_visc_sim.call_args.kwargs["server_kwargs"] == {}


@patch("amorphouspy.pipelines.viscosity.downsample_log", side_effect=lambda a: a[::2])
@patch("amorphouspy.pipelines.viscosity.get_viscosity")
@patch("amorphouspy.pipelines.viscosity.viscosity_simulation", return_value={"result": {}})
@patch("amorphouspy.pipelines.viscosity.melt_quench_simulation")
def test_run_viscosity_workflow_downsamples_saved_series(
    mock_mq: MagicMock,
    mock_visc_sim: MagicMock,
    mock_get_visc: MagicMock,
    mock_downsample: MagicMock,
) -> None:
    """Lag-time and viscosity integral series are downsampled before return."""
    structure = Atoms("Si", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    mock_mq.return_value = {"structure": structure}
    mock_get_visc.return_value = {
        "viscosity": 1.2,
        "max_lag": 20.0,
        "lag_time_ps": [0.0, 1.0, 2.0, 3.0],
        "viscosity_integral": [0.0, 0.5, 1.0, 1.5],
    }

    out = run_viscosity_workflow(
        structure=structure,
        potential=MagicMock(),
        temperatures=[2000.0],
        heating_rate=1e12,
        cooling_rate=1e12,
    )

    assert mock_visc_sim is not None
    assert mock_downsample is not None

    assert out["lag_times_ps"] == [[0.0, 2.0]]
    assert out["viscosity_integral"] == [[0.0, 1.0]]
