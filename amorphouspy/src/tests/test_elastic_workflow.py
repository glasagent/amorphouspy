"""Tests for workflow-level logic in amorphouspy.properties.elastic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from amorphouspy.properties import elastic as elastic_module
from ase import Atoms


def _structure() -> Atoms:
    return Atoms("SiO2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)


def _potential(name: str = "pmmcs", config: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame({"Name": [name], "Config": [config or []]})


def _matrix_with_component(i: int, j: int, value: float) -> np.ndarray:
    mat = np.zeros((3, 3))
    mat[i, j] = value
    if i != j:
        mat[j, i] = value
    return mat


@patch("amorphouspy.properties.elastic._run_lammps_md")
def test_elastic_simulation_uses_central_difference_stress_path(mock_run_lammps_md: MagicMock) -> None:
    """elastic_simulation computes Cij from alternating (+/-) strained MD pressure responses."""
    call_count = {"n": 0}

    def _side_effect(*args, **kwargs):
        del args, kwargs
        idx = call_count["n"]
        call_count["n"] += 1

        if idx == 0:
            # Initial equilibration call reads only the volume series.
            return _structure(), {"generic": {"volume": [125.0] * 4}}

        pressure_value = 2.0 if idx % 2 == 1 else 1.0
        pressures = np.full((4, 3, 3), pressure_value)
        return _structure(), {"generic": {"pressures": pressures}}

    mock_run_lammps_md.side_effect = _side_effect

    strain = 1e-3
    result = elastic_module.elastic_simulation(
        structure=_structure(),
        potential=_potential(),
        equilibration_steps=4,
        production_steps=4,
        strain=strain,
    )

    expected = -1.0 / (2.0 * strain)
    assert result["Cij"][0, 0] == pytest.approx(expected)
    assert result["Cij"][1, 1] == pytest.approx(expected)
    assert result["Cij"][2, 2] == pytest.approx(expected)
    assert result["Cij"][3, 3] == pytest.approx(expected)
    assert result["Cij"][4, 4] == pytest.approx(expected)
    assert result["Cij"][5, 5] == pytest.approx(expected)


@patch("amorphouspy.properties.elastic._run_strained_md")
@patch("amorphouspy.properties.elastic._run_lammps_md")
def test_elastic_simulation_warns_when_c11_c22_c33_differ(
    mock_run_lammps_md: MagicMock,
    mock_run_strained_md: MagicMock,
) -> None:
    """A warning is emitted when normal elastic constants break cubic symmetry."""
    mock_run_lammps_md.return_value = (_structure(), {"generic": {"volume": [125.0] * 4}})

    strain = 1e-3
    # Normal blocks: C11=1, C22=2, C33=3
    normal = [
        _matrix_with_component(0, 0, 2 * strain * 1),
        _matrix_with_component(1, 1, 2 * strain * 2),
        _matrix_with_component(2, 2, 2 * strain * 3),
    ]
    # Shear blocks equal: C44=C55=C66=4
    shear = [
        _matrix_with_component(1, 2, 2 * strain * 4),
        _matrix_with_component(0, 2, 2 * strain * 4),
        _matrix_with_component(0, 1, 2 * strain * 4),
    ]
    mock_run_strained_md.side_effect = normal + shear

    with pytest.warns(UserWarning, match="C11 != C22 != C33"):
        elastic_module.elastic_simulation(
            structure=_structure(),
            potential=_potential(),
            equilibration_steps=10,
            production_steps=10,
            strain=strain,
        )


@patch("amorphouspy.properties.elastic._run_strained_md")
@patch("amorphouspy.properties.elastic._run_lammps_md")
def test_elastic_simulation_warns_when_c44_c55_c66_differ(
    mock_run_lammps_md: MagicMock,
    mock_run_strained_md: MagicMock,
) -> None:
    """A warning is emitted when shear elastic constants break cubic symmetry."""
    mock_run_lammps_md.return_value = (_structure(), {"generic": {"volume": [125.0] * 4}})

    strain = 1e-3
    # Normal blocks equal: C11=C22=C33=2
    normal = [
        _matrix_with_component(0, 0, 2 * strain * 2),
        _matrix_with_component(1, 1, 2 * strain * 2),
        _matrix_with_component(2, 2, 2 * strain * 2),
    ]
    # Shear blocks differ: C44=4, C55=5, C66=6
    shear = [
        _matrix_with_component(1, 2, 2 * strain * 4),
        _matrix_with_component(0, 2, 2 * strain * 5),
        _matrix_with_component(0, 1, 2 * strain * 6),
    ]
    mock_run_strained_md.side_effect = normal + shear

    with pytest.warns(UserWarning, match="C44 != C55 != C66"):
        elastic_module.elastic_simulation(
            structure=_structure(),
            potential=_potential(),
            equilibration_steps=10,
            production_steps=10,
            strain=strain,
        )


@patch("amorphouspy.properties.elastic._run_strained_md")
@patch("amorphouspy.properties.elastic._run_lammps_md")
def test_elastic_simulation_filters_shik_patterns_before_run(
    mock_run_lammps_md: MagicMock,
    mock_run_strained_md: MagicMock,
) -> None:
    """SHIK config lines for early melt-quench setup are removed before running elasticity."""
    mock_run_lammps_md.return_value = (_structure(), {"generic": {"volume": [125.0] * 4}})
    mock_run_strained_md.return_value = np.zeros((3, 3))

    keep_line = "pair_style table spline 500"
    potential = _potential(
        "shik",
        [
            "fix langevin all langevin 5000 5000 0.01 48279",
            "fix ensemble all nve/limit 0.5",
            "run 10000",
            "unfix langevin",
            "unfix ensemble",
            keep_line,
        ],
    )

    elastic_module.elastic_simulation(
        structure=_structure(),
        potential=potential,
        equilibration_steps=10,
        production_steps=10,
    )

    filtered = mock_run_lammps_md.call_args.kwargs["potential"].loc[0, "Config"]
    assert keep_line in filtered
    assert "run 10000" not in filtered
    assert all("langevin" not in line for line in filtered)
    assert all("nve/limit" not in line for line in filtered)
