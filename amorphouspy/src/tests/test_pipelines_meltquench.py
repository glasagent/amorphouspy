"""Tests for amorphouspy.pipelines.meltquench."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from amorphouspy.pipelines.meltquench import generate_structure, run_melt_quench
from ase import Atoms

# ---------------------------------------------------------------------------
# generate_structure
# ---------------------------------------------------------------------------


class TestGenerateStructure:
    """Tests for generate_structure."""

    @patch("amorphouspy.pipelines.meltquench.generate_potential")
    @patch("amorphouspy.pipelines.meltquench.get_ase_structure")
    @patch("amorphouspy.pipelines.meltquench.get_structure_dict")
    def test_returns_expected_keys(
        self,
        mock_struct_dict: MagicMock,
        mock_ase: MagicMock,
        mock_potential: MagicMock,
    ) -> None:
        """generate_structure returns atoms_dict, structure, and potential."""
        mock_struct_dict.return_value = {"Si": 10, "O": 20}
        mock_ase.return_value = Atoms("Si")
        mock_potential.return_value = MagicMock()

        result = generate_structure(
            composition={"SiO2": 100.0},
            n_atoms=100,
            potential_type="pmmcs",
        )

        assert set(result.keys()) == {"atoms_dict", "structure", "potential"}
        mock_struct_dict.assert_called_once()
        mock_ase.assert_called_once()
        mock_potential.assert_called_once()

    @patch("amorphouspy.pipelines.meltquench.generate_potential")
    @patch("amorphouspy.pipelines.meltquench.get_ase_structure")
    @patch("amorphouspy.pipelines.meltquench.get_structure_dict")
    def test_forwards_parameters(
        self,
        mock_struct_dict: MagicMock,
        mock_ase: MagicMock,
        mock_potential: MagicMock,
    ) -> None:
        """Parameters are forwarded to the underlying functions."""
        mock_struct_dict.return_value = {}
        mock_ase.return_value = Atoms("Si")
        mock_potential.return_value = MagicMock()

        generate_structure(
            composition={"SiO2": 100.0},
            n_atoms=500,
            potential_type="bks",
            density=2.2,
            structure_seed=99,
        )

        _, kwargs = mock_struct_dict.call_args
        assert kwargs["target_atoms"] == 500
        assert kwargs["density"] == 2.2
        assert kwargs["random_seed"] == 99

        _, kwargs = mock_potential.call_args
        assert kwargs["potential_type"] == "bks"


# ---------------------------------------------------------------------------
# run_melt_quench
# ---------------------------------------------------------------------------


class TestRunMeltQuench:
    """Tests for run_melt_quench."""

    @staticmethod
    def _mock_mq_result(n_steps: int = 50) -> dict:
        return {
            "structure": Atoms("SiO2", positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]], cell=[5, 5, 5], pbc=True),
            "result": [
                None,
                {
                    "temperature": np.full(n_steps, 300.0).tolist(),
                    "steps": list(range(n_steps)),
                },
            ],
        }

    @patch("amorphouspy.pipelines.meltquench.melt_quench_simulation")
    def test_returns_expected_keys(self, mock_mq: MagicMock) -> None:
        """run_melt_quench returns all expected result keys."""
        mock_mq.return_value = self._mock_mq_result()

        result = run_melt_quench(
            structure=Atoms("Si"),
            potential=MagicMock(),
        )

        expected_keys = {
            "final_structure",
            "mean_temperature",
            "simulation_steps",
            "temperature_trajectory",
            "steps_trajectory",
            "simulation_history",
            "timestep",
            "cooling_rate",
            "heating_rate",
            "temperature_high",
            "temperature_low",
            "n_averaging_frames",
        }
        assert set(result.keys()) == expected_keys

    @patch("amorphouspy.pipelines.meltquench.melt_quench_simulation")
    def test_default_temperature_high_from_protocol(self, mock_mq: MagicMock) -> None:
        """Default temperature_high is looked up from the protocol."""
        mock_mq.return_value = self._mock_mq_result()

        run_melt_quench(
            structure=Atoms("Si"),
            potential=MagicMock(),
            potential_type="pmmcs",
        )

        _, kwargs = mock_mq.call_args
        # Should use DEFAULT_MELT_TEMPERATURES["pmmcs"] or fallback 5000.0
        assert kwargs["temperature_high"] is not None

    @patch("amorphouspy.pipelines.meltquench.melt_quench_simulation")
    def test_explicit_temperature_high(self, mock_mq: MagicMock) -> None:
        """Explicit temperature_high is forwarded to the simulation."""
        mock_mq.return_value = self._mock_mq_result()

        run_melt_quench(
            structure=Atoms("Si"),
            potential=MagicMock(),
            temperature_high=8000.0,
        )

        _, kwargs = mock_mq.call_args
        assert kwargs["temperature_high"] == 8000.0

    @patch("amorphouspy.pipelines.meltquench.melt_quench_simulation")
    def test_mean_temperature_computed(self, mock_mq: MagicMock) -> None:
        """Mean temperature and step count are correctly computed."""
        n = 100
        mock_mq.return_value = self._mock_mq_result(n_steps=n)

        result = run_melt_quench(structure=Atoms("Si"), potential=MagicMock())

        assert result["mean_temperature"] == pytest.approx(300.0)
        assert result["simulation_steps"] == n
