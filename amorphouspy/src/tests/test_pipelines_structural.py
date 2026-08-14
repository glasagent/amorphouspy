"""Tests for amorphouspy.pipelines.structural."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from amorphouspy.fabrication.meltquench import extract_equilibration_frames
from amorphouspy.properties.structural.all import run_structural_analysis
from ase import Atoms

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def glass_atoms() -> Atoms:
    """Minimal Si-O structure with PBC."""
    positions = [[0, 0, 0], [1.6, 0, 0], [3.2, 0, 0]]
    return Atoms("SiOO", positions=positions, cell=[6, 6, 6], pbc=True)


# ---------------------------------------------------------------------------
# extract_equilibration_frames
# ---------------------------------------------------------------------------


class TestExtractEquilibrationFrames:
    """Tests for extract_equilibration_frames."""

    def test_no_history_returns_single_frame(self, glass_atoms: Atoms) -> None:
        """No history yields a single-frame list with the input structure."""
        result = extract_equilibration_frames(glass_atoms, simulation_history=None)
        assert len(result) == 1
        assert result[0] is glass_atoms

    def test_empty_history_returns_single_frame(self, glass_atoms: Atoms) -> None:
        """Empty history list falls back to single frame."""
        result = extract_equilibration_frames(glass_atoms, simulation_history=[])
        assert len(result) == 1

    def test_history_without_positions_returns_single_frame(self, glass_atoms: Atoms) -> None:
        """History stage without position data falls back to single frame."""
        history = [{"temperature": [300.0], "steps": [0]}]
        result = extract_equilibration_frames(glass_atoms, simulation_history=history)
        assert len(result) == 1

    def test_history_all_none_returns_single_frame(self, glass_atoms: Atoms) -> None:
        """All-None history falls back to single frame."""
        result = extract_equilibration_frames(glass_atoms, simulation_history=[None, None])
        assert len(result) == 1

    def test_single_frame_in_history_returns_single(self, glass_atoms: Atoms) -> None:
        """History with exactly one frame returns a single-element list."""
        pos = glass_atoms.get_positions()
        cell = glass_atoms.get_cell().array
        history = [{"positions": [pos], "cells": [cell]}]
        result = extract_equilibration_frames(glass_atoms, simulation_history=history)
        assert len(result) == 1

    def test_multiple_frames_extracted(self, glass_atoms: Atoms) -> None:
        """Multiple frames in history produce matching Atoms objects."""
        n_frames = 5
        base_pos = glass_atoms.get_positions()
        cell = glass_atoms.get_cell().array
        positions = [base_pos + i * 0.01 for i in range(n_frames)]
        cells = [cell for _ in range(n_frames)]
        history = [None, {"positions": positions, "cells": cells}]

        result = extract_equilibration_frames(glass_atoms, simulation_history=history)

        assert len(result) == n_frames
        for frame in result:
            assert isinstance(frame, Atoms)
            assert len(frame) == len(glass_atoms)
            assert all(frame.get_pbc())

    def test_frames_have_distinct_positions(self, glass_atoms: Atoms) -> None:
        """Extracted frames carry distinct atomic positions."""
        base_pos = glass_atoms.get_positions()
        cell = glass_atoms.get_cell().array
        positions = [base_pos, base_pos + 1.0]
        cells = [cell, cell]
        history = [{"positions": positions, "cells": cells}]

        result = extract_equilibration_frames(glass_atoms, simulation_history=history)
        assert not np.allclose(result[0].get_positions(), result[1].get_positions())


# ---------------------------------------------------------------------------
# run_structural_analysis
# ---------------------------------------------------------------------------


class TestRunStructuralAnalysis:
    """Tests for run_structural_analysis."""

    @patch("amorphouspy.properties.structural.all.analyze_structure")
    def test_single_frame_path(self, mock_analyze: MagicMock, glass_atoms: Atoms) -> None:
        """Single-frame analysis skips frame averaging."""
        mock_mean = MagicMock()
        mock_analyze.return_value = (mock_mean, None)

        mean_data, sem_data, n_frames, sampling_history = run_structural_analysis(glass_atoms)

        assert n_frames == 1
        assert mean_data is mock_mean
        assert sem_data is None
        assert sampling_history is None
        mock_analyze.assert_called_once()
        # Single frame: frame_averaging should not be passed as True
        _, kwargs = mock_analyze.call_args
        assert kwargs.get("frame_averaging", False) is not True

    @patch("amorphouspy.properties.structural.all._run_lammps_md")
    @patch("amorphouspy.properties.structural.all.analyze_structure")
    def test_multi_frame_path(self, mock_analyze: MagicMock, mock_run_md: MagicMock, glass_atoms: Atoms) -> None:
        """Multi-frame analysis enables frame averaging."""
        mock_mean = MagicMock()
        mock_sem = MagicMock()
        mock_analyze.return_value = (mock_mean, mock_sem)

        base_pos = glass_atoms.get_positions()
        cell = glass_atoms.get_cell().array
        mock_run_md.return_value = (
            glass_atoms,
            {"generic": {"positions": [base_pos, base_pos + 0.01, base_pos + 0.02], "cells": [cell] * 3}},
        )

        mean_data, sem_data, n_frames, sampling_history = run_structural_analysis(
            glass_atoms,
            potential=MagicMock(),
            n_averaging_frames=3,
        )

        assert n_frames == 3
        assert mean_data is mock_mean
        assert sem_data is mock_sem
        assert sampling_history is not None
        assert len(sampling_history) == 1
        assert sampling_history[0]["positions"] == mock_run_md.return_value[1]["generic"]["positions"]
        assert sampling_history[0]["cells"] == mock_run_md.return_value[1]["generic"]["cells"]
        _, kwargs = mock_analyze.call_args
        assert kwargs.get("frame_averaging") is True

    @patch("amorphouspy.properties.structural.all._run_lammps_md")
    @patch("amorphouspy.properties.structural.all.analyze_structure")
    def test_multi_frame_path_accepts_numpy_geometry_arrays(
        self, mock_analyze: MagicMock, mock_run_md: MagicMock, glass_atoms: Atoms
    ) -> None:
        """Real parsed NVT output may provide geometry series as NumPy arrays."""
        mock_mean = MagicMock()
        mock_sem = MagicMock()
        mock_analyze.return_value = (mock_mean, mock_sem)

        base_pos = glass_atoms.get_positions()
        cell = glass_atoms.get_cell().array
        mock_run_md.return_value = (
            glass_atoms,
            {
                "generic": {
                    "positions": np.array([base_pos, base_pos + 0.01, base_pos + 0.02]),
                    "cells": np.array([cell, cell, cell]),
                }
            },
        )

        mean_data, sem_data, n_frames, sampling_history = run_structural_analysis(
            glass_atoms,
            potential=MagicMock(),
            n_averaging_frames=3,
        )

        assert n_frames == 3
        assert mean_data is mock_mean
        assert sem_data is mock_sem
        assert sampling_history is not None
        assert np.allclose(sampling_history[0]["positions"], mock_run_md.return_value[1]["generic"]["positions"])
        assert np.allclose(sampling_history[0]["cells"], mock_run_md.return_value[1]["generic"]["cells"])

    def test_n_averaging_frames_gt_one_requires_potential(self, glass_atoms: Atoms) -> None:
        """Requesting multi-frame averaging without potential raises a clear error."""
        with pytest.raises(ValueError, match="potential must be provided"):
            run_structural_analysis(glass_atoms, n_averaging_frames=2)

    def test_n_averaging_frames_less_than_one_raises(self, glass_atoms: Atoms) -> None:
        """n_averaging_frames < 1 raises a clear error."""
        with pytest.raises(ValueError, match="n_averaging_frames must be >= 1"):
            run_structural_analysis(glass_atoms, n_averaging_frames=0)

    @patch("amorphouspy.properties.structural.all._run_lammps_md")
    @patch("amorphouspy.properties.structural.all.analyze_structure")
    def test_n_jobs_larger_than_n_averaging_frames_is_capped(
        self,
        mock_analyze: MagicMock,
        mock_run_md: MagicMock,
        glass_atoms: Atoms,
    ) -> None:
        """Oversized n_jobs should be capped automatically instead of raising."""
        mock_mean = MagicMock()
        mock_sem = MagicMock()
        mock_analyze.return_value = (mock_mean, mock_sem)

        base_pos = glass_atoms.get_positions()
        cell = glass_atoms.get_cell().array
        mock_run_md.return_value = (
            glass_atoms,
            {"generic": {"positions": [base_pos, base_pos + 0.01, base_pos + 0.02], "cells": [cell] * 3}},
        )

        run_structural_analysis(
            glass_atoms,
            potential=MagicMock(),
            n_averaging_frames=2,
            n_jobs=99,
        )

        assert mock_analyze.call_args.kwargs["n_jobs"] == 3

    @patch("amorphouspy.properties.structural.all._run_lammps_md")
    @patch("amorphouspy.properties.structural.all.analyze_structure")
    def test_n_jobs_auto_selects_half_of_allocated_cores(
        self,
        mock_analyze: MagicMock,
        mock_run_md: MagicMock,
        glass_atoms: Atoms,
    ) -> None:
        """Default n_jobs uses about half of allocated cores and caps by frame count."""
        mock_mean = MagicMock()
        mock_sem = MagicMock()
        mock_analyze.return_value = (mock_mean, mock_sem)

        base_pos = glass_atoms.get_positions()
        cell = glass_atoms.get_cell().array
        mock_run_md.return_value = (
            glass_atoms,
            {"generic": {"positions": [base_pos, base_pos + 0.01], "cells": [cell, cell]}},
        )

        run_structural_analysis(
            glass_atoms,
            potential=MagicMock(),
            n_averaging_frames=2,
            n_jobs=None,
            server_kwargs={"cores": 24},
        )

        # max frame jobs = n_averaging_frames + 1 = 3; half of 24 would be 12.
        assert mock_analyze.call_args.kwargs["n_jobs"] == 3

    @patch("amorphouspy.properties.structural.all._run_lammps_md")
    @patch("amorphouspy.properties.structural.all.analyze_structure")
    def test_n_jobs_auto_uses_half_cores_when_many_frames(
        self,
        mock_analyze: MagicMock,
        mock_run_md: MagicMock,
        glass_atoms: Atoms,
    ) -> None:
        """Auto n_jobs should use half cores when frame count is not the limiting factor."""
        mock_mean = MagicMock()
        mock_sem = MagicMock()
        mock_analyze.return_value = (mock_mean, mock_sem)

        base_pos = glass_atoms.get_positions()
        cell = glass_atoms.get_cell().array
        positions = [base_pos + i * 0.01 for i in range(20)]
        cells = [cell] * 20
        mock_run_md.return_value = (glass_atoms, {"generic": {"positions": positions, "cells": cells}})

        run_structural_analysis(
            glass_atoms,
            potential=MagicMock(),
            n_averaging_frames=20,
            n_jobs=None,
            server_kwargs={"cores": 24},
        )

        assert mock_analyze.call_args.kwargs["n_jobs"] == 12

    @patch("amorphouspy.properties.structural.all._run_lammps_md")
    def test_multi_frame_path_requires_positions_and_cells(self, mock_run_md: MagicMock, glass_atoms: Atoms) -> None:
        """Missing geometry arrays in NVT output should raise a clear error."""
        mock_run_md.return_value = (glass_atoms, {"generic": {"temperature": [300.0]}})

        with pytest.raises(ValueError, match="missing required 'positions'/'cells'"):
            run_structural_analysis(
                glass_atoms,
                potential=MagicMock(),
                n_averaging_frames=3,
            )

    @patch("amorphouspy.properties.structural.all._run_lammps_md")
    def test_multi_frame_path_requires_non_empty_aligned_geometry(
        self, mock_run_md: MagicMock, glass_atoms: Atoms
    ) -> None:
        """Empty or mismatched geometry arrays in NVT output should raise a clear error."""
        mock_run_md.return_value = (
            glass_atoms,
            {"generic": {"positions": [], "cells": [glass_atoms.get_cell().array]}},
        )

        with pytest.raises(ValueError, match="non-empty"):
            run_structural_analysis(
                glass_atoms,
                potential=MagicMock(),
                n_averaging_frames=3,
            )
