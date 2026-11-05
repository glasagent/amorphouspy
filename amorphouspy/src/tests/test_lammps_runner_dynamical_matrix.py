"""Tests for _run_lammps_dynamical_matrix in amorphouspy.lammps.runner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from amorphouspy.lammps.runner import _run_lammps_dynamical_matrix
from ase import Atoms

if TYPE_CHECKING:
    from collections.abc import Callable

SHIFT = np.array([0.1, 0.0, 0.0])
STATIC_FORCES = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
MIN_SUMMARY = {"max_force_ev_per_a": 99.0}
MATRIX = np.arange(36, dtype=float)  # 9 N² float64 values for N = 2


def _structure() -> Atoms:
    return Atoms("SiO", positions=[[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True)


def _minimised(structure: Atoms) -> Atoms:
    minimised = structure.copy()
    minimised.positions = minimised.positions + SHIFT
    return minimised


def _fake_lammps(content: bytes | None = MATRIX.tobytes()) -> Callable[..., dict]:
    """Stand-in for run_lammps_with_error_capture: writes the matrix file (unless None), returns step-0 forces."""

    def run(working_directory: str, **kwargs: Any) -> dict:
        if content is not None:
            name = kwargs["input_control_file"]["dynamical_matrix"].split()[-3]
            Path(working_directory, name).write_bytes(content)
        return {"generic": {"forces": np.array([STATIC_FORCES])}, "lammps": {}}

    return run


@pytest.mark.parametrize("minimize", [True, False])
@patch("amorphouspy.lammps.runner._run_lammps_minimize")
@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_dynamical_matrix_uses_minimised_or_input_structure(
    mock_run: MagicMock, mock_minimize: MagicMock, tmp_path: Path, *, minimize: bool
) -> None:
    """minimize=True forwards the kwargs and differentiates the minimised structure; False uses the input."""
    structure = _structure()
    minimised = _minimised(structure)
    mock_minimize.return_value = (minimised, {}, MIN_SUMMARY)
    mock_run.side_effect = _fake_lammps()
    options: dict[str, Any] = {"minimize_kwargs": {"force_tolerance": 1e-6}} if minimize else {"minimize": False}

    returned, summary = _run_lammps_dynamical_matrix(
        structure, "dummy", tmp_path / "d.bin", server_kwargs={"cores": 2}, tmp_working_directory=tmp_path, **options
    )

    expected = minimised if minimize else structure
    sent = mock_run.call_args.kwargs["structure"]
    assert sent is not expected
    np.testing.assert_array_equal(sent.get_positions(), expected.get_positions())
    assert returned is expected
    if minimize:
        mock_minimize.assert_called_once_with(
            structure, "dummy", server_kwargs={"cores": 2}, tmp_working_directory=tmp_path, force_tolerance=1e-6
        )
        assert summary["minimization"] is MIN_SUMMARY
    else:
        mock_minimize.assert_not_called()
        assert summary["minimization"] is None


@patch("amorphouspy.lammps.runner._run_lammps_minimize")
@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_dynamical_matrix_static_run_moves_file_and_summarises(
    mock_run: MagicMock, mock_minimize: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The static run gets the exact input lines; the file is moved to matrix_path; summary uses step-0 forces."""
    structure = _structure()
    mock_minimize.return_value = (_minimised(structure), {}, MIN_SUMMARY)
    mock_run.side_effect = _fake_lammps()
    (tmp_path / "out").mkdir()
    monkeypatch.chdir(tmp_path)

    _, summary = _run_lammps_dynamical_matrix(
        structure, "dummy", "out/d.bin", displacement=2e-5, tmp_working_directory=tmp_path
    )

    kwargs = mock_run.call_args.kwargs
    assert kwargs["calc_mode"] == "static"
    assert kwargs["dump_final_structure"] is False
    assert kwargs["input_control_file"] == {
        "units": "metal\natom_modify map array",
        "dynamical_matrix": "all regular 2e-05 file dynamical_matrix.bin binary yes",
    }
    matrix_path = tmp_path / "out" / "d.bin"
    assert matrix_path.read_bytes() == MATRIX.tobytes()
    assert list(Path(kwargs["working_directory"]).iterdir()) == []
    wall_time_s = summary.pop("wall_time_s")
    assert wall_time_s >= 0
    assert summary == {
        "matrix_path": matrix_path,
        "matrix_bytes": 288,
        "n_atoms": 2,
        "displacement_a": 2e-5,
        "style": "regular",
        "max_force_ev_per_a": 5.0,
        "force_norm_ev_per_a": pytest.approx(np.sqrt(26.0)),
        "minimization": MIN_SUMMARY,
    }


@pytest.mark.parametrize(
    ("content", "match"),
    [(None, "did not write"), (MATRIX.tobytes()[:-1], "287 bytes, expected 288")],
    ids=["missing", "wrong-size"],
)
@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_dynamical_matrix_rejects_missing_or_wrong_size_file(
    mock_run: MagicMock, tmp_path: Path, content: bytes | None, match: str
) -> None:
    """A missing or wrongly sized matrix file raises, and nothing reaches matrix_path."""
    mock_run.side_effect = _fake_lammps(content)
    matrix_path = tmp_path / "d.bin"

    with pytest.raises(RuntimeError, match=match):
        _run_lammps_dynamical_matrix(_structure(), "dummy", matrix_path, minimize=False)

    assert not matrix_path.exists()


@pytest.mark.parametrize(
    ("path", "options", "match"),
    [
        ("d.bin", {"displacement": 0.0}, "displacement"),
        ("d.bin", {"displacement": float("nan")}, "displacement"),
        ("exists.bin", {}, "already exists"),
        ("missing/d.bin", {}, "parent directory"),
        ("d.bin", {"minimize": False, "minimize_kwargs": {}}, "minimize=False"),
        ("d.bin", {"minimize_kwargs": {"server_kwargs": {"cores": 2}}}, "server_kwargs"),
        ("d.bin", {"minimize_kwargs": {"tmp_working_directory": "x"}}, "tmp_working_directory"),
    ],
    ids=[
        "displacement-zero",
        "displacement-nan",
        "path-exists",
        "parent-missing",
        "kwargs-without-minimize",
        "kwargs-server",
        "kwargs-tmp-dir",
    ],
)
@patch("amorphouspy.lammps.runner._run_lammps_minimize")
@patch("amorphouspy.lammps.runner.run_lammps_with_error_capture")
def test_dynamical_matrix_rejects_invalid_arguments(
    mock_run: MagicMock, mock_minimize: MagicMock, tmp_path: Path, path: str, options: dict, match: str
) -> None:
    """Invalid arguments raise ValueError before any LAMMPS run."""
    (tmp_path / "exists.bin").touch()

    with pytest.raises(ValueError, match=match):
        _run_lammps_dynamical_matrix(_structure(), "dummy", tmp_path / path, **options)

    mock_minimize.assert_not_called()
    mock_run.assert_not_called()
