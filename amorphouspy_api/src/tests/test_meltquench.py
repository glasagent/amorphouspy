"""Unit tests for meltquench API functionality.

Tests insert tasks directly into the task store rather than mocking the executor,
except for tests that specifically exercise the /submit endpoint.
"""

import time
from collections.abc import Generator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from amorphouspy_api.app import app
from amorphouspy_api.database import get_task_store
from amorphouspy_api.models import MeltquenchRequest
from amorphouspy_api.routers.meltquench import get_meltquench_hash

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_mock_structure_dict() -> dict[str, Any]:
    """Create a mock structure dictionary."""
    return {
        "numbers": [14] * 50 + [8] * 100,  # Si and O atoms
        "positions": [[0.0, 0.0, 0.0]] * 150,  # Simple positions
        "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],  # 10x10x10 box
        "pbc": [True, True, True],
    }


def create_mock_structural_analysis_data() -> dict[str, Any]:
    """Create mock structural analysis data."""
    return {
        "density": 2.5,
        "coordination": {"oxygen": {}, "formers": {}, "modifiers": {}},
        "network": {
            "Qn_distribution": {},
            "Qn_distribution_partial": {},
            "connectivity": 0.0,
        },
        "distributions": {"bond_angles": {}, "rings": {}},
        "rdfs": {"r": [], "rdfs": {}, "cumulative_coordination": {}},
        "elements": {"formers": [], "modifiers": [], "cutoffs": {}},
    }


def create_mock_result(
    composition: str = "0.6SiO2-0.25CaO-0.15Al2O3",
) -> dict[str, Any]:
    """Create a complete mock meltquench result."""
    return {
        "composition": composition,
        "final_structure": create_mock_structure_dict(),
        "mean_temperature": 302.3333333333,
        "simulation_steps": 3,
        "structural_analysis": create_mock_structural_analysis_data(),
    }


def insert_completed_task(
    task_id: str,
    *,
    request_hash: str = "test-hash",
    composition: str = "0.6SiO2-0.25CaO-0.15Al2O3",
    request_data: dict[str, Any] | None = None,
) -> None:
    """Insert a completed task into the task store."""
    get_task_store().set(
        task_id,
        {
            "state": "complete",
            "request_hash": request_hash,
            "request_data": request_data,
            "result": create_mock_result(composition),
        },
    )


def insert_running_task(
    task_id: str,
    *,
    request_hash: str = "test-hash-running",
    request_data: dict[str, Any] | None = None,
) -> None:
    """Insert a running task into the task store."""
    if request_data is None:
        request_data = {
            "components": ["SiO2"],
            "values": [100.0],
            "unit": "wt",
            "n_atoms": 3,
            "potential_type": "pmmcs",
            "heating_rate": 1e12,
            "cooling_rate": 1e12,
            "n_print": 100,
        }
    get_task_store().set(
        task_id,
        {
            "state": "running",
            "request_hash": request_hash,
            "request_data": request_data,
        },
    )


def validate_result_structure(result: dict[str, Any]) -> None:
    """Validate the structure of a meltquench result."""
    assert "composition" in result
    assert "final_structure" in result
    assert "mean_temperature" in result
    assert "structural_analysis" in result
    assert "simulation_steps" in result

    assert isinstance(result["mean_temperature"], float)
    if isinstance(result["structural_analysis"], dict):
        assert isinstance(result["structural_analysis"]["density"], float)
    else:
        assert isinstance(result["structural_analysis"].density, float)
    assert isinstance(result["simulation_steps"], int)


# ---------------------------------------------------------------------------
# /submit/meltquench tests
# ---------------------------------------------------------------------------


@contextmanager
def _mock_executor_context() -> Generator[SimpleNamespace, None, None]:
    """Context manager that patches get_executor and run_meltquench_workflow."""
    mock_future = MagicMock()
    mock_future.result.return_value = create_mock_result()

    with (
        patch("amorphouspy_api.routers.meltquench.get_executor") as mock_get_exe,
        patch(
            "amorphouspy_api.routers.meltquench.run_meltquench_workflow",
            return_value=mock_future,
        ) as mock_workflow,
    ):
        mock_get_exe.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_get_exe.return_value.__exit__ = MagicMock(return_value=False)
        yield SimpleNamespace(mock_workflow=mock_workflow, mock_future=mock_future)


def test_submit_meltquench_new_task() -> None:
    """Test submitting a new task runs the executor and returns completed."""
    with _mock_executor_context():
        payload = {
            "components": ["SiO2", "CaO", "Al2O3"],
            "values": [60.0, 25.0, 15.0],
            "unit": "wt",
        }
        response = client.post("/submit/meltquench", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "completed"
    assert data["result"] is not None
    validate_result_structure(data["result"])

    # Verify task was stored as complete
    stored = get_task_store().get(data["task_id"])
    assert stored is not None
    assert stored["state"] == "complete"


def test_submit_meltquench_returns_cached() -> None:
    """Test that submitting a duplicate request returns the cached result."""
    # Pre-insert a completed task with a known hash
    request = MeltquenchRequest(
        components=["SiO2", "BaO"],
        values=[80.0, 20.0],
        unit="wt",
    )
    request_hash = get_meltquench_hash(request)
    insert_completed_task("cached-task-1", request_hash=request_hash, composition="0.8SiO2-0.2BaO")

    # Submit with the same parameters — should return cached
    response = client.post("/submit/meltquench", json=request.model_dump())

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed_from_cache"
    assert data["task_id"] == "cached-task-1"


def test_submit_meltquench_stores_request_data() -> None:
    """Test that submitting a new task stores request_data."""
    with _mock_executor_context():
        payload = {
            "components": ["SiO2", "ZnO"],
            "values": [90.0, 10.0],
            "unit": "wt",
        }
        response = client.post("/submit/meltquench", json=payload)

    assert response.status_code == 200
    stored = get_task_store().get(response.json()["task_id"])
    assert stored is not None
    assert stored["request_data"]["components"] == ["SiO2", "ZnO"]
    assert stored["request_data"]["values"] == [90.0, 10.0]


def test_submit_meltquench_executor_error_returns_500() -> None:
    """Test that an executor error returns HTTP 500 and stores the error."""
    with patch("amorphouspy_api.routers.meltquench.get_executor", side_effect=RuntimeError("LAMMPS crashed")):
        payload = {
            "components": ["SiO2", "TiO2"],
            "values": [95.0, 5.0],
            "unit": "wt",
        }
        response = client.post("/submit/meltquench", json=payload)

    assert response.status_code == 500
    assert "LAMMPS crashed" in response.json()["detail"]


def test_invalid_payload() -> None:
    """Test submitting an invalid payload."""
    payload = {
        "components": ["SiO2"],
        "values": [60.0, 25.0],  # Mismatched lengths
        "unit": "wt",
    }
    response = client.post("/submit/meltquench", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# /check/{task_id} tests
# ---------------------------------------------------------------------------


def test_check_completed_task() -> None:
    """Test that checking a completed task returns the stored result."""
    insert_completed_task("check-complete-1", request_hash="hash-check-1")

    response = client.get("/check/check-complete-1")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["result"] is not None
    validate_result_structure(data["result"])


def test_check_running_task_resubmits() -> None:
    """Test that checking a running task re-submits to executor and completes."""
    insert_running_task("check-running-1", request_hash="hash-check-running-1")

    with _mock_executor_context():
        response = client.get("/check/check-running-1")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["result"] is not None
    validate_result_structure(data["result"])

    # Verify task store was updated to complete
    stored = get_task_store().get("check-running-1")
    assert stored["state"] == "complete"


def test_check_errored_task() -> None:
    """Test that checking an errored task returns the error."""
    get_task_store().set(
        "check-error-1",
        {
            "state": "error",
            "request_hash": "hash-check-error-1",
            "error": "LAMMPS crashed",
        },
    )

    response = client.get("/check/check-error-1")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["error"] == "LAMMPS crashed"


def test_check_nonexistent_task() -> None:
    """Test checking a task that doesn't exist."""
    response = client.get("/check/nonexistent-task-id")
    assert response.status_code == 404
    assert "Task not found" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /cache/meltquench tests
# ---------------------------------------------------------------------------


def test_cache_hit() -> None:
    """Test cache endpoint returns a result when one exists."""
    request = MeltquenchRequest(
        components=["SiO2", "K2O"],
        values=[85.0, 15.0],
        unit="wt",
    )
    request_hash = get_meltquench_hash(request)
    insert_completed_task("cache-hit-1", request_hash=request_hash, composition="0.85SiO2-0.15K2O")

    response = client.post("/cache/meltquench", json=request.model_dump())
    assert response.status_code == 200
    data = response.json()
    assert data is not None
    assert data["composition"] == "0.85SiO2-0.15K2O"


def test_cache_miss() -> None:
    """Test cache endpoint returns null when no result exists."""
    payload = {
        "components": ["SiO2", "Li2O"],
        "values": [90.0, 10.0],
        "unit": "wt",
    }
    response = client.post("/cache/meltquench", json=payload)
    assert response.status_code == 200
    assert response.json() is None


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------


@patch("amorphouspy.workflows.structural_analysis.plot_analysis_results_plotly")
def test_visualization_endpoint(mock_plot_analysis_results_plotly: MagicMock) -> None:
    """Test the visualization endpoint returns HTML for a completed task."""
    mock_fig = MagicMock()
    mock_fig.to_dict.return_value = {"data": [], "layout": {}}
    mock_plot_analysis_results_plotly.return_value = mock_fig

    task_id = f"viz-task-{int(time.time() * 1000)}"
    get_task_store().set(
        task_id,
        {
            "state": "complete",
            "request_hash": f"viz-hash-{task_id}",
            "result": {
                **create_mock_result("0.75SiO2-0.25Na2O"),
                "structural_analysis": {
                    **create_mock_structural_analysis_data(),
                    "density": 2.65,
                },
            },
        },
    )

    response = client.get(f"/visualize/meltquench/{task_id}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"

    html_content = response.text
    assert "Melt-Quench Simulation Results" in html_content
    assert task_id in html_content
    assert "plotlyData" in html_content or "plotly-div" in html_content
    mock_plot_analysis_results_plotly.assert_called_once()


def test_visualization_task_not_found() -> None:
    """Test visualization endpoint with non-existent task."""
    response = client.get("/visualize/meltquench/nonexistent-task")
    assert response.status_code == 404
    assert "Task not found" in response.json()["detail"]


def test_visualization_incomplete_task() -> None:
    """Test visualization endpoint with an incomplete task."""
    task_id = "viz-incomplete-task"
    insert_running_task(task_id, request_hash="viz-incomplete-hash")

    response = client.get(f"/visualize/meltquench/{task_id}")
    assert response.status_code == 400
    assert "not completed yet" in response.json()["detail"]


# ---------------------------------------------------------------------------
# General tests
# ---------------------------------------------------------------------------


def test_root_redirect() -> None:
    """Test that root redirects to docs."""
    response = client.get("/")
    assert response.status_code == 200
    assert "swagger" in response.text.lower() or "openapi" in response.text.lower()
