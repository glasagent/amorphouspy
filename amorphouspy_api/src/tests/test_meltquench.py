"""Unit tests for meltquench API functionality."""

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from amorphouspy_api.app import app
from amorphouspy_api.models import MeltquenchRequest

client = TestClient(app)


@pytest.fixture(autouse=True)
def _patch_job_manager(monkeypatch) -> None:
    """Replace JobManager.submit_meltquench with a mock that returns completed result.

    This keeps tests fully in-process and avoids spawning real executorlib jobs.
    """
    from amorphouspy_api import jobs as jobs_module

    def fake_submit_meltquench(self, request_data: dict) -> dict:
        return {
            "state": "complete",
            "status": "Completed",
            "result": {
                "composition": "0.6SiO2-0.25CaO-0.15Al2O3",
                "final_structure": create_mock_structure_dict(),
                "mean_temperature": 302.3333333333,
                "simulation_steps": 3,
                "structural_analysis": create_mock_structural_analysis_data(),
            },
        }

    monkeypatch.setattr(jobs_module.JobManager, "submit_meltquench", fake_submit_meltquench)
    monkeypatch.setattr(jobs_module.JobManager, "check_status", fake_submit_meltquench)


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


def validate_result_structure(result: dict[str, Any]) -> None:
    """Validate the structure of a meltquench result."""
    assert "composition" in result
    assert "final_structure" in result
    assert "mean_temperature" in result
    assert "structural_analysis" in result
    assert "simulation_steps" in result

    # Validate numerical values
    assert isinstance(result["mean_temperature"], float)
    # Handle both dict and StructureData object cases
    if isinstance(result["structural_analysis"], dict):
        assert isinstance(result["structural_analysis"]["density"], float)
    else:
        assert isinstance(result["structural_analysis"].density, float)
    assert isinstance(result["simulation_steps"], int)


def test_submit_meltquench_and_check() -> None:
    """Test the complete meltquench workflow with mocked job manager."""
    payload = {
        "components": ["SiO2", "CaO", "Al2O3"],
        "values": [60.0, 25.0, 15.0],
        "unit": "wt",
    }
    response = client.post("/submit/meltquench", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert "status" in data

    # Mock returns "completed" immediately
    assert data["status"] in ["completed", "completed_from_cache"]
    assert "result" in data
    validate_result_structure(data["result"])


def test_check_running_then_complete() -> None:
    """Test the running → complete flow by directly manipulating the task store."""
    from amorphouspy_api.database import get_task_store

    task_store = get_task_store()
    task_id = "test-running-to-complete-task"

    # Insert a "running" task directly into the task store
    task_store.set(
        task_id,
        {
            "state": "running",
            "status": "Running simulation",
            "request_data": {"components": ["SiO2"], "values": [100.0], "unit": "wt"},
            "request_hash": "test-hash-running",
        },
    )

    # Check that the task is running
    check_response = client.get(f"/check/{task_id}")
    assert check_response.status_code == 200
    check_data = check_response.json()
    assert check_data["state"] == "running"

    # Simulate completion by updating the task store entry
    task_store.set(
        task_id,
        {
            "state": "complete",
            "status": "Completed",
            "result": {
                "composition": "1.0SiO2",
                "final_structure": create_mock_structure_dict(),
                "mean_temperature": 300.0,
                "simulation_steps": 3,
                "structural_analysis": create_mock_structural_analysis_data(),
            },
        },
    )

    # Check again - should now be complete
    check_response = client.get(f"/check/{task_id}")
    assert check_response.status_code == 200
    check_data = check_response.json()
    assert check_data["state"] == "complete"
    assert check_data["result"] is not None
    validate_result_structure(check_data["result"])


def test_check_nonexistent_task() -> None:
    """Test checking a task that doesn't exist."""
    response = client.get("/check/nonexistent-task-id")
    assert response.status_code == 404
    assert "Task not found" in response.json()["detail"]


def test_invalid_payload() -> None:
    """Test submitting an invalid payload."""
    payload = {
        "components": ["SiO2"],
        "values": [60.0, 25.0],  # Mismatched lengths
        "unit": "wt",
    }
    response = client.post("/submit/meltquench", json=payload)
    assert response.status_code == 422  # Validation error


def test_root_redirect() -> None:
    """Test that root redirects to docs."""
    # FastAPI TestClient follows redirects by default, so we need to check differently
    # We can verify that accessing "/" eventually serves docs content
    response = client.get("/")
    assert response.status_code == 200
    # The response should contain swagger/docs content when redirected
    assert "swagger" in response.text.lower() or "openapi" in response.text.lower()


def validate_cached_result(data: dict[str, Any] | None) -> None:
    """Validate cached result structure if it exists."""
    if data is not None:
        assert "composition" in data
        assert "structural_analysis" in data
        # Handle both dict and StructureData object cases
        if isinstance(data["structural_analysis"], dict):
            assert "density" in data["structural_analysis"]
        else:
            assert hasattr(data["structural_analysis"], "density")
        assert "final_structure" in data
        assert "mean_temperature" in data
        assert "simulation_steps" in data


def test_check_cached_result_found() -> None:
    """Test checking for cached results with a specific composition."""
    payload = {
        "components": ["SiO2", "K2O"],  # Different from other tests
        "values": [85.0, 15.0],
        "unit": "wt",
    }

    response = client.post("/cache/meltquench", json=payload)
    assert response.status_code == 200
    validate_cached_result(response.json())


def test_check_cached_result_not_found() -> None:
    """Test checking for cached results with another unique composition."""
    payload = {
        "components": ["SiO2", "Li2O"],  # Different from other tests
        "values": [90.0, 10.0],
        "unit": "wt",
    }

    response = client.post("/cache/meltquench", json=payload)
    assert response.status_code == 200
    validate_cached_result(response.json())


def test_caching_behavior() -> None:
    """Test that caching actually works by submitting and then checking cache."""
    unique_payload = {
        "components": ["SiO2", "MgO"],
        "values": [70.0, 30.0],
        "unit": "wt",
        "heating_rate": int(1e15),  # Fast for testing
        "cooling_rate": int(1e15),
        "n_print": 100,
    }

    # Check cache first
    cache_response = client.post("/cache/meltquench", json=unique_payload)
    assert cache_response.status_code == 200

    # Submit the simulation (will be mocked by autouse fixture)
    submit_response = client.post("/submit/meltquench", json=unique_payload)
    assert submit_response.status_code == 200
    submit_data = submit_response.json()

    # Should either start a new task or return cached/completed result
    assert "task_id" in submit_data
    assert "status" in submit_data
    assert submit_data["status"] in ["started", "completed", "completed_from_cache"]


@patch("amorphouspy.workflows.structural_analysis.plot_analysis_results_plotly")
def test_visualization_endpoint(mock_plot_analysis_results_plotly: MagicMock) -> None:
    """Test the visualization endpoint with mocked plot generation."""
    # Create a mock figure for the plot
    mock_fig = MagicMock()
    mock_fig.to_dict.return_value = {
        "data": [],
        "layout": {},
    }  # Mock Plotly figure dict
    mock_plot_analysis_results_plotly.return_value = mock_fig

    # Submit task with unique payload to avoid caching
    unique_suffix = str(int(time.time() * 1000))  # millisecond timestamp
    payload = {
        "components": ["SiO2", "Na2O"],
        "values": [75.0, 25.0],
        "unit": "wt",
        "heating_rate": int(unique_suffix[-6:]),  # Use last 6 digits
    }

    submit_response = client.post("/submit/meltquench", json=payload)
    assert submit_response.status_code == 200
    submit_data = submit_response.json()
    task_id = submit_data["task_id"]

    # Overwrite task result directly to tailor the visualization data
    from amorphouspy_api.database import get_task_store

    get_task_store().set(
        task_id,
        {
            "state": "complete",
            "status": "Completed",
            "result": {
                "composition": "0.75SiO2-0.25Na2O",
                "final_structure": create_mock_structure_dict(),
                "mean_temperature": 300.0,
                "simulation_steps": 3,
                "structural_analysis": {
                    **create_mock_structural_analysis_data(),
                    "density": 2.65,
                },
            },
        },
    )

    # Test the visualization endpoint
    viz_response = client.get(f"/visualize/meltquench/{task_id}")
    assert viz_response.status_code == 200

    # Check that we get HTML content
    assert viz_response.headers["content-type"] == "text/html; charset=utf-8"
    html_content = viz_response.text

    # Verify HTML contains expected elements
    assert "Melt-Quench Simulation Results" in html_content
    assert task_id in html_content
    assert "plotlyData" in html_content or "plotly-div" in html_content

    # Verify the plot function was called
    mock_plot_analysis_results_plotly.assert_called_once()


def test_visualization_endpoint_task_not_found() -> None:
    """Test visualization endpoint with non-existent task."""
    response = client.get("/visualize/meltquench/nonexistent-task")
    assert response.status_code == 404
    assert "Task not found" in response.json()["detail"]


def test_visualization_endpoint_incomplete_task() -> None:
    """Test visualization endpoint with incomplete task."""
    # Create a task manually in the database with 'running' state
    from amorphouspy_api.app import get_meltquench_hash
    from amorphouspy_api.database import get_task_store

    task_store = get_task_store()
    fake_task_id = "test-incomplete-task-123"

    # Create a proper request to generate hash
    request_data = {"components": ["SiO2"], "values": [100.0], "unit": "wt"}
    request = MeltquenchRequest(**request_data)
    request_hash = get_meltquench_hash(request)

    # Add incomplete task to database
    task_store.set(
        fake_task_id,
        {
            "state": "running",
            "request_data": request_data,
            "request_hash": request_hash,
        },
    )

    # Try to visualize incomplete task
    viz_response = client.get(f"/visualize/meltquench/{fake_task_id}")
    assert viz_response.status_code == 400
    assert "not completed yet" in viz_response.json()["detail"]
