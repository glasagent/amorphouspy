"""Unit tests for meltquench API functionality."""

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from pyiron_glass_api.app import app
from pyiron_glass_api.models import MeltquenchRequest

client = TestClient(app)


@pytest.fixture(autouse=True)
def _patch_worker(monkeypatch) -> None:
    """Replace background worker with a no-op that writes a completed result.

    This keeps tests fully in-process and avoids spawning real child processes.
    """
    from pyiron_glass_api import app as app_module

    async def fake_worker(task_id: str, request: MeltquenchRequest) -> None:
        from pyiron_glass_api.database import get_task_store

        ts = get_task_store()
        ts.set(
            task_id,
            {
                "state": "complete",
                "status": "Completed",
                "result": {
                    "composition": "0.6SiO2-0.25CaO-0.15Al2O3",
                    "final_structure": create_mock_structure_dict(),
                    "mean_temperature": 302.3333333333,
                    "simulation_steps": 3,
                    "structural_analysis": create_mock_structural_analysis_data(),
                },
            },
        )

    monkeypatch.setattr(app_module, "_meltquench_worker", fake_worker)


class MockAtoms:
    """Mock ASE Atoms-like object that can be serialized."""

    def __init__(self, atoms_dict: dict[str, Any]) -> None:
        """Initialize mock atoms with dictionary data."""
        self._dict = atoms_dict

    def get_masses(self) -> object:
        """Return a mock that has a sum method."""

        class MockMasses:
            def sum(self) -> int:
                return 1000  # mock mass

        return MockMasses()

    def __str__(self) -> str:
        """Return string representation of mock atoms."""
        return "Mock ASE structure with 100 atoms"

    def __getstate__(self) -> dict[str, Any]:
        """Return a fully serializable dictionary - avoid any ASE objects."""
        return {
            "numbers": self._dict["numbers"],
            "positions": self._dict["positions"],
            "cell": self._dict["cell"],  # Keep as nested list, not Cell object
            "pbc": self._dict["pbc"],
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore state from serialized dictionary."""
        self._dict = state


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
        "network": {"Qn_distribution": {}, "Qn_distribution_partial": {}, "connectivity": 0.0},
        "distributions": {"bond_angles": {}, "rings": {}},
        "rdfs": {"r": [], "rdfs": {}, "cumulative_coordination": {}},
        "elements": {"formers": [], "modifiers": [], "cutoffs": {}},
    }


def create_mock_result_data() -> dict[str, Any]:
    """Create mock simulation result data."""
    return {
        "structure": create_mock_structure_dict(),
        "result": {
            "volume": [1000, 1000, 1000],  # cm³
            "temperature": [300, 305, 302],  # K
            "steps": [1, 2, 3],
        },
    }


def setup_common_mocks(
    mock_project: MagicMock,
    mock_get_structure_dict: MagicMock,
    mock_get_ase_structure: MagicMock,
    mock_generate_potential: MagicMock,
    mock_melt_quench_simulation: MagicMock,
    mock_analyze_structure: MagicMock,
) -> None:
    """Set up common mock objects for meltquench tests."""
    # Mock the pyiron components
    mock_atoms_dict = {"atoms": [{"element": "Si", "position": [0, 0, 0]}] * 100}
    mock_get_structure_dict.return_value.pull.return_value = mock_atoms_dict

    # Create mock structure
    mock_structure_dict = create_mock_structure_dict()
    mock_structure = MockAtoms(mock_structure_dict)
    mock_get_ase_structure.return_value = mock_structure

    # Mock potential
    mock_potential = "mock_potential_content"
    mock_generate_potential.return_value = mock_potential

    # Mock structural analysis
    mock_analyze_structure.return_value.pull.return_value = create_mock_structural_analysis_data()

    # Mock simulation result
    mock_melt_quench_simulation.return_value.pull.return_value = create_mock_result_data()


def wait_for_task_completion(task_id: str, max_wait: float = 10.0) -> dict[str, Any]:
    """Wait for a task to complete and return the final check data."""
    waited = 0.0
    while waited < max_wait:
        check_response = client.get(f"/check/{task_id}")
        assert check_response.status_code == 200
        check_data = check_response.json()

        if check_data["state"] == "complete":
            return check_data
        if check_data["state"] == "error":
            pytest.fail(f"Simulation failed: {check_data.get('error')}")

        time.sleep(0.5)
        waited += 0.5

    pytest.fail(f"Task {task_id} did not complete within {max_wait} seconds")


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
    """Test the complete meltquench workflow without real background processes."""
    # Submit meltquench task
    payload = {"components": ["SiO2", "CaO", "Al2O3"], "values": [60.0, 25.0, 15.0], "unit": "wt"}
    response = client.post("/submit/meltquench", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert "status" in data

    # Handle cached results
    if data["status"] == "completed_from_cache":
        assert "result" in data
        validate_result_structure(data["result"])
        return

    # Wait for completion and validate
    assert data["status"] == "started"
    check_data = wait_for_task_completion(data["task_id"])

    assert check_data["task_id"] == data["task_id"]
    assert check_data["state"] == "complete"
    assert check_data["result"] is not None

    # Validate the result structure
    validate_result_structure(check_data["result"])

    # Validate composition format
    assert check_data["result"]["composition"] == "0.6SiO2-0.25CaO-0.15Al2O3"


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

    # Should either start a new task or return cached result
    assert "task_id" in submit_data
    assert "status" in submit_data
    assert submit_data["status"] in ["started", "completed_from_cache"]


@patch("amorphouspy.workflows.structural_analysis.plot_analysis_results_plotly")
def test_visualization_endpoint(mock_plot_analysis_results_plotly: MagicMock) -> None:
    """Test the visualization endpoint with mocked plot generation."""
    # Create a mock figure for the plot
    mock_fig = MagicMock()
    mock_fig.to_dict.return_value = {"data": [], "layout": {}}  # Mock Plotly figure dict
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
    from pyiron_glass_api.database import get_task_store

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
                "structural_analysis": {**create_mock_structural_analysis_data(), "density": 2.65},
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
    from pyiron_glass_api.app import get_meltquench_hash
    from pyiron_glass_api.database import get_task_store
    from pyiron_glass_api.models import MeltquenchRequest

    task_store = get_task_store()
    fake_task_id = "test-incomplete-task-123"

    # Create a proper request to generate hash
    request_data = {"components": ["SiO2"], "values": [100.0], "unit": "wt"}
    request = MeltquenchRequest(**request_data)
    request_hash = get_meltquench_hash(request)

    # Add incomplete task to database
    task_store.set(fake_task_id, {"state": "running", "request_data": request_data, "request_hash": request_hash})

    # Try to visualize incomplete task
    viz_response = client.get(f"/visualize/meltquench/{fake_task_id}")
    assert viz_response.status_code == 400
    assert "not completed yet" in viz_response.json()["detail"]
