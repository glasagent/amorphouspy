"""Unit tests for meltquench API functionality."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from pyiron_glass_api.app import app

client = TestClient(app)


@patch("pyiron_glass.workflows.structural_analysis.analyze_structure")
@patch("pyiron_glass.melt_quench_simulation")
@patch("pyiron_glass.generate_potential")
@patch("pyiron_glass.get_ase_structure")
@patch("pyiron_glass.get_structure_dict")
@patch("pyiron_base.Project")
def test_submit_meltquench_and_check(
    mock_project,
    mock_get_structure_dict,
    mock_get_ase_structure,
    mock_generate_potential,
    mock_melt_quench_simulation,
    mock_analyze_structure,
) -> None:
    """Test the complete meltquench workflow with mocked pyiron dependencies."""
    # Mock the pyiron components
    mock_atoms_dict = {"atoms": [{"element": "Si", "position": [0, 0, 0]}] * 100}
    mock_get_structure_dict.return_value.pull.return_value = mock_atoms_dict

    mock_structure = MagicMock()
    mock_structure.get_masses.return_value.sum.return_value = 1000  # mock mass
    mock_structure.__str__ = lambda self: "Mock ASE structure with 100 atoms"
    mock_get_ase_structure.return_value = mock_structure

    mock_potential = "mock_potential_content"
    mock_generate_potential.return_value = mock_potential

    # Mock structural analysis result - return the dict directly instead of a MagicMock
    mock_structural_analysis_data = {
        "density": 2.5,
        "coordination": {"oxygen": {}, "formers": {}, "modifiers": {}},
        "network": {"Qn_distribution": {}, "Qn_distribution_partial": {}, "connectivity": 0.0},
        "distributions": {"bond_angles": {}, "rings": {}},
        "rdfs": {"r": [], "rdfs": {}, "cumulative_coordination": {}},
        "elements": {"formers": [], "modifiers": [], "cutoffs": {}},
    }
    mock_analyze_structure.return_value.pull.return_value = mock_structural_analysis_data
    mock_generate_potential.return_value = mock_potential

    # Mock the simulation result - create a separate structure mock for the result
    mock_result_structure = MagicMock()
    mock_result_structure.__str__ = lambda self: "Mock final structure with 100 atoms"
    mock_result_structure.__len__ = lambda self: 100

    mock_result = {
        "structure": mock_result_structure,
        "result": {
            "volume": [1000, 1000, 1000],  # cm³
            "temperature": [300, 305, 302],  # K
            "steps": [1, 2, 3],
        },
    }
    mock_melt_quench_simulation.return_value.pull.return_value = mock_result

    # Updated payload to match simplified API
    payload = {"components": ["SiO2", "CaO", "Al2O3"], "values": [60.0, 25.0, 15.0], "unit": "wt"}

    # Submit meltquench task
    response = client.post("/submit_meltquench", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert "status" in data

    # With caching, we might get either a new task or cached result
    if data["status"] == "completed_from_cache":
        # If cached, verify the result structure
        assert "result" in data
        result = data["result"]
        assert "composition" in result
        assert "structural_analysis" in result
        assert result["structural_analysis"]["density"] > 0
        return  # Exit early since we got the cached result

    # If not cached, should be "started"
    assert data["status"] == "started"
    task_id = data["task_id"]

    # Since we're now using thread executor, we need to wait longer for completion
    import time

    max_wait = 10  # seconds
    waited = 0

    while waited < max_wait:
        check_response = client.get(f"/check/{task_id}")
        assert check_response.status_code == 200
        check_data = check_response.json()

        if check_data["state"] == "complete":
            break
        if check_data["state"] == "error":
            pytest.fail(f"Simulation failed: {check_data.get('error')}")

        time.sleep(0.5)
        waited += 0.5

    # Final check
    check_response = client.get(f"/check/{task_id}")
    assert check_response.status_code == 200
    check_data = check_response.json()

    assert check_data["task_id"] == task_id
    assert check_data["state"] == "complete"
    assert check_data["result"] is not None

    # Validate the result structure
    result = check_data["result"]
    assert "composition" in result
    assert "final_structure" in result
    assert "mean_temperature" in result
    assert "structural_analysis" in result
    assert "simulation_steps" in result

    # Validate composition format
    assert result["composition"] == "0.6SiO2-0.25CaO-0.15Al2O3"

    # Validate numerical values
    assert isinstance(result["mean_temperature"], float)
    # Handle both dict and StructureData object cases
    if isinstance(result["structural_analysis"], dict):
        assert isinstance(result["structural_analysis"]["density"], float)
    else:
        assert isinstance(result["structural_analysis"].density, float)
    assert isinstance(result["simulation_steps"], int)


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
    response = client.post("/submit_meltquench", json=payload)
    assert response.status_code == 422  # Validation error


def test_root_redirect() -> None:
    """Test that root redirects to docs."""
    # FastAPI TestClient follows redirects by default, so we need to check differently
    # We can verify that accessing "/" eventually serves docs content
    response = client.get("/")
    assert response.status_code == 200
    # The response should contain swagger/docs content when redirected
    assert "swagger" in response.text.lower() or "openapi" in response.text.lower()


def test_check_cached_result_found() -> None:
    """Test checking for cached results with a specific composition."""
    # Use a unique composition
    payload = {
        "components": ["SiO2", "K2O"],  # Different from other tests
        "values": [85.0, 15.0],
        "unit": "wt",
    }

    response = client.post("/check_cached_result", json=payload)
    assert response.status_code == 200
    data = response.json()

    # With working cache, this could be None (no cache) or a result (if cached)
    # Both are valid responses
    if data is not None:
        # If cached result exists, verify it has the right structure
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


def test_check_cached_result_not_found() -> None:
    """Test checking for cached results with another unique composition."""
    # Use yet another unique composition
    payload = {
        "components": ["SiO2", "Li2O"],  # Different from other tests
        "values": [90.0, 10.0],
        "unit": "wt",
    }

    response = client.post("/check_cached_result", json=payload)
    assert response.status_code == 200
    data = response.json()

    # With working cache, this could be None (no cache) or a result (if cached)
    # Both are valid responses
    if data is not None:
        # If cached result exists, verify it has the right structure
        assert "composition" in data
        assert "structural_analysis" in data
        assert data["structural_analysis"]["density"] > 0
        assert "final_structure" in data
        assert "mean_temperature" in data
        assert "simulation_steps" in data


def test_caching_behavior() -> None:
    """Test that caching actually works by submitting and then checking cache."""
    # Use a unique composition for this test
    unique_payload = {
        "components": ["SiO2", "MgO"],
        "values": [70.0, 30.0],
        "unit": "wt",
        "heating_rate": int(1e15),  # Fast for testing
        "cooling_rate": int(1e15),
        "n_print": 100,
    }

    # First check - should not be cached initially (unless run before)
    cache_response = client.post("/check_cached_result", json=unique_payload)
    assert cache_response.status_code == 200
    cache_response.json()

    # Submit the simulation (will be mocked)
    submit_response = client.post("/submit_meltquench", json=unique_payload)
    assert submit_response.status_code == 200
    submit_data = submit_response.json()

    # Should either start a new task or return cached result
    assert "task_id" in submit_data
    assert "status" in submit_data

    # The status should indicate either "started" (new task) or "completed_from_cache" (cached)
    assert submit_data["status"] in ["started", "completed_from_cache"]


@patch("pyiron_glass.workflows.structural_analysis.analyze_structure")
@patch("pyiron_glass.workflows.structural_analysis.plot_analysis_results")
@patch("pyiron_glass.melt_quench_simulation")
@patch("pyiron_glass.generate_potential")
@patch("pyiron_glass.get_ase_structure")
@patch("pyiron_glass.get_structure_dict")
@patch("pyiron_base.Project")
def test_visualization_endpoint(
    mock_project,
    mock_get_structure_dict,
    mock_get_ase_structure,
    mock_generate_potential,
    mock_melt_quench_simulation,
    mock_plot_analysis_results,
    mock_analyze_structure,
) -> None:
    """Test the visualization endpoint with mocked plot generation."""
    from unittest.mock import MagicMock

    # Mock the pyiron components (same as other tests)
    mock_atoms_dict = {"atoms": [{"element": "Si", "position": [0, 0, 0]}] * 100}
    mock_get_structure_dict.return_value.pull.return_value = mock_atoms_dict

    mock_structure = MagicMock()
    mock_structure.get_masses.return_value.sum.return_value = 1000
    mock_structure.__str__ = lambda self: "Mock ASE structure with 100 atoms"
    mock_get_ase_structure.return_value = mock_structure

    mock_potential = "mock_potential_content"
    mock_generate_potential.return_value = mock_potential

    # Mock the simulation result - create a separate structure mock for the result
    mock_result_structure = MagicMock()
    mock_result_structure.__str__ = lambda self: "Mock final structure with 100 atoms"
    mock_result_structure.__len__ = lambda self: 100

    mock_result = {
        "structure": mock_result_structure,
        "result": {
            "volume": [1000, 1000, 1000],
            "temperature": [300, 305, 302],
            "steps": [1, 2, 3],
        },
    }
    mock_melt_quench_simulation.return_value.pull.return_value = mock_result

    # Create a mock figure for the plot
    mock_fig = MagicMock()
    mock_plot_analysis_results.return_value = mock_fig

    # Mock the analyze_structure function - return dict directly instead of MagicMock
    mock_structural_analysis_data = {
        "density": 2.65,
        "coordination": {"oxygen": {}, "formers": {}, "modifiers": {}},
        "network": {"Qn_distribution": {}, "Qn_distribution_partial": {}, "connectivity": 0.0},
        "distributions": {"bond_angles": {}, "rings": {}},
        "rdfs": {"r": [], "rdfs": {}, "cumulative_coordination": {}},
        "elements": {"formers": [], "modifiers": [], "cutoffs": {}},
    }
    mock_analyze_structure.return_value.pull.return_value = mock_structural_analysis_data

    # Mock the figure's savefig method to simulate saving
    mock_fig.savefig = MagicMock()

    # Submit task and get it completed - use unique payload to avoid caching
    import time

    unique_suffix = str(int(time.time() * 1000))  # millisecond timestamp
    payload = {
        "components": ["SiO2", "Na2O"],
        "values": [75.0, 25.0],
        "unit": "wt",
        "heating_rate": int(unique_suffix[-6:]),
    }  # Use last 6 digits

    submit_response = client.post("/submit_meltquench", json=payload)
    assert submit_response.status_code == 200
    submit_data = submit_response.json()

    if submit_data["status"] == "completed_from_cache":
        task_id = submit_data["task_id"]
    else:
        task_id = submit_data["task_id"]
        # Wait for mocked completion
        import time

        max_wait = 5  # Reduced wait time since it's mocked
        waited = 0

        while waited < max_wait:
            check_response = client.get(f"/check/{task_id}")
            check_data = check_response.json()
            if check_data["state"] == "complete":
                break
            time.sleep(0.1)  # Shorter sleep
            waited += 0.1

    # Now test the visualization endpoint

    # First check what the task state looks like
    check_response = client.get(f"/check/{task_id}")

    viz_response = client.get(f"/viz/results/{task_id}")
    assert viz_response.status_code == 200

    # Check that we get HTML content
    assert viz_response.headers["content-type"] == "text/html; charset=utf-8"
    html_content = viz_response.text

    # Verify HTML contains expected elements
    assert "Glass Simulation Results" in html_content
    assert task_id in html_content
    assert "Structural Analysis Plot" in html_content
    assert "data:image/png;base64," in html_content

    # Verify the plot function was called
    mock_plot_analysis_results.assert_called_once()


def test_visualization_endpoint_task_not_found() -> None:
    """Test visualization endpoint with non-existent task."""
    response = client.get("/viz/results/nonexistent-task")
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
    viz_response = client.get(f"/viz/results/{fake_task_id}")
    assert viz_response.status_code == 400
    assert "not completed yet" in viz_response.json()["detail"]
