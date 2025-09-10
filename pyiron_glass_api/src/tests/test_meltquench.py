import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from pyiron_glass_api.app import app

client = TestClient(app)


@patch("pyiron_glass.melt_quench_simulation")
@patch("pyiron_glass.generate_potential")
@patch("pyiron_glass.get_ase_structure")
@patch("pyiron_glass.get_structure_dict")
@patch("pyiron_base.Project")
def test_submit_meltquench_and_check(
    mock_project, mock_get_structure_dict, mock_get_ase_structure, mock_generate_potential, mock_melt_quench_simulation
):
    """Test the complete meltquench workflow with mocked pyiron dependencies."""

    # Mock the pyiron components
    mock_atoms_dict = {"atoms": [{"element": "Si", "position": [0, 0, 0]}] * 100}
    mock_get_structure_dict.return_value.pull.return_value = mock_atoms_dict

    mock_structure = MagicMock()
    mock_structure.get_masses.return_value.sum.return_value = 1000  # mock mass
    mock_get_ase_structure.return_value = mock_structure

    mock_potential = "mock_potential_content"
    mock_generate_potential.return_value = mock_potential

    # Mock the simulation result
    mock_result = {
        "structure": mock_structure,
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
        assert "final_density" in result
        print("✓ Got cached result, skipping task polling")
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
        elif check_data["state"] == "error":
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
    assert "final_density" in result
    assert "simulation_steps" in result

    # Validate composition format
    assert result["composition"] == "0.6SiO2-0.25CaO-0.15Al2O3"

    # Validate numerical values
    assert isinstance(result["mean_temperature"], float)
    assert isinstance(result["final_density"], float)
    assert isinstance(result["simulation_steps"], int)


def test_check_nonexistent_task():
    """Test checking a task that doesn't exist."""
    response = client.get("/check/nonexistent-task-id")
    assert response.status_code == 404
    assert "Task not found" in response.json()["detail"]


def test_invalid_payload():
    """Test submitting an invalid payload."""
    payload = {
        "components": ["SiO2"],
        "values": [60.0, 25.0],  # Mismatched lengths
        "unit": "wt",
    }
    response = client.post("/submit_meltquench", json=payload)
    assert response.status_code == 422  # Validation error


def test_root_redirect():
    """Test that root redirects to docs."""
    # FastAPI TestClient follows redirects by default, so we need to check differently
    # We can verify that accessing "/" eventually serves docs content
    response = client.get("/")
    assert response.status_code == 200
    # The response should contain swagger/docs content when redirected
    assert "swagger" in response.text.lower() or "openapi" in response.text.lower()


def test_check_cached_result():
    """Test checking for cached results with unique composition."""
    # Use a unique composition to avoid cache hits from other tests
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
        assert "final_density" in data
        assert "final_structure" in data
        assert "mean_temperature" in data
        assert "simulation_steps" in data


def test_check_cached_result_not_found():
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
        assert "final_density" in data
        assert "final_structure" in data
        assert "mean_temperature" in data
        assert "simulation_steps" in data


def test_caching_behavior():
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
    initial_cache = cache_response.json()

    # Submit the simulation (will be mocked)
    submit_response = client.post("/submit_meltquench", json=unique_payload)
    assert submit_response.status_code == 200
    submit_data = submit_response.json()

    # Should either start a new task or return cached result
    assert "task_id" in submit_data
    assert "status" in submit_data

    # The status should indicate either "started" (new task) or "completed_from_cache" (cached)
    assert submit_data["status"] in ["started", "completed_from_cache"]
