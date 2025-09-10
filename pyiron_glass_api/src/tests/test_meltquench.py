import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from pyiron_glass_api.app import app

client = TestClient(app)

@patch('pyiron_glass.melt_quench_simulation')
@patch('pyiron_glass.generate_potential')
@patch('pyiron_glass.get_ase_structure')
@patch('pyiron_glass.get_structure_dict')
@patch('pyiron_base.Project')
def test_submit_meltquench_and_check(mock_project, mock_get_structure_dict, 
                                    mock_get_ase_structure, mock_generate_potential, 
                                    mock_melt_quench_simulation):
    """Test the complete meltquench workflow with mocked pyiron dependencies."""
    
    # Mock the pyiron components
    mock_atoms_dict = {'atoms': [{'element': 'Si', 'position': [0, 0, 0]}] * 100}
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
            "steps": [1, 2, 3]
        }
    }
    mock_melt_quench_simulation.return_value.pull.return_value = mock_result
    
    # Updated payload to match simplified API
    payload = {
        "components": ["SiO2", "CaO", "Al2O3"],
        "values": [60.0, 25.0, 15.0],
        "unit": "wt"
    }
    
    # Submit meltquench task
    response = client.post("/submit_meltquench", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert "status" in data
    assert data["status"] == "started"
    task_id = data["task_id"]

    # Since we're using asyncio.create_task, we need to give it a moment to complete
    import time
    time.sleep(0.1)
    
    # Check the task result
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
        "unit": "wt"
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
