import pytest
from fastapi.testclient import TestClient
from pyiron_glass_api.app import app

client = TestClient(app)

def test_submit_meltquench_and_check():
    # Example meltquench request payload
    payload = {
        "components": ["SiO2", "CaO", "Al2O3"],
        "values": [60.0, 25.0, 15.0],
        "unit": "wt",
        "n_molecules": 100,
        "density": 2.5,
        "temperature_high": 4000,
        "temperature_low": 300
    }
    # Submit meltquench task
    response = client.post("/submit_meltquench", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    task_id = data["task_id"]

    # Poll for completion, fail if error or timeout
    import time
    timeout = 30  # seconds
    poll_interval = 1  # seconds
    start = time.time()
    while True:
        check_response = client.get(f"/check/{task_id}")
        assert check_response.status_code == 200
        check_data = check_response.json()
        assert check_data["task_id"] == task_id
        state = check_data["state"]
        if state == "complete":
            assert check_data["result"] is not None
            break
        if state == "error":
            raise AssertionError(f"Meltquench task errored: {check_data.get('error')}")
        if time.time() - start > timeout:
            raise AssertionError(f"Meltquench task did not complete within {timeout} seconds. Last state: {state}")
        time.sleep(poll_interval)
