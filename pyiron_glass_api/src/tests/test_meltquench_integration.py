import os
import time
import requests
import pytest

def is_api_server_running(url):
    try:
        r = requests.get(url)
        return r.status_code == 200
    except Exception:
        return False

@pytest.mark.integration
def test_meltquench_api_integration():
    """
    Full integration test for the meltquench API using a running server.
    Requires: API server running in main thread with PYIRON_GLASS_INTEGRATION=1
    Example:
        PYIRON_GLASS_INTEGRATION=1 uvicorn pyiron_glass_api.src.pyiron_glass_api.app:app --host 127.0.0.1 --port 8002
        pytest -m integration src/tests/test_run_meltquench_integration.py
    """
    API_URL = "http://127.0.0.1:8002"
    root_url = f"{API_URL}/"
    print("Checking API server status...")
    if not is_api_server_running(root_url):
        pytest.skip("API server not running at http://127.0.0.1:8002/")

    payload = {
        "components": ["SiO2", "CaO", "Al2O3"],
        "values": [60.0, 25.0, 15.0],
        "unit": "wt",
        "n_molecules": 100,
        "density": 2.5,
        "temperature_high": 4000,
        "temperature_low": 300
    }
    print("Submitting meltquench task...")
    r = requests.post(f"{API_URL}/submit_meltquench", json=payload)
    r.raise_for_status()
    data = r.json()
    assert "task_id" in data
    task_id = data["task_id"]
    print(f"Task ID: {task_id}")

    timeout = 300  # seconds
    poll_interval = 5  # seconds
    start = time.time()
    while True:
        r = requests.get(f"{API_URL}/check/{task_id}")
        r.raise_for_status()
        check_data = r.json()
        state = check_data["state"]
        print(f"Polling: state={state}")
        if state == "complete":
            print(f"Result: {check_data['result']}")
            assert check_data["result"] is not None
            break
        if state == "error":
            print(f"Meltquench task errored: {check_data.get('error')}")
            pytest.fail(f"Meltquench task errored: {check_data.get('error')}")
        if time.time() - start > timeout:
            print(f"Timeout: Meltquench task did not complete within {timeout} seconds. Last state: {state}")
            pytest.fail(f"Meltquench task did not complete within {timeout} seconds. Last state: {state}")
        time.sleep(poll_interval)
