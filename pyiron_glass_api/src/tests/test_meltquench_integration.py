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
        PYIRON_GLASS_INTEGRATION=1 uvicorn pyiron_glass_api.src.pyiron_glass_api.app:app --port 8002
        pytest -m integration
    """
    API_URL = "http://127.0.0.1:8002"
    root_url = f"{API_URL}/"
    print("Checking API server status...")
    if not is_api_server_running(root_url):
        pytest.skip("API server not running at http://127.0.0.1:8002/")

    # Use faster rates for integration testing
    payload = {
        "components": ["SiO2", "CaO", "Al2O3"],
        "values": [60.0, 25.0, 15.0],
        "unit": "wt",
        "heating_rate": int(1e15),  # 10x faster than default
        "cooling_rate": int(1e15),  # 10x faster than default
        "n_print": 1000,
    }
    print(f"Submitting meltquench task with faster rates: {payload['heating_rate']}")
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
            break
        if state == "error":
            print(f"Meltquench task errored: {check_data.get('error')}")
            pytest.fail(f"Meltquench task errored: {check_data.get('error')}")
        if time.time() - start > timeout:
            print(f"Timeout: Meltquench task did not complete within {timeout} seconds. Last state: {state}")
            pytest.fail(f"Meltquench task did not complete within {timeout} seconds. Last state: {state}")
        time.sleep(poll_interval)

    result = check_data["result"]
    assert result is not None

    # Validate result structure
    assert "composition" in result
    assert "final_structure" in result
    assert "mean_temperature" in result
    assert "final_density" in result
    assert "simulation_steps" in result

    # Validate composition string format (should be fractions with components)
    composition = result["composition"]
    assert isinstance(composition, str)
    assert "SiO2" in composition
    assert "CaO" in composition
    assert "Al2O3" in composition
    assert composition.startswith("0.6SiO2")  # 60% -> 0.6

    # Validate numerical results are reasonable
    temp = result["mean_temperature"]
    density = result["final_density"]
    steps = result["simulation_steps"]

    assert isinstance(temp, (int, float))
    assert isinstance(density, (int, float))
    assert isinstance(steps, int)

    # Temperature should be around room temperature (final equilibration)
    assert 250 < temp < 400, f"Temperature {temp}K seems unreasonable"

    # Density should be reasonable for glass (~2-4 g/cm³)
    assert 1.5 < density < 5.0, f"Density {density} g/cm³ seems unreasonable"

    # Should have completed some simulation steps
    assert steps > 0, f"No simulation steps completed: {steps}"

    print(f"✓ Composition: {composition}")
    print(f"✓ Temperature: {temp:.1f} K")
    print(f"✓ Density: {density:.2f} g/cm³")
    print(f"✓ Steps: {steps}")
