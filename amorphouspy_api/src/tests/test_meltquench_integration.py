"""Integration tests for meltquench API with live server."""

import logging
import os
import time

import pytest
import requests

logger = logging.getLogger(__name__)


def is_api_server_running(url: str) -> bool:
    """Check if the API server is running at the given URL.

    Args:
        url: The URL to check

    Returns:
        True if server is running, False otherwise

    """
    try:
        r = requests.get(url, timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False


@pytest.mark.integration
def test_meltquench_api_integration() -> None:
    """Full integration test for the meltquench API using a running server.
    Requires: API server running in main thread with AMORPHOUSPY_INTEGRATION=1
    Example:
        AMORPHOUSPY_INTEGRATION=1 uvicorn amorphouspy_api.src.amorphouspy_api.app:app --port 8002
        pytest -m integration.
    """
    API_URL = "http://127.0.0.1:8002"
    root_url = f"{API_URL}/"
    logger.info("Checking API server status...")
    if not is_api_server_running(root_url):
        if os.environ.get("AMORPHOUSPY_INTEGRATION"):
            pytest.fail(
                "API server not running at http://127.0.0.1:8002/ "
                "but AMORPHOUSPY_INTEGRATION is set — the server should have started"
            )
        pytest.skip("API server not running at http://127.0.0.1:8002/")

    # Use faster rates for integration testing
    payload = {
        "components": ["SiO2", "CaO", "Al2O3"],
        "values": [60.0, 25.0, 15.0],
        "unit": "wt",
        "heating_rate": int(1e15),  # 10x faster than default
        "cooling_rate": int(1e15),  # 10x faster than default
        "n_print": 1000,
        "n_atoms": 100,
    }
    logger.info("Submitting meltquench task with faster rates: %s", payload["heating_rate"])
    r = requests.post(f"{API_URL}/submit/meltquench", json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    assert "task_id" in data
    task_id = data["task_id"]
    logger.info("Task ID: %s", task_id)

    # Handle cached result case
    if task_id == "cached" and "result" in data:
        logger.info("Found cached result, skipping polling")
        result = data["result"]
    else:
        # Poll for completion
        timeout = 300  # seconds
        poll_interval = 5  # seconds
        start = time.time()
        while True:
            r = requests.get(f"{API_URL}/check/{task_id}", timeout=30)
            r.raise_for_status()
            check_data = r.json()
            status = check_data["status"]
            logger.info("Polling: status=%s", status)
            if status == "completed":
                logger.info("Result: %s", check_data["result"])
                result = check_data["result"]
                break
            if status == "error":
                logger.error("Meltquench task errored: %s", check_data.get("error"))
                pytest.fail(f"Meltquench task errored: {check_data.get('error')}")
            if time.time() - start > timeout:
                logger.error(
                    "Timeout: Meltquench task did not complete within %s seconds. Last status: %s",
                    timeout,
                    status,
                )
                pytest.fail(f"Meltquench task did not complete within {timeout} seconds. Last status: {status}")
            time.sleep(poll_interval)

    assert result is not None

    # Validate result structure
    assert "composition" in result
    assert "final_structure" in result
    assert "mean_temperature" in result
    assert "simulation_steps" in result
    assert "structural_analysis" in result

    # Validate composition string format (should be fractions with components)
    composition = result["composition"]
    assert isinstance(composition, str)
    assert "SiO2" in composition
    assert "CaO" in composition
    assert "Al2O3" in composition
    assert composition.startswith("0.6SiO2")  # 60% -> 0.6

    # Validate numerical results are reasonable
    temp = result["mean_temperature"]
    steps = result["simulation_steps"]
    structural_analysis = result["structural_analysis"]

    # Check that structural analysis contains density
    assert "density" in structural_analysis
    density = structural_analysis["density"]

    # Validate additional structural properties (now nested)
    assert "network" in structural_analysis
    assert "elements" in structural_analysis
    assert "connectivity" in structural_analysis["network"]
    assert "formers" in structural_analysis["elements"]
    assert "modifiers" in structural_analysis["elements"]

    assert isinstance(temp, (int, float))
    assert isinstance(density, (int, float))
    assert isinstance(steps, int)

    # Temperature should be around room temperature (final equilibration)
    assert 250 < temp < 400, f"Temperature {temp}K seems unreasonable"

    # Density should be reasonable for glass (~2-4 g/cm³)
    assert 1.5 < density < 5.0, f"Density {density} g/cm³ seems unreasonable"

    # Should have completed some simulation steps
    assert steps > 0, f"No simulation steps completed: {steps}"

    logger.info("✓ Composition: %s", composition)
    logger.info("✓ Temperature: %.1f K", temp)
    logger.info("✓ Density: %.2f g/cm³", density)
    logger.info("✓ Steps: %s", steps)
    logger.info(
        "✓ Structural analysis: %s",
        {k: v for k, v in structural_analysis.items() if k != "error"},
    )
