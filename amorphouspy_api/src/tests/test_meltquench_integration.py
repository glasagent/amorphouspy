"""Integration tests for the jobs API with a live server."""

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
def test_jobs_api_integration() -> None:
    """Full integration test for the jobs API using a running server.

    Requires: API server running with AMORPHOUSPY_INTEGRATION=1
    Example:
        AMORPHOUSPY_INTEGRATION=1 uvicorn amorphouspy_api.app:app --port 8002
        pytest -m integration
    """
    api_url = "http://127.0.0.1:8002"
    root_url = f"{api_url}/"
    logger.info("Checking API server status...")
    if not is_api_server_running(root_url):
        if os.environ.get("AMORPHOUSPY_INTEGRATION"):
            pytest.fail(
                "API server not running at http://127.0.0.1:8002/ "
                "but AMORPHOUSPY_INTEGRATION is set - the server should have started"
            )
        pytest.skip("API server not running at http://127.0.0.1:8002/")

    # Submit a job with fast rates for integration testing
    payload = {
        "composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15},
        "potential": "pmmcs",
        "simulation": {
            "melt_temperature": 5000,
            "quench_rate": 1e15,
            "n_atoms": 100,
            "timestep": 1.0,
            "equilibration_steps": 10_000,
        },
        "analyses": [{"type": "structure_characterization"}],
    }
    logger.info("Submitting job with fast rates: quench_rate=%s", payload["simulation"]["quench_rate"])
    r = requests.post(f"{api_url}/jobs?force=true", json=payload, timeout=200)
    r.raise_for_status()
    data = r.json()
    assert "id" in data
    job_id = data["id"]
    logger.info("Job ID: %s, status: %s", job_id, data["status"])

    # Poll for completion
    timeout = 300  # seconds
    poll_interval = 5  # seconds
    start = time.time()
    while True:
        r = requests.get(f"{api_url}/jobs/{job_id}", timeout=30)
        r.raise_for_status()
        status_data = r.json()
        status = status_data["status"]
        logger.info("Polling: status=%s, progress=%s", status, status_data.get("progress"))
        if status == "completed":
            break
        if status == "failed":
            logger.error("Job failed: %s", status_data.get("errors"))
            pytest.fail(f"Job failed: {status_data.get('errors')}")
        if status == "cancelled":
            pytest.fail("Job was cancelled")
        if time.time() - start > timeout:
            logger.error(
                "Timeout: Job did not complete within %s seconds. Last status: %s",
                timeout,
                status,
            )
            pytest.fail(f"Job did not complete within {timeout} seconds. Last status: {status}")
        time.sleep(poll_interval)

    # Fetch results
    r = requests.get(f"{api_url}/jobs/{job_id}/results", timeout=30)
    r.raise_for_status()
    result = r.json()

    assert result is not None
    assert result["job_id"] == job_id
    assert "composition" in result

    # Validate structural analysis results
    structure = result.get("analyses", {}).get("structure_characterization")
    assert structure is not None, "Expected structure analysis results"
    assert "density" in structure

    density = structure["density"]
    assert isinstance(density, (int, float))
    assert 1.5 < density < 5.0, f"Density {density} g/cm3 seems unreasonable"

    # Validate additional structural properties
    assert "network" in structure
    assert "elements" in structure
    assert "connectivity" in structure["network"]
    assert "formers" in structure["elements"]
    assert "modifiers" in structure["elements"]

    logger.info("Composition: %s", result["composition"])
    logger.info("Density: %.2f g/cm3", density)

    # Verify structure export works
    r = requests.get(f"{api_url}/jobs/{job_id}/structure?format=xyz", timeout=30)
    r.raise_for_status()
    assert len(r.text) > 0, "Structure export returned empty content"
    logger.info("Structure export: %d bytes", len(r.text))

    # Fetch per-analysis results
    r = requests.get(f"{api_url}/jobs/{job_id}/results/structure_characterization", timeout=30)
    r.raise_for_status()
    per_analysis = r.json()
    assert per_analysis["job_id"] == job_id
    assert "structure_characterization" in per_analysis
    logger.info("Per-analysis result keys: %s", list(per_analysis["structure_characterization"].keys()))

    # Fetch visualization page
    r = requests.get(f"{api_url}/jobs/{job_id}/visualize", timeout=30)
    r.raise_for_status()
    assert "text/html" in r.headers.get("content-type", "")
    assert len(r.text) > 0, "Visualization page returned empty content"
    logger.info("Visualization page: %d bytes", len(r.text))

    # Verify glasses endpoint shows this composition
    r = requests.get(f"{api_url}/glasses", timeout=30)
    r.raise_for_status()
    glasses = r.json()
    assert "glasses" in glasses
    compositions = [g["composition"] for g in glasses["glasses"]]
    logger.info("Known compositions: %s", compositions)
    assert len(compositions) > 0, "Expected at least one composition in glasses list"
