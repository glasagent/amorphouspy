"""Unit tests for viscosity API functionality."""

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pyiron_glass_api.app import app, get_viscosity_hash
from pyiron_glass_api.models import MeltquenchRequest, ViscosityRequest

client = TestClient(app)


@pytest.fixture(autouse=True)
def _patch_viscosity_worker(monkeypatch) -> None:
    """Replace viscosity background worker with a no-op that writes a completed result.

    This keeps tests fully in-process and avoids spawning real child processes.
    """
    from pyiron_glass_api import app as app_module
    from pyiron_glass_api.models import ViscosityResult

    async def fake_worker(task_id: str, request: ViscosityRequest) -> None:
        from pyiron_glass_api.database import get_task_store

        ts = get_task_store()
        result = ViscosityResult(
            composition="0.6SiO2-0.25CaO-0.15Al2O3",
            temperatures=[1500.0],
            viscosities=[1.23e3],
            max_lag=[[10.0, 12.0, 11.0]],
            simulation_steps=[1_000_000],
        )
        ts.set(
            task_id,
            {
                "state": "complete",
                "status": "Completed",
                "result": result.model_dump(),
            },
        )

    monkeypatch.setattr(app_module, "_viscosity_worker", fake_worker)


def wait_for_task_completion(task_id: str, max_wait: float = 10.0) -> dict[str, Any]:
    """Wait for a viscosity task to complete and return the final check data."""
    waited = 0.0
    while waited < max_wait:
        check_response = client.get(f"/check/{task_id}")
        assert check_response.status_code == 200
        check_data = check_response.json()

        if check_data["state"] == "complete":
            return check_data
        if check_data["state"] == "error":
            pytest.fail(f"Viscosity simulation failed: {check_data.get('error')}")

        time.sleep(0.5)
        waited += 0.5

    pytest.fail(f"Viscosity task {task_id} did not complete within {max_wait} seconds")


def validate_viscosity_result(result: dict[str, Any]) -> None:
    """Validate the structure of a viscosity result."""
    assert "kind" in result
    assert result["kind"] == "viscosity"
    assert "temperatures" in result
    assert "viscosities" in result
    assert "max_lag" in result
    assert "simulation_steps" in result

    assert isinstance(result["temperatures"], list)
    assert isinstance(result["viscosities"], list)
    assert isinstance(result["simulation_steps"], list)
    assert isinstance(result["max_lag"], list)


def test_submit_viscosity_with_meltquench_and_check() -> None:
    """Test the complete viscosity workflow using a nested meltquench request."""
    mq_payload = {
        "components": ["SiO2", "CaO", "Al2O3"],
        "values": [60.0, 25.0, 15.0],
        "unit": "wt",
    }
    payload = {
        "meltquench_request": mq_payload,
        "temperatures": [1500.0],
        "timestep": 1.0,
        "n_timesteps": 1_000_000,
        "n_print": 1000,
        "potential_type": "pmmcs",
    }

    response = client.post("/submit/viscosity", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert "status" in data

    if data["status"] == "completed_from_cache":
        assert "result" in data
        validate_viscosity_result(data["result"])
        return

    assert data["status"] == "started"
    check_data = wait_for_task_completion(data["task_id"])

    assert check_data["task_id"] == data["task_id"]
    assert check_data["state"] == "complete"
    assert check_data["result"] is not None

    validate_viscosity_result(check_data["result"])


def test_viscosity_hash_consistency() -> None:
    """Test that identical viscosity requests produce identical hashes."""
    mq_req = MeltquenchRequest(components=["SiO2"], values=[100.0], unit="wt")

    request1 = ViscosityRequest(
        meltquench_request=mq_req,
        temperatures=[1500.0],
        timestep=1.0,
        n_timesteps=1_000_000,
        n_print=1000,
        potential_type="pmmcs",
    )

    request2 = ViscosityRequest(
        meltquench_request=mq_req,
        temperatures=[1500.0],
        timestep=1.0,
        n_timesteps=1_000_000,
        n_print=1000,
        potential_type="pmmcs",
    )

    hash1 = get_viscosity_hash(request1)
    hash2 = get_viscosity_hash(request2)

    assert hash1 == hash2
    assert len(hash1) == 16


def test_viscosity_invalid_payload_missing_source() -> None:
    """Test that submitting a viscosity request without source information fails validation."""
    payload = {
        "temperature_sim": 1500.0,
        "timestep": 1.0,
        "n_timesteps": 1_000_000,
        "n_print": 1000,
        "potential_type": "pmmcs",
    }
    response = client.post("/submit/viscosity", json=payload)
    assert response.status_code == 422


def test_viscosity_visualization_endpoint() -> None:
    """Test the viscosity visualization endpoint."""
    # First, submit a viscosity task (will be completed immediately by the fake worker)
    mq_payload = {
        "components": ["SiO2"],
        "values": [100.0],
        "unit": "wt",
    }
    payload = {
        "meltquench_request": mq_payload,
        "temperatures": [1500.0],
        "timestep": 1.0,
        "n_timesteps": 1_000_000,
        "n_print": 1000,
        "potential_type": "pmmcs",
    }

    submit_response = client.post("/submit/viscosity", json=payload)
    assert submit_response.status_code == 200
    submit_data = submit_response.json()
    task_id = submit_data["task_id"]

    # Call the viscosity visualization endpoint
    viz_response = client.get(f"/visualize/viscosity/{task_id}")
    assert viz_response.status_code == 200
    assert "Viscosity Results" in viz_response.text
    assert "viscosity-plot" in viz_response.text
