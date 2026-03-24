"""Tests for the new /jobs and /glasses API endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from amorphouspy_api.app import app
from amorphouspy_api.database import Job, get_job_store

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_structural_analysis() -> dict[str, Any]:
    return {
        "density": 2.5,
        "coordination": {"oxygen": {}, "formers": {}, "modifiers": {}},
        "network": {
            "Qn_distribution": {},
            "Qn_distribution_partial": {},
            "connectivity": 0.0,
        },
        "distributions": {"bond_angles": {}, "rings": {}},
        "rdfs": {"r": [], "rdfs": {}, "cumulative_coordination": {}},
        "elements": {"formers": [], "modifiers": [], "cutoffs": {}},
    }


def _mock_result() -> dict[str, Any]:
    return {
        "composition": "SiO2 60 - CaO 25 - Al2O3 15",
        "final_structure": {
            "numbers": [14] * 50 + [8] * 100,
            "positions": [[0.0, 0.0, 0.0]] * 150,
            "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            "pbc": [True, True, True],
        },
        "mean_temperature": 302.3,
        "simulation_steps": 3,
        "structural_analysis": _mock_structural_analysis(),
    }


def _insert_completed_job(
    job_id: str = "j-test-1",
    *,
    composition: str = "Al2O3 15 - CaO 25 - SiO2 60",
    potential: str = "pmmcs",
    request_hash: str = "testhash1234",
) -> None:
    store = get_job_store()
    store.create_job(
        Job(
            job_id=job_id,
            request_hash=request_hash,
            composition=composition,
            potential=potential,
            status="completed",
            request_data={
                "composition": composition,
                "potential": potential,
                "simulation": {},
                "analyses": [{"type": "structure"}],
            },
            progress={
                "structure_generation": "completed",
                "melt_quench": "completed",
                "structure_analysis": "completed",
            },
            result_data=_mock_result(),
            completed_at=datetime.now(UTC),
        )
    )


def _insert_running_job(job_id: str = "j-running-1") -> None:
    store = get_job_store()
    store.create_job(
        Job(
            job_id=job_id,
            request_hash="runhash123",
            composition="SiO2 100",
            potential="pmmcs",
            status="running",
            request_data={"composition": "SiO2 100", "potential": "pmmcs"},
            progress={
                "structure_generation": "completed",
                "melt_quench": "running",
                "structure_analysis": "pending",
            },
        )
    )


# ---------------------------------------------------------------------------
# POST /jobs (submit)
# ---------------------------------------------------------------------------


def test_submit_job_new() -> None:
    """Test submitting a new job via the executor."""
    mock_future = MagicMock()
    mock_future.done.return_value = True
    mock_future.exception.return_value = None
    mock_future.result.return_value = _mock_result()

    with (
        patch("amorphouspy_api.routers.jobs.get_executor") as mock_exe,
        patch(
            "amorphouspy_api.routers.jobs.run_meltquench_workflow",
            return_value=mock_future,
        ),
    ):
        mock_exe.return_value.shutdown = MagicMock()

        resp = client.post(
            "/jobs",
            json={
                "composition": "SiO2 60 - CaO 25 - Al2O3 15",
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert data["status"] in ("pending", "completed")
    assert data["composition"] == "Al2O3 15 - CaO 25 - SiO2 60"  # normalised


def test_submit_job_returns_cached() -> None:
    """Test that submitting a duplicate request returns the cached job."""
    from amorphouspy_api.composition import normalize_composition
    from amorphouspy_api.models import JobSubmission
    from amorphouspy_api.routers.jobs import _job_hash

    # Build the same submission the client will send
    sub = JobSubmission(composition="SiO2 60 - CaO 25 - Al2O3 15")
    norm = normalize_composition(sub.composition)
    real_hash = _job_hash(sub, norm)

    _insert_completed_job("j-cached", composition=norm, request_hash=real_hash)

    resp = client.post(
        "/jobs",
        json={
            "composition": "SiO2 60 - CaO 25 - Al2O3 15",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "j-cached"
    assert data["status"] == "completed"


# ---------------------------------------------------------------------------
# GET /jobs/{id}
# ---------------------------------------------------------------------------


def test_get_job_status_completed() -> None:
    _insert_completed_job("j-status-1")

    resp = client.get("/jobs/j-status-1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["progress"]["melt_quench"] == "completed"


def test_get_job_status_not_found() -> None:
    resp = client.get("/jobs/nonexistent")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /jobs/{id}:cancel
# ---------------------------------------------------------------------------


def test_cancel_running_job() -> None:
    _insert_running_job("j-cancel-1")

    resp = client.post("/jobs/j-cancel-1:cancel")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "cancelled"
    assert data["progress"]["melt_quench"] == "cancelled"
    assert data["progress"]["structure_analysis"] == "cancelled"
    # Already-completed steps stay completed
    assert data["progress"]["structure_generation"] == "completed"


def test_cancel_completed_job_fails() -> None:
    _insert_completed_job("j-cancel-done")
    resp = client.post("/jobs/j-cancel-done:cancel")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /jobs/{id}/results
# ---------------------------------------------------------------------------


def test_get_results_completed() -> None:
    _insert_completed_job("j-results-1")

    resp = client.get("/jobs/j-results-1/results")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == "j-results-1"
    assert data["structure"] is not None


def test_get_results_no_data() -> None:
    store = get_job_store()
    store.create_job(
        Job(
            job_id="j-no-result",
            request_hash="noresult",
            composition="SiO2 100",
            potential="pmmcs",
            status="running",
        )
    )
    resp = client.get("/jobs/j-no-result/results")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /jobs/{id}/results/{analysis}
# ---------------------------------------------------------------------------


def test_get_single_result() -> None:
    _insert_completed_job("j-single-1")

    resp = client.get("/jobs/j-single-1/results/structure")
    assert resp.status_code == 200
    data = resp.json()
    assert "structure" in data


def test_get_single_result_missing() -> None:
    _insert_completed_job("j-single-2")
    resp = client.get("/jobs/j-single-2/results/viscosity")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /jobs/{id}/structure
# ---------------------------------------------------------------------------


def test_get_structure_xyz() -> None:
    _insert_completed_job("j-struct-1")

    resp = client.get("/jobs/j-struct-1/structure?format=xyz")
    assert resp.status_code == 200
    assert "150" in resp.text  # atom count


def test_get_structure_bad_format() -> None:
    _insert_completed_job("j-struct-2")
    resp = client.get("/jobs/j-struct-2/structure?format=pdb")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /jobs:search
# ---------------------------------------------------------------------------


def test_search_jobs_exact_match() -> None:
    _insert_completed_job("j-search-1")

    resp = client.post(
        "/jobs:search",
        json={
            "composition": "SiO2 60 - CaO 25 - Al2O3 15",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["matches"]) >= 1
    assert data["matches"][0]["job_id"] == "j-search-1"


def test_search_jobs_no_match() -> None:
    resp = client.post(
        "/jobs:search",
        json={
            "composition": "B2O3 100",
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["matches"]) == 0


# ---------------------------------------------------------------------------
# GET /glasses
# ---------------------------------------------------------------------------


def test_list_glasses() -> None:
    _insert_completed_job("j-glass-1", composition="Al2O3 15 - CaO 25 - SiO2 60")
    _insert_completed_job("j-glass-2", composition="SiO2 100", request_hash="other")

    resp = client.get("/glasses")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["glasses"]) >= 2


def test_get_glass_properties() -> None:
    _insert_completed_job("j-glass-prop")

    resp = client.get("/glasses", params={"composition": "SiO2 60 - CaO 25 - Al2O3 15"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["composition"] == "Al2O3 15 - CaO 25 - SiO2 60"
    assert "structure" in data["properties"]


def test_get_glass_properties_not_found() -> None:
    resp = client.get("/glasses", params={"composition": "B2O3 100"})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------


def test_root_redirect() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "swagger" in resp.text.lower() or "openapi" in resp.text.lower()
