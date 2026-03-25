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
    """Mock melt-quench result (no analysis — that's a separate step now)."""
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
    }


def _insert_completed_job(
    job_id: str = "j-test-1",
    *,
    composition: str = "Al2O3 15 - CaO 25 - SiO2 60",
    potential: str = "pmmcs",
    request_hash: str = "testhash1234",
) -> None:
    store = get_job_store()
    result = _mock_result()
    result["structure"] = _mock_structural_analysis()
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
                "structure": "completed",
            },
            result_data=result,
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
            request_data={
                "composition": "SiO2 100",
                "potential": "pmmcs",
                "analyses": [{"type": "structure"}],
            },
            progress={
                "structure_generation": "completed",
                "melt_quench": "running",
                "structure": "pending",
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
        patch.dict(
            "amorphouspy_api.routers.jobs._ANALYSIS_RUNNERS",
            {"structure": lambda _s, _c, _r: _mock_structural_analysis()},
        ),
    ):
        mock_exe.return_value.shutdown = MagicMock()

        resp = client.post(
            "/jobs",
            json={
                "composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15},
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert data["status"] in ("pending", "completed")
    assert data["composition"] == {"Al2O3": 15.0, "CaO": 25.0, "SiO2": 60.0}


def test_submit_job_returns_cached() -> None:
    """Test that submitting a duplicate request returns the cached job."""
    from amorphouspy_api.models import JobSubmission
    from amorphouspy_api.routers.jobs import _job_hash

    # Build the same submission the client will send
    sub = JobSubmission(composition={"SiO2": 60, "CaO": 25, "Al2O3": 15})
    norm = sub.composition.canonical
    real_hash = _job_hash(sub, norm)

    _insert_completed_job("j-cached", composition=norm, request_hash=real_hash)

    resp = client.post(
        "/jobs",
        json={
            "composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15},
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
    assert data["progress"]["analyses"]["structure"] == "completed"


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
    assert data["progress"]["analyses"]["structure"] == "cancelled"
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
    assert data["analyses"]["structure"] is not None


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
            "composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15},
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
            "composition": {"B2O3": 100},
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

    resp = client.post("/glasses:lookup", json={"composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15}})
    assert resp.status_code == 200
    data = resp.json()
    assert data["composition"] == {"Al2O3": 15.0, "CaO": 25.0, "SiO2": 60.0}
    assert "structure" in data["properties"]


def test_get_glass_properties_not_found() -> None:
    resp = client.post("/glasses:lookup", json={"composition": {"B2O3": 100}})
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------


def test_root_redirect() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "swagger" in resp.text.lower() or "openapi" in resp.text.lower()


# ---------------------------------------------------------------------------
# Viscosity integration
# ---------------------------------------------------------------------------


def _mock_viscosity_result() -> dict[str, Any]:
    return {
        "temperatures": [2500.0, 2000.0, 1500.0],
        "viscosities": [1.0e-2, 5.0e-1, 1.2e3],
        "max_lag": [100.0, 120.0, 110.0],
        "simulation_steps": [10_000_000, 10_000_000, 10_000_000],
        "lag_times_ps": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        "sacf_data": [[1.0, 0.5], [1.0, 0.4], [1.0, 0.3]],
        "viscosity_running": [[0.0, 0.01], [0.0, 0.5], [0.0, 1200.0]],
    }


def _insert_completed_viscosity_job(job_id: str = "j-visc-1") -> None:
    store = get_job_store()
    result = _mock_result()
    result["structure"] = _mock_structural_analysis()
    result["viscosity"] = _mock_viscosity_result()
    store.create_job(
        Job(
            job_id=job_id,
            request_hash="vischash1234",
            composition="Al2O3 15 - CaO 25 - SiO2 60",
            potential="pmmcs",
            status="completed",
            request_data={
                "composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15},
                "potential": "pmmcs",
                "simulation": {},
                "analyses": [
                    {"type": "structure"},
                    {"type": "viscosity", "temperatures": [1500, 2000, 2500]},
                ],
            },
            progress={
                "structure_generation": "completed",
                "melt_quench": "completed",
                "structure": "completed",
                "viscosity": "completed",
            },
            result_data=result,
            completed_at=datetime.now(UTC),
        )
    )


def test_submit_job_with_viscosity() -> None:
    """Test submitting a job that includes viscosity analysis."""
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
        patch.dict(
            "amorphouspy_api.routers.jobs._ANALYSIS_RUNNERS",
            {
                "structure": lambda _s, _c, _r: _mock_structural_analysis(),
                "viscosity": lambda _s, _c, _r: _mock_viscosity_result(),
            },
        ),
    ):
        mock_exe.return_value.shutdown = MagicMock()

        resp = client.post(
            "/jobs",
            json={
                "composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15},
                "analyses": [
                    {"type": "structure"},
                    {"type": "viscosity", "temperatures": [1500, 2000, 2500]},
                ],
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert data["status"] in ("pending", "completed")


def test_get_results_with_viscosity() -> None:
    """Test that the results endpoint returns viscosity data."""
    _insert_completed_viscosity_job("j-visc-results")

    resp = client.get("/jobs/j-visc-results/results")
    assert resp.status_code == 200
    data = resp.json()
    assert data["analyses"]["structure"] is not None
    assert data["analyses"]["viscosity"] is not None
    assert data["analyses"]["viscosity"]["temperatures"] == [2500.0, 2000.0, 1500.0]
    assert len(data["analyses"]["viscosity"]["viscosities"]) == 3


def test_get_single_viscosity_result() -> None:
    """Test retrieving only the viscosity analysis result."""
    _insert_completed_viscosity_job("j-visc-single")

    resp = client.get("/jobs/j-visc-single/results/viscosity")
    assert resp.status_code == 200
    data = resp.json()
    assert "viscosity" in data
    assert data["viscosity"]["temperatures"] == [2500.0, 2000.0, 1500.0]


def test_viscosity_progress_tracking() -> None:
    """Test that viscosity step appears in progress when requested."""
    _insert_completed_viscosity_job("j-visc-progress")

    resp = client.get("/jobs/j-visc-progress")
    assert resp.status_code == 200
    data = resp.json()
    assert data["progress"]["analyses"]["viscosity"] == "completed"


def test_job_hash_differs_with_viscosity() -> None:
    """Test that the job hash changes when viscosity analysis is added."""
    from amorphouspy_api.models import JobSubmission
    from amorphouspy_api.routers.jobs import _job_hash

    sub_no_visc = JobSubmission(
        composition={"SiO2": 60, "CaO": 25, "Al2O3": 15},
    )
    sub_with_visc = JobSubmission(
        composition={"SiO2": 60, "CaO": 25, "Al2O3": 15},
        analyses=[
            {"type": "structure"},
            {"type": "viscosity", "temperatures": [1500, 2000]},
        ],
    )

    hash1 = _job_hash(sub_no_visc, sub_no_visc.composition.canonical)
    hash2 = _job_hash(sub_with_visc, sub_with_visc.composition.canonical)

    assert hash1 != hash2


def test_job_without_viscosity_has_no_viscosity_progress() -> None:
    """Test that jobs without viscosity analysis don't have viscosity in progress."""
    _insert_completed_job("j-no-visc")

    resp = client.get("/jobs/j-no-visc")
    assert resp.status_code == 200
    data = resp.json()
    assert "viscosity" not in data["progress"]["analyses"]
