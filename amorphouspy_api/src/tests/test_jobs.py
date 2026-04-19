"""Tests for the new /jobs and /glasses API endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

from amorphouspy_api.app import app
from amorphouspy_api.database import Job, get_job_store
from fastapi.testclient import TestClient

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


def _mock_result(*, include_structure: bool = True) -> dict[str, Any]:
    """Mock full pipeline result in nested format."""
    result: dict[str, Any] = {
        "structure_generation": {
            "atoms_dict": {},
            "structure": {},
            "potential": {},
        },
        "melt_quench": {
            "composition": {"SiO2": 60.0, "CaO": 25.0, "Al2O3": 15.0},
            "final_structure": {
                "numbers": [14] * 50 + [8] * 100,
                "positions": [[0.0, 0.0, 0.0]] * 150,
                "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                "pbc": [True, True, True],
            },
            "mean_temperature": 302.3,
            "simulation_steps": 3,
        },
    }
    if include_structure:
        result["structure_characterization"] = _mock_structural_analysis()
    return result


def _insert_completed_job(
    job_id: str = "j-test-1",
    *,
    composition: str = "Al2O3 15 - CaO 25 - SiO2 60",
    potential: str = "pmmcs",
    request_hash: str = "testhash1234",
) -> None:
    from amorphouspy_api.routers.jobs_helpers import _elem_dict_to_vector, elemental_fractions_from_job

    store = get_job_store()
    result = _mock_result(include_structure=True)
    job = Job(
        job_id=job_id,
        request_hash=request_hash,
        composition=composition,
        potential=potential,
        status="completed",
        request_data={
            "composition": composition,
            "potential": potential,
            "simulation": {},
            "analyses": [{"type": "structure_characterization"}],
        },
        progress={
            "structure_generation": "completed",
            "melt_quench": "completed",
            "structure_characterization": "completed",
        },
        result_data=result,
        completed_at=datetime.now(UTC),
    )
    # Pre-compute elemental vector (mirrors what _update_from_resolved does)
    fracs = elemental_fractions_from_job(job)
    job.elemental_vector = _elem_dict_to_vector(fracs) if fracs else None
    store.create_job(job)


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
                "analyses": [{"type": "structure_characterization"}],
            },
            progress={
                "structure_generation": "completed",
                "melt_quench": "running",
                "structure_characterization": "pending",
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
        patch("amorphouspy_api.routers.jobs_helpers.get_executor") as mock_exe,
        patch(
            "amorphouspy_api.routers.jobs_helpers.submit_pipeline",
            return_value=mock_future,
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
    from amorphouspy.structure.composition import extract_composition
    from amorphouspy.structure.density import get_glass_density_from_model
    from amorphouspy_api.models import Composition, JobSubmission
    from amorphouspy_api.routers.jobs_helpers import _job_hash

    # Build the same submission the client will send, mirroring the
    # normalisation and density resolution that submit_job performs.
    sub = JobSubmission(composition={"SiO2": 60, "CaO": 25, "Al2O3": 15})
    normalised = extract_composition(sub.composition.root, tolerance=0.03)
    sub.composition = Composition({ox: frac * 100 for ox, frac in normalised.items()})
    sub.simulation.target_density = get_glass_density_from_model(sub.composition.root)
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
    assert data["progress"]["analyses"]["structure_characterization"] == "completed"


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
    assert data["progress"]["analyses"]["structure_characterization"] == "cancelled"
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
    assert data["analyses"]["structure_characterization"] is not None


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

    resp = client.get("/jobs/j-single-1/results/structure_characterization")
    assert resp.status_code == 200
    data = resp.json()
    assert "structure_characterization" in data


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
            "threshold": 0,
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()["matches"]) == 0


def test_search_jobs_close_match() -> None:
    """A nearby composition should appear as a close match."""
    _insert_completed_job("j-close-1", composition="Al2O3 15 - CaO 25 - SiO2 60")

    # The mock structure has only Si+O atoms, so the elemental distance
    # from an Al2O3-CaO-SiO2 query is ~0.18 — use threshold 0.2.
    resp = client.post(
        "/jobs:search",
        json={
            "composition": {"SiO2": 62, "CaO": 23, "Al2O3": 15},
            "threshold": 0.2,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    close = [m for m in data["matches"] if m["match_type"] == "close"]
    assert len(close) >= 1
    assert close[0]["job_id"] == "j-close-1"
    assert close[0]["distance"] > 0
    assert close[0]["similarity"] < 1.0


def test_search_jobs_close_match_outside_threshold() -> None:
    """A composition outside the threshold should not appear."""
    _insert_completed_job("j-far-1", composition="SiO2 100")

    resp = client.post(
        "/jobs:search",
        json={
            "composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15},
            "threshold": 0.01,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    close = [m for m in data["matches"] if m["match_type"] == "close"]
    assert all(m["job_id"] != "j-far-1" for m in close)


def test_search_jobs_threshold_zero_exact_only() -> None:
    """threshold=0 should return only exact matches, no close ones."""
    _insert_completed_job("j-exact-1", composition="Al2O3 15 - CaO 25 - SiO2 60")

    resp = client.post(
        "/jobs:search",
        json={
            "composition": {"SiO2": 62, "CaO": 23, "Al2O3": 15},
            "threshold": 0,
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
    assert "structure_characterization" in data["properties"]


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
        "viscosity_integral": [[0.0, 0.01], [0.0, 0.5], [0.0, 1200.0]],
    }


def _insert_completed_viscosity_job(job_id: str = "j-visc-1") -> None:
    store = get_job_store()
    result = _mock_result(include_structure=True)
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
                    {"type": "structure_characterization"},
                    {"type": "viscosity", "temperatures": [1500, 2000, 2500]},
                ],
            },
            progress={
                "structure_generation": "completed",
                "melt_quench": "completed",
                "structure_characterization": "completed",
                "viscosity": "completed",
            },
            result_data=result,
            completed_at=datetime.now(UTC),
        )
    )


def test_submit_job_with_viscosity() -> None:
    """Test submitting a job that includes viscosity analysis."""
    result = _mock_result()
    result["viscosity"] = _mock_viscosity_result()

    mock_future = MagicMock()
    mock_future.done.return_value = True
    mock_future.exception.return_value = None
    mock_future.result.return_value = result

    with (
        patch("amorphouspy_api.routers.jobs_helpers.get_executor") as mock_exe,
        patch(
            "amorphouspy_api.routers.jobs_helpers.submit_pipeline",
            return_value=mock_future,
        ),
    ):
        mock_exe.return_value.shutdown = MagicMock()

        resp = client.post(
            "/jobs",
            json={
                "composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15},
                "analyses": [
                    {"type": "structure_characterization"},
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
    assert data["analyses"]["structure_characterization"] is not None
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
    from amorphouspy_api.routers.jobs_helpers import _job_hash

    sub_no_visc = JobSubmission(
        composition={"SiO2": 60, "CaO": 25, "Al2O3": 15},
    )
    sub_with_visc = JobSubmission(
        composition={"SiO2": 60, "CaO": 25, "Al2O3": 15},
        analyses=[
            {"type": "structure_characterization"},
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


# ---------------------------------------------------------------------------
# CTE integration
# ---------------------------------------------------------------------------


def _mock_cte_fluctuations_result() -> dict[str, Any]:
    return {
        "summary": {
            "CTE_V_mean": 2.5e-5,
            "CTE_x_mean": 8.3e-6,
            "CTE_y_mean": 8.4e-6,
            "CTE_z_mean": 8.3e-6,
            "CTE_V_uncertainty": 5e-7,
            "CTE_x_uncertainty": 2e-7,
            "CTE_y_uncertainty": 2e-7,
            "CTE_z_uncertainty": 2e-7,
            "is_converged": "True",
            "convergence_criterion": 1e-6,
        },
        "data": {
            "run_index": [1, 2, 3],
            "steps": [200_000, 200_000, 200_000],
            "T": [300.1, 300.2, 300.0],
            "V": [1000.0, 1000.1, 999.9],
            "CTE_V": [2.5e-5, 2.5e-5, 2.5e-5],
            "CTE_x": [8.3e-6, 8.3e-6, 8.3e-6],
            "CTE_y": [8.4e-6, 8.4e-6, 8.4e-6],
            "CTE_z": [8.3e-6, 8.3e-6, 8.3e-6],
        },
    }


def _insert_completed_cte_job(job_id: str = "j-cte-1") -> None:
    store = get_job_store()
    result = _mock_result(include_structure=True)
    result["cte"] = _mock_cte_fluctuations_result()
    store.create_job(
        Job(
            job_id=job_id,
            request_hash="ctehash1234",
            composition="SiO2 60 - CaO 25 - Al2O3 15",
            potential="pmmcs",
            status="completed",
            request_data={
                "composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15},
                "potential": "pmmcs",
                "simulation": {},
                "analyses": [
                    {"type": "structure_characterization"},
                    {"type": "cte"},
                ],
            },
            progress={
                "structure_generation": "completed",
                "melt_quench": "completed",
                "structure_characterization": "completed",
                "cte": "completed",
            },
            result_data=result,
            completed_at=datetime.now(UTC),
        )
    )


def test_submit_job_with_cte() -> None:
    """Test submitting a job that includes CTE analysis."""
    result = _mock_result()
    result["cte"] = _mock_cte_fluctuations_result()

    mock_future = MagicMock()
    mock_future.done.return_value = True
    mock_future.exception.return_value = None
    mock_future.result.return_value = result

    with (
        patch("amorphouspy_api.routers.jobs_helpers.get_executor") as mock_exe,
        patch(
            "amorphouspy_api.routers.jobs_helpers.submit_pipeline",
            return_value=mock_future,
        ),
    ):
        mock_exe.return_value.shutdown = MagicMock()

        resp = client.post(
            "/jobs",
            json={
                "composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15},
                "analyses": [
                    {"type": "structure_characterization"},
                    {"type": "cte"},
                ],
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert data["status"] in ("pending", "completed")


def test_submit_job_with_cte_temperature_scan() -> None:
    """Test submitting a CTE job using the temperature_scan method."""
    result = _mock_result()
    result["cte"] = {"01_300K": {"run01": {"CTE_V": 2.5e-5}}}

    mock_future = MagicMock()
    mock_future.done.return_value = True
    mock_future.exception.return_value = None
    mock_future.result.return_value = result

    with (
        patch("amorphouspy_api.routers.jobs_helpers.get_executor") as mock_exe,
        patch(
            "amorphouspy_api.routers.jobs_helpers.submit_pipeline",
            return_value=mock_future,
        ),
    ):
        mock_exe.return_value.shutdown = MagicMock()

        resp = client.post(
            "/jobs",
            json={
                "composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15},
                "analyses": [
                    {"type": "cte", "method": "temperature_scan", "temperatures": [300, 500, 700]},
                ],
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data


def test_get_results_with_cte() -> None:
    """Test that the results endpoint returns CTE data."""
    _insert_completed_cte_job("j-cte-results")

    resp = client.get("/jobs/j-cte-results/results")
    assert resp.status_code == 200
    data = resp.json()
    assert data["analyses"]["cte"] is not None
    assert data["analyses"]["cte"]["summary"]["CTE_V_mean"] == 2.5e-5
    assert data["analyses"]["cte"]["summary"]["is_converged"] == "True"


def test_get_single_cte_result() -> None:
    """Test retrieving only the CTE analysis result."""
    _insert_completed_cte_job("j-cte-single")

    resp = client.get("/jobs/j-cte-single/results/cte")
    assert resp.status_code == 200
    data = resp.json()
    assert "cte" in data
    assert data["cte"]["summary"]["CTE_V_mean"] == 2.5e-5


def test_cte_progress_tracking() -> None:
    """Test that CTE step appears in progress when requested."""
    _insert_completed_cte_job("j-cte-progress")

    resp = client.get("/jobs/j-cte-progress")
    assert resp.status_code == 200
    data = resp.json()
    assert data["progress"]["analyses"]["cte"] == "completed"


# ---------------------------------------------------------------------------
# Elastic integration
# ---------------------------------------------------------------------------


def _mock_elastic_result() -> dict[str, Any]:
    return {
        "Cij": [[100, 30, 30, 0, 0, 0]] * 3 + [[0, 0, 0, 35, 0, 0]] * 3,
        "moduli": {
            "B": 53.3,
            "G": 35.0,
            "E": 85.6,
            "nu": 0.222,
        },
    }


def _insert_completed_elastic_job(job_id: str = "j-elastic-1") -> None:
    store = get_job_store()
    result = _mock_result(include_structure=True)
    result["elastic"] = _mock_elastic_result()
    store.create_job(
        Job(
            job_id=job_id,
            request_hash="elastichash1234",
            composition="SiO2 60 - CaO 25 - Al2O3 15",
            potential="pmmcs",
            status="completed",
            request_data={
                "composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15},
                "potential": "pmmcs",
                "simulation": {},
                "analyses": [
                    {"type": "structure_characterization"},
                    {"type": "elastic"},
                ],
            },
            progress={
                "structure_generation": "completed",
                "melt_quench": "completed",
                "structure_characterization": "completed",
                "elastic": "completed",
            },
            result_data=result,
            completed_at=datetime.now(UTC),
        )
    )


def test_submit_job_with_elastic() -> None:
    """Test submitting a job that includes elastic analysis."""
    result = _mock_result()
    result["elastic"] = _mock_elastic_result()

    mock_future = MagicMock()
    mock_future.done.return_value = True
    mock_future.exception.return_value = None
    mock_future.result.return_value = result

    with (
        patch("amorphouspy_api.routers.jobs_helpers.get_executor") as mock_exe,
        patch(
            "amorphouspy_api.routers.jobs_helpers.submit_pipeline",
            return_value=mock_future,
        ),
    ):
        mock_exe.return_value.shutdown = MagicMock()

        resp = client.post(
            "/jobs",
            json={
                "composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15},
                "analyses": [
                    {"type": "structure_characterization"},
                    {"type": "elastic"},
                ],
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "id" in data
    assert data["status"] in ("pending", "completed")


def test_get_results_with_elastic() -> None:
    """Test that the results endpoint returns elastic data."""
    _insert_completed_elastic_job("j-elastic-results")

    resp = client.get("/jobs/j-elastic-results/results")
    assert resp.status_code == 200
    data = resp.json()
    assert data["analyses"]["elastic"] is not None
    assert data["analyses"]["elastic"]["moduli"]["B"] == 53.3
    assert data["analyses"]["elastic"]["moduli"]["E"] == 85.6


def test_get_single_elastic_result() -> None:
    """Test retrieving only the elastic analysis result."""
    _insert_completed_elastic_job("j-elastic-single")

    resp = client.get("/jobs/j-elastic-single/results/elastic")
    assert resp.status_code == 200
    data = resp.json()
    assert "elastic" in data
    assert data["elastic"]["moduli"]["G"] == 35.0
    assert data["elastic"]["moduli"]["nu"] == 0.222


def test_elastic_progress_tracking() -> None:
    """Test that elastic step appears in progress when requested."""
    _insert_completed_elastic_job("j-elastic-progress")

    resp = client.get("/jobs/j-elastic-progress")
    assert resp.status_code == 200
    data = resp.json()
    assert data["progress"]["analyses"]["elastic"] == "completed"


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


def _insert_completed_job_with_tags(
    job_id: str = "j-tag-1",
    *,
    tags: list[str] | None = None,
    composition: str = "Al2O3 15 - CaO 25 - SiO2 60",
    request_hash: str = "taghash1234",
) -> None:
    from amorphouspy_api.routers.jobs_helpers import _elem_dict_to_vector, elemental_fractions_from_job

    store = get_job_store()
    result = _mock_result(include_structure=True)
    job = Job(
        job_id=job_id,
        request_hash=request_hash,
        composition=composition,
        potential="pmmcs",
        status="completed",
        request_data={
            "composition": composition,
            "potential": "pmmcs",
            "simulation": {},
            "analyses": [{"type": "structure_characterization"}],
        },
        progress={
            "structure_generation": "completed",
            "melt_quench": "completed",
            "structure_characterization": "completed",
        },
        result_data=result,
        tags=tags,
        completed_at=datetime.now(UTC),
    )
    fracs = elemental_fractions_from_job(job)
    job.elemental_vector = _elem_dict_to_vector(fracs) if fracs else None
    store.create_job(job)


def test_submit_job_with_tags() -> None:
    """Tags provided at submission time are stored and returned."""
    mock_future = MagicMock()
    mock_future.done.return_value = True
    mock_future.exception.return_value = None
    mock_future.result.return_value = _mock_result()

    with (
        patch("amorphouspy_api.routers.jobs_helpers.get_executor") as mock_exe,
        patch(
            "amorphouspy_api.routers.jobs_helpers.submit_pipeline",
            return_value=mock_future,
        ),
    ):
        mock_exe.return_value.shutdown = MagicMock()

        resp = client.post(
            "/jobs",
            json={
                "composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15},
                "tags": ["project-alpha", "batch-1"],
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert sorted(data["tags"]) == ["batch-1", "project-alpha"]


def test_submit_job_without_tags() -> None:
    """Submitting without tags returns an empty tag list."""
    mock_future = MagicMock()
    mock_future.done.return_value = True
    mock_future.exception.return_value = None
    mock_future.result.return_value = _mock_result()

    with (
        patch("amorphouspy_api.routers.jobs_helpers.get_executor") as mock_exe,
        patch(
            "amorphouspy_api.routers.jobs_helpers.submit_pipeline",
            return_value=mock_future,
        ),
    ):
        mock_exe.return_value.shutdown = MagicMock()

        resp = client.post(
            "/jobs",
            json={"composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15}},
        )

    assert resp.status_code == 200
    assert resp.json()["tags"] == []


def test_get_job_status_includes_tags() -> None:
    """GET /jobs/{id} includes tags in the response."""
    _insert_completed_job_with_tags("j-tag-status", tags=["my-project"])

    resp = client.get("/jobs/j-tag-status")
    assert resp.status_code == 200
    assert resp.json()["tags"] == ["my-project"]


def test_update_tags() -> None:
    """PUT /jobs/{id}/tags replaces the tag set."""
    _insert_completed_job_with_tags("j-tag-update", tags=["old-tag"])

    resp = client.put(
        "/jobs/j-tag-update/tags",
        json={"tags": ["new-tag", "project-beta"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == "j-tag-update"
    assert sorted(data["tags"]) == ["new-tag", "project-beta"]

    # Verify persisted
    resp2 = client.get("/jobs/j-tag-update")
    assert sorted(resp2.json()["tags"]) == ["new-tag", "project-beta"]


def test_update_tags_not_found() -> None:
    """PUT /jobs/{id}/tags returns 404 for unknown job."""
    resp = client.put("/jobs/nonexistent/tags", json={"tags": ["x"]})
    assert resp.status_code == 404


def test_update_tags_deduplicates() -> None:
    """Duplicate tags are collapsed."""
    _insert_completed_job_with_tags("j-tag-dedup", tags=[])

    resp = client.put(
        "/jobs/j-tag-dedup/tags",
        json={"tags": ["a", "b", "a"]},
    )
    assert resp.status_code == 200
    assert resp.json()["tags"] == ["a", "b"]


def test_search_jobs_returns_tags() -> None:
    """POST /jobs:search includes tags on each match."""
    _insert_completed_job_with_tags(
        "j-tag-search",
        tags=["project-x"],
        request_hash="tagsearchhash",
    )

    resp = client.post(
        "/jobs:search",
        json={"composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15}},
    )
    assert resp.status_code == 200
    match = next(m for m in resp.json()["matches"] if m["job_id"] == "j-tag-search")
    assert match["tags"] == ["project-x"]


def test_search_jobs_filter_by_tags() -> None:
    """POST /jobs:search with tags filter returns only matching jobs."""
    _insert_completed_job_with_tags(
        "j-tag-filter-yes",
        tags=["project-x", "batch-1"],
        request_hash="tagfilter1",
    )
    _insert_completed_job_with_tags(
        "j-tag-filter-no",
        tags=["project-y"],
        request_hash="tagfilter2",
    )

    resp = client.post(
        "/jobs:search",
        json={
            "composition": {"SiO2": 60, "CaO": 25, "Al2O3": 15},
            "tags": ["project-x"],
        },
    )
    assert resp.status_code == 200
    ids = [m["job_id"] for m in resp.json()["matches"]]
    assert "j-tag-filter-yes" in ids
    assert "j-tag-filter-no" not in ids


# ---------------------------------------------------------------------------
# ElectrostaticsParams
# ---------------------------------------------------------------------------


def test_electrostatics_params_to_config_roundtrip():
    """ElectrostaticsParams.to_electrostatics_config() returns matching ElectrostaticsConfig."""
    from amorphouspy_api.models import ElectrostaticsParams, LongRangeMethod

    from amorphouspy import ElectrostaticsConfig

    params = ElectrostaticsParams(method=LongRangeMethod.pppm, long_range_cutoff=9.0, kspace_accuracy=1e-4)
    config = params.to_electrostatics_config()

    assert isinstance(config, ElectrostaticsConfig)
    assert config.method == "pppm"
    assert config.long_range_cutoff == 9.0
    assert config.kspace_accuracy == 1e-4


def test_job_submission_accepts_electrostatics():
    """JobSubmission with a non-default electrostatics field serialises and deserialises correctly."""
    from amorphouspy_api.models import ElectrostaticsParams, JobSubmission, LongRangeMethod

    submission = JobSubmission(
        composition={"SiO2": 70, "Na2O": 30},
        electrostatics=ElectrostaticsParams(method=LongRangeMethod.wolf, alpha=0.3, long_range_cutoff=10.0),
    )
    data = submission.model_dump()
    roundtrip = JobSubmission.model_validate(data)

    assert roundtrip.electrostatics.method == LongRangeMethod.wolf
    assert roundtrip.electrostatics.alpha == 0.3
    assert roundtrip.electrostatics.long_range_cutoff == 10.0
