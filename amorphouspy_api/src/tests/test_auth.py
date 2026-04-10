"""Tests for bearer-token authentication."""

from __future__ import annotations

from unittest.mock import patch

from amorphouspy_api.database import Job, get_job_store
from fastapi.testclient import TestClient

TEST_TOKEN = "test-secret-token"  # noqa: S105


def _make_client() -> TestClient:
    """Create a fresh app + client so module-level auth config is re-evaluated."""
    # Re-import with patched API_TOKEN so the HTTPBearer scheme picks it up.
    import importlib

    import amorphouspy_api.auth
    import amorphouspy_api.routers.glasses
    import amorphouspy_api.routers.jobs

    importlib.reload(amorphouspy_api.auth)
    importlib.reload(amorphouspy_api.routers.glasses)
    importlib.reload(amorphouspy_api.routers.jobs)
    importlib.reload(importlib.import_module("amorphouspy_api.app"))

    from amorphouspy_api.app import app

    return TestClient(app)


def _seed_job(job_id: str = "j-auth-1") -> None:
    store = get_job_store()
    store.create_job(
        Job(
            job_id=job_id,
            request_hash="h-auth",
            composition="SiO2 100",
            potential="pmmcs",
            status="completed",
            result_data={"structure_characterization": {"density": 2.2}},
        ),
    )


@patch("amorphouspy_api.auth.API_TOKEN", TEST_TOKEN)
@patch("amorphouspy_api.config.API_TOKEN", TEST_TOKEN)
def test_protected_endpoint_rejects_missing_token():
    """POST /jobs without a token returns 401/403."""
    client = _make_client()
    resp = client.post("/jobs", json={"composition": {"SiO2": 100}, "potential": "CHGNet"})
    assert resp.status_code in (401, 403)


@patch("amorphouspy_api.auth.API_TOKEN", TEST_TOKEN)
@patch("amorphouspy_api.config.API_TOKEN", TEST_TOKEN)
def test_protected_endpoint_rejects_wrong_token():
    """POST /jobs with a wrong token returns 403."""
    client = _make_client()
    resp = client.post(
        "/jobs",
        json={"composition": {"SiO2": 100}, "potential": "CHGNet"},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 403


@patch("amorphouspy_api.auth.API_TOKEN", TEST_TOKEN)
@patch("amorphouspy_api.config.API_TOKEN", TEST_TOKEN)
def test_open_endpoint_allows_without_token():
    """GET /jobs/{id} (read-only) works without a token even when auth is enabled."""
    client = _make_client()
    _seed_job()
    resp = client.get("/jobs/j-auth-1")
    assert resp.status_code == 200


@patch("amorphouspy_api.auth.API_TOKEN", TEST_TOKEN)
@patch("amorphouspy_api.config.API_TOKEN", TEST_TOKEN)
def test_glasses_protected():
    """GET /glasses requires a token when API_TOKEN is set."""
    client = _make_client()
    resp = client.get("/glasses")
    assert resp.status_code in (401, 403)

    resp = client.get("/glasses", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
    assert resp.status_code == 200
