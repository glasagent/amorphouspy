"""Shared test fixtures for amorphouspy_api tests."""

from pathlib import Path

import pytest

from amorphouspy_api.database import close_job_store, init_job_store


@pytest.fixture(autouse=True)
def _fresh_job_store(tmp_path: Path) -> None:
    """Provide a fresh temporary job store for every test."""
    db_path = tmp_path / "test_jobs.db"
    init_job_store(db_path)
    yield
    close_job_store()
