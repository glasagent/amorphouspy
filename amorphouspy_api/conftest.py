"""Shared test fixtures for amorphouspy_api tests."""

from pathlib import Path

import pytest

from amorphouspy_api.database import close_task_store, init_task_store


@pytest.fixture(autouse=True)
def _fresh_task_store(tmp_path: Path) -> None:
    """Provide a fresh temporary task store for every test.

    This ensures tests are isolated from each other and from any
    persistent database left over from previous runs.
    """
    # Re-initialise the singleton so every call to get_task_store()
    # (in routers, visualization, tests, …) returns the fresh instance.
    db_path = tmp_path / "test_tasks.db"
    init_task_store(db_path)
    yield
    close_task_store()
