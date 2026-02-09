"""Shared test fixtures for amorphouspy_api tests."""

from pathlib import Path

import pytest

from amorphouspy_api import app as app_module
from amorphouspy_api.database import close_task_store, init_task_store


@pytest.fixture(autouse=True)
def _fresh_task_store(tmp_path: Path) -> None:
    """Provide a fresh temporary task store for every test.

    This ensures tests are isolated from each other and from any
    persistent database left over from previous runs.
    """
    # Close the existing store (created at app import time) to avoid resource warnings
    old_store = app_module._task_store
    if old_store is not None:
        old_store.close()

    db_path = tmp_path / "test_tasks.db"
    store = init_task_store(db_path)
    # Update the module-level reference used by the app endpoints
    app_module._task_store = store
    yield
    close_task_store()
