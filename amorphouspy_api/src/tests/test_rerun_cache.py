"""Tests for selective executor-cache cleanup on ``rerun=failed``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from amorphouspy_api.routers.jobs import _clear_executor_cache
from executorlib.standalone.hdf import dump, get_queue_id

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

HASH = "deadbeefcafef00d"


def _write_step(cache_dir: Path, step: str, *, success: bool, queue_id: int) -> Path:
    """Create an executorlib ``_o.h5`` (and matching ``_i.h5``) for a step."""
    out = cache_dir / f"{HASH}_{step}_o.h5"
    inp = cache_dir / f"{HASH}_{step}_i.h5"
    payload = {"output": {"value": 1}} if success else {"error": RuntimeError("boom")}
    dump(file_name=str(out), data_dict={**payload, "queue_id": queue_id})
    dump(file_name=str(inp), data_dict={"queue_id": queue_id})
    return out


def test_failed_only_strips_queue_id_from_kept_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful steps are kept but lose their stale ``queue_id``."""
    from amorphouspy_api import config

    monkeypatch.setattr(config, "MELTQUENCH_PROJECT_DIR", tmp_path)

    good = _write_step(tmp_path, "melt_quench", success=True, queue_id=96213)
    bad = _write_step(tmp_path, "elastic", success=False, queue_id=96214)

    assert get_queue_id(file_name=str(good)) == 96213

    _clear_executor_cache(HASH, failed_only=True)

    # Successful step kept, but its stale queue_id is gone so it no longer
    # injects a dead ``afterok`` dependency on resubmission.
    assert good.exists()
    assert get_queue_id(file_name=str(good)) is None

    # Failed step (and its input) removed so executorlib re-submits it.
    assert not bad.exists()
    assert not (tmp_path / f"{HASH}_elastic_i.h5").exists()


def test_failed_only_keeps_successful_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cached result payload of a kept step is preserved."""
    import h5py

    from amorphouspy_api import config

    monkeypatch.setattr(config, "MELTQUENCH_PROJECT_DIR", tmp_path)

    good = _write_step(tmp_path, "structure_generation", success=True, queue_id=96212)

    _clear_executor_cache(HASH, failed_only=True)

    assert good.exists()
    with h5py.File(good, "r") as hdf:
        assert "output" in hdf
        assert "queue_id" not in hdf


def test_rerun_all_removes_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``failed_only=False`` deletes every cache file for the hash."""
    from amorphouspy_api import config

    monkeypatch.setattr(config, "MELTQUENCH_PROJECT_DIR", tmp_path)

    _write_step(tmp_path, "melt_quench", success=True, queue_id=96213)
    _write_step(tmp_path, "elastic", success=False, queue_id=96214)

    _clear_executor_cache(HASH, failed_only=False)

    assert list(tmp_path.glob(f"{HASH}*.h5")) == []
