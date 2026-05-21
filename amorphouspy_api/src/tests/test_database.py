"""Tests for the JobStore database layer."""

import sqlite3
import tempfile
import threading
from pathlib import Path

from amorphouspy_api.database import Job, JobStore


def test_job_store_create_and_get() -> None:
    """Test basic create and get operations."""
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "test.db")
        job = Job(
            job_id="j-1",
            request_hash="abc123",
            composition="SiO2 70 - Na2O 30",
            potential="pmmcs",
            status="pending",
            progress={"structure_generation": "pending"},
        )
        store.create_job(job)

        retrieved = store.get_job("j-1")
        assert retrieved is not None
        assert retrieved.composition == "SiO2 70 - Na2O 30"
        assert retrieved.status == "pending"
        store.close()


def test_job_store_update() -> None:
    """Test updating job fields."""
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "test.db")
        store.create_job(
            Job(
                job_id="j-2",
                request_hash="def456",
                composition="SiO2 100",
                potential="shik",
                status="running",
            )
        )

        store.update_job("j-2", status="completed", result_data={"density": 2.2})
        job = store.get_job("j-2")
        assert job.status == "completed"
        assert job.result_data == {"density": 2.2}
        store.close()


def test_find_completed_by_hash() -> None:
    """Test cache lookup by request hash."""
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "test.db")
        store.create_job(
            Job(
                job_id="j-3",
                request_hash="hash-a",
                composition="SiO2 80 - Na2O 20",
                potential="pmmcs",
                status="completed",
                result_data={"structure_characterization": {"density": 2.5}},
            )
        )

        found = store.find_completed_by_hash("hash-a")
        assert found is not None
        assert found.job_id == "j-3"

        assert store.find_completed_by_hash("nonexistent") is None
        store.close()


def test_search_by_composition() -> None:
    """Test searching completed jobs by composition."""
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "test.db")
        for i, pot in enumerate(["pmmcs", "shik"]):
            store.create_job(
                Job(
                    job_id=f"j-search-{i}",
                    request_hash=f"h-{i}",
                    composition="SiO2 70 - Na2O 30",
                    potential=pot,
                    status="completed",
                    result_data={},
                )
            )
        store.create_job(
            Job(
                job_id="j-other",
                request_hash="h-other",
                composition="SiO2 100",
                potential="pmmcs",
                status="completed",
                result_data={},
            )
        )

        results = store.search_by_composition("SiO2 70 - Na2O 30")
        assert len(results) == 2

        results_filtered = store.search_by_composition("SiO2 70 - Na2O 30", "shik")
        assert len(results_filtered) == 1
        assert results_filtered[0].potential == "shik"
        store.close()


def test_list_compositions() -> None:
    """Test listing all compositions with completed jobs."""
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "test.db")
        for i in range(3):
            store.create_job(
                Job(
                    job_id=f"j-list-{i}",
                    request_hash=f"h-list-{i}",
                    composition="SiO2 70 - Na2O 30",
                    potential="pmmcs",
                    status="completed",
                    result_data={},
                )
            )
        store.create_job(
            Job(
                job_id="j-list-other",
                request_hash="h-list-other",
                composition="SiO2 100",
                potential="pmmcs",
                status="completed",
                result_data={},
            )
        )
        # Pending job should not appear
        store.create_job(
            Job(
                job_id="j-list-pending",
                request_hash="h-list-pending",
                composition="B2O3 100",
                potential="pmmcs",
                status="pending",
            )
        )

        comps = store.list_compositions()
        assert len(comps) == 2
        by_comp = {c["composition"]: c["n_jobs"] for c in comps}
        assert by_comp["SiO2 70 - Na2O 30"] == 3
        assert by_comp["SiO2 100"] == 1
        store.close()


def test_list_completed_vectors() -> None:
    """Test lightweight vector query for fuzzy search."""
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "test.db")
        # Build fixed-length vectors (119 elements, indexed by Z)
        vec_sio2 = [0.0] * 119
        vec_sio2[8] = 0.667  # O
        vec_sio2[14] = 0.333  # Si
        vec_binary = [0.0] * 119
        vec_binary[8] = 0.63  # O
        vec_binary[11] = 0.17  # Na
        vec_binary[14] = 0.2  # Si

        store.create_job(
            Job(
                job_id="j-vec-1",
                request_hash="h-vec-1",
                composition="SiO2 100",
                potential="pmmcs",
                status="completed",
                result_data={},
                elemental_vector=vec_sio2,
                request_data={"analyses": [{"type": "structure_characterization"}]},
            )
        )
        store.create_job(
            Job(
                job_id="j-vec-2",
                request_hash="h-vec-2",
                composition="SiO2 70 - Na2O 30",
                potential="shik",
                status="completed",
                result_data={},
                elemental_vector=vec_binary,
                request_data={"analyses": [{"type": "structure_characterization"}]},
            )
        )
        # No vector -> should not appear
        store.create_job(
            Job(
                job_id="j-vec-none",
                request_hash="h-vec-none",
                composition="B2O3 100",
                potential="pmmcs",
                status="completed",
                result_data={},
            )
        )

        rows = store.list_completed_vectors()
        assert len(rows) == 2

        rows_pmmcs = store.list_completed_vectors("pmmcs")
        assert len(rows_pmmcs) == 1
        assert rows_pmmcs[0][0] == "j-vec-1"  # job_id
        assert rows_pmmcs[0][1] == vec_sio2  # elemental_vector
        store.close()


def test_concurrent_writes() -> None:
    """Test thread safety of concurrent writes."""
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "test.db")
        errors: list[Exception] = []
        n = 10

        def create(i: int) -> None:
            try:
                store.create_job(
                    Job(
                        job_id=f"j-thread-{i}",
                        request_hash=f"h-thread-{i}",
                        composition="SiO2 100",
                        potential="pmmcs",
                        status="pending",
                    )
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        store.close()


def test_persistence() -> None:
    """Test that data persists across JobStore instances."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        store1 = JobStore(db_path)
        store1.create_job(
            Job(
                job_id="j-persist",
                request_hash="h-persist",
                composition="SiO2 100",
                potential="pmmcs",
                status="completed",
                result_data={},
            )
        )
        store1.close()

        store2 = JobStore(db_path)
        job = store2.get_job("j-persist")
        assert job is not None
        assert job.status == "completed"
        store2.close()


def test_list_glasses() -> None:
    """Test list_glasses returns correct lightweight summaries."""
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "test.db")
        store.create_job(
            Job(
                job_id="j-lg-1",
                request_hash="h-lg-1",
                composition="SiO2 70 - Na2O 30",
                potential="pmmcs",
                status="completed",
                progress={
                    "structure_generation": "completed",
                    "melt_quench": "completed",
                    "structure_characterization": "completed",
                    "viscosity": "completed",
                },
                tags=["batch-1", "project-a"],
            )
        )
        store.create_job(
            Job(
                job_id="j-lg-2",
                request_hash="h-lg-2",
                composition="SiO2 100",
                potential="shik",
                status="completed",
                progress={
                    "structure_generation": "completed",
                    "melt_quench": "completed",
                    "structure_characterization": "completed",
                },
                tags=["batch-2"],
            )
        )
        # Pending job should not appear
        store.create_job(
            Job(
                job_id="j-lg-pending",
                request_hash="h-lg-pending",
                composition="B2O3 100",
                potential="pmmcs",
                status="pending",
            )
        )

        # No tag filter
        results = store.list_glasses()
        assert len(results) == 2
        ids = {r["job_id"] for r in results}
        assert ids == {"j-lg-1", "j-lg-2"}

        # Check analyses are extracted from progress (excluding non-analysis steps)
        by_id = {r["job_id"]: r for r in results}
        assert set(by_id["j-lg-1"]["analyses"]) == {"structure_characterization", "viscosity"}
        assert by_id["j-lg-1"]["tags"] == ["batch-1", "project-a"]

        # Tag filter
        results_b1 = store.list_glasses(tag="batch-1")
        assert len(results_b1) == 1
        assert results_b1[0]["job_id"] == "j-lg-1"

        results_b2 = store.list_glasses(tag="batch-2")
        assert len(results_b2) == 1
        assert results_b2[0]["job_id"] == "j-lg-2"

        results_none = store.list_glasses(tag="nonexistent")
        assert results_none == []
        store.close()


def test_list_glasses_uses_covering_index() -> None:
    """Verify the list_glasses query uses the covering index and never touches row data."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        store = JobStore(db_path)
        store.create_job(
            Job(
                job_id="j-idx",
                request_hash="h-idx",
                composition="SiO2 100",
                potential="pmmcs",
                status="completed",
                progress={"structure_generation": "completed", "melt_quench": "completed"},
            )
        )
        store.close()

        conn = sqlite3.connect(str(db_path))
        # The exact query issued by list_glasses (via ORM)
        plan_rows = conn.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT jobs.job_id, jobs.composition, jobs.potential, "
            "jobs.tags, jobs.completed_at, jobs.progress "
            "FROM jobs "
            "WHERE jobs.status = 'completed' "
            "ORDER BY jobs.composition, jobs.created_at DESC"
        ).fetchall()
        plan_text = " ".join(str(r) for r in plan_rows)
        assert "COVERING INDEX" in plan_text, f"Expected covering index usage, got: {plan_text}"
        assert "ix_jobs_glasses_listing" in plan_text
        conn.close()


def test_simulation_history_separate_column() -> None:
    """Test that simulation_history is extracted from result_data into its own column."""
    with tempfile.TemporaryDirectory() as tmp:
        store = JobStore(Path(tmp) / "test.db")
        store.create_job(
            Job(
                job_id="j-hist",
                request_hash="h-hist",
                composition="SiO2 100",
                potential="shik",
                status="running",
            )
        )

        history = [{"positions": [[0, 0, 0]], "temperature": [300.0]}]
        store.update_job(
            "j-hist",
            status="completed",
            result_data={
                "melt_quench": {
                    "final_structure": {},
                    "simulation_history": history,
                },
                "structure_characterization": {"density": 2.2},
            },
        )

        # get_job should not include simulation_history and result_data should be clean
        job = store.get_job("j-hist")
        assert job is not None
        assert "simulation_history" not in job.result_data.get("melt_quench", {})
        assert job.result_data["structure_characterization"]["density"] == 2.2

        # get_job_with_history should include it
        job_full = store.get_job_with_history("j-hist")
        assert job_full is not None
        assert job_full.simulation_history == history
        store.close()


def test_get_job_defers_simulation_history() -> None:
    """Test that get_job does not load the simulation_history column."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        store = JobStore(db_path)
        store.create_job(
            Job(
                job_id="j-defer",
                request_hash="h-defer",
                composition="SiO2 100",
                potential="pmmcs",
                status="completed",
                result_data={"melt_quench": {"final_structure": {}}},
                simulation_history=[{"positions": [[1, 2, 3]]}],
            )
        )

        # get_job should work without loading history
        job = store.get_job("j-defer")
        assert job is not None
        assert job.status == "completed"

        # get_job_with_history should return the history
        job_full = store.get_job_with_history("j-defer")
        assert job_full.simulation_history == [{"positions": [[1, 2, 3]]}]
        store.close()
