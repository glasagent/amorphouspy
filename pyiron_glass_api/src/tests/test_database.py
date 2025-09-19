"""Test database functionality for the task store."""

import tempfile
from pathlib import Path

from pyiron_glass_api.database import TaskStore


def test_task_store_basic_operations() -> None:
    """Test basic task store operations."""
    # Use temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_tasks.db"
        store = TaskStore(db_path)

        # Test set and get
        task_data = {"state": "processing", "status": "Starting", "request_hash": "abc123def456"}

        store.set("test_task_1", task_data)
        retrieved = store.get("test_task_1")

        assert retrieved is not None
        assert retrieved["state"] == "processing"
        assert retrieved["status"] == "Starting"
        assert retrieved["request_hash"] == "abc123def456"


def test_task_store_cached_result_lookup() -> None:
    """Test efficient cached result lookup by hash."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_tasks.db"
        store = TaskStore(db_path)

        # Create a completed task with results (matching current API format)
        # Create a simple mock structure that can be validated as ASE Atoms
        mock_structure = {
            "numbers": [14, 8, 8],  # Si, O, O
            "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "cell": [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            "pbc": [True, True, True],
        }

        result_data = {
            "composition": "0.75SiO2-0.25Na2O",
            "final_structure": mock_structure,
            "mean_temperature": 300.0,
            "simulation_steps": 1000,
            "structural_analysis": {
                "density": 2.5,
                "coordination": {"oxygen": {}, "formers": {}, "modifiers": {}},
                "network": {"connectivity": 3.0, "Qn_distribution": {}, "Qn_distribution_partial": {}},
                "distributions": {"bond_angles": {}, "rings": {}},
                "rdfs": {"r": [], "rdfs": {}, "cumulative_coordination": {}},
                "elements": {"formers": ["Si"], "modifiers": ["Na"], "cutoffs": {}},
            },
        }

        completed_task = {
            "state": "complete",
            "status": "Completed",
            "request_hash": "test_hash_123",
            "result": result_data,
        }

        store.set("completed_task", completed_task)

        # Test cache lookup
        cached_result = store.find_cached_result("test_hash_123")
        assert cached_result is not None
        # find_cached_result now returns tuple[str, MeltquenchResult]
        task_id, result = cached_result
        assert task_id == "completed_task"
        assert result.composition == "0.75SiO2-0.25Na2O"
        assert result.structural_analysis is not None
        # Handle both dict and StructureData object cases
        if isinstance(result.structural_analysis, dict):
            assert result.structural_analysis["density"] == 2.5
        else:
            assert result.structural_analysis.density == 2.5

        # Test cache miss
        no_result = store.find_cached_result("nonexistent_hash")
        assert no_result is None


def test_task_store_items() -> None:
    """Test getting all tasks."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_tasks.db"
        store = TaskStore(db_path)

        # Add multiple tasks
        store.set("task1", {"state": "processing", "request_hash": "hash1"})
        store.set("task2", {"state": "complete", "request_hash": "hash2"})

        # Get all items
        items = store.items()
        assert len(items) == 2

        task_ids = [item[0] for item in items]
        assert "task1" in task_ids
        assert "task2" in task_ids


def test_task_store_persistence() -> None:
    """Test that data persists across TaskStore instances."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_tasks.db"

        # Create store and add data
        store1 = TaskStore(db_path)
        store1.set("persistent_task", {"state": "complete", "status": "Done", "request_hash": "persistent_hash"})

        # Create new store instance with same database
        store2 = TaskStore(db_path)
        retrieved = store2.get("persistent_task")

        assert retrieved is not None
        assert retrieved["state"] == "complete"
        assert retrieved["status"] == "Done"
        assert retrieved["request_hash"] == "persistent_hash"
