"""Database models and utilities for amorphouspy API.

This module provides SQLAlchemy models for persisting task metadata and cached results,
replacing the ephemeral in-memory task store with persistent SQLite storage.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Index, String, Text, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .models import MeltquenchResult, serialize_atoms

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base class."""


class Task(Base):
    """SQLAlchemy model for task metadata and results.

    Stores both active task state and completed results for caching purposes.
    """

    __tablename__ = "tasks"

    task_id = Column(String(36), primary_key=True)  # UUID4 string
    request_hash = Column(String(16), nullable=False, index=True)  # For cache lookups
    state = Column(String(20), nullable=False, default="processing")  # processing, complete, error
    status = Column(String(100), nullable=True)  # Human-readable status message

    # Store the original request for reference
    request_data = Column(JSON, nullable=True)

    # Store results as JSON when completed
    result_data = Column(JSON, nullable=True)

    # Store error information if failed
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    # Index for efficient cache lookups
    __table_args__ = (Index("ix_request_hash_state", "request_hash", "state"),)


class TaskStore:
    """SQLAlchemy-based task store that provides the same interface as the multiprocessing manager dict.

    This class handles all database operations for task storage and retrieval,
    with special focus on efficient cache lookups by request hash.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the task store with SQLite database.

        Args:
            db_path: Path to SQLite database file. If None, uses 'tasks.db' in current directory.
        """
        if db_path is None:
            db_path = Path("tasks.db")

        # Create database URL
        self.db_url = f"sqlite:///{db_path}"

        # Create engine with SQLite-specific settings
        self.engine = create_engine(
            self.db_url,
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,  # Verify connections before use
            connect_args={
                "check_same_thread": False,  # Allow use from multiple threads
                "timeout": 30,  # 30 second timeout for busy database
            },
        )

        # Create session factory
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Create tables
        self._create_tables()

        logger.info("Initialized task store with database: %s", db_path)

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created/verified")
        except SQLAlchemyError:
            logger.exception("Error creating database tables")
            raise

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def get(self, task_id: str) -> dict[str, Any] | None:
        """Get task data by task ID.

        Args:
            task_id: Task identifier.

        Returns:
            Task data dict or None if not found.
        """
        try:
            with self.get_session() as session:
                task = session.get(Task, task_id)
                if task:
                    return self._task_to_dict(task)
                return None
        except SQLAlchemyError:
            logger.exception("Error getting task %s", task_id)
            return None

    def set(self, task_id: str, task_data: dict[str, Any]) -> None:
        """Set or update task data.

        Args:
            task_id: Task identifier.
            task_data: Task data dictionary.

        Raises:
            SQLAlchemyError: If a database error occurs.
        """
        try:
            with self.get_session() as session:
                task = session.get(Task, task_id)

                if task:
                    # Update existing task
                    self._update_task_from_dict(task, task_data)
                else:
                    # Create new task
                    task = Task(task_id=task_id)
                    self._update_task_from_dict(task, task_data)
                    session.add(task)

                session.commit()
                logger.debug("Updated task %s", task_id)

        except SQLAlchemyError:
            logger.exception("Error setting task %s", task_id)
            raise

    def items(self) -> list[tuple]:
        """Get all tasks as (task_id, task_data) tuples.

        Returns:
            List of (task_id, task_data) tuples.
        """
        try:
            with self.get_session() as session:
                tasks = session.query(Task).all()
                return [(task.task_id, self._task_to_dict(task)) for task in tasks]
        except SQLAlchemyError:
            logger.exception("Error getting all tasks")
            return []

    def find_cached_result(self, request_hash: str) -> tuple[str, MeltquenchResult] | None:
        """Find a completed task with matching request hash for cache lookup.

        Args:
            request_hash: Hash of the request parameters.

        Returns:
            Tuple of (task_id, MeltquenchResult) if cached result found, None otherwise.
        """
        try:
            with self.get_session() as session:
                task = (
                    session.query(Task)
                    .filter(
                        Task.request_hash == request_hash,
                        Task.state == "complete",
                        Task.result_data.isnot(None),
                    )
                    .first()
                )

                if task and task.result_data:
                    logger.info(
                        "Found cached result for hash %s in task %s",
                        request_hash,
                        task.task_id,
                    )
                    return (task.task_id, MeltquenchResult(**task.result_data))

                return None

        except SQLAlchemyError:
            logger.exception("Error finding cached result for hash %s", request_hash)
            return None

    def cleanup_old_tasks(self, days: int = 30) -> int:
        """Clean up old completed/error tasks older than specified days.

        Args:
            days: Number of days to keep tasks. Defaults to 30.

        Returns:
            Number of tasks deleted.
        """
        try:
            cutoff_date = datetime.now(UTC).replace(day=datetime.now(UTC).day - days)

            with self.get_session() as session:
                deleted_count = (
                    session.query(Task)
                    .filter(
                        Task.state.in_(["complete", "error"]),
                        Task.updated_at < cutoff_date,
                    )
                    .delete()
                )

                session.commit()

                if deleted_count > 0:
                    logger.info("Cleaned up %s old tasks", deleted_count)

                return deleted_count

        except SQLAlchemyError:
            logger.exception("Error cleaning up old tasks")
            return 0

    def _task_to_dict(self, task: Task) -> dict[str, Any]:
        """Convert Task model to dictionary format expected by the API."""
        task_dict = {
            "state": task.state,
            "status": task.status,
            "request_hash": task.request_hash,
        }

        if task.result_data:
            task_dict["result"] = task.result_data

        if task.error_message:
            task_dict["error"] = task.error_message

        return task_dict

    def _update_task_from_dict(self, task: Task, task_data: dict[str, Any]) -> None:
        """Update Task model from dictionary data."""
        if "state" in task_data:
            task.state = task_data["state"]

        if "status" in task_data:
            task.status = task_data["status"]

        if "request_hash" in task_data:
            task.request_hash = task_data["request_hash"]

        if "result" in task_data:
            result = task_data["result"]
            if result is not None:
                # Handle ASE Atoms serialization in final_structure
                result_data = result.copy()
                if "final_structure" in result_data:
                    from ase import Atoms

                    if isinstance(result_data["final_structure"], Atoms):
                        # Serialize ASE Atoms to JSON string for storage
                        result_data["final_structure"] = serialize_atoms(result_data["final_structure"])
                task.result_data = result_data
            else:
                task.result_data = None

        if "error" in task_data:
            task.error_message = task_data["error"]

        if "request_data" in task_data:
            task.request_data = task_data["request_data"]

        # Always update the timestamp
        task.updated_at = datetime.now(UTC)


# Global task store instance
_task_store_instance: TaskStore | None = None


def get_task_store() -> TaskStore:
    """Get the global task store instance."""
    global _task_store_instance
    if _task_store_instance is None:
        # Initialize with default database path
        db_path = Path("tasks.db")
        _task_store_instance = TaskStore(db_path)
    return _task_store_instance


def init_task_store(db_path: Path | None = None) -> TaskStore:
    """Initialize the global task store with custom database path.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        TaskStore instance.
    """
    global _task_store_instance
    _task_store_instance = TaskStore(db_path)
    return _task_store_instance
