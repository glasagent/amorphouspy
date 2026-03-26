"""Database models and store for the amorphouspy API.

Single ``Job`` table backs both ``/jobs`` and ``/glasses`` endpoints.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Column, DateTime, Index, String, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool

from .models import serialize_atoms

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base class."""


class Job(Base):
    """A simulation job."""

    __tablename__ = "jobs"

    job_id = Column(String(36), primary_key=True)
    # Deterministic hash of (normalised composition + potential + simulation params)
    request_hash = Column(String(64), nullable=False, index=True)
    # Canonical composition string (via Composition.canonical)
    composition = Column(String(256), nullable=False, index=True)
    potential = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False, default="pending")

    # Full JobSubmission.model_dump()
    request_data = Column(JSON, nullable=True)

    # Per-step progress: {"structure_generation": "completed", …}
    progress = Column(JSON, nullable=True)

    # Analysis results keyed by type: {"structure": {…}}
    result_data = Column(JSON, nullable=True)

    # Errors keyed by step name: {"viscosity": "…"}
    errors = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    completed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_jobs_composition_status", "composition", "status"),
        Index("ix_jobs_hash_status", "request_hash", "status"),
    )


# ---------------------------------------------------------------------------
# JobStore — thin wrapper around SQLAlchemy sessions
# ---------------------------------------------------------------------------


class JobStore:
    """SQLAlchemy-backed store for Job records."""

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialise the store, creating the DB file if needed."""
        if db_path is None:
            db_path = Path("jobs.db")

        self.db_url = f"sqlite:///{db_path}"
        self.engine = create_engine(
            self.db_url,
            echo=False,
            poolclass=NullPool,
            connect_args={
                "check_same_thread": False,
                "timeout": 30,
            },
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        logger.info("Initialised job store: %s", db_path)

    def close(self) -> None:
        """Dispose of the SQLAlchemy engine."""
        if self.engine:
            self.engine.dispose()
            logger.info("Closed job store connection")

    def session(self) -> Session:
        """Create a new database session."""
        return self.SessionLocal()

    # -- CRUD ---------------------------------------------------------------

    def create_job(self, job: Job) -> None:
        """Insert a new job record."""
        with self.session() as s:
            s.add(job)
            s.commit()

    def get_job(self, job_id: str) -> Job | None:
        """Fetch a job by ID, or ``None``."""
        with self.session() as s:
            return s.get(Job, job_id)

    def update_job(self, job_id: str, **fields: object) -> None:
        """Update arbitrary columns on a job record."""
        with self.session() as s:
            job = s.get(Job, job_id)
            if job is None:
                return
            for k, val in fields.items():
                # Serialise ASE Atoms if present inside result_data
                effective = _serialise_atoms_in_result(val) if k == "result_data" and isinstance(val, dict) else val
                setattr(job, k, effective)
            s.commit()

    # -- query helpers ------------------------------------------------------

    def find_by_hash(self, request_hash: str) -> list[Job]:
        """Return all jobs with this request_hash (any status)."""
        with self.session() as s:
            return list(s.query(Job).filter(Job.request_hash == request_hash).order_by(Job.created_at.desc()).all())

    def find_completed_by_hash(self, request_hash: str) -> Job | None:
        """Return the most recent completed job for this hash, or None."""
        with self.session() as s:
            return (
                s.query(Job)
                .filter(
                    Job.request_hash == request_hash,
                    Job.status == "completed",
                    Job.result_data.isnot(None),
                )
                .order_by(Job.created_at.desc())
                .first()
            )

    def search_by_composition(
        self,
        composition: str,
        potential: str | None = None,
    ) -> list[Job]:
        """Find completed jobs matching a normalised composition."""
        with self.session() as s:
            q = s.query(Job).filter(
                Job.composition == composition,
                Job.status == "completed",
            )
            if potential:
                q = q.filter(Job.potential == potential)
            return list(q.order_by(Job.created_at.desc()).all())

    def list_compositions(self) -> list[dict[str, Any]]:
        """Return ``[{composition, n_jobs}, …]`` for all completed jobs."""
        with self.session() as s:
            rows = (
                s.query(Job.composition, func.count(Job.job_id))
                .filter(Job.status == "completed")
                .group_by(Job.composition)
                .all()
            )
            return [{"composition": comp, "n_jobs": n} for comp, n in rows]


# ---------------------------------------------------------------------------
# ASE Atoms serialisation inside result dicts
# ---------------------------------------------------------------------------


def _serialise_atoms_in_result(result: dict) -> dict:
    """Deep-copy *result* and recursively convert non-JSON-serialisable objects."""
    import copy

    import numpy as np
    from ase import Atoms

    def _walk(obj: object) -> object:
        if isinstance(obj, Atoms):
            return serialize_atoms(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(v) for v in obj]
        return _walk(obj.to_dict()) if hasattr(obj, "to_dict") else obj

    return _walk(copy.deepcopy(result))


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_store: JobStore | None = None


def get_job_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore()
    return _store


def init_job_store(db_path: Path | None = None) -> JobStore:
    global _store
    _store = JobStore(db_path)
    return _store


def close_job_store() -> None:
    global _store
    if _store is not None:
        _store.close()
        _store = None
