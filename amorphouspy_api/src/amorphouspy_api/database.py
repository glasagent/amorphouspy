"""Database models and store for the amorphouspy API.

Single ``Job`` table backs both ``/jobs`` and ``/glasses`` endpoints.

Trajectory dataflow overview:
1. The simulation pipeline computes structural analyses from full in-memory
    trajectories.
2. On persistence, ``result_data['melt_quench']['simulation_history']`` is
    extracted into the dedicated ``Job.simulation_history`` column.
3. Storage policy of trajectory (dump) data is controlled by ``trajectory_storage_mode``:
    - ``"all_frames_all_data"``
    - ``"all_frames_drop_velocities_and_forces"``
    - ``"last_frame_all_data"``
    - ``"last_frame_drop_velocities_and_forces"``
    For the melt-quench workflow, the selected mode is applied independently to
    every stage entry in ``simulation_history``.
4. ``result_data`` is kept lightweight and no longer carries the trajectory.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from sqlalchemy import JSON, DateTime, Index, String, create_engine, func, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, load_only, mapped_column, sessionmaker
from sqlalchemy.orm import defer as _defer
from sqlalchemy.pool import NullPool

from .models import (
    MeltQuenchTrajectoryStorageMode,
    StructuralAnalysisTrajectoryStorageMode,
    serialize_atoms,
)

TrajectoryStorageMode = MeltQuenchTrajectoryStorageMode | StructuralAnalysisTrajectoryStorageMode | str

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base class."""


class Job(Base):
    """A simulation job.

    The row stores both request/analysis metadata and optional heavy trajectory
    payload. Large trajectory data is isolated in ``simulation_history`` so
    routine status/listing queries can avoid deserialising megabytes of JSON.
    """

    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # Deterministic hash of (normalised composition + potential + simulation params)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # Canonical composition string (via Composition.canonical)
    composition: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    potential: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")

    # Full JobSubmission.model_dump()
    request_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Per-step progress: {"structure_generation": "completed", …}
    progress: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Analysis results keyed by type: {"structure_characterization": {…}}
    result_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Errors keyed by step name: {"viscosity": "…"}
    errors: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Fixed-length elemental atom-fraction vector (length 119, indexed by Z).
    # Populated on completion from the actual structure.
    elemental_vector: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # User-defined tags for labelling / grouping jobs (e.g. project names).
    tags: Mapped[list | None] = mapped_column(JSON, nullable=True)

    # Heavy trajectory data is extracted from result_data and stored here.
    # Depending on persistence settings this contains either full geometry
    # history or only the final frame geometry; both keep thermo/step series.
    simulation_history: Mapped[list | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_jobs_composition_status", "composition", "status"),
        Index("ix_jobs_hash_status", "request_hash", "status"),
        # Covering index for search_jobs.  Led by ``created_at`` so the
        # default ``ORDER BY created_at DESC`` (and created_after/before
        # range filters) is served by a backward index scan, while also
        # covering every column the search returns.  Because ``created_at``
        # and ``completed_at`` are stored *after* the huge JSON blob columns
        # in each row record, reading them from the table would force SQLite
        # to walk hundreds of MB of overflow pages per row; this covering
        # index avoids touching the table entirely.
        Index(
            "ix_jobs_search",
            "created_at",
            "job_id",
            "composition",
            "potential",
            "status",
            "tags",
            "progress",
            "completed_at",
        ),
        # Covering index for GET /glasses - avoids reading huge result_data
        # overflow pages on every row scan.
        Index(
            "ix_jobs_glasses_listing",
            "status",
            "composition",
            "created_at",
            "job_id",
            "potential",
            "tags",
            "completed_at",
            "progress",
        ),
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
        self._ensure_indexes()
        logger.info("Initialised job store: %s", db_path)

    def _ensure_indexes(self) -> None:
        """Create indexes that may be missing on a pre-existing database.

        ``Base.metadata.create_all`` only creates indexes when it first
        creates the table, so indexes added to the model later never appear
        on an already-populated database.  Creating them explicitly and
        idempotently keeps searches fast on large databases.  The first run
        on a big database can take a while to build the covering index (it
        must scan every row once); subsequent runs are no-ops thanks to
        ``IF NOT EXISTS``.
        """
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_jobs_search "
                    "ON jobs (created_at, job_id, composition, potential, "
                    "status, tags, progress, completed_at)"
                )
            )
            # Superseded by the covering ix_jobs_search above, which serves
            # the same ordering / range filters without any table lookups.
            conn.execute(text("DROP INDEX IF EXISTS ix_jobs_created_at"))

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
        """Fetch a job by ID without loading heavy trajectory data.

        Use this for regular status/result endpoints where ``simulation_history``
        is not needed.
        """
        with self.session() as s:
            return s.query(Job).options(_defer(Job.simulation_history)).filter(Job.job_id == job_id).first()

    def get_job_with_history(self, job_id: str) -> Job | None:
        """Fetch a job by ID including ``simulation_history``.

        Use this only for trajectory-heavy endpoints.
        """
        with self.session() as s:
            return s.get(Job, job_id)

    def get_raw_simulation_history(self, job_id: str) -> str | None:
        """Fetch the simulation_history column as raw JSON text.

        This bypasses SQLAlchemy's JSON deserialization to avoid the
        costly json.loads() -> json.dumps() round-trip for large payloads.
        Returns None if the job doesn't exist or has no trajectory.
        """
        with self.session() as s:
            row = s.execute(
                text("SELECT simulation_history FROM jobs WHERE job_id = :jid"),
                {"jid": job_id},
            ).fetchone()
        if row is None or row[0] is None:
            return None
        return row[0]

    def update_job(
        self,
        job_id: str,
        *,
        trajectory_storage_mode: MeltQuenchTrajectoryStorageMode = MeltQuenchTrajectoryStorageMode.LAST_FRAME_ALL_DATA,
        **fields: object,
    ) -> None:
        """Update arbitrary columns on a job record.

        For ``result_data`` updates, this method also applies the melt-quench
        trajectory persistence dataflow:
        1. Serialize non-JSON objects (e.g. ASE Atoms).
        2. Extract ``melt_quench.simulation_history`` out of ``result_data`` and
           persist it in ``Job.simulation_history``.
        3. Apply trajectory storage policy controlled by
            ``trajectory_storage_mode`` for the melt-quench workflow,
            independently to each stage in ``simulation_history``.
        """
        with self.session() as s:
            job = s.get(Job, job_id)
            if job is None:
                return
            for k, val in fields.items():
                # Serialise ASE Atoms if present inside result_data
                effective = _serialise_atoms_in_result(val) if k == "result_data" and isinstance(val, dict) else val
                # Extract simulation_history from result_data into its own column.
                # Unless explicitly requested, avoid storing full structure
                # trajectories in the DB. Keep only the last frame so one
                # representative structure is always available.
                if k == "result_data" and isinstance(effective, dict):
                    mq = cast("dict[str, Any]", effective).get("melt_quench")
                    if isinstance(mq, dict) and "simulation_history" in mq:
                        history = mq.pop("simulation_history")
                        if isinstance(history, list):
                            history = _apply_trajectory_storage_mode(
                                history,
                                trajectory_storage_mode=trajectory_storage_mode,
                            )
                        job.simulation_history = history
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
        statuses: list[str] | None = None,
        *,
        light: bool = False,
    ) -> list[Job]:
        """Find jobs matching a normalised composition, optionally filtered by status.

        The heavy ``simulation_history`` (trajectory) column is never loaded.
        When *light* is True, only the lightweight columns needed to build
        search results are loaded (``result_data`` is skipped too), keeping
        the query fast on very large databases.
        """
        with self.session() as s:
            q = s.query(Job)
            if light:
                q = q.options(
                    load_only(
                        Job.job_id,
                        Job.composition,
                        Job.potential,
                        Job.status,
                        Job.tags,
                        Job.progress,
                        Job.created_at,
                        Job.completed_at,
                    )
                )
            else:
                q = q.options(_defer(Job.simulation_history))
            q = q.filter(Job.composition == composition)
            if statuses:
                q = q.filter(Job.status.in_(statuses))
            if potential:
                q = q.filter(Job.potential == potential)
            return list(q.order_by(Job.created_at.desc()).all())

    def search_jobs(
        self,
        composition: str | None = None,
        potential: str | None = None,
        statuses: list[str] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[Job]:
        """Search jobs with optional filters.

        Only loads the lightweight columns needed to build search results;
        the requested analyses are derived from the small ``progress`` column
        instead of the heavy ``request_data`` blob.  All loaded columns are
        covered by ``ix_jobs_glasses_listing``, so the heavy JSON columns
        (``request_data``, ``result_data`` and ``simulation_history``, which
        can be hundreds of MB per row) are never touched — keeping the query
        fast even on very large databases with no filters applied.
        """
        with self.session() as s:
            q = s.query(Job).options(
                load_only(
                    Job.job_id,
                    Job.composition,
                    Job.potential,
                    Job.status,
                    Job.tags,
                    Job.progress,
                    Job.created_at,
                    Job.completed_at,
                )
            )
            if composition:
                q = q.filter(Job.composition == composition)
            if statuses:
                q = q.filter(Job.status.in_(statuses))
            if potential:
                q = q.filter(Job.potential == potential)
            if created_after is not None:
                q = q.filter(Job.created_at >= created_after)
            if created_before is not None:
                q = q.filter(Job.created_at <= created_before)
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

    def list_glasses(self, tag: str | None = None) -> list[dict[str, Any]]:
        """Return lightweight summaries for all completed jobs.

        Only fetches the columns needed for the listing — the heavy
        ``result_data`` and ``elemental_vector`` columns are never
        touched.  Completed analysis names are derived from the small
        ``progress`` JSON column instead.
        """
        _NON_ANALYSIS_STEPS = {"structure_generation", "melt_quench"}

        with self.session() as s:
            q = s.query(
                Job.job_id,
                Job.composition,
                Job.potential,
                Job.tags,
                Job.completed_at,
                Job.progress,
            ).filter(Job.status == "completed")
            if tag is not None:
                q = q.filter(Job.tags.isnot(None))
            rows = q.order_by(Job.composition, Job.created_at.desc()).all()

            results = []
            for job_id, comp, potential, tags, completed_at, progress in rows:
                tag_list = tags or []
                if tag is not None and tag not in tag_list:
                    continue
                analyses = [k for k, v in (progress or {}).items() if k not in _NON_ANALYSIS_STEPS and v == "completed"]
                results.append(
                    {
                        "job_id": job_id,
                        "composition": comp,
                        "potential": potential,
                        "tags": tag_list,
                        "analyses": analyses,
                        "completed_at": completed_at,
                    }
                )
            return results

    def list_completed_vectors(
        self,
        potential: str | None = None,
    ) -> list[tuple[str, list | None, str, str, dict | None, datetime | None]]:
        """Return (job_id, elemental_vector, composition, potential, request_data, completed_at) for fuzzy search.

        Only includes completed jobs that have an elemental_vector stored.
        """
        with self.session() as s:
            q = s.query(
                Job.job_id,
                Job.elemental_vector,
                Job.composition,
                Job.potential,
                Job.request_data,
                Job.completed_at,
            ).filter(
                Job.status == "completed",
                Job.elemental_vector.isnot(None),
            )
            if potential:
                q = q.filter(Job.potential == potential)
            rows = q.order_by(Job.created_at.desc()).all()
            return cast("list[tuple[str, list | None, str, str, dict | None, datetime | None]]", rows)

    def get_tags_for_jobs(self, job_ids: list[str]) -> dict[str, list[str]]:
        """Return {job_id: tags} for the given IDs."""
        if not job_ids:
            return {}
        with self.session() as s:
            rows = s.query(Job.job_id, Job.tags).filter(Job.job_id.in_(job_ids)).all()
            return {jid: (tags or []) for jid, tags in rows}


# ---------------------------------------------------------------------------
# ASE Atoms serialisation inside result dicts
# ---------------------------------------------------------------------------


def _apply_trajectory_storage_mode(
    history: list[Any],
    *,
    trajectory_storage_mode: TrajectoryStorageMode,
) -> list[Any]:
    """Transform simulation history according to persistence mode.

    Args:
        history: Stage-wise melt-quench history.
        trajectory_storage_mode: Explicit storage mode for simulation history,
            applied independently to each stage entry for the melt-quench
            workflow.

    Returns:
        A transformed history according to selected modes.

    Notes:
        Default behaviour (``trajectory_storage_mode="last_frame_all_data"``)
        reduces configured frame-series keys to their final entry per stage.
    """
    _validate_trajectory_storage_mode(trajectory_storage_mode)

    trajectory_storage_mode_value = str(trajectory_storage_mode)

    if trajectory_storage_mode_value == MeltQuenchTrajectoryStorageMode.ALL_FRAMES_ALL_DATA.value:
        return history

    if trajectory_storage_mode_value == MeltQuenchTrajectoryStorageMode.ALL_FRAMES_DROP_VELOCITIES_AND_FORCES.value:
        return _drop_velocities_and_forces(history)

    if trajectory_storage_mode_value == MeltQuenchTrajectoryStorageMode.LAST_FRAME_DROP_VELOCITIES_AND_FORCES.value:
        without_force_velocity = _drop_velocities_and_forces(history)
        return _reduce_selected_keys_to_last_frame_per_stage(without_force_velocity)

    return _reduce_selected_keys_to_last_frame_per_stage(history)


def _validate_trajectory_storage_mode(
    trajectory_storage_mode: TrajectoryStorageMode,
) -> None:
    allowed_values = {
        MeltQuenchTrajectoryStorageMode.ALL_FRAMES_ALL_DATA.value,
        MeltQuenchTrajectoryStorageMode.ALL_FRAMES_DROP_VELOCITIES_AND_FORCES.value,
        MeltQuenchTrajectoryStorageMode.LAST_FRAME_ALL_DATA.value,
        MeltQuenchTrajectoryStorageMode.LAST_FRAME_DROP_VELOCITIES_AND_FORCES.value,
    }
    if str(trajectory_storage_mode) not in allowed_values:
        msg = f"Invalid trajectory_storage_mode={trajectory_storage_mode!r}; expected one of {sorted(allowed_values)}"
        raise ValueError(msg)


def _drop_velocities_and_forces(history: list[Any]) -> list[Any]:
    """Drop per-stage ``velocities`` and ``forces`` while preserving all other data."""
    filtered: list[Any] = []
    for stage in history:
        if not isinstance(stage, dict):
            filtered.append(stage)
            continue
        filtered.append({k: v for k, v in stage.items() if k not in {"forces", "velocities"}})
    return filtered


def _reduce_selected_keys_to_last_frame_per_stage(history: list[Any]) -> list[Any]:
    """Reduce dump-related frame-series to their last entry on every stage.

    Keys are reduced only when present and list-like with at least one value.
    Non-stage entries and all other keys are preserved unchanged.
    """
    keys_to_reduce = {
        "natoms",
        "cells",
        "indices",
        "forces",
        "velocities",
        "unwrapped_positions",
        "positions",
    }

    reduced: list[Any] = []
    for stage in history:
        if not isinstance(stage, dict):
            reduced.append(stage)
            continue

        stage_copy = dict(stage)
        for key in keys_to_reduce:
            value = stage_copy.get(key)
            if isinstance(value, list) and value:
                stage_copy[key] = [value[-1]]
        reduced.append(stage_copy)
    return reduced


def _serialise_atoms_in_result(result: dict) -> dict:
    """Deep-copy *result* and recursively convert non-JSON-serialisable objects."""
    import copy
    from typing import Any

    import numpy as np
    from ase import Atoms

    def _walk(obj: Any) -> Any:  # noqa: ANN401
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
