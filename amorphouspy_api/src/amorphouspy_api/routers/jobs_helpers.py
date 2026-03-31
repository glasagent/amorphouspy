"""Helpers for the jobs router — hashing, progress tracking, executor interaction."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
from amorphouspy.structure import element_counts_from_formula_units, normalize
from ase.data import chemical_symbols

from amorphouspy_api.config import MELTQUENCH_PROJECT_DIR
from amorphouspy_api.database import get_job_store
from amorphouspy_api.executor import get_executor, get_future_from_cache
from amorphouspy_api.models import (
    JobProgress,
    JobSubmission,
    StepStatus,
)
from amorphouspy_api.workflows import ANALYSES, BASE_STEPS, submit_pipeline

if TYPE_CHECKING:
    from amorphouspy_api.database import Job

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _job_hash(submission: JobSubmission, normalized_comp: str) -> str:
    """Deterministic hash from (normalised composition + potential + sim params + analyses)."""
    analyses_dump = [a.model_dump() for a in submission.analyses]
    payload = json.dumps(
        {
            "composition": normalized_comp,
            "potential": submission.potential.value,
            "simulation": submission.simulation.model_dump(),
            "analyses": analyses_dump,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def composition_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """Euclidean distance between two compositions in elemental atom-fraction space.

    Each dict maps element symbols to atom fractions (values summing to 1).
    Missing elements are treated as 0.
    """
    all_elements = set(a) | set(b)
    return sum((a.get(el, 0.0) - b.get(el, 0.0)) ** 2 for el in all_elements) ** 0.5


# Number of elements in the periodic table (index 0 unused, 1..118 = H..Og).
N_ELEMENTS = len(chemical_symbols)  # 119 (index 0 is dummy '')


def _elem_dict_to_vector(d: dict[str, float]) -> list[float]:
    """Convert a sparse {symbol: fraction} dict to a fixed-length list[float]."""
    vec = [0.0] * N_ELEMENTS
    for el, val in d.items():
        vec[chemical_symbols.index(el)] = val
    return vec


def _vector_to_elem_dict(vec: list[float]) -> dict[str, float]:
    """Convert a fixed-length vector back to a sparse dict (non-zero entries only)."""
    return {chemical_symbols[i]: v for i, v in enumerate(vec) if v != 0.0}


def oxide_to_elemental_vector(oxide_comp: dict[str, float]) -> np.ndarray:
    """Convert oxide mol-% composition to a fixed-length numpy vector."""
    fracs = oxide_to_elemental_fractions(oxide_comp)
    vec = np.zeros(N_ELEMENTS)
    for el, val in fracs.items():
        vec[chemical_symbols.index(el)] = val
    return vec


def _filter_candidates(
    rows: list,
    exclude_ids: set[str],
) -> list[tuple[str, list, str, str, dict | None, object]]:
    """Filter rows from list_completed_vectors, dropping excluded/empty entries."""
    return [
        (job_id, evec, comp, potential, req_data, completed_at)
        for job_id, evec, comp, potential, req_data, completed_at in rows
        if job_id not in exclude_ids and evec is not None
    ]


def find_close_matches(
    query_vec: np.ndarray,
    rows: list,
    *,
    exclude_ids: set[str],
    threshold: float,
    max_results: int,
) -> list[tuple[float, str, str, str, dict | None, object]]:
    """Vectorised close-match search using numpy.

    Args:
        query_vec: Fixed-length (N_ELEMENTS,) numpy array for the query composition.
        rows: Output of ``store.list_completed_vectors()``.
        exclude_ids: Job IDs to skip (already matched exactly).
        threshold: Maximum Euclidean distance to include.
        max_results: Cap on returned matches.

    Returns:
        List of ``(distance, job_id, composition, potential, request_data, completed_at)``
        sorted by ascending distance.
    """
    candidates = _filter_candidates(rows, exclude_ids)
    if not candidates:
        return []

    # Stack pre-aligned vectors directly — no element-index building needed
    mat = np.array([evec for _, evec, *_ in candidates])
    distances = np.linalg.norm(mat - query_vec, axis=1)

    # Filter by threshold and pick top-k
    idxs = np.where(distances <= threshold)[0]
    if len(idxs) == 0:
        return []
    if len(idxs) > max_results:
        top_k = np.argpartition(distances[idxs], max_results)[:max_results]
        idxs = idxs[top_k]
    idxs = idxs[np.argsort(distances[idxs])]

    return [
        (
            float(distances[idx]),
            candidates[idx][0],  # job_id
            candidates[idx][2],  # composition
            candidates[idx][3],  # potential
            candidates[idx][4],  # request_data
            candidates[idx][5],  # completed_at
        )
        for idx in idxs
    ]


def oxide_to_elemental_fractions(oxide_comp: dict[str, float]) -> dict[str, float]:
    """Convert an oxide mol-% composition to normalised elemental atom fractions.

    Reuses :func:`~amorphouspy.structure.element_counts_from_formula_units`
    (which accepts float "formula units" at runtime) and
    :func:`~amorphouspy.structure.normalize`.

    Example:
    -------
    >>> oxide_to_elemental_fractions({"SiO2": 100})
    {'Si': 0.333..., 'O': 0.666...}
    """
    mol_frac = normalize(oxide_comp)
    raw_counts = element_counts_from_formula_units(mol_frac)  # type: ignore[arg-type]
    return normalize(raw_counts)


def elemental_fractions_from_job(job: Job) -> dict[str, float] | None:
    """Extract normalised elemental atom fractions from a completed job.

    Prefers the actual structure (atomic numbers) stored in
    ``result_data["melt_quench"]["final_structure"]``; falls back to the
    requested oxide composition.
    """
    result = job.result_data or {}
    mq = result.get("melt_quench", {})
    struct = mq.get("final_structure")

    if struct and isinstance(struct, dict) and "numbers" in struct:
        counts = Counter(chemical_symbols[z] for z in struct["numbers"])
        return normalize(dict(counts))

    # Fallback: derive from the stored oxide composition
    stored_comp = mq.get("composition")
    if stored_comp and isinstance(stored_comp, dict):
        return oxide_to_elemental_fractions(stored_comp)

    # Last resort: derive from the canonical composition on the job record
    from amorphouspy_api.models import Composition

    return oxide_to_elemental_fractions(Composition.from_canonical(job.composition).root)


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()


def _progress_from_dict(d: dict | None) -> JobProgress:
    """Convert a flat progress dict (from DB) into a JobProgress model."""
    if d is None:
        return JobProgress()
    base: dict[str, StepStatus] = {}
    analyses: dict[str, StepStatus] = {}
    for k, v in d.items():
        sv = StepStatus(v) if v is not None else StepStatus.PENDING
        if k in BASE_STEPS:
            base[k] = sv
        else:
            analyses[k] = sv
    return JobProgress(**base, analyses=analyses)


def _analyses_list(job: Job) -> list[str]:
    """Extract the list of requested analysis type names from a stored job."""
    req = job.request_data or {}
    return [a.get("type", "structure") for a in req.get("analyses", [])]


def _initial_progress(submission: JobSubmission) -> dict[str, str]:
    """Build the initial progress dict based on requested analyses."""
    progress: dict[str, str] = {
        "structure_generation": "pending",
        "melt_quench": "pending",
    }
    for a in submission.analyses:
        progress[a.type] = "pending"
    return progress


# ---------------------------------------------------------------------------
# Executor interaction
# ---------------------------------------------------------------------------


def _resolve_future(future, job_id: str) -> dict:
    """Extract state from an executorlib future."""
    if not future.done():
        return {"status": "running"}
    exc = future.exception()
    if exc is not None:
        logger.error("Job %s failed: %s", job_id, exc)
        return {"status": "failed", "error": str(exc)}
    return {"status": "completed", "result": future.result()}


def _submit_to_executor(
    submission: JobSubmission,
    job_id: str,
    request_hash: str,
) -> dict:
    """Submit the full pipeline to executorlib and return resolved status dict."""
    exe = get_executor(cache_directory=MELTQUENCH_PROJECT_DIR)
    future = submit_pipeline(exe, submission, cache_key=request_hash)
    resolved = _resolve_future(future, job_id)
    exe.shutdown(wait=False, cancel_futures=False)
    return resolved


def _update_from_resolved(job_id: str, resolved: dict, submission: JobSubmission | None = None) -> None:
    """Persist pipeline outcome into the job store."""
    store = get_job_store()
    status = resolved["status"]

    # Build the expected step list
    all_steps = list(BASE_STEPS)
    if submission:
        all_steps.extend(a.type for a in submission.analyses if a.type in ANALYSES)

    if status == "completed":
        result = resolved["result"]
        progress = dict.fromkeys(all_steps, "completed")

        # Compute elemental atom-fraction vector from the result
        from amorphouspy_api.database import Job as _Job

        _tmp = _Job(result_data=result, composition=submission.composition.canonical if submission else "")
        evec_dict = elemental_fractions_from_job(_tmp)
        evec = _elem_dict_to_vector(evec_dict) if evec_dict else None

        store.update_job(
            job_id,
            status="completed",
            progress=progress,
            result_data=result,
            elemental_vector=evec,
            completed_at=datetime.now(UTC),
        )
    elif status == "failed":
        progress = dict.fromkeys(all_steps, "failed")
        store.update_job(
            job_id,
            status="failed",
            progress=progress,
            errors={"pipeline": resolved.get("error", "unknown error")},
        )
    else:
        progress = dict.fromkeys(all_steps, "pending")
        progress["structure_generation"] = "running"
        store.update_job(job_id, status="running", progress=progress)


def _probe_step_caches(
    job: Job,
    all_steps: list[str],
) -> tuple[dict[str, str], dict[str, object], bool]:
    """Probe per-step caches and return (progress, partial_results, has_failure)."""
    cache_dir = str(MELTQUENCH_PROJECT_DIR)
    cache_key = job.request_hash
    progress: dict[str, str] = {}
    partial_results: dict[str, object] = dict(job.result_data or {})
    has_failure = False
    for step_name in all_steps:
        step_key = f"{cache_key}_{step_name}"
        try:
            step_future = get_future_from_cache(
                cache_directory=cache_dir,
                cache_key=step_key,
            )
            step_resolved = _resolve_future(step_future, job.job_id)
            progress[step_name] = step_resolved["status"]
            if step_resolved["status"] == "completed":
                # Each step result is already a dict keyed by step name(s),
                # e.g. _accumulate_step → {"structure_generation": ..., "melt_quench": ...}
                #       _run_analysis   → {"structure": ...}
                partial_results.update(step_resolved["result"])
            elif step_resolved["status"] == "failed":
                has_failure = True
        except FileNotFoundError:
            progress[step_name] = "pending"
        except Exception:
            logger.exception("Job %s step %s failed (cached error)", job.job_id, step_name)
            progress[step_name] = "failed"
            has_failure = True
    return progress, partial_results, has_failure


def refresh_job_from_cache(job: Job) -> None:
    """Re-check executor cache for a running job and update DB if resolved.

    Checks the merged result first (fast path).  If that is not ready yet,
    probes each step's individual cache key for granular progress.
    """
    cache_dir = str(MELTQUENCH_PROJECT_DIR)
    cache_key = job.request_hash

    # Fast path — final merged result available?
    try:
        future = get_future_from_cache(
            cache_directory=cache_dir,
            cache_key=cache_key,
        )
        resolved = _resolve_future(future, job.job_id)
        if resolved["status"] in ("completed", "failed"):
            submission = JobSubmission(**job.request_data) if job.request_data else None
            _update_from_resolved(job.job_id, resolved, submission)
            return
    except FileNotFoundError:
        pass
    except Exception as exc:
        # get_future_from_cache raises stored errors directly; treat as failed.
        logger.exception("Job %s failed (cached error)", job.job_id)
        submission = JobSubmission(**job.request_data) if job.request_data else None
        resolved = {"status": "failed", "error": str(exc)}
        _update_from_resolved(job.job_id, resolved, submission)
        return

    # Slow path — probe per-step cache keys
    submission = JobSubmission(**job.request_data) if job.request_data else None
    all_steps = list(BASE_STEPS)
    if submission:
        all_steps.extend(a.type for a in submission.analyses if a.type in ANALYSES)

    progress, partial_results, has_failure = _probe_step_caches(job, all_steps)

    store = get_job_store()
    updates: dict[str, object] = {"progress": progress}
    if partial_results:
        updates["result_data"] = partial_results
    if has_failure:
        updates["status"] = "failed"
    else:
        updates["status"] = "running"
    store.update_job(job.job_id, **updates)


def _add_optional_analyses(context: dict, result_data: dict) -> None:
    """Populate context with viscosity / CTE / elastic plots if available."""
    import json

    from amorphouspy_api.workflows.analyses.cte import prepare_cte_plots
    from amorphouspy_api.workflows.analyses.elastic import prepare_elastic_plots
    from amorphouspy_api.workflows.analyses.viscosity import prepare_viscosity_plots

    visc_data = result_data.get("viscosity")
    if visc_data:
        context["viscosity_plots"] = prepare_viscosity_plots(visc_data)

    cte_data = result_data.get("cte")
    if cte_data:
        context["cte_plots"] = prepare_cte_plots(cte_data)
        summary = cte_data.get("summary")
        if summary:
            context["cte_summary"] = json.dumps(summary)

    elastic_data = result_data.get("elastic")
    if elastic_data:
        context["elastic_plots"] = prepare_elastic_plots(elastic_data)
        moduli = elastic_data.get("moduli")
        if moduli:
            context["elastic_moduli"] = moduli


def build_visualization_context(job_id: str, result_data: dict, *, request_hash: str = "") -> dict:
    """Build the Jinja2 template context from job result data.

    Handles partial results gracefully — sections whose data is not yet
    available are simply omitted from the context so the template can
    conditionally skip them.
    """
    from amorphouspy_api.workflows.analyses.meltquench_viz import (
        build_temperature_time_plot,
        prepare_timing_context,
    )
    from amorphouspy_api.workflows.analyses.structure import prepare_structure_context

    mq = result_data.get("melt_quench", {})

    context: dict[str, object] = {"job_id": job_id}

    _STRUCTURE_DEFAULTS: dict[str, object] = {
        "plotly_json": None,
        "structure_xyz": "",
        "density": "N/A",
        "network_connectivity": "N/A",
        "network_formers": [],
        "modifiers": [],
    }

    # --- Structure analysis (only when available) ---
    if result_data.get("structure"):
        try:
            context.update(prepare_structure_context(result_data))
        except Exception:
            logger.warning("Could not render structure analysis for job %s", job_id)
            context.update(_STRUCTURE_DEFAULTS)
    else:
        context.update(_STRUCTURE_DEFAULTS)

    # Melt-quench metadata
    mean_temperature = mq.get("mean_temperature", "N/A")
    if isinstance(mean_temperature, (int, float)):
        mean_temperature = f"{mean_temperature:.1f}"
    simulation_steps = mq.get("simulation_steps", "N/A")
    if isinstance(simulation_steps, int):
        simulation_steps = f"{simulation_steps:,}"

    context.update(
        {
            "composition": mq.get("composition", "N/A"),
            "mean_temperature": mean_temperature,
            "simulation_steps": simulation_steps,
        }
    )

    # --- Step timings from executorlib cache ---
    if request_hash:
        timing_ctx = prepare_timing_context(request_hash)
        context.update(timing_ctx)

    # --- Temperature-time diagram ---
    if mq:
        tt_plot = build_temperature_time_plot(mq)
        if tt_plot:
            context["temperature_time_plot"] = tt_plot

    # --- Optional analyses ---
    _add_optional_analyses(context, result_data)

    return context
