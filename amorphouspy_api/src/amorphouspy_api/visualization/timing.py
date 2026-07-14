"""Simulation timing helpers.

Reconstructs the executorlib task DAG from the on-disk cache and derives
per-step and workflow-level timings.  Every cached task self-describes its
runtime, core count, and dependency edges (via stored ``FutureItem`` inputs),
so the whole workflow is timed generically with no per-workflow special casing.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _TimingNode:
    """A single executorlib task in the timing DAG.

    ``label`` is the step name derived from the cache filename (``""`` for the
    final merge node).  ``deps`` are the labels of the tasks this one consumed,
    recovered from the stored ``FutureItem`` inputs.
    """

    label: str
    runtime: float
    cores: int
    deps: list[str] = field(default_factory=list)

    @property
    def core_seconds(self) -> float:
        return self.runtime * self.cores


def _format_runtime(seconds: float) -> str:
    """Format runtime as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    hours = seconds / 3600
    return f"{hours:.1f} h"


def _label_from_filename(name: str, request_hash: str) -> str | None:
    """Derive a step label from an executorlib output filename.

    ``{hash}_o.h5`` -> ``""`` (the final merge node), ``{hash}_melt_quench_o.h5``
    -> ``"melt_quench"``.  Returns *None* for files not belonging to *request_hash*.
    """
    if not name.endswith("_o.h5"):
        return None
    core = name[: -len("_o.h5")]
    if core == request_hash:
        return ""
    prefix = f"{request_hash}_"
    if not core.startswith(prefix):
        return None
    return core[len(prefix) :]


def _read_h5_node(h5_path: Path, request_hash: str) -> _TimingNode | None:
    """Read a timing node (runtime, cores, dependency labels) from a cache file.

    Dependency edges are recovered by scanning the pickled task inputs for
    ``FutureItem`` objects (duck-typed via their ``_file_name`` attribute), so
    the workflow DAG is reconstructed without any knowledge of its shape.

    Returns *None* if the file is missing, unreadable, not part of this request,
    or has a non-positive runtime.
    """
    import h5py

    label = _label_from_filename(h5_path.name, request_hash)
    if label is None or not h5_path.exists():
        return None
    try:
        with h5py.File(h5_path, "r") as hdf:
            runtime = pickle.loads(hdf["runtime"][()])  # noqa: S301
            if runtime <= 0:
                return None
            cores = 1
            if "resource_dict" in hdf:
                rd = pickle.loads(hdf["resource_dict"][()])  # noqa: S301
                cores = max(rd.get("cores", 1), rd.get("threads_per_core", 1))
            deps: list[str] = []
            for key in ("input_args", "input_kwargs"):
                if key not in hdf:
                    continue
                raw = pickle.loads(hdf[key][()])  # noqa: S301
                values = raw.values() if isinstance(raw, dict) else raw
                for value in values:
                    file_name = getattr(value, "_file_name", None)
                    if isinstance(file_name, str):
                        dep_label = _label_from_filename(Path(file_name).name, request_hash)
                        if dep_label is not None:
                            deps.append(dep_label)
            return _TimingNode(label=label, runtime=runtime, cores=cores, deps=deps)
    except Exception:
        logger.debug("Could not read cache for %s", h5_path)
        return None


def _load_timing_dag(cache_dir: Path, request_hash: str) -> dict[str, _TimingNode]:
    """Load every cached task for *request_hash* into a ``label -> node`` map."""
    paths = list(cache_dir.glob(f"{request_hash}_*_o.h5"))
    root = cache_dir / f"{request_hash}_o.h5"
    if root.exists():
        paths.append(root)

    nodes: dict[str, _TimingNode] = {}
    for path in paths:
        node = _read_h5_node(path, request_hash)
        if node is not None:
            nodes[node.label] = node
    return nodes


def _critical_path_seconds(nodes: dict[str, _TimingNode]) -> float:
    """Longest dependency chain through *nodes*, weighted by task runtime.

    This is the true wall-clock time: tasks on parallel branches overlap, so
    only the slowest chain counts.  Edges to labels outside *nodes* are ignored,
    which lets the same routine measure a whole workflow or a single subtree.
    """
    memo: dict[str, float] = {}

    def longest(label: str) -> float:
        if label in memo:
            return memo[label]
        node = nodes.get(label)
        if node is None:
            return 0.0
        memo[label] = 0.0  # cycle guard
        upstream = max((longest(dep) for dep in node.deps), default=0.0)
        memo[label] = node.runtime + upstream
        return memo[label]

    return max((longest(label) for label in nodes), default=0.0)


def _fold_steps(nodes: dict[str, _TimingNode]) -> dict[str, tuple[float, float]]:
    """Group sub-jobs under their parent step and time each group.

    A label is a sub-job when a shorter label ``L`` exists such that it starts
    with ``L + "_"``; it is then folded into ``L``.  The final merge node
    (empty label) is excluded from the per-step breakdown but still contributes
    to the workflow-level totals computed elsewhere.
    """
    labels = [label for label in nodes if label]

    def parent_of(label: str) -> str | None:
        parent: str | None = None
        for other in labels:
            if other != label and label.startswith(f"{other}_") and (parent is None or len(other) > len(parent)):
                parent = other
        return parent

    steps: dict[str, tuple[float, float]] = {}
    for label in labels:
        if parent_of(label) is not None:
            continue
        members = [label, *(m for m in labels if m.startswith(f"{label}_"))]
        subtree = {m: nodes[m] for m in members}
        wall = _critical_path_seconds(subtree)
        core_seconds = sum(nodes[m].core_seconds for m in members)
        steps[label] = (wall, core_seconds)
    return steps


def get_step_timings(request_hash: str) -> dict[str, tuple[float, float]]:
    """Return per-step ``(wall_seconds, core_seconds)`` from the executorlib cache.

    The workflow DAG is reconstructed from the cache files, then fan-out steps
    are folded into their parent generically: any task whose label is prefixed
    by a shorter task's label (e.g. ``viscosity_2500K`` under ``viscosity``) is
    grouped with it.  A folded step's wall time is the critical path through its
    own subtree (parallel sub-jobs overlap); its core-seconds sum across the
    subtree.  This needs no per-workflow special casing.

    Returns:
        Dict mapping top-level step label to ``(wall_seconds, core_seconds)``.
    """
    from amorphouspy_api.config import MELTQUENCH_PROJECT_DIR

    nodes = _load_timing_dag(MELTQUENCH_PROJECT_DIR, request_hash)
    return _fold_steps(nodes)


def prepare_timing_context(request_hash: str) -> dict[str, Any]:
    """Build template context for step timings.

    Wall time is the workflow critical path (parallel branches overlap), while
    core hours sum every task's core-seconds.

    Returns:
        Dict with ``step_timings`` (list of ``{name, runtime}``),
        ``total_runtime`` and ``total_core_hours``.
    """
    from amorphouspy_api.config import MELTQUENCH_PROJECT_DIR
    from amorphouspy_api.pipeline import DISPLAY_NAMES

    nodes = _load_timing_dag(MELTQUENCH_PROJECT_DIR, request_hash)
    steps = _fold_steps(nodes)
    if not steps:
        return {}

    order = list(DISPLAY_NAMES)

    def sort_key(label: str) -> tuple[int, str]:
        return (order.index(label) if label in order else len(order), label)

    items = [
        {"name": DISPLAY_NAMES.get(label, label), "runtime": _format_runtime(steps[label][0])}
        for label in sorted(steps, key=sort_key)
    ]

    total_wall = _critical_path_seconds(nodes)
    core_hours = sum(node.core_seconds for node in nodes.values()) / 3600
    return {
        "step_timings": items,
        "total_runtime": _format_runtime(total_wall),
        "total_core_hours": f"{core_hours:.1f} h" if core_hours >= 1 else f"{core_hours * 60:.0f} min",
    }
