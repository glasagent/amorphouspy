"""Job submission utilities for amorphouspy API.

This module provides utilities for selecting and configuring executorlib executors
(TestClusterExecutor for local or SlurmClusterExecutor for SLURM).

Both executors use wait=False to allow non-blocking exit from the context manager,
enabling the API to check job status without blocking.

Configure via environment variables:
    EXECUTOR_TYPE: "test" (default), "slurm", "flux", or "single"
    LAMMPS_MAX_CORES: Maximum number of MPI cores available for a single LAMMPS
        simulation (e.g. the cores on a node or the per-job limit in the SLURM
        queue). The actual core count is scaled down from this maximum when the
        workload is too small to use it efficiently (see ``compute_lammps_cores``).
        Defaults to 4.
    SLURM_PARTITION: SLURM partition name (optional, slurm only)
    SLURM_RUN_TIME_MAX: Max run time per job in seconds (optional, slurm only)
    SLURM_MEMORY_MAX: Max memory per job in GB (optional, slurm only)

For advanced SLURM customization, place a Jinja2 submission template at
``<AMORPHOUSPY_PROJECTS>/submission_template.sh``. If present, it is
automatically used for all SLURM job submissions.
"""

import logging
import os
from pathlib import Path
from typing import Any

import executorlib
from executorlib import get_future_from_cache  # noqa: F401 — re-exported
from executorlib.api import TestClusterExecutor

from amorphouspy_api.config import PROJECTS_FOLDER

logger = logging.getLogger(__name__)


def get_max_cores() -> int:
    """Return the system-wide maximum number of cores for a LAMMPS simulation.

    Read from ``LAMMPS_MAX_CORES``, defaulting to 4. Always at least 1.
    """
    return max(1, int(os.environ.get("LAMMPS_MAX_CORES", "4")))


def compute_lammps_cores(potential: str, n_atoms: int, cores: int | None = None) -> int:
    """Pick the number of MPI cores for a LAMMPS run.

    When *cores* is given it is honoured directly (clamped to at least 1); a
    warning is logged if it exceeds the system maximum. Otherwise the system
    maximum (:func:`get_max_cores`) is used unless that would push the workload
    below the potential's minimum atoms-per-core, in which case the core count
    is scaled down to preserve scaling efficiency. The result is always at
    least 1.

    Args:
        potential: Potential identifier (e.g. ``"pmmcs"``, ``"shik"``).
        n_atoms: Number of atoms in the simulation cell.
        cores: Explicit job-level core count override. ``None`` means
            auto-select.

    Returns:
        The chosen core count.
    """
    if cores is not None:
        chosen = max(1, cores)
        max_cores = get_max_cores()
        if chosen > max_cores:
            logger.warning(
                "Requested cores=%d exceeds LAMMPS_MAX_CORES=%d; using %d anyway.",
                chosen,
                max_cores,
                chosen,
            )
        return chosen

    from amorphouspy.lammps.potentials.potential import get_min_atoms_per_core

    max_cores = get_max_cores()
    min_atoms_per_core = get_min_atoms_per_core(potential)
    # floor keeps the realised atoms/core at or above the minimum.
    efficient_cores = max(1, n_atoms // min_atoms_per_core)
    return min(max_cores, efficient_cores)


def get_executor_class() -> type:
    """Get the appropriate executor class based on environment.

    Note: the executor classes behave differently with respect to cache and `wait`ing:
    - Only the SlurmClusterExecutor and the FluxClusterExecutor support cache and `wait`ing as expected
    - SingleNodeExecutor: uses socket-based communication, so cache is created only once results are computed
      and calling `get_future_from_cache` earlier results in `FileNotFoundError`.
      Executor will wait for futures to be done when exiting the executor context.
    - TestClusterExecutor: uses Python's `subprocess` module which does not provide task dependency management.
      When chaining futures, the next future is thus submitted only once the previous one is completed

    Returns:
        BaseExecutor subclass based on environment.
    """
    executor_type = os.environ.get("EXECUTOR_TYPE", "test").lower()

    executor_classes = {
        "slurm": executorlib.SlurmClusterExecutor,
        "flux": executorlib.FluxClusterExecutor,
        "single": executorlib.SingleNodeExecutor,
        "test": TestClusterExecutor,
    }

    if executor_type not in executor_classes:
        msg = f"Unknown EXECUTOR_TYPE '{executor_type}'. Valid options are: {list(executor_classes.keys())}"
        raise ValueError(msg)

    return executor_classes[executor_type]


def _is_slurm() -> bool:
    """Return True when the executor is configured for SLURM."""
    return os.environ.get("EXECUTOR_TYPE", "test").lower() == "slurm"


def get_base_resource_dict() -> dict[str, Any]:
    """Get base resource dictionary shared by all steps.

    SLURM-specific keys (partition, time limits, submission template) are
    only included when ``EXECUTOR_TYPE=slurm``.
    """
    resource_dict: dict[str, Any] = {}
    if not _is_slurm():
        return resource_dict
    if os.environ.get("SLURM_PARTITION"):
        resource_dict["partition"] = os.environ["SLURM_PARTITION"]
    if os.environ.get("SLURM_RUN_TIME_MAX"):
        resource_dict["run_time_limit"] = int(os.environ["SLURM_RUN_TIME_MAX"])
    if os.environ.get("SLURM_MEMORY_MAX"):
        resource_dict["memory_max"] = int(os.environ["SLURM_MEMORY_MAX"])
    template_path = PROJECTS_FOLDER / "submission_template.sh"
    if template_path.is_file():
        resource_dict["submission_template"] = template_path.read_text()
    return resource_dict


def get_lammps_resource_dict(potential: str, n_atoms: int, cores: int | None = None) -> dict[str, Any]:
    """Get resource dictionary for LAMMPS simulations.

    On SLURM, uses ``threads_per_core`` so that executorlib runs a **single**
    Python process while SBATCH still allocates enough CPUs for LAMMPS's
    internal ``mpiexec -n <cores>``.  On other executors the dict is empty
    (LAMMPS core count is handled via ``get_lammps_server_kwargs`` instead).

    Args:
        potential: Potential identifier used to pick the minimum atoms-per-core.
        n_atoms: Number of atoms in the simulation cell.
        cores: Explicit job-level core count override (``None`` = auto-select).

    Returns:
        Dictionary with LAMMPS-specific resource settings.
    """
    base = get_base_resource_dict()
    if _is_slurm():
        base["threads_per_core"] = compute_lammps_cores(potential, n_atoms, cores)
    return base


def get_lammps_server_kwargs(potential: str, n_atoms: int, cores: int | None = None) -> dict[str, int]:
    """Get the ``server_kwargs`` for amorphouspy LAMMPS functions.

    Returns ``{"cores": N}`` as expected by
    :func:`amorphouspy.workflows.shared.get_lammps_command`, where ``N`` is
    chosen by :func:`compute_lammps_cores` from the explicit *cores* override
    or, when ``None``, the system maximum and the potential's minimum
    atoms-per-core.
    """
    return {"cores": compute_lammps_cores(potential, n_atoms, cores)}


def get_executor(cache_directory: Path) -> executorlib.BaseExecutor:
    """Create a fresh executor instance.

    Args:
        cache_directory: Directory for executor disk cache.

    Returns:
        The executor instance.
    """
    # Create new executor each time to properly detect cached results
    executor_class = get_executor_class()

    logger.info(
        "Creating executor: %s with cache_directory=%s",
        executor_class.__name__,
        cache_directory,
    )

    return executor_class(cache_directory=cache_directory)
