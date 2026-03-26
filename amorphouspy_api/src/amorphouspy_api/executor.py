"""Job submission utilities for amorphouspy API.

This module provides utilities for selecting and configuring executorlib executors
(TestClusterExecutor for local or SlurmClusterExecutor for SLURM).

Both executors use wait=False to allow non-blocking exit from the context manager,
enabling the API to check job status without blocking.

Configure via environment variables:
    EXECUTOR_TYPE: "test" (default), "slurm", "flux", or "single"
    LAMMPS_CORES: Number of MPI cores for LAMMPS simulations (default: 4)
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


def get_executor_class() -> type:
    """Get the appropriate executor class based on environment.

    Note: the executor classes behave differently with respect to cache and `wait`ing:
    - Only the SlurmClusterExecutor and the FluxClusterExecutor support cache and `wait`ing as expected
    - SingleNodeExecutor: uses socket-based communication, so cache is created only once results are computed
      and calling `get_future_from_cache` earlier results in `FileNotFoundError`
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


def get_lammps_resource_dict() -> dict[str, Any]:
    """Get resource dictionary for LAMMPS simulations.

    These are passed as ``resource_dict`` to ``executor.submit()`` and control
    the SLURM job allocation for compute-intensive LAMMPS steps.

    Returns:
        Dictionary with LAMMPS-specific resource settings.
    """
    resource_dict: dict[str, Any] = {
        "cores": int(os.environ.get("LAMMPS_CORES", "4")),
    }
    if os.environ.get("SLURM_PARTITION"):
        resource_dict["partition"] = os.environ["SLURM_PARTITION"]
    if os.environ.get("SLURM_RUN_TIME_MAX"):
        resource_dict["run_time_max"] = int(os.environ["SLURM_RUN_TIME_MAX"])
    if os.environ.get("SLURM_MEMORY_MAX"):
        resource_dict["memory_max"] = int(os.environ["SLURM_MEMORY_MAX"])
    template_path = PROJECTS_FOLDER / "submission_template.sh"
    if template_path.is_file():
        resource_dict["submission_template"] = template_path.read_text()
    return resource_dict


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
