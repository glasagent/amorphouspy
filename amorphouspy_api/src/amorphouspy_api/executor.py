"""Job submission utilities for amorphouspy API.

This module provides utilities for selecting and configuring executorlib executors
(TestClusterExecutor for local or SlurmClusterExecutor for SLURM).

Both executors use wait=False to allow non-blocking exit from the context manager,
enabling the API to check job status without blocking.

Configure via environment variables:
    EXECUTOR_TYPE: "local" (default) or "slurm"
    EXECUTOR_CORES: Number of cores per worker (default: 4)
    LAMMPS_CORES: Number of cores for LAMMPS simulations (default: EXECUTOR_CORES or 4)
    SLURM_PARTITION: SLURM partition name (optional, slurm only)
    SLURM_TIME: SLURM time limit (optional, slurm only)
"""

import logging
import os
from pathlib import Path
from typing import Any

import executorlib
from executorlib import get_future_from_cache  # noqa: F401 — re-exported
from executorlib.api import TestClusterExecutor

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
    executor_type = os.environ.get("EXECUTOR_TYPE", "local").lower()

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


def get_executor_config() -> dict[str, Any]:
    """Build executor configuration from environment variables.

    Returns:
        Dictionary of executor configuration options.
    """
    config: dict[str, Any] = {}
    cores = os.environ.get("EXECUTOR_CORES")
    if cores:
        config["cores_per_worker"] = int(cores)

    # SLURM-specific config
    if os.environ.get("EXECUTOR_TYPE", "local").lower() == "slurm":
        if os.environ.get("SLURM_PARTITION"):
            config["partition"] = os.environ["SLURM_PARTITION"]
        if os.environ.get("SLURM_TIME"):
            config["time"] = os.environ["SLURM_TIME"]

    return config


def get_lammps_resource_dict() -> dict[str, Any]:
    """Get resource dictionary for LAMMPS simulations.

    Returns:
        Dictionary with LAMMPS-specific resource settings.
    """
    cores = int(os.environ.get("LAMMPS_CORES", os.environ.get("EXECUTOR_CORES", "4")))
    return {"cores": cores}


def get_executor(cache_directory: Path) -> executorlib.BaseExecutor:
    """Create a fresh executor instance.

    Args:
        cache_directory: Directory for executor disk cache.

    Returns:
        The executor instance.
    """
    # Create new executor each time to properly detect cached results
    executor_class = get_executor_class()
    executor_config = get_executor_config()

    logger.info(
        "Creating executor: %s with cache_directory=%s",
        executor_class.__name__,
        cache_directory,
    )

    return executor_class(cache_directory=cache_directory, **executor_config)
