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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from executorlib.api import TestClusterExecutor

logger = logging.getLogger(__name__)


def get_executor_class() -> type:
    """Get the appropriate executor class based on environment.

    Returns:
        TestClusterExecutor (local) or SlurmClusterExecutor class.
    """
    executor_type = os.environ.get("EXECUTOR_TYPE", "local").lower()

    if executor_type == "slurm":
        from executorlib import SlurmClusterExecutor

        return SlurmClusterExecutor
    elif executor_type == "flux":
        from executorlib import FluxClusterExecutor

        return FluxClusterExecutor
    else:
        # Use TestClusterExecutor for local - it supports wait=False
        # (SingleNodeExecutor does not support wait=False)
        # from executorlib.api import TestClusterExecutor

        # return TestClusterExecutor
        from executorlib import SingleNodeExecutor

        return SingleNodeExecutor


def get_executor_config() -> dict[str, Any]:
    """Build executor configuration from environment variables.

    Returns:
        Dictionary of executor configuration options.
    """
    config: dict[str, Any] = {}

    # Common config: allow non-blocking exit (recommended by executorlib author)
    config["wait"] = False

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


def get_executor(cache_directory: Path) -> "TestClusterExecutor":
    """Create a fresh executor instance.

    A new executor is created for each call to properly detect cached results.
    With wait=False, futures from a previous executor instance don't update
    their done() status when background jobs complete. Creating a fresh
    executor allows it to check the disk cache and return done()=True
    immediately if results are cached.

    Args:
        cache_directory: Directory for executor disk cache.

    Returns:
        The executor instance (already entered via __enter__).
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
