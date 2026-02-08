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
    from executorlib.executor.single import TestClusterExecutor

logger = logging.getLogger(__name__)

# Singleton executor instance
_executor_instance: "TestClusterExecutor | None" = None
_executor_cache_dir: Path | None = None


def get_executor_class() -> type:
    """Get the appropriate executor class based on environment.

    Returns:
        TestClusterExecutor (local) or SlurmClusterExecutor class.
    """
    executor_type = os.environ.get("EXECUTOR_TYPE", "local").lower()

    if executor_type == "slurm":
        from executorlib import SlurmClusterExecutor

        return SlurmClusterExecutor
    else:
        # Use TestClusterExecutor for local - it supports wait=False
        # (SingleNodeExecutor does not support wait=False)
        from executorlib.executor.single import TestClusterExecutor

        return TestClusterExecutor


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
    """Get or create the singleton executor instance.

    The executor is created once and reused for all submissions.
    This allows multiple jobs to share the same executor context
    and enables proper dependency tracking between jobs.

    Args:
        cache_directory: Directory for executor disk cache.

    Returns:
        The executor instance (already entered via __enter__).
    """
    global _executor_instance, _executor_cache_dir

    # If executor exists and cache dir matches, return it
    if _executor_instance is not None and _executor_cache_dir == cache_directory:
        return _executor_instance

    # Close existing executor if cache dir changed
    if _executor_instance is not None:
        try:
            _executor_instance.__exit__(None, None, None)
        except Exception:
            logger.exception("Error closing previous executor")
        _executor_instance = None

    # Create new executor
    executor_class = get_executor_class()
    executor_config = get_executor_config()

    logger.info(
        "Creating singleton executor: %s with cache_directory=%s",
        executor_class.__name__,
        cache_directory,
    )

    _executor_instance = executor_class(cache_directory=cache_directory, **executor_config)
    _executor_cache_dir = cache_directory

    # Enter context manager
    _executor_instance.__enter__()

    return _executor_instance


def shutdown_executor() -> None:
    """Shutdown the singleton executor if it exists.

    Call this during application shutdown to clean up resources.
    """
    global _executor_instance, _executor_cache_dir

    if _executor_instance is not None:
        try:
            _executor_instance.__exit__(None, None, None)
        except Exception:
            logger.exception("Error shutting down executor")
        finally:
            _executor_instance = None
            _executor_cache_dir = None
