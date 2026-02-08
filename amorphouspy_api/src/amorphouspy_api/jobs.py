"""Job submission utilities for amorphouspy API.

This module provides utilities for selecting and configuring executorlib executors
(TestClusterExecutor for local or SlurmClusterExecutor for SLURM).

Both executors use wait=False to allow non-blocking exit from the context manager,
enabling the API to check job status without blocking.

Configure via environment variables:
    EXECUTOR_TYPE: "local" (default) or "slurm"
    EXECUTOR_CORES: Number of cores per worker (default: 4)
    SLURM_PARTITION: SLURM partition name (optional, slurm only)
    SLURM_TIME: SLURM time limit (optional, slurm only)
"""

import logging
import os
from typing import Any

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
