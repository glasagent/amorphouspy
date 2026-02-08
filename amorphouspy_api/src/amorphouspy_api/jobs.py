"""Job submission module for amorphouspy API.

This module provides job management using executorlib executors
(SingleNodeExecutor or SlurmClusterExecutor).

Configure via environment variables:
    EXECUTOR_TYPE: "local" (default) or "slurm"
    EXECUTOR_CORES: Number of cores per worker (default: 4)
    SLURM_PARTITION: SLURM partition name (optional, slurm only)
    SLURM_TIME: SLURM time limit (optional, slurm only)
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .workflows import run_meltquench_workflow

if TYPE_CHECKING:
    from executorlib import SingleNodeExecutor, SlurmClusterExecutor

logger = logging.getLogger(__name__)


def _get_executor_class() -> type:
    """Get the appropriate executor class based on environment."""
    executor_type = os.environ.get("EXECUTOR_TYPE", "local").lower()

    if executor_type == "slurm":
        from executorlib import SlurmClusterExecutor

        return SlurmClusterExecutor
    else:
        from executorlib import SingleNodeExecutor

        return SingleNodeExecutor


def _get_executor_config() -> dict[str, Any]:
    """Build executor configuration from environment variables."""
    config = {}

    # Common config
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


class JobManager:
    """Manages job submission and status checking using executorlib.

    Supports SingleNodeExecutor (local) and SlurmClusterExecutor based on
    the EXECUTOR_TYPE environment variable.
    """

    def __init__(self, cache_directory: Path) -> None:
        """Initialize the job manager.

        Args:
            cache_directory: Directory for caching job results.
        """
        self.cache_directory = cache_directory
        self._executor = None
        self._executor_class = _get_executor_class()
        self._config = _get_executor_config()
        logger.info(
            "JobManager initialized with executor=%s, config=%s",
            self._executor_class.__name__,
            self._config,
        )

    def _get_executor(self) -> "SingleNodeExecutor | SlurmClusterExecutor":
        """Get or create the executor instance."""
        if self._executor is None:
            self._executor = self._executor_class(
                cache_directory=self.cache_directory,
                **self._config,
            )
        return self._executor

    def submit_meltquench(
        self,
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Submit a meltquench job.

        The key insight is that executorlib's caching mechanism means
        submitting the same job twice will return the cached result if
        complete, or the running future if still in progress.

        Args:
            request_data: Dictionary containing the meltquench request parameters.
                Must include: components, values, n_atoms, potential_type,
                heating_rate, cooling_rate, n_print.

        Returns:
            Dictionary with job status information:
            - 'state': 'running', 'complete', or 'error'
            - 'result': Result dict if complete
            - 'error': Error message if failed
        """
        exe = self._get_executor()

        try:
            future = exe.submit(
                run_meltquench_workflow,
                components=request_data["components"],
                values=request_data["values"],
                n_atoms=request_data["n_atoms"],
                potential_type=request_data["potential_type"],
                heating_rate=request_data["heating_rate"],
                cooling_rate=request_data["cooling_rate"],
                n_print=request_data["n_print"],
            )

            # Check if the future is still running
            # cancelled() returns True if the job is still running
            if future.cancelled():
                return {
                    "state": "running",
                    "status": "Job submitted, waiting for completion",
                }

            # If not cancelled, check if done
            if future.done():
                try:
                    result = future.result()
                    return {
                        "state": "complete",
                        "status": "Completed",
                        "result": result,
                    }
                except Exception as e:
                    return {
                        "state": "error",
                        "status": "Failed",
                        "error": str(e),
                    }

            # Job is pending/queued
            return {
                "state": "running",
                "status": "Job queued",
            }

        except Exception as e:
            logger.exception("Error submitting job")
            return {
                "state": "error",
                "status": "Submission failed",
                "error": str(e),
            }

    def check_status(
        self,
        request_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Check the status of a meltquench job by re-submitting.

        Since executorlib uses caching, re-submitting the same parameters
        will return:
        - The cached result if complete
        - The running future if still in progress

        Args:
            request_data: Dictionary containing the meltquench request parameters.

        Returns:
            Dictionary with job status information.
        """
        # Re-submitting with same parameters will hit the cache
        return self.submit_meltquench(request_data=request_data)

    def close(self) -> None:
        """Close the executor and clean up resources."""
        if self._executor is not None:
            self._executor.__exit__(None, None, None)
            self._executor = None
