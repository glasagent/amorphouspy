"""Shared configuration for the amorphouspy API."""

import logging
import os
from importlib.metadata import version
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# amorphouspy version (used in project directory naming)
# ---------------------------------------------------------------------------

try:
    amorphouspy_version = version("amorphouspy")
    logger.info("Using amorphouspy version: %s", amorphouspy_version)
except Exception:
    amorphouspy_version = "unknown"
    logger.warning("Could not determine amorphouspy version, using 'unknown'")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECTS_FOLDER = Path(
    os.environ.get(
        "AMORPHOUSPY_PROJECTS",
        str(Path(__file__).resolve().parent.parent.parent / "projects"),
    ),
)

MELTQUENCH_PROJECT_DIR = PROJECTS_FOLDER / f"amorphouspy_{amorphouspy_version}" / "meltquench"

DB_PATH = PROJECTS_FOLDER / "tasks.db"

# ---------------------------------------------------------------------------
# API base URL for visualization links (e.g. behind a reverse proxy)
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "")
