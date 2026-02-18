"""amorphouspy Simulation API.

FastAPI application that manages long-running glass simulation tasks.
Routers handle the individual simulation types (meltquench, etc.).
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi_mcp import FastApiMCP

from .config import DB_PATH, PROJECTS_FOLDER
from .database import close_task_store, init_task_store
from .routers.meltquench import router as meltquench_router
from .visualization import router as visualization_router

# Configure logging - use stream handler by default, file handler only if not in test
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logger.info("Using project directory: %s", PROJECTS_FOLDER)

# Ensure the projects directory exists
PROJECTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Initialize persistent task store
logger.info("Task store database path: %s", DB_PATH)
init_task_store(DB_PATH)


# Create FastAPI app
app = FastAPI(
    title="amorphouspy Simulation API",
    description="API for managing long-running glass simulation tasks using amorphouspy",
    version="0.1.0",
)

# Enable CORS for all origins (customize as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include routers
app.include_router(meltquench_router, tags=["meltquench"])
app.include_router(visualization_router, tags=["visualization"])


mcp = FastApiMCP(app, include_tags=["tool"])
mcp.mount_http(mount_path="/mcp")


@app.on_event("shutdown")
def shutdown_event() -> None:
    """Close database connections on app shutdown."""
    logger.info("Closing task store database connection")
    close_task_store()


@app.get("/")
def root() -> RedirectResponse:
    """Root endpoint redirects to API documentation."""
    return RedirectResponse(url="/docs")
