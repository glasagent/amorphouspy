"""amorphouspy Simulation API.

FastAPI application that manages long-running glass simulation jobs.
See ``docs/api-spec.md`` for the full specification.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi_mcp import FastApiMCP

from .config import DB_PATH, PROJECTS_FOLDER
from .database import close_job_store, init_job_store
from .routers.glasses import router as glasses_router
from .routers.jobs import router as jobs_router

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logger.info("Using project directory: %s", PROJECTS_FOLDER)
PROJECTS_FOLDER.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan — startup and shutdown."""
    logger.info("Job store database path: %s", DB_PATH)
    init_job_store(DB_PATH)
    yield
    logger.info("Closing job store database connection")
    close_job_store()


app = FastAPI(
    title="amorphouspy Simulation API",
    description="API for managing glass simulation jobs using amorphouspy",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Routers
app.include_router(jobs_router)
app.include_router(glasses_router)

# MCP (expose all "tool"-tagged endpoints)
mcp = FastApiMCP(app, include_tags=["tool"])
mcp.mount_http(mount_path="/mcp")


@app.get("/")
def root() -> RedirectResponse:
    """Root endpoint redirects to API documentation."""
    return RedirectResponse(url="/docs")
