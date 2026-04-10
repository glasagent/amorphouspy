"""amorphouspy Simulation API.

FastAPI application that manages long-running glass simulation jobs.
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles


class PrettyJSONResponse(JSONResponse):
    """JSONResponse subclass that renders indented JSON for readability."""

    def render(self, content) -> bytes:
        """Serialize *content* as indented JSON."""
        return json.dumps(content, ensure_ascii=False, indent=2).encode("utf-8")


from .config import API_TOKEN, DB_PATH, PROJECTS_FOLDER
from .database import close_job_store, init_job_store
from .mcp_server import MCPRouteMiddleware
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
    if API_TOKEN:
        logger.info("Bearer-token authentication is enabled")
    else:
        logger.warning("API_TOKEN is not set — the API is open with no authentication!")
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
    default_response_class=PrettyJSONResponse,
)

app.add_middleware(MCPRouteMiddleware)

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


@app.get("/")
def root() -> RedirectResponse:
    """Root endpoint redirects to API documentation."""
    return RedirectResponse(url="/docs")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions so the API always returns JSON."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
