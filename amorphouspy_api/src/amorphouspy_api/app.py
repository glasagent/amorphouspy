"""amorphouspy Simulation API.

FastAPI application that manages long-running glass simulation jobs.
"""

import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi_mcp import FastApiMCP
from starlette.middleware.base import BaseHTTPMiddleware

from .config import API_TOKEN, DB_PATH, PROJECTS_FOLDER
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


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """Reject requests without a valid ``Authorization: Bearer <token>`` header."""

    async def dispatch(self, request: Request, call_next):
        """Check the Authorization header and reject invalid tokens."""
        # Allow OpenAPI docs / schema without auth
        if request.url.path in ("/docs", "/redoc", "/openapi.json", "/"):
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Missing bearer token"})

        token = auth_header.removeprefix("Bearer ")
        if not secrets.compare_digest(token, API_TOKEN):  # type: ignore[arg-type]
            return JSONResponse(status_code=403, content={"detail": "Invalid bearer token"})

        return await call_next(request)


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
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if API_TOKEN:
    app.add_middleware(BearerTokenMiddleware)

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


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions so the API always returns JSON."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
