"""MCP server — exposes amorphouspy API tools to LLM clients.

Uses the ``mcp`` SDK's :class:`~mcp.server.fastmcp.FastMCP` to register the
existing FastAPI endpoint functions as MCP tools. Synchronous endpoints are
wrapped by :func:`_offload_sync` so their (potentially blocking) IO runs in a
worker thread instead of on the event loop.

The :class:`MCPRouteMiddleware` ASGI middleware routes ``/mcp`` requests to the
MCP server while all other traffic passes through to the FastAPI application.
"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP
from starlette.concurrency import run_in_threadpool

from .config import MCP_HOST

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------

MCP_INSTRUCTIONS = """\
You are interacting with the **amorphouspy glass simulation API**.

**Always present the `visualization_url`** (or the `urls.visualization` field)
from every job response to the user — it links to an interactive HTML
dashboard with RDF plots, bond-angle distributions, and 3-D structure viewers.

Typical workflow:
1. `submit_job` — submit a simulation (returns job ID + URLs).
2. `search_jobs` — find existing jobs by exact filters (composition, tags, status, time window).
3. `get_job_status` — confirm live per-job status before fetching results.
4. `get_job_results` — retrieve analysis data.
5. `get_job_settings` — inspect the original submission parameters.
6. `list_glasses` / `lookup_glass` — browse completed compositions/results.

Notes:
- `search_jobs` is a listing/filter tool, not a similarity search.
- For nearest/close composition matching among completed results, use `lookup_glass`.
- When status accuracy matters for jobs returned by `search_jobs`, call `get_job_status` for each job ID.
"""

mcp = FastMCP(
    "amorphouspy",
    instructions=MCP_INSTRUCTIONS,
    stateless_http=True,
    host=MCP_HOST,
)


def _offload_sync(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a synchronous endpoint so MCP runs it in a worker thread.

    FastMCP executes synchronous tool functions inline on the asyncio event
    loop (``func_metadata.call_fn_with_arg_validation`` calls ``fn(...)``
    directly when the function is not a coroutine). Any blocking IO inside such
    a tool — e.g. reading executorlib HDF5 caches — would therefore freeze the
    entire single-worker server.

    Wrapping the function in an ``async`` shim that offloads to Starlette's
    worker-thread pool keeps the event loop responsive. ``functools.wraps``
    preserves ``__wrapped__`` so ``inspect.signature`` (used by FastMCP to build
    the tool schema) still resolves the original signature and annotations.
    """
    if inspect.iscoroutinefunction(fn):
        return fn

    @functools.wraps(fn)
    async def _async_tool(**kwargs: object) -> object:
        return await run_in_threadpool(fn, **kwargs)

    return _async_tool


def register_tools() -> None:
    """Register FastAPI endpoint functions as MCP tools.

    Called from ``app.py`` after routers are loaded to avoid circular imports.
    Only data-returning endpoints are registered — human-facing HTML
    (``visualize_job``) and raw file downloads (``get_structure``) are excluded.
    """
    from amorphouspy_api.routers.glasses import list_glasses, lookup_glass
    from amorphouspy_api.routers.jobs import (
        cancel_job,
        get_job_results,
        get_job_settings,
        get_job_status,
        get_single_result,
        search_jobs,
        submit_job,
    )

    for fn in [
        submit_job,
        search_jobs,
        get_job_status,
        cancel_job,
        get_job_settings,
        get_job_results,
        get_single_result,
        list_glasses,
        lookup_glass,
    ]:
        mcp.add_tool(_offload_sync(fn))

    # Strip auto-generated 'title' fields from MCP tool schemas.
    # Pydantic titles duplicate the field/class name and waste LLM tokens.
    for tool in mcp._tool_manager._tools.values():
        _strip_titles(tool.parameters)


# ---------------------------------------------------------------------------
# Schema helpers — reduce MCP tool token cost
# ---------------------------------------------------------------------------


def _strip_titles(obj: dict | list) -> None:
    """Remove ``title`` keys from a JSON schema dict in-place (recursive)."""
    if isinstance(obj, dict):
        obj.pop("title", None)
        for v in obj.values():
            if isinstance(v, (dict, list)):
                _strip_titles(v)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (dict, list)):
                _strip_titles(v)


# ---------------------------------------------------------------------------
# ASGI middleware for routing /mcp -> MCP server
# ---------------------------------------------------------------------------


class MCPRouteMiddleware:
    """ASGI middleware that routes ``/mcp`` requests to the MCP server.

    All other requests are passed through to the wrapped application.
    """

    def __init__(self, app) -> None:  # noqa: D107
        self.app = app
        self.mcp_app = mcp.streamable_http_app()

    async def __call__(self, scope, receive, send) -> None:
        """Route ``/mcp`` to the MCP app, everything else to the wrapped app."""
        if scope["type"] == "http" and scope["path"].rstrip("/") == "/mcp":
            # MCP app routes at /mcp already
            await self.mcp_app(scope, receive, send)
        else:
            await self.app(scope, receive, send)
