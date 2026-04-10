"""MCP server — exposes amorphouspy API tools to LLM clients.

Uses the ``mcp`` SDK's :class:`~mcp.server.fastmcp.FastMCP` to register the
existing FastAPI endpoint functions directly as MCP tools (no wrappers needed).

The :class:`MCPRouteMiddleware` ASGI middleware routes ``/mcp`` requests to the
MCP server while all other traffic passes through to the FastAPI application.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

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
2. `get_job_status` — poll until status is "completed".
3. `get_job_results` — retrieve analysis data.
4. `search_jobs` — find existing results for similar compositions.
5. `list_glasses` / `lookup_glass` — browse available compositions.
"""

mcp = FastMCP(
    "amorphouspy",
    instructions=MCP_INSTRUCTIONS,
    stateless_http=True,
)


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
        get_job_results,
        get_single_result,
        list_glasses,
        lookup_glass,
    ]:
        mcp.add_tool(fn)


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
