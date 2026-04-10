from __future__ import annotations

import asyncio

from amorphouspy_api.app import app
from amorphouspy_api.mcp_server import MCP_INSTRUCTIONS, mcp
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# MCP server configuration
# ---------------------------------------------------------------------------


def test_mcp_instructions_set() -> None:
    """The MCP server must have instructions for LLM clients."""
    assert MCP_INSTRUCTIONS
    assert "visualization" in MCP_INSTRUCTIONS.lower()


def test_mcp_exposes_expected_tools() -> None:
    """Key API endpoints should be exposed as MCP tools."""
    tools = asyncio.run(mcp.list_tools())
    tool_names = {t.name for t in tools}
    assert len(tools) > 0, "MCP server has no tools registered"
    # Core workflow tools should be present
    assert "submit_job" in tool_names
    assert "search_jobs" in tool_names
    assert "get_job_status" in tool_names
    assert "get_job_results" in tool_names


def test_mcp_excludes_human_facing_endpoints() -> None:
    """HTML visualization and raw structure download should not be MCP tools."""
    tools = asyncio.run(mcp.list_tools())
    tool_names = {t.name for t in tools}
    assert not any("visualize" in n for n in tool_names), (
        f"visualize endpoint should not be an MCP tool, found: {[n for n in tool_names if 'visualize' in n]}"
    )
    assert not any("structure" in n and "characterization" not in n for n in tool_names), (
        f"structure download endpoint should not be an MCP tool, found: "
        f"{[n for n in tool_names if 'structure' in n and 'characterization' not in n]}"
    )


# ---------------------------------------------------------------------------
# MCP HTTP transport
# ---------------------------------------------------------------------------


def test_mcp_endpoint_reachable() -> None:
    """The /mcp endpoint should respond (not 404) and handle JSON-RPC POSTs."""
    with TestClient(app, raise_server_exceptions=False) as c:
        # Plain GET should not be 404 (proves the middleware works)
        resp = c.get("/mcp")
        assert resp.status_code != 404

        # JSON-RPC POST should also not be 404
        resp = c.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "clientInfo": {"name": "test", "version": "0.1"},
                    "capabilities": {},
                },
                "id": 1,
            },
        )
        assert resp.status_code != 404
