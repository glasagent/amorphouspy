from uuid import uuid4
from typing import Dict
import anyio
from fastmcp import FastMCP
from fastmcp.utilities import types

mcp = FastMCP("long_runner")

# in-memory task store ---------------------------------
_task_store: Dict[str, dict] = {}

async def _worker(task_id: str, n: int) -> None:
    try:
        await anyio.sleep(n)          # do the expensive thing
        _task_store[task_id]["state"] = "complete"
        _task_store[task_id]["result"] = f"slept {n}s"
    except Exception as exc:
        _task_store[task_id]["state"] = "error"
        _task_store[task_id]["error"] = str(exc)

# tools ------------------------------------------------
@mcp.tool(title="Start task")
async def start_sleep(seconds: int) -> types.TextContent:
    task_id = str(uuid4())
    _task_store[task_id] = {"state": "processing"}
    anyio.create_task_group().start_soon(_worker, task_id, seconds)
    return types.TextContent(text=task_id)

@mcp.tool(title="Check status")
async def check(task_id: str) -> types.TextContent:
    meta = _task_store.get(task_id)
    if not meta:
        return types.TextContent(text="unknown task")
    if meta["state"] == "complete":
        return types.TextContent(text=meta["result"])
    if meta["state"] == "error":
        return types.TextContent(text=f"ERROR: {meta['error']}")
    return types.TextContent(text="processing")

