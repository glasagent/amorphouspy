"""amorphouspy_api - API package for meltquench simulations."""

try:
    import fastapi  # noqa: F401
except ImportError:
    msg = "API dependencies not installed. Install with: pip install amorphouspy[api]"
    raise ImportError(msg) from None
