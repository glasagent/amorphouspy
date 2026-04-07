"""Export the OpenAPI schema from the FastAPI app to a JSON file."""

import json
from pathlib import Path

from amorphouspy_api.app import app

schema = app.openapi()
out = Path(__file__).resolve().parent / "openapi.json"
out.write_text(json.dumps(schema, indent=2))
print(f"Wrote {out}")
