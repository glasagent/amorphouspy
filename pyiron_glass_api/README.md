# pyiron-glass-api

API for atomistic modeling of oxide glasses using the pyiron-glass workflows.

## Installation

```
pip install -e ../pyiron-glass/   # local install
#pip install -e https://github.com/glasagent/pyiron-glass/   # will start working when published
pip install -e .
```

## Launch API (including MCP server)

```
python -m uvicorn pyiron_glass_api.app:app
```

## Developer setup
In addition, install
```
pip install -r requirements-dev.txt
pre-commit install
```


## Run tests
unit tests
```
pytest
```

integration tests (requires a working pyiron-glass installation)
```
# start API
uvicorn pyiron_glass_api.app:app --port 8002 --reload
# run tests
pytest -m integration -s
```



