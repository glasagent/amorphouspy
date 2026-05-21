## Environment

All commands run inside the pixi environment. Prefix every command with `pixi run`, or activate once with `pixi shell`. Do not invoke `python`, `pip`, `pytest`, or `ruff` directly — they will resolve to the wrong interpreter.

## Lint and format

Before finishing any task that touched Python files:

```bash
pixi run pre-commit run -a
```

This runs ruff check, ruff format, and ty type-check on all changed files.

## Tests

When adding new features or fixing bugs, add or update tests. Run the relevant test files with:

```bash
pixi run pytest amorphouspy_api/src/tests/test_database.py -v
pixi run pytest amorphouspy_api/src/tests/test_jobs.py -v
pixi run pytest amorphouspy/src/tests/ -v
```