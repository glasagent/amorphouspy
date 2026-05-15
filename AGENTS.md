## Environment

All commands run inside the pixi environment. Prefix every command with `pixi run`, or activate once with `pixi shell`. Do not invoke `python`, `pip`, `pytest`, or `ruff` directly — they will resolve to the wrong interpreter.

## Lint and format

Before finishing any task that touched Python files:

```bash
pixi run pre-commit run -a
```

This runs ruff check, ruff format, and ty type-check on all changed files.