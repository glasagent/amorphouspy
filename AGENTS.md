## Environment

All commands run inside the pixi environment. Prefix every command with `pixi run`, or activate once with `pixi shell`. Do not invoke `python`, `pip`, `pytest`, or `ruff` directly — they will resolve to the wrong interpreter.

## Lint and format

Before finishing any task that touched Python files:

```bash
pixi run ruff format .
pixi run ruff check --fix .
```

Both must pass cleanly. `ruff format` first, then `ruff check --fix` — the order matters because some lint rules are formatting-adjacent.