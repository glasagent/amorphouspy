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

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/) for every commit message,
matching the style already used on the branch. The subject line is
`<type>(<optional scope>): <description>` with a lowercase description, e.g.:

```
feat: parallelize viscosity calculation
fix(api): attribute pipeline failures to the failing step
refactor(lammps): caller owns provided tmp_working_directory
perf: lower MIN_ATOMS_PER_CORE to 500
chore: update pixi.lock
```

Use the same prefixes as the branch names documented in
`docs/developers/contributing.md` (these drive PR labels and release notes):
`feat`, `perf`, `fix`, `docs`, `refactor`, `style`, `ci`, `chore`, `test`,
`dep`/`deps`.