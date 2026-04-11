# Contributing

## Developer Setup

```bash
# Clone the repository
git clone https://github.com/glasagent/amorphouspy.git
cd amorphouspy

# Install the environment
pixi install

# Set up pre-commit hooks
pixi run -- pre-commit install
```

## Running Tests

```bash
# amorphouspy unit tests
pixi run test-lib

# API unit tests
pixi run test-api

# Notebook integration tests
pixi run test-notebooks
```

## Linting & Formatting

```bash
pixi run lint
pixi run format-check
```

## Building the Docs

```bash
# Build (strict mode)
pixi run docs-build

# Serve locally with live reload
pixi run docs-serve
```

## Tagging Pull Requests

Every pull request should have a label so it lands in the right section of the release notes.

| Category | Labels |
|---|---|
| 💥 Breaking Changes | `breaking` |
| 🚀 Features | `feature`, `enhancement` |
| 🐛 Bug Fixes | `bug` |
| 📚 Documentation | `documentation`, `docs` |
| 🔧 Maintenance | `dependencies`, `ci`, `refactor` |

Use `skip-changelog` to exclude a PR from the release notes entirely.

## Making a Release (maintainers only)

Releases are intentional acts — not every merged PR needs one. When you are ready to cut a release, add one of the version-bump labels to the PR before merging:

| Label | Effect |
|---|---|
| `patch` | Bumps `0.4.1` → `0.4.2` |
| `minor` | Bumps `0.4.1` → `0.5.0` |
| `major` | Bumps `0.4.1` → `1.0.0` |

These labels trigger the version-bump workflow, which updates `pyproject.toml`, commits the change, and pushes a new tag. Regular contributors should not add these labels.

## Making a Release

Releases are made whenever convenient, and can be made by any maintainer (push rights to the repository).

**Release procedure:**

1. Merge a pull request labeled with `patch`, `minor`, or `major`. This automatically bumps the version in `pyproject.toml`, commits it, and creates a new tag.
2. Go to [GitHub Releases](https://github.com/glasagent/amorphouspy/releases)
3. Edit the latest draft release (pre-populated automatically by the release drafter action) and select the newly created tag
    - Note: you can also use the "Generate release notes" button, but it won't filter commits into different categories
4. Publish the release

