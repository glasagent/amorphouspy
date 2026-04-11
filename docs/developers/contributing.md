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

Pull requests should be labeled for two purposes: categorizing release notes and triggering automatic version bumps.

### Release notes labels

| Category | Labels |
|---|---|
| 💥 Breaking Changes | `major`, `breaking` |
| 🚀 Features | `minor`, `feature`, `enhancement` |
| 🐛 Bug Fixes | `patch`, `fix`, `bugfix`, `bug` |
| 📚 Documentation | `documentation`, `docs` |
| 🔧 Maintenance | `chore`, `dependencies`, `ci`, `refactor` |

Use `skip-changelog` to exclude a PR from the release notes entirely.

### Version bump labels

The labels `patch`, `minor`, and `major` also trigger an automatic version bump when the PR is merged. Other categorization labels (e.g. `bug`, `enhancement`) do **not** trigger a bump on their own — add one of the three version labels as well if you want the bump to happen automatically.

## Making a Release

Releases are made whenever convenient, and can be made by any maintainer (push rights to the repository).

**Release procedure:**

1. Merge a pull request labeled with `patch`, `minor`, or `major`. This automatically bumps the version in `pyproject.toml`, commits it, and creates a new tag.
2. Go to [GitHub Releases](https://github.com/glasagent/amorphouspy/releases)
3. Edit the latest draft release (pre-populated automatically by the release drafter action) and select the newly created tag
    - Note: you can also use the "Generate release notes" button, but it won't filter commits into different categories
4. Publish the release

