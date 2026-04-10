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

Pull requests should always be labeled so the release drafter can categorize changes automatically. The following labels are recognized:

| Category | Labels |
|---|---|
| 💥 Breaking Changes | `breaking` |
| 🚀 Features | `enhancement` |
| 🐛 Bug Fixes | `bug` |
| 📚 Documentation | `documentation` |
| 🔧 Maintenance | `dependencies`, `ci`, `refactor` |

Use `skip-changelog` to exclude a PR from the release notes entirely.

Labels also drive automatic version bumping (see below).


## Making a Release

Releases are made whenever convenient, and can be made by any maintainer (push rights to the repository).

**Release procedure:**

1. Merge a pull request that is labeled with one of the version labels (`patch`, `minor`, or `major`). This automatically bumps the version and creates a new tag.
2. Go to [GitHub Releases](https://github.com/glasagent/amorphouspy/releases)
3. Edit the latest draft release (pre-populated automatically by the release drafter action) and select the newly created tag
    - Note: you can also use the "Generate release notes" button, but it won't filter commits into different categories
4. Publish the release

