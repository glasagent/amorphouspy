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

## Making a Release

Releases are made whenever convenient, and can be made by any maintainer (push rights to the repository).

**Release procedure:**

1. Go to [GitHub Releases](https://github.com/glasagent/amorphouspy/releases)
2. From the tag dropdown, select **"Create new tag"** (e.g. `v0.3.0`)
3. Edit the latest draft release (pre-populated automatically by the release drafter action)
    - Note: you can also use the "Generate release notes" button, but it has no concept of patch/minor/major changes
4. Publish the release

## Tagging Pull Requests

Pull requests should always be labeled so the release drafter can categorize changes automatically. The following labels are recognized:

| Category | Labels |
|---|---|
| 💥 Breaking Changes | `breaking`, `major` |
| 🚀 Features | `enhancement`, `minor` |
| 🐛 Bug Fixes | `bug`, `patch` |
| 📚 Documentation | `documentation` |
| 🔧 Maintenance | `dependencies`, `ci`, `refactor` |

Labels also drive automatic version bumping: `major`/`breaking` bump the major version, `minor`/`feature`/`enhancement` bump the minor version, and `patch`/`fix`/`bugfix`/`bug` bump the patch version (default).

Use `skip-changelog` to exclude a PR from the release notes entirely.
