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

## Labelling Pull Requests

Every pull request should have a label so it lands in the right section of the release notes.
Labels are applied **automatically** based on the branch name prefix:

| Branch prefix | Label | Release-notes category |
|---|---|---|
| `feat/`, `perf/` | `type: feature` | 🚀 Features |
| `fix/` | `type: bug` | 🐛 Bug Fixes |
| `docs/` | `type: documentation` | 📚 Documentation |
| `refactor/`, `style/` | `type: refactor` | 🔧 Maintenance |
| `ci/`, `chore/` | `type: ci` | 🔧 Maintenance |
| `test/` | `type: tests` | 🔧 Maintenance |
| `dep/`, `deps/` | `type: dependencies` | 🔧 Maintenance |

You can always add or override labels manually. The `type: breaking` label (💥 Breaking Changes) must be added by hand.

Use `skip-changelog` to exclude a PR from the release notes entirely.

## Making a Release (maintainers only)

Releases are intentional acts — not every merged PR needs one.

### Version resolution

The release drafter resolves the next version automatically based on PR labels:

| Labels | Version bump |
|---|---|
| `type: breaking`, `semver: major` | major (`0.4.1` → `1.0.0`) |
| `type: feature`, `semver: minor` | minor (`0.4.1` → `0.5.0`) |
| `type: bug`, `semver: patch` | patch (`0.4.1` → `0.4.2`) |
| *(none of the above)* | patch (default) |

### Release procedure

1. **Trigger the version bump.** Either:
    - Add a `patch`, `minor`, or `major` label to a PR before merging it, **or**
    - Run the *Version Bump* workflow manually from the Actions tab (`workflow_dispatch`).

    This does **not** tag or release directly. Instead it opens a new PR (e.g. *"chore: bump version to 0.5.0"*) that updates `pyproject.toml` and `pixi.lock`. The PR is labelled `skip-changelog` so it stays out of the release notes.

2. **Merge the version-bump PR.** When it merges, a separate job detects the title, creates and pushes a `v<version>` tag.

3. **Publish the release.** Go to [GitHub Releases](https://github.com/glasagent/amorphouspy/releases), edit the latest draft (pre-populated by the release drafter action), select the newly created tag, and publish.

Regular contributors should not add `patch` / `minor` / `major` labels.

