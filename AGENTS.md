# Agent Instructions

Apply these requirements to new and edited code. Read the relevant source, tests,
and configuration before changing behavior. Keep changes focused; do not reformat
unrelated files or add abstractions and dependencies without a demonstrated need.
Do not assume existing code complies with these rules, but do not modify unrelated
files just to make them compliant. Treat `pyproject.toml` and CI as the source of
truth for tooling. Use clear Conventional Commit messages when asked to create a
commit.

## Project Overview

`graphomotor` is a Python toolkit for analyzing graphomotor data collected via
Curious. It reads drawing-response data, extracts task features, and provides
visualization utilities. Preserve compatibility with Curious input formats and
the documented command-line and Python APIs unless the task explicitly changes
them.

## Architecture and Structure

The project uses a `src` layout:

| Location | Responsibility |
| --- | --- |
| `src/graphomotor/__main__.py` | CLI entry point |
| `src/graphomotor/core/` | Configuration, shared models, CLI, and task orchestration |
| `src/graphomotor/io/` | Curious file reading and input validation |
| `src/graphomotor/features/` | Shared and task-specific feature extraction |
| `src/graphomotor/utils/` | Reusable processing and geometry helpers |
| `src/graphomotor/plot/` | Feature and drawing visualizations |
| `tests/unit/` | Isolated tests for modules and behavior |
| `tests/smoke/` | End-to-end workflow tests |
| `tests/sample_data/` | Sample task input data and fixtures |
| `tests/sample_batch_data/` | Sample batch-processing inputs |

The four primary tasks are **Spiral**, **Trails**, **AtoZ**, and **Digit Symbol
Substitution (DSYM)**. Keep task-specific feature extraction functions in that
task's submodule under `src/graphomotor/features/` (for example,
`features/spiral/` or `features/trails/`). Add task submodules for AtoZ and DSYM
as their task-specific feature work is implemented; do not place their features
in another task's package. Put features shared across tasks in
`src/graphomotor/features/shared_features.py`. Put reusable non-feature helpers
in `src/graphomotor/utils/`, rather than duplicating task-independent logic.

Current implementation is not equally complete for all tasks: Spiral has a
feature extraction orchestrator; Trails has task-specific feature functions and
processing utilities; AtoZ and DSYM have input/model groundwork but not yet the
same task-specific feature packages. Extend the existing architecture without
implying that unfinished task pipelines are already available.

Keep input parsing and validation in `io`, reusable data models and coordination
in `core`, and plotting concerns in `plot`. Avoid coupling feature calculation
to CSV parsing or visualization. When changing data models, parsing, task
features, or CLI behavior, check all callers and update relevant tests and
documentation.

## Python Style

- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).
- Keep code compatible with Python `>=3.11,<3.14`.
- Add type annotations to functions and public interfaces. Use Google-style
	docstrings, including module docstrings for new modules; document public
	behavior and non-obvious edge cases without repeating type information.
- Use `snake_case` for functions, modules, and variables, `PascalCase` for
	classes, and `SCREAMING_SNAKE_CASE` for constants.
- Follow the project Ruff configuration: four-space indentation, 88-character
	line length, and the configured lint rules for annotations, docstrings,
	imports, and common errors.
- Prefer the standard library and existing dependencies. If a dependency is
	necessary, update project metadata and the lockfile consistently.
- Preserve clear input validation and actionable error reporting. Batch
	processing should continue to handle per-file failures without silently
	misrepresenting unsuccessful inputs as successful results.

## Setup and Commands

Run commands from the repository root. Install/synchronize the project and
development dependencies with `uv sync`. Use these checks as relevant to the
change:

| Command | Purpose |
| --- | --- |
| `uv run pytest` | Run unit and smoke tests |
| `uv run ruff format --check` | Check Python formatting |
| `uv run ruff check` | Run Ruff lint checks |
| `uv run mypy .` | Run static type checks |
| `uv run pre-commit run --all-files` | Run configured contribution hooks |

CI tests Python 3.11, 3.12, and 3.13 across Linux, Windows, and macOS, and runs
pytest with coverage and debug logging. Run the focused tests while iterating,
then run the relevant full checks before finishing. Report commands run and any
failures or environment blockers accurately.

## Tests and Data

- Add tests under `tests/unit/` or `tests/smoke/`, following the `test_*.py`
	naming pattern and existing pytest fixtures in `tests/conftest.py`.
- Use the existing pytest patterns such as fixtures, parametrization, and
	`tmp_path`; test normal behavior and relevant invalid-input and failure cases.
- Add or adapt sample data only when needed. Do not commit real participant data
	or other sensitive responses; use approved, de-identified samples.
- Keep tests independent of external services and machine-specific paths.
- Documentation-only and configuration-only changes do not need artificial
	unit tests.

## Documentation and Contributions

Update the README or relevant documentation when user-facing commands, input
requirements, task support, or public APIs change. Follow `CONTRIBUTING.md` for
contribution expectations and the repository license. Do not claim a task,
feature, or command is available unless the implementation and tests support it.

