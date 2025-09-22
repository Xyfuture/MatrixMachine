# Repository Guidelines

## Project Structure & Module Organization
MatrixMachine couples a lightweight core with vendored dependencies. Core data models live in `description.py`, and the discrete-event shim lives in `sim_engine.py`. Tiling strategies such as `GridTilingStrategy` reside in `strategy/`. Example-driven integration tests are in `test_grid_tiling.py`. Third-party engines are vendored under `packages/Desim` and `packages/PerfTracer`; treat them as git submodules and install in editable mode before running simulations.

## Build, Test, and Development Commands
Create an isolated environment (`python -m venv .venv && source .venv/bin/activate`) before installing dependencies. Install the simulation backends via `pip install -e packages/Desim` and `pip install -e packages/PerfTracer`. Run quick smoke tests with `python test_grid_tiling.py`; it prints mapping stats and validates bidirectional consistency. Use `pytest` for the full suite, and `pytest --cov=description.py,strategy` when you need coverage. Format and lint prior to review with `black .`, `isort .`, and `mypy .`.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation and descriptive snake_case for functions and variables. Classes remain in CapWords (see `MatrixShape`, `GridTilingStrategy`). Keep modules focused and avoid cross-package imports outside the documented boundaries. Prefer type hints and dataclasses for configuration objects. Let `black` and `isort` enforce formatting, and address `mypy` feedback before pushing.

## Testing Guidelines
Unit and integration checks use `pytest`; place new tests alongside implementation modules or extend `test_grid_tiling.py` when validating strategies end-to-end. Name tests `test_<feature>` and keep fixtures minimal. Ensure new tiling logic calls `Mapping.check_all()`; include negative tests when adding validation paths. Aim for coverage above 85% on modified files and document gaps in the pull request if lower.

## Commit & Pull Request Guidelines
Write commits in imperative mood (`Add grid balancer`) and keep subjects under 50 characters; elaborate in the body with wrapped prose when rationale or trade-offs are non-obvious. Reference issues with `Fixes #<id>` when applicable. PRs should summarize behavior changes, list validation commands, and attach Perfetto traces or CLI screenshots when altering simulation timing. Request review early if touching vendored packages so owners can audit divergences.

## Simulation & Tracing Notes
When instrumenting new stages, register Perfetto units via `PerfettoTracer.get_global_tracer()` and scope events around `SimModule.wait_time` calls. Generated trace files open cleanly at `ui.perfetto.dev`; include a short walkthrough in review notes when trace interpretation informs acceptance.
