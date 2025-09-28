# Repository Guidelines

## Project Structure & Module Organization
- Core domain and sim shim: `matrixmachine/core/description.py` (MatrixShape, Mapping, Chip/Die) and `matrixmachine/core/sim_engine.py` (Desim-based shim; Perfetto tracing).
- Tiling strategies: `matrixmachine/strategy/` (e.g., `trivial.py`, `gpt_grid_search.py`).
- Tests: `tests/` (e.g., `tests/test_grid_tiling.py`).
- Vendored engines: `packages/Desim`, `packages/PerfTracer` — treat as submodules; do not edit internals without review.

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv && source .venv/bin/activate`.
- Install backends (editable): `pip install -e packages/Desim -e packages/PerfTracer`.
- Smoke test: `python tests/test_grid_tiling.py` (prints mapping stats; validates bidirectional consistency).
- Full tests: `pytest`.
- Coverage: `pytest --cov=matrixmachine/core/description.py,matrixmachine/strategy`.
- Format/lint: `black . && isort . && mypy .`.

## Coding Style & Naming Conventions
- Python 3, PEP 8, 4-space indent; use type hints and dataclasses for configs.
- Names: functions/vars `snake_case`; classes `CapWords` (e.g., `MatrixShape`).
- Keep modules focused; avoid cross-package imports outside documented boundaries.

## Testing Guidelines
- Framework: `pytest`; name files `tests/test_<feature>.py`.
- New tiling logic must call `Mapping.check_all()`; include negative cases (invalid grids, bounds, batch splits).
- Target >85% coverage on modified files; note gaps in PRs if lower.

## Commit & Pull Request Guidelines
- Commits: imperative mood; subject ≤50 chars; reference issues (e.g., "Fixes #42").
- PRs: summarize behavior changes, list validation commands, and attach Perfetto traces or CLI screenshots for timing changes.
- If touching vendored packages, open a PR early for owners to audit divergences.

## Simulation & Tracing Notes
- Use `perf_tracer.PerfettoTracer.get_global_tracer()` or an injected tracer; scope events around `SimModule.wait_time`.
- Load traces at `ui.perfetto.dev`; include a brief walkthrough when traces inform acceptance.

