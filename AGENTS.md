# Repository Guidelines

## Project Structure & Module Organization
- Core models live in `description.py`; the discrete‑event shim is `sim_engine.py`.
- Tiling strategies (e.g., `GridTilingStrategy`) are in `strategy/`.
- Example-driven integration tests: `test_grid_tiling.py`. Add new unit tests next to the module they cover or extend this file for end‑to‑end checks.
- Vendored engines: `packages/Desim`, `packages/PerfTracer`. Treat as submodules; do not modify their internals without review.

## Build, Test, and Development Commands
- Create venv: `python -m venv .venv && source .venv/bin/activate`.
- Install backends (editable): `pip install -e packages/Desim -e packages/PerfTracer`.
- Smoke test: `python test_grid_tiling.py` (prints mapping stats, validates bidirectional consistency).
- Full tests: `pytest`; coverage: `pytest --cov=description.py,strategy`.
- Format/lint: `black .` · `isort .` · `mypy .`.

## Coding Style & Naming Conventions
- Python, PEP 8, 4‑space indent; prefer type hints and dataclasses for configs.
- Names: functions/vars `snake_case`; classes `CapWords` (e.g., `MatrixShape`).
- Keep modules focused; avoid cross‑package imports outside documented boundaries.

## Testing Guidelines
- Framework: `pytest`; name tests `test_<feature>`.
- New tiling logic must call `Mapping.check_all()` in tests; include negative cases for validation paths.
- Aim for >85% coverage on modified files; document gaps in PR if lower.
- Keep fixtures minimal; prioritize example‑driven checks mirroring `test_grid_tiling.py`.

## Commit & Pull Request Guidelines
- Commits: imperative mood, subject ≤50 chars (e.g., "Add grid balancer"); elaborate in body when trade‑offs exist. Reference issues (e.g., `Fixes #42`).
- PRs: summarize behavior changes, list validation commands, and attach Perfetto traces or CLI screenshots when timing changes.
- If touching vendored packages, open PR early for owners to audit divergences.

## Simulation & Tracing Notes
- Instrument via `PerfettoTracer.get_global_tracer()`; scope around `SimModule.wait_time`.
- Generated traces open at `ui.perfetto.dev`; add a short walkthrough when trace reading informs acceptance.
