# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MatrixMachine is a discrete event simulation framework for matrix computation on hardware accelerators. The project simulates matrix operations distributed across multiple compute dies with configurable compute power and bandwidth constraints.

## Common Commands

### Development Setup
```bash
# Install Desim dependency
cd packages/Desim && pip install -e .

# Install PerfTracer dependency
cd packages/PerfTracer && pip install -e .

# Run the main simulation
python main.py

# Run example tests
python tests/test_grid_tiling.py
```

### Code Quality
```bash
# Format code with black
black .

# Sort imports with isort
isort .

# Type checking with mypy
mypy .

# Run tests
pytest

# Run tests with coverage
pytest --cov=.

# Run single test file
pytest tests/test_grid_tiling.py
```

## Architecture

### Core Components

1. **matrixmachine/description.py**: Defines the data model for matrix computation
   - `MatrixShape`: Matrix dimensions (rows, cols)
   - `Tile`: Submatrix regions with half-open intervals [row0, row1), [col0, col1)
   - `ComputeDie`: Hardware compute unit with power and bandwidth specs
   - `Chip`: Container for multiple compute dies
   - `Mapping`: Complete mapping of tiles to compute dies with validation

2. **matrixmachine/sim_engine.py**: Simulation engine using Desim framework
   - `SimComputeDie`: Simulates compute die behavior with input/compute/output phases
   - Uses FIFO queues for task processing
   - Integrates with PerfettoTracer for performance tracing

3. **packages/Desim/**: Discrete event simulation library (SystemC-like)
   - Core simulation engine with coroutines and time management
   - Modules: FIFO, Pipeline, Memory
   - Sync primitives for coordination

4. **packages/PerfTracer/**: Performance tracing library
   - Generates Perfetto-compatible trace files
   - Supports scoped events and complete events
   - Chrome Trace Event JSON format

### Key Design Patterns

- **Tile-based decomposition**: Large matrices are divided into rectangular tiles
- **Half-open intervals**: Tiles use [start, end) ranges for easy concatenation
- **Validation**: Mapping class provides comprehensive validation for bounds, overlap, and coverage
- **Coroutine-based simulation**: Uses greenlets for concurrent simulation of compute dies
- **Performance tracing**: All simulation events are traced for performance analysis

### Simulation Flow

1. Define matrix shape and compute die configurations
2. Create mapping of tiles to compute dies
3. Validate mapping (bounds, overlap, coverage)
4. Run simulation with Desim engine
5. Generate Perfetto traces for analysis

## Development Notes

- Python 3.8+ required
- Uses type hints extensively
- Dataclasses for immutable configuration objects
- Simulation time is measured in cycles
- Performance traces can be viewed at ui.perfetto.dev

## File Structure

```
MatrixMachine/
├── main.py                     # User entry point
├── matrixmachine/              # Core project package
│   ├── __init__.py            # Package exports
│   ├── description.py         # Core data model
│   ├── sim_engine.py          # Simulation engine
│   └── strategy/              # Tiling strategies
│       ├── __init__.py
│       ├── grid.py            # Grid-based tiling (placeholder)
│       └── trivial.py         # Basic tiling
├── tests/                      # Test files
│   ├── __init__.py
│   └── test_grid_tiling.py    # Grid tiling examples and tests
└── packages/                   # Third-party dependencies
    ├── Desim/                 # Discrete event simulation library
    └── PerfTracer/            # Performance tracing library
```