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

# Run design space exploration (DSE) example
python dse_main.py

# Run manual mapping example
python manual_mapping_example.py

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

1. **matrixmachine/core/description.py**: Defines the data model for matrix computation
   - `MatrixShape`: Matrix dimensions (rows, cols, batch_size) with 3D batch support
   - `Tile`: Submatrix regions with half-open intervals [row0, row1), [col0, col1), [batch0, batch1)
   - `ComputeDieSpec`: Hardware compute unit specifications (compute power, bandwidth)
   - `ChipSpec`: Hardware chip specification containing multiple compute dies
   - `Chip`: Container for multiple compute dies with validation
   - `Mapping`: Complete mapping of tiles to compute dies with comprehensive validation
   - `TileAssignmentInput`: Input specification for tile assignments

2. **matrixmachine/core/sim_engine.py**: Simulation engine using Desim framework
   - `SimComputeDie`: Simulates compute die behavior with input/compute/output phases
   - `simulate()`: Main simulation function that orchestrates the entire process
   - Uses FIFO queues for task processing
   - Integrates with PerfettoTracer for performance tracing

3. **matrixmachine/core/utils.py**: Utility functions for simulation analysis
   - `MappingResult`: Results from mapping validation and metrics
   - `calculate_compute_utilization()`: Computes hardware utilization metrics
   - Various helper functions for performance analysis

4. **matrixmachine/strategy/**: Tiling strategy implementations
   - `grid.py`: Grid-based tiling strategy with configurable tile sizes
   - `trivial.py`: Basic tiling implementation for simple use cases

5. **packages/Desim/**: Discrete event simulation library (SystemC-like)
   - Core simulation engine with coroutines and time management
   - Modules: FIFO, Pipeline, Memory
   - Sync primitives for coordination

6. **packages/PerfTracer/**: Performance tracing library
   - Generates Perfetto-compatible trace files
   - Supports scoped events and complete events
   - Chrome Trace Event JSON format

### Key Design Patterns

- **Tile-based decomposition**: Large matrices are divided into rectangular tiles with 3D batch support
- **Half-open intervals**: Tiles use [start, end) ranges for easy concatenation across all dimensions
- **Validation**: Mapping class provides comprehensive validation for bounds, overlap, and coverage
- **Coroutine-based simulation**: Uses greenlets for concurrent simulation of compute dies
- **Performance tracing**: All simulation events are traced for performance analysis
- **Modular strategies**: Tiling strategies are pluggable and configurable
- **Compute utilization analysis**: Built-in metrics for hardware efficiency evaluation

### Simulation Flow

1. Define matrix shape (with optional 3D batch dimensions) and compute die configurations
2. Choose and configure a tiling strategy (grid-based or trivial)
3. Create mapping of tiles to compute dies with validation
4. Run simulation with Desim engine
5. Generate Perfetto traces and compute utilization metrics for analysis

## Entry Points

- **main.py**: Basic simulation example with default configurations
- **dse_main.py**: Design space exploration script for testing different configurations
- **manual_mapping_example.py**: Example showing manual tile mapping for batch processing

## Development Notes

- Python 3.8+ required
- Uses type hints extensively
- Dataclasses for immutable configuration objects
- Simulation time is measured in cycles
- Performance traces can be viewed at ui.perfetto.dev

## File Structure

```
MatrixMachine/
├── main.py                     # Basic simulation entry point
├── dse_main.py                 # Design space exploration example
├── manual_mapping_example.py   # Manual tile mapping example
├── matrixmachine/              # Core project package
│   ├── __init__.py            # Package exports
│   ├── core/                  # Core simulation components
│   │   ├── description.py     # Data model and hardware specs
│   │   ├── sim_engine.py      # Simulation engine
│   │   └── utils.py           # Analysis utilities
│   └── strategy/              # Tiling strategies
│       ├── grid.py            # Grid-based tiling strategy
│       └── trivial.py         # Basic tiling implementation
├── tests/                      # Test files
│   └── test_grid_tiling.py    # Grid tiling examples and tests
└── packages/                   # Third-party dependencies
    ├── Desim/                 # Discrete event simulation library
    └── PerfTracer/            # Performance tracing library
```