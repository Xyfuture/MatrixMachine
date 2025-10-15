# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MatrixMachine is a discrete event simulation framework for matrix computation on hardware accelerators. The project simulates matrix operations distributed across multiple compute dies with configurable compute power and bandwidth constraints.

这个项目旨在解决一个问题, 找到矩阵在给定硬件上最快的执行方式, 找到一个映射 mapping, 能够将矩阵拆分到有个 ComputeDie 上, 实现最快的执行速度, 达成最高的算力利用率.
我现在的硬件架构(Chip)是这样的:
-  整体上看是一个中心化的架构, 有用一个中心的 IO die, 和 c 个 compute die,  每个 compute die 通过独立的链路与 IO die 相连. 这个架构用来做矩阵运算
-  每个 Compute Die 拥有一个独立的 Memory, 拥有 4 个参数, 分别是 与 IOdie 的输入/输出带宽, 自身 Memory 的带宽, 自身的算力. 
-  ComputeDie 输入, 运算, 输出是能够流水运算的
-  IO Die 每次给 Compute Die 发送输入数据, 多个 ComputeDie 并行运算, 最后 IO Die 收集输出结果, IO Die 上拥有 reduce 模块, 可以对 Compute Die 的输出结果按需要进行累加. 
我现在有一个任务, 我有一个大的矩阵, 矩阵的尺寸是 M x N,  我需要将这个矩阵拆分为多个 tile, 然后将 tile 映射到多个 Compute Die 上, 然后以 执行时间最长的 Compute Die 的执行时间作为最终的延迟. 

在 matrixmachine/core/description.py 中, 定义了矩阵的描述, tile 的描述, 硬件的描述 和 mapping 的描述.
在 matrixmachine/core/sim_engine.py 中, 定义了模拟引擎, 模拟引擎使用 Desim 框架, 可以在给定硬件和 mapping 的情况下仿真出运行时间.
在 matrixmachine/core/utils.py 中, 封装了算力利用率计算函数.
在 matrixmachine/strategy/ 目录下放置了各种映射算法. 
在 matrixmachine/workload/ 目录下存放了一些硬件配置参数和一些用于测试的矩阵.

## Common Commands

### Development Setup
```bash
# Install Desim dependency
cd packages/Desim && pip install -e .

# Install PerfTracer dependency
cd packages/PerfTracer && pip install -e .

# Activate virtual environment (if present)
source .venv/bin/activate  # On macOS/Linux
```

### Running Simulations
```bash
# Basic simulation example
python main.py

# Agent Grid Search test (configurable)
python agent_test.py --die-count 8 --batch-size 16

# H2-LLM strategy utilization test
python test_h2llm_utilization.py

# Agent strategy utilization test
python test_agent_utilization.py

# Simulation engine tests
python sim_engine_test.py

# H2-LLM demonstration
python h2.py
```

### Testing and Code Quality
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

### Performance Analysis
```bash
# Generate and view traces (requires Chrome/Chromium)
# Open trace files at: https://ui.perfetto.dev
# Trace files are saved in ./trace/ directory
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
   - `trivial.py`: Basic tiling implementation for simple use cases
   - `agent_grid_search.py`: Advanced recursive DSE strategy using grid splits and round-robin assignment
   - `h2llm_mapping.py`: H2-LLM analytical tiling strategy based on ISCA 2025 paper

5. **packages/Desim/**: Discrete event simulation library (SystemC-like)
   - Core simulation engine with coroutines and time management
   - Modules: FIFO, Pipeline, Memory
   - Sync primitives for coordination

6. **packages/PerfTracer/**: Performance tracing library
   - Generates Perfetto-compatible trace files
   - Supports scoped events and complete events
   - Chrome Trace Event JSON format

7. **matrixmachine/workload/**: Hardware configurations and test matrices
   - `hardware/h2llm.py`: H2-LLM hardware configuration with configurable compute dies
   - `matrix/models.py`: Real-world matrix models (Llama variants, transformer layers)

### Key Design Patterns

- **Tile-based decomposition**: Large matrices are divided into rectangular tiles with 3D batch support
- **Half-open intervals**: Tiles use [start, end) ranges for easy concatenation across all dimensions
- **Validation**: Mapping class provides comprehensive validation for bounds, overlap, and coverage
- **Coroutine-based simulation**: Uses greenlets for concurrent simulation of compute dies
- **Performance tracing**: All simulation events are traced for performance analysis
- **Modular strategies**: Tiling strategies are pluggable and configurable
- **Compute utilization analysis**: Built-in metrics for hardware efficiency evaluation
- **Recursive DSE**: Agent Grid Search uses recursive design space exploration with memoization
- **Analytical optimization**: H2-LLM strategy uses closed-form solutions for optimal tiling factors
- **Round-robin assignment**: Load balancing through intelligent tile distribution strategies

### Simulation Flow

1. Define matrix shape (with optional 3D batch dimensions) and compute die configurations
2. Choose and configure a tiling strategy (grid-based or trivial)
3. Create mapping of tiles to compute dies with validation
4. Run simulation with Desim engine
5. Generate Perfetto traces and compute utilization metrics for analysis

## Entry Points

- **main.py**: Basic simulation example with default configurations
- **agent_test.py**: Configurable Agent Grid Search utilization testing with command-line arguments
- **test_h2llm_utilization.py**: H2-LLM strategy performance testing
- **test_agent_utilization.py**: Agent strategy utilization testing
- **sim_engine_test.py**: Simulation engine testing and validation
- **h2.py**: H2-LLM demonstration and hardware showcase

## Development Notes

- Python 3.8+ required
- Uses type hints extensively
- Dataclasses for immutable configuration objects
- Simulation time is measured in cycles
- Performance traces can be viewed at https://ui.perfetto.dev
- Virtual environment recommended (`.venv/` directory present)
- Logging configured for detailed algorithm execution analysis

## Advanced Features

### Agent Grid Search Strategy
- **Recursive Design Space Exploration**: Explores multiple tiling configurations recursively
- **Round-Robin Assignment**: Distributes tiles evenly across compute dies
- **Memoization**: Caches results to avoid redundant computation
- **Tail Region Handling**: Geometrically precise handling of remaining tiles
- **Configurable Parameters**: Split candidates, iteration limits, fallback options

### H2-LLM Strategy
- **Analytical Optimization**: Uses closed-form solutions from ISCA 2025 paper
- **Bandwidth-Aware Tiling**: Considers input/output bandwidth constraints
- **Automatic Parameter Extraction**: Derives bandwidth from hardware specifications
- **Multi-Dimension Splitting**: Optimizes across K and N dimensions
- **Batch Processing**: Supports 3D matrix shapes with batch dimensions

### Hardware Configurations
- **H2-LLM Chip**: Configurable number of compute dies with realistic specifications
- **Shared vs Separate Bandwidth**: Support for both shared I/O and separate input/output bandwidth
- **Memory Hierarchy**: Internal memory bandwidth vs external I/O bandwidth modeling
- **Real-World Models**: Llama transformer layer dimensions for testing

## File Structure

```
MatrixMachine/
├── main.py                     # Basic simulation entry point
├── agent_test.py               # Configurable Agent Grid Search testing
├── test_h2llm_utilization.py   # H2-LLM strategy performance testing
├── test_agent_utilization.py   # Agent strategy utilization testing
├── sim_engine_test.py          # Simulation engine testing
├── h2.py                       # H2-LLM demonstration
├── matrixmachine/              # Core project package
│   ├── __init__.py            # Package exports
│   ├── core/                  # Core simulation components
│   │   ├── description.py     # Data model and hardware specs
│   │   ├── sim_engine.py      # Simulation engine
│   │   └── utils.py           # Analysis utilities
│   ├── strategy/              # Tiling strategies
│   │   ├── trivial.py         # Basic tiling implementation
│   │   ├── agent_grid_search.py # Advanced recursive DSE strategy
│   │   └── h2llm_mapping.py   # H2-LLM analytical strategy
│   └── workload/              # Hardware configurations and test matrices
│       ├── hardware/          # Hardware specifications
│       │   └── h2llm.py       # H2-LLM chip configuration
│       └── matrix/            # Test matrix models
│           └── models.py      # Real-world matrix dimensions
├── tests/                      # Test files
│   └── test_grid_tiling.py    # Grid tiling examples and tests
├── trace/                      # Performance trace files output
├── docs/                       # Documentation
│   ├── H2LLM_README.md        # H2-LLM implementation details
│   ├── matrix_shape.md         # Matrix shape specifications
│   └── log_usage.md           # Logging configuration
└── packages/                   # Third-party dependencies
    ├── Desim/                 # Discrete event simulation library
    └── PerfTracer/            # Performance tracing library
```