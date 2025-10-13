#!/usr/bin/env python3
"""
Agent Grid Search Hardware Utilization Test (Configurable)

This script evaluates the Agent Grid Search mapping strategy on configurable chip configurations
with configurable test matrices.
"""

import argparse
import logging
import os
from datetime import datetime
from typing import List, Tuple

from matrixmachine.core.description import MatrixShape, DataFormat, DataType
from matrixmachine.core.sim_engine import simulate
from matrixmachine.strategy.agent_grid_search import AgentGridSearchStrategy
from matrixmachine.workload.hardware.h2llm import create_h2llm_chip
from matrixmachine.workload.matrix import models

# Configure logging to show detailed algorithm execution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

agent_grid_search_logger = logging.getLogger("agent_grid_search")
# agent_grid_search_logger.disabled = True



# ================================
# CONFIGURATION PARAMETERS
# ================================

# Hardware Configuration
HARDWARE_CONFIG = {
    'die_count': 5,  # Number of compute dies
    'chip_name': 'H2-LLM',  # Chip name for display
    'compute_power': 9.6,  # Total compute power in TFLOPS
    'io_bandwidth': 12.5,  # I/O bandwidth per die in GB/s
}

# Strategy Configuration
STRATEGY_CONFIG = {
    'num_split_row_candidates': [2,],  # Row split candidates
    'num_split_col_candidates': [3],  # Column split candidates
    'max_iterations': 1,  # Maximum recursion depth
    'strategy_name': 'Agent Grid Search (Recursive DSE)',  # Strategy name for display
}

# Matrix Configuration
MATRIX_CONFIG = {
    'batch_size': 16,  # Batch size for all matrices
    'input_dtype': DataType.FP16,  # Input data type
    'output_dtype': DataType.FP16,  # Output data type
    'weight_dtype': DataType.INT4,  # Weight data type
    'test_matrices': [
        ("Llama-3.2 1B Q_PROJ", models.LLAMA32_1B_Q_PROJ),
        # Uncomment to add more test matrices
        # ("Llama-3 8B Q_PROJ", models.LLAMA3_8B_Q_PROJ),
        # ("Llama-3 8B GATE_UP", models.LLAMA3_8B_GATE_UP),
        # ("Llama-3 70B Q_PROJ", models.LLAMA3_70B_Q_PROJ),
        # ("Llama-3 8B K_PROJ (GQA)", models.LLAMA3_8B_K_PROJ),
    ]
}

# Output Configuration
OUTPUT_CONFIG = {
    'show_detailed_mapping': True,  # Show detailed mapping information
    'show_summary_statistics': True,  # Show summary statistics
    'show_hardware_info': True,  # Show hardware configuration
    'show_algorithm_info': True,  # Show algorithm configuration
    'table_width': 90,  # Width for table formatting
    'matrix_name_width': 26,  # Width for matrix name column
    'utilization_width': 15,  # Width for utilization column
}

# Trace Configuration
TRACE_CONFIG = {
    'enable_trace': False,  # Enable trace generation
    'save_trace_file': True,  # Save trace to file
    'trace_directory': 'trace',  # Directory to save trace files
    'trace_filename_prefix': 'agent_test',  # Prefix for trace filenames
    'show_trace_info': True,  # Show trace file information in output
}


def print_header():
    """Print test header with configuration information."""
    width = OUTPUT_CONFIG['table_width']

    print("\n" + "="*width)
    print(f"Agent Grid Search {HARDWARE_CONFIG['die_count']}-Die Chip Utilization Test (Configurable)")
    print("="*width)

    if OUTPUT_CONFIG['show_hardware_info']:
        print(f"Hardware: {HARDWARE_CONFIG['die_count']}-die {HARDWARE_CONFIG['chip_name']} chip "
              f"({HARDWARE_CONFIG['compute_power']} TFLOPS total, {HARDWARE_CONFIG['io_bandwidth']} GB/s I/O per die)")

    if OUTPUT_CONFIG['show_algorithm_info']:
        print(f"Strategy: {STRATEGY_CONFIG['strategy_name']}")
        print(f"Batch size: {MATRIX_CONFIG['batch_size']}")
        print(f"Data format: {MATRIX_CONFIG['input_dtype'].name} (input/output), {MATRIX_CONFIG['weight_dtype'].name} (weight)")

    print(f"Verbose mapping: {OUTPUT_CONFIG['show_detailed_mapping']}")
    print("="*width)


def print_result_summary(matrix_name: str, matrix_shape: MatrixShape, result, batch_size: int):
    """Print formatted result summary for a single matrix."""
    width = OUTPUT_CONFIG['table_width']

    # Format output strings
    shape_str = f"{matrix_shape.rows}×{matrix_shape.cols}×{batch_size}"
    latency_str = f"{result.latency:.0f} cycles"
    utilization = result.get_compute_utilization()
    util_str = f"{utilization:.2%}"
    tiles_str = f"{len(result.mapping.tiles)}"

    print(f"\n{'='*width}")
    print(f"RESULT: {matrix_name}")
    print(f"  Shape: {shape_str}")
    print(f"  Latency: {latency_str}")
    print(f"  Utilization: {util_str}")
    print(f"  Total tiles: {tiles_str}")
    print(f"{'='*width}\n")

    return utilization


def print_summary_statistics(results: List[Tuple], chip, strategy):
    """Print summary statistics table."""
    if not OUTPUT_CONFIG['show_summary_statistics'] or not results:
        return

    width = OUTPUT_CONFIG['table_width']
    name_width = OUTPUT_CONFIG['matrix_name_width']
    util_width = OUTPUT_CONFIG['utilization_width']

    # Calculate average utilization (handle both old and new result formats)
    total_utilization = 0
    count = 0
    for result in results:
        if len(result) >= 2:
            total_utilization += result[1]  # utilization is second element
            count += 1

    avg_utilization = total_utilization / count if count > 0 else 0

    print("\n" + "="*width)
    print("SUMMARY STATISTICS")
    print("="*width)

    # Check if we have trace information
    has_trace = any(len(result) >= 3 for result in results)

    if has_trace:
        print(f"\nResults Table:")
        print(f"{'Matrix Name':<{name_width}} {'Utilization':<{util_width}} {'Trace File':<25}")
        print("-" * width)
        for result in results:
            if len(result) >= 3:
                matrix_name, utilization, trace_info = result
                trace_file = os.path.basename(trace_info['trace_file']) if trace_info and trace_info.get('trace_file') else "No trace"
                print(f"{matrix_name:<{name_width}} {utilization:>{util_width-2}.2%} {trace_file:<25}")
            else:
                matrix_name, utilization = result
                print(f"{matrix_name:<{name_width}} {utilization:>{util_width-2}.2%} {'No trace':<25}")
        print("-" * width)
    else:
        print(f"\nResults Table:")
        print(f"{'Matrix Name':<{name_width}} {'Utilization':<{util_width}}")
        print("-" * width)
        for result in results:
            matrix_name, utilization = result[:2]  # Take first two elements
            print(f"{matrix_name:<{name_width}} {utilization:>{util_width-2}.2%}")
        print("-" * width)

    print(f"{'Average':<{name_width}} {avg_utilization:>{util_width-2}.2%}\n")

    if OUTPUT_CONFIG['show_hardware_info']:
        print(f"Hardware Configuration:")
        print(f"  Total dies: {len(chip.compute_dies)}")
        print(f"  Total compute power: {chip.total_compute_power} TFLOPS")

    if OUTPUT_CONFIG['show_algorithm_info']:
        print(f"\nAlgorithm Configuration:")
        print(f"  Strategy: {STRATEGY_CONFIG['strategy_name']}")
        print(f"  Max recursion depth: {strategy.max_iterations}")
        print(f"  Split candidates (row): {strategy.num_split_row_candidates}")
        print(f"  Split candidates (col): {strategy.num_split_col_candidates}")

    if TRACE_CONFIG['enable_trace'] and has_trace:
        print(f"\nTrace Configuration:")
        print(f"  Trace generation: {'Enabled' if TRACE_CONFIG['enable_trace'] else 'Disabled'}")
        print(f"  Trace directory: {TRACE_CONFIG['trace_directory']}")
        print(f"  Total trace files: {sum(1 for r in results if len(r) >= 3 and r[2] and r[2].get('trace_file'))}")

    print("="*width + "\n")


def setup_trace_directory():
    """Create trace directory if it doesn't exist."""
    if TRACE_CONFIG['save_trace_file'] and TRACE_CONFIG['trace_directory']:
        os.makedirs(TRACE_CONFIG['trace_directory'], exist_ok=True)


def generate_trace_filename(matrix_name: str) -> str:
    """Generate trace filename based on matrix name and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean matrix name for filename
    clean_name = matrix_name.replace(" ", "_").replace("-", "_").replace(".", "_")
    filename = f"{TRACE_CONFIG['trace_filename_prefix']}_{clean_name}_{timestamp}.json"

    if TRACE_CONFIG['trace_directory']:
        filename = os.path.join(TRACE_CONFIG['trace_directory'], filename)

    return filename


def run_trace_simulation(matrix_name: str, chip, result):
    """Run detailed simulation with trace generation."""
    if not TRACE_CONFIG['enable_trace']:
        return None

    trace_filename = None
    if TRACE_CONFIG['save_trace_file']:
        trace_filename = generate_trace_filename(matrix_name)

    try:
        print(f"Running detailed simulation with trace generation...")
        sim_cycles = simulate(
            chip=chip,
            mapping=result.mapping,
            save_trace=TRACE_CONFIG['save_trace_file'],
            trace_filename=trace_filename or "temp_trace.json"
        )

        if TRACE_CONFIG['show_trace_info']:
            print(f"  Simulation cycles: {sim_cycles}")
            if trace_filename:
                print(f"  Trace file saved: {trace_filename}")
                file_size = os.path.getsize(trace_filename) if os.path.exists(trace_filename) else 0
                print(f"  Trace file size: {file_size:,} bytes")

        return {
            'cycles': sim_cycles,
            'trace_file': trace_filename,
            'file_size': os.path.getsize(trace_filename) if trace_filename and os.path.exists(trace_filename) else 0
        }

    except Exception as e:
        print(f"  Warning: Trace simulation failed: {e}")
        return None


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Agent Grid Search Hardware Utilization Test (Configurable)')
    parser.add_argument('--die-count', type=int, default=HARDWARE_CONFIG['die_count'],
                       help=f'Number of compute dies (default: {HARDWARE_CONFIG["die_count"]})')
    parser.add_argument('--batch-size', type=int, default=MATRIX_CONFIG['batch_size'],
                       help=f'Batch size for matrices (default: {MATRIX_CONFIG["batch_size"]})')
    parser.add_argument('--max-iterations', type=int, default=STRATEGY_CONFIG['max_iterations'],
                       help=f'Maximum recursion depth (default: {STRATEGY_CONFIG["max_iterations"]})')
    parser.add_argument('--enable-trace', action='store_true', default=TRACE_CONFIG['enable_trace'],
                       help='Enable trace generation')
    parser.add_argument('--no-trace', action='store_true',
                       help='Disable trace generation')
    parser.add_argument('--trace-dir', type=str, default=TRACE_CONFIG['trace_directory'],
                       help=f'Trace file directory (default: {TRACE_CONFIG["trace_directory"]})')

    args = parser.parse_args()

    # Override configurations with command line arguments
    HARDWARE_CONFIG['die_count'] = args.die_count
    MATRIX_CONFIG['batch_size'] = args.batch_size
    STRATEGY_CONFIG['max_iterations'] = args.max_iterations

    # Handle trace configuration
    if args.no_trace:
        TRACE_CONFIG['enable_trace'] = False
        TRACE_CONFIG['save_trace_file'] = False
    elif args.enable_trace:
        TRACE_CONFIG['enable_trace'] = True
        TRACE_CONFIG['save_trace_file'] = True

    if args.trace_dir:
        TRACE_CONFIG['trace_directory'] = args.trace_dir

    # Setup trace directory
    setup_trace_directory()

    print_header()

    # Create chip and strategy
    chip = create_h2llm_chip(die_count=HARDWARE_CONFIG['die_count'])
    strategy = AgentGridSearchStrategy(
        num_split_row_candidates=STRATEGY_CONFIG['num_split_row_candidates'],
        num_split_col_candidates=STRATEGY_CONFIG['num_split_col_candidates'],
        max_iterations=STRATEGY_CONFIG['max_iterations']
    )

    # Configure data format
    data_format = DataFormat(
        input_dtype=MATRIX_CONFIG['input_dtype'],
        output_dtype=MATRIX_CONFIG['output_dtype'],
        weight_dtype=MATRIX_CONFIG['weight_dtype']
    )

    # Run tests and collect results
    results = []
    test_matrices = MATRIX_CONFIG['test_matrices']
    batch_size = MATRIX_CONFIG['batch_size']
    width = OUTPUT_CONFIG['table_width']

    for idx, (matrix_name, matrix_shape) in enumerate(test_matrices, 1):
        print(f"\n{'#'*width}")
        print(f"# Test {idx}/{len(test_matrices)}: {matrix_name}")
        print(f"{'#'*width}\n")

        # Create matrix with batch size and custom data format
        matrix_with_batch = MatrixShape(
            rows=matrix_shape.rows,
            cols=matrix_shape.cols,
            batch_size=batch_size,
            data_format=data_format
        )

        print(f"Matrix dimensions: {matrix_with_batch.rows}×{matrix_with_batch.cols}×{batch_size}")
        print(f"Available dies: {len(chip.compute_dies)}")
        print(f"Starting {STRATEGY_CONFIG['strategy_name']} optimization...\n")

        # Find optimal mapping and simulate
        result = strategy.find_optimal_mapping(matrix_with_batch, chip)

        if result is None:
            print(f"\n{matrix_name:<26} FAILED")
            continue

        # Print result summary using helper function
        utilization = print_result_summary(matrix_name, matrix_with_batch, result, batch_size)
        results.append((matrix_name, utilization))

        # Run detailed simulation with trace generation
        trace_info = run_trace_simulation(matrix_name, chip, result)
        if trace_info:
            results[-1] = (matrix_name, utilization, trace_info)  # Add trace info to results

        # Print detailed mapping information if verbose
        if OUTPUT_CONFIG['show_detailed_mapping']:
            print(f"Detailed mapping for {matrix_name}:")
            result.mapping.display()
            print()

    # Print summary statistics using helper function
    print_summary_statistics(results, chip, strategy)


if __name__ == "__main__":
    main()
