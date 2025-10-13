#!/usr/bin/env python3
"""
Agent Grid Search Hardware Utilization Test

This script evaluates the Agent Grid Search mapping strategy on an 8-die chip configuration
and reports compute utilization for various model matrices.
"""

import argparse
import logging
import os
from typing import List, Tuple

from matrixmachine.core.description import MatrixShape, DataFormat, DataType
from matrixmachine.core.sim_engine import simulate
from matrixmachine.strategy.agent_grid_search import AgentGridSearchStrategy
from matrixmachine.workload.hardware.h2llm import create_h2llm_chip
from matrixmachine.workload.matrix import models

# Configure logging to suppress all warnings
# logging.basicConfig(level=logging.ERROR)

agent_grid_search_logger = logging.getLogger("agent_grid_search")
agent_grid_search_logger.disabled = True



def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Agent Grid Search Hardware Utilization Test')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed mapping information for each matrix')
    parser.add_argument('--save-trace', '-t', action='store_true',
                       help='Save trace files for best mappings to trace/ directory')
    args = parser.parse_args()

    print("\n" + "="*90)
    print("Agent Grid Search 8-Die Chip Utilization Test")
    print("="*90)
    print(f"Hardware: 8-die H2-LLM chip (9.6 TFLOPS total, 12.5 GB/s I/O per die)")
    print(f"Strategy: Agent Grid Search (Recursive DSE)")
    print(f"Batch size: 16")
    print(f"Data format: FP16 (input/output), INT4 (weight)")
    print(f"Verbose mapping: {args.verbose}")
    print(f"Save trace files: {args.save_trace}")
    print("="*90)

    # Create chip and strategy
    chip = create_h2llm_chip(die_count=8)
    strategy = AgentGridSearchStrategy(
        num_split_row_candidates=[1,2,4,6,8],
        num_split_col_candidates=[1,2,4,6,8],
        max_iterations=2
    )
    batch_size = 16

    # Configure data format: FP16 for input/output, INT4 for weight
    data_format = DataFormat(
        input_dtype=DataType.FP16,
        output_dtype=DataType.FP16,
        weight_dtype=DataType.INT4
    )

    # Select representative matrices
    test_matrices: List[Tuple[str, MatrixShape]] = [
        ("Llama-3.2 1B Q_PROJ", models.LLAMA32_1B_Q_PROJ),
        ("Llama-3.2 1B GATE_UP", models.LLAMA32_1B_GATE_UP),
        ("Gemma-2 2B Q_PROJ", models.GEMMA2_2B_Q_PROJ),
        ("Llama-3 8B Q_PROJ", models.LLAMA3_8B_Q_PROJ),
        ("Llama-3 8B GATE_UP", models.LLAMA3_8B_GATE_UP),
        ("Qwen3 8B GATE_UP", models.QWEN3_8B_GATE_UP),
        ("Llama-3 70B Q_PROJ", models.LLAMA3_70B_Q_PROJ),
        ("Llama-3 70B GATE_UP", models.LLAMA3_70B_GATE_UP),
        ("Llama-3 8B K_PROJ (GQA)", models.LLAMA3_8B_K_PROJ),
        ("Llama-3 70B K_PROJ (GQA)", models.LLAMA3_70B_K_PROJ),
    ]

    # Print table header
    if args.save_trace:
        print(f"\n{'Matrix Name':<26} {'Shape':<18} {'Latency':<12} {'Utilization':<13} {'Tiles':<8} {'Trace':<7}")
        print("-" * 97)
    else:
        print(f"\n{'Matrix Name':<26} {'Shape':<18} {'Latency':<12} {'Utilization':<13} {'Tiles':<8}")
        print("-" * 90)

    # Run tests and collect results
    results = []
    for matrix_name, matrix_shape in test_matrices:
        # print(f"\n[{idx}/{len(test_matrices)}] Testing {matrix_name}...", flush=True)

        # Create matrix with batch size and custom data format
        matrix_with_batch = MatrixShape(
            rows=matrix_shape.rows,
            cols=matrix_shape.cols,
            batch_size=batch_size,
            data_format=data_format
        )

        # Find optimal mapping and simulate
        result = strategy.find_optimal_mapping(matrix_with_batch, chip)

        if result is None:
            print(f"{matrix_name:<26} FAILED")
            continue

        # Save trace file for the best mapping if requested
        if args.save_trace:
            trace_filename = f"trace/{matrix_with_batch.rows}x{matrix_with_batch.cols}x{batch_size}.json"
            try:
                # Simulate again with trace saving enabled
                simulate(chip, result.mapping, save_trace=True, trace_filename=trace_filename)
            except Exception as e:
                print(f"Warning: Failed to save trace for {matrix_name}: {e}")

        # Format output
        shape_str = f"{matrix_with_batch.rows}×{matrix_with_batch.cols}×{batch_size}"
        latency_str = f"{result.latency:.0f} cycles"
        utilization = result.get_compute_utilization()
        util_str = f"{utilization:.2%}"
        tiles_str = f"{len(result.mapping.tiles)}"

        output_line = f"{matrix_name:<26} {shape_str:<18} {latency_str:<12} {util_str:<13} {tiles_str:<8}"
        if args.save_trace:
            output_line += " ✓trace"
        print(output_line)
        results.append((matrix_name, utilization))

        # Print detailed mapping information if verbose mode is enabled
        if args.verbose:
            result.mapping.display()

    # Calculate and print summary
    if results:
        avg_utilization = sum(util for _, util in results) / len(results)
        print("="*90)
        print(f"\nSummary Statistics:")
        print(f"  Total matrices tested: {len(results)}")
        print(f"  Average utilization: {avg_utilization:.2%}")
        print(f"  Total compute power: {chip.total_compute_power} TFLOPS")
        print(f"  Strategy: Recursive grid search with dynamic tiling")
        print(f"  Max recursion depth: {strategy.max_iterations}")
        print("="*90 + "\n")


if __name__ == "__main__":
    main()
