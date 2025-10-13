#!/usr/bin/env python3
"""
H2-LLM Hardware Utilization Test

This script evaluates the H2-LLM mapping strategy on an 8-die chip configuration
and reports compute utilization for various model matrices.
"""

import argparse
import logging
from typing import List, Tuple

from matrixmachine.core.description import MatrixShape, DataFormat, DataType
from matrixmachine.strategy.h2llm_mapping import H2LLMTilingStrategy
from matrixmachine.workload.hardware.h2llm import create_h2llm_chip
from matrixmachine.workload.matrix import models

# Configure logging to reduce verbosity
# logging.basicConfig(level=logging.WARNING)

h2llm_logger = logging.getLogger("h2llm_mapping")
h2llm_logger.disabled = True    


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='H2-LLM Hardware Utilization Test')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed mapping information for each matrix')
    args = parser.parse_args()

    print("\n" + "="*90)
    print("H2-LLM 8-Die Chip Utilization Test")
    print("="*90)
    print(f"Hardware: 8-die H2-LLM chip (9.6 TFLOPS total, 12.5 GB/s I/O per die)")
    print(f"Strategy: H2-LLM Analytical Tiling")
    print(f"Batch size: 16")
    print(f"Data format: FP16 (input/output), INT4 (weight)")
    print(f"Verbose mapping: {args.verbose}")
    print("="*90)

    # Create chip and strategy
    chip = create_h2llm_chip(die_count=8)
    strategy = H2LLMTilingStrategy(element_size=2.0)
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
    print(f"\n{'Matrix Name':<26} {'Shape':<18} {'Latency':<12} {'Utilization':<13} {'Tiles':<8}")
    print("-" * 90)

    # Run tests and collect results
    results = []
    for matrix_name, matrix_shape in test_matrices:
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

        # Format output
        shape_str = f"{matrix_with_batch.rows}×{matrix_with_batch.cols}×{batch_size}"
        latency_str = f"{result.latency:.0f} cycles"
        utilization = result.get_compute_utilization()
        util_str = f"{utilization:.2%}"
        tiles_str = f"{len(result.mapping.tiles)}"

        print(f"{matrix_name:<26} {shape_str:<18} {latency_str:<12} {util_str:<13} {tiles_str:<8}")
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
        print(f"  Tiles per matrix: 8 (1 per die)")
        print(f"  Tiling pattern: T_K=4, T_N=2")
        print("="*90 + "\n")


if __name__ == "__main__":
    main()