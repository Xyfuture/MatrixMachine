#!/usr/bin/env python3
"""
Batch test for multiple matrix configurations
"""
import logging
from matrixmachine.core.description import ComputeDieSpec, ChipSpec, Chip
from matrixmachine.strategy.agent_grid_search import AgentGridSearchStrategy
from matrixmachine.core.utils import calculate_compute_utilization
from matrixmachine.workload.matrix.models import (
    LLAMA3_8B_Q_PROJ,
    LLAMA3_8B_O_PROJ,
    LLAMA3_8B_GATE_UP,
    LLAMA32_3B_Q_PROJ,
    LLAMA32_3B_O_PROJ,
    GEMMA2_2B_Q_PROJ,
    PHI2_Q_PROJ,
)

logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("agent_grid_search").setLevel(logging.CRITICAL)

def test_matrix_batch():
    """Test multiple matrix configurations and report best utilization."""
    print("Matrix Batch Test - Computing Best Utilization")
    print("=" * 60)

    die_spec = ComputeDieSpec(
        compute_power=128,
        input_bandwidth=128,
        output_bandwidth=128,
        memory_bandwidth=128,
    )
    chip_spec = ChipSpec(die_count=8, die_spec=die_spec)
    chip = Chip.create_from_spec(chip_spec)

    test_matrices = [
        ("LLAMA3_8B_Q_PROJ", LLAMA3_8B_Q_PROJ),
        ("LLAMA3_8B_O_PROJ", LLAMA3_8B_O_PROJ),
        ("LLAMA3_8B_GATE_UP", LLAMA3_8B_GATE_UP),
        ("LLAMA32_3B_Q_PROJ", LLAMA32_3B_Q_PROJ),
        ("LLAMA32_3B_O_PROJ", LLAMA32_3B_O_PROJ),
        ("GEMMA2_2B_Q_PROJ", GEMMA2_2B_Q_PROJ),
        ("PHI2_Q_PROJ", PHI2_Q_PROJ),
    ]

    strategy = AgentGridSearchStrategy(
        num_split_row_candidates=[2, 3, 4, 6, 8],
        num_split_col_candidates=[2, 3, 4, 6, 8],
        max_iterations=2,
    )

    results = []

    for name, matrix in test_matrices:
        print(f"\n{'─' * 60}")
        print(f"Testing: {name}")
        print(f"Matrix: {matrix.rows}×{matrix.cols}")

        result = strategy.find_optimal_mapping(matrix, chip)

        if result:
            utilization = calculate_compute_utilization(result)
            results.append((name, matrix, utilization, result.latency))
            print(f"✓ Latency: {result.latency:,} cycles")
            print(f"✓ Utilization: {utilization:.2%}")
        else:
            results.append((name, matrix, 0.0, float('inf')))
            print(f"✗ No valid mapping found")

    print(f"\n{'=' * 60}")
    print("Summary - Best Compute Utilization")
    print(f"{'=' * 60}")
    print(f"{'Matrix':<20} {'Dimensions':<15} {'Utilization':<15} {'Latency':<15}")
    print(f"{'-' * 60}")

    for name, matrix, util, lat in sorted(results, key=lambda x: x[2], reverse=True):
        dim = f"{matrix.rows}×{matrix.cols}"
        lat_str = f"{lat:,.0f}" if lat != float('inf') else "N/A"
        print(f"{name:<20} {dim:<15} {util:>12.2%}   {lat_str:>12}")


if __name__ == "__main__":
    test_matrix_batch()