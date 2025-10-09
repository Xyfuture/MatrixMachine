#!/usr/bin/env python3
"""
Batch test comparing CodeSearchStrategy vs AgentGridSearchStrategy.
Uses the same hardware and matrix set as batch_test_matrices.py.
"""

import logging
from matrixmachine.core.description import ComputeDieSpec, ChipSpec, Chip
from matrixmachine.strategy.code_search import CodeSearchStrategy
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


def compare_batch():
    print("CodeSearch vs AgentGridSearch - Utilization Comparison")
    print("=" * 70)

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

    agent = AgentGridSearchStrategy(
        num_split_row_candidates=[2, 3, 4, 6, 8],
        num_split_col_candidates=[2, 3, 4, 6, 8],
        max_iterations=2,
    )
    code = CodeSearchStrategy(
        num_split_row_candidates=[1, 2, 3, 4, 6, 8],
        num_split_col_candidates=[1, 2, 3, 4, 6, 8],
    )

    rows = []
    for name, matrix in test_matrices:
        print(f"\n{'─' * 70}")
        print(f"Testing: {name}  ({matrix.rows}×{matrix.cols})")

        a = agent.find_optimal_mapping(matrix, chip)
        c = code.find_optimal_mapping(matrix, chip)

        if a is not None:
            au = calculate_compute_utilization(a)
            alat = a.latency
        else:
            au, alat = 0.0, float("inf")

        if c is not None:
            cu = calculate_compute_utilization(c)
            clat = c.latency
        else:
            cu, clat = 0.0, float("inf")

        rows.append((name, matrix, au, alat, cu, clat))
        print(f"Agent  -> util: {au:>8.2%}  latency: {alat if alat!=float('inf') else 'N/A'}")
        print(f"Code   -> util: {cu:>8.2%}  latency: {clat if clat!=float('inf') else 'N/A'}")

    print(f"\n{'=' * 70}")
    print("Summary - Higher utilization expected for CodeSearch")
    print(f"{'=' * 70}")
    print(f"{'Matrix':<22} {'Dims':<15} {'Agent':>10} {'Code':>10} {'Better?':>10}")
    print(f"{'-' * 70}")
    for name, matrix, au, alat, cu, clat in rows:
        dim = f"{matrix.rows}×{matrix.cols}"
        better = "Yes" if cu >= au else "No"
        print(f"{name:<22} {dim:<15} {au:>9.2%} {cu:>10.2%} {better:>10}")


if __name__ == "__main__":
    compare_batch()

