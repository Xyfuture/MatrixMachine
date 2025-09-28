#!/usr/bin/env python3
"""
Agent Grid Search Strategy Test

Test the new unified AgentGridSearchStrategy that combines the best features
from both GPT and Codex implementations.
"""

from matrixmachine.core.description import (
    MatrixShape,
    ComputeDieSpec,
    ChipSpec,
    Chip,
)
from matrixmachine.strategy.agent_grid_search import AgentGridSearchStrategy
from matrixmachine.core.utils import calculate_compute_utilization


def test_agent_grid_search():
    """Test the unified agent grid search strategy."""
    print("🤖 Agent Grid Search Strategy Test")
    print("="*40)

    # Test with 4K x 4K matrix
    matrix = MatrixShape(rows=4096, cols=4096, batch_size=1)
    print(f"Matrix: {matrix.rows}x{matrix.cols} ({matrix.area():,} elements)")

    # Standard hardware configuration
    die_spec = ComputeDieSpec(
        compute_power=128,  # 128 TFLOPS
        input_bandwidth=128,  # 128 GB/s
        output_bandwidth=128,  # 128 GB/s
        memory_bandwidth=1.0,  # 1.0 TB/s
    )
    chip_spec = ChipSpec(die_count=8, die_spec=die_spec)
    chip = Chip.create_from_spec(chip_spec)

    print(f"Hardware: {len(chip.compute_dies)} dies, {chip.total_compute_power} TFLOPS total")

    # Test with limited search space for fast execution
    strategy = AgentGridSearchStrategy(
        num_split_row_candidates=[2, 4],
        num_split_col_candidates=[2, 4],
    )

    print(f"Search space: {len(strategy.num_split_row_candidates) * len(strategy.num_split_col_candidates)} configurations")
    print("Starting optimization search...")

    try:
        result = strategy.find_optimal_mapping(matrix, chip)

        if result:
            print("\n✅ Optimal mapping found!")
            print(f"🚀 Execution latency: {result.latency:,} cycles")

            # Performance analysis
            utilization = calculate_compute_utilization(result)
            print(f"⚡ Compute utilization: {utilization:.2%}")
            print(f"🔢 Matrix operations: {result.get_matrix_operation_count():,}")
            print(f"💪 Total compute power: {result.get_chip_total_compute_power_gops():,} GOPS")

            # Workload distribution
            print(f"\n📊 Workload Distribution:")
            die_loads = result.mapping.die_loads()
            die_volumes = result.mapping.die_volumes()

            for die_id in sorted(die_loads.keys()):
                print(f"  {die_id}: {die_loads[die_id]} tiles, {die_volumes[die_id]:,} volume")

            # Balance analysis
            volumes = list(die_volumes.values())
            if volumes:
                max_vol = max(volumes)
                min_vol = min(volumes)
                balance_ratio = min_vol / max_vol if max_vol > 0 else 0
                imbalance = (1 - balance_ratio) * 100

                print(f"\n⚖️  Load Balance Analysis:")
                print(f"  Balance ratio: {balance_ratio:.3f}")
                print(f"  Load imbalance: {imbalance:.1f}%")

                if balance_ratio > 0.95:
                    print("  Status: ✅ Excellent balance")
                elif balance_ratio > 0.8:
                    print("  Status: ✅ Good balance")
                else:
                    print("  Status: ⚠️  Poor balance")

            # Algorithm performance
            print(f"\n🔍 Algorithm Performance:")
            print(f"  Cache entries: {len(strategy.memo)}")
            print(f"  Search efficiency: Memoization enabled")

            return True

        else:
            print("❌ No valid mapping found")
            return False

    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        return False


def test_small_matrix():
    """Quick test with small matrix for validation."""
    print("\n🔬 Small Matrix Validation Test")
    print("="*35)

    # Small 16x16 matrix
    matrix = MatrixShape(rows=16, cols=16, batch_size=1)
    print(f"Matrix: {matrix.rows}x{matrix.cols}")

    # 4 compute dies
    die_spec = ComputeDieSpec(
        compute_power=32,
        input_bandwidth=32,
        output_bandwidth=32,
        memory_bandwidth=0.5
    )
    chip_spec = ChipSpec(die_count=4, die_spec=die_spec)
    chip = Chip.create_from_spec(chip_spec)

    # Full search for small problem
    strategy = AgentGridSearchStrategy(
        num_split_row_candidates=[1, 2, 4],
        num_split_col_candidates=[1, 2, 4],
    )

    result = strategy.find_optimal_mapping(matrix, chip)

    if result:
        print("✅ Small matrix test passed!")
        utilization = calculate_compute_utilization(result)
        print(f"Latency: {result.latency} cycles, Utilization: {utilization:.1%}")
        return True
    else:
        print("❌ Small matrix test failed")
        return False


if __name__ == "__main__":
    # Run validation test first
    if test_small_matrix():
        # Run main test
        test_agent_grid_search()
    else:
        print("Validation failed, skipping main test")