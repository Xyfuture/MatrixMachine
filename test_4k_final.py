#!/usr/bin/env python3
"""
4K Matrix Test with Agent Grid Search
"""

from matrixmachine.core.description import (
    MatrixShape,
    ComputeDieSpec,
    ChipSpec,
    Chip,
)
from matrixmachine.strategy.agent_grid_search import AgentGridSearchStrategy
from matrixmachine.core.utils import calculate_compute_utilization


def test_4k_matrix_optimized():
    """Test 4K matrix with minimal search space."""
    print("4K Matrix Agent Grid Search Test")
    print("="*35)

    # 4K x 4K matrix
    matrix = MatrixShape(rows=4096, cols=4096, batch_size=1)
    print(f"Matrix: {matrix.rows}x{matrix.cols} ({matrix.area():,} elements)")

    # 8 compute dies with realistic specs
    die_spec = ComputeDieSpec(
        compute_power=128,  # 128 TFLOPS
        input_bandwidth=128,  # 128 GB/s
        output_bandwidth=128,  # 128 GB/s
        memory_bandwidth=128,  # 1.0 TB/s
    )
    chip_spec = ChipSpec(die_count=8, die_spec=die_spec)
    chip = Chip.create_from_spec(chip_spec)

    print(f"Hardware: {len(chip.compute_dies)} dies, {chip.total_compute_power} TFLOPS total")

    # Test only one configuration to avoid excessive search time
    strategy = AgentGridSearchStrategy(
        num_split_row_candidates=[2,3,4,6],  # Only 4x4 split
        num_split_col_candidates=[2,3,4,6],
        max_iterations= 2 , 
    )

    print("Testing 4x4 split configuration...")

    result = strategy.find_optimal_mapping(matrix, chip)

    if result:
        print("\n✅ 4K Matrix Mapping Found!")
        print(f"Execution latency: {result.latency:,} cycles")

        # Performance metrics
        utilization = calculate_compute_utilization(result)
        print(f"Compute utilization: {utilization:.2%}")
        print(f"Matrix operations: {result.get_matrix_operation_count():,}")
        print(f"Total compute power: {result.get_chip_total_compute_power_gops():,} GOPS")

        # Workload distribution analysis
        print("\n📊 Workload Distribution:")
        die_volumes = result.mapping.die_volumes()
        die_loads = result.mapping.die_loads()

        for die_id in sorted(die_loads.keys()):
            print(f"  {die_id}: {die_loads[die_id]} tiles, {die_volumes[die_id]:,} volume")

        # Detailed tile shapes for each die
        print("\n🔍 Detailed Tile Shapes:")
        for die_id in sorted(result.mapping.placement.keys()):
            tiles = result.mapping.placement[die_id]
            print(f"  {die_id} ({len(tiles)} tiles):")
            for i, tile in enumerate(tiles):
                rows = tile.row1 - tile.row0
                cols = tile.col1 - tile.col0
                batches = tile.batch1 - tile.batch0
                print(f"    Tile {i}: [{tile.row0}:{tile.row1}, {tile.col0}:{tile.col1}, {tile.batch0}:{tile.batch1}] -> {rows}×{cols}×{batches}")
            print()

        # Balance analysis
        volumes = list(die_volumes.values())
        if volumes:
            max_vol = max(volumes)
            min_vol = min(volumes)
            balance = min_vol / max_vol if max_vol > 0 else 0
            imbalance = (1 - balance) * 100

            print(f"\n⚖️  Load Balance:")
            print(f"  Perfect balance ratio: {balance:.3f}")
            print(f"  Load imbalance: {imbalance:.1f}%")

        print(f"\n🔍 Algorithm Performance:")
        print(f"  Configurations tested: 1")
        print(f"  Cache entries: {len(strategy.memo)}")

        return True
    else:
        print("❌ No mapping found for 4K matrix")
        return False


if __name__ == "__main__":
    test_4k_matrix_optimized()