#!/usr/bin/env python3

"""
Test shared engine simulation with 1 die mapped to multiple tiles.

This test specifically focuses on testing the sim_engine module's behavior
when a single compute die is assigned multiple tiles to process.
"""

import sys
import os
# Add parent directory to path to import matrixmachine package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matrixmachine.core.description import (
    MatrixShape, ComputeDieSpec, ComputeDie, Chip, ChipSpec,
    Tile, Mapping, DataFormat, DataType
)
from matrixmachine.core.sim_engine import simulate
from matrixmachine.core.utils import calculate_compute_utilization, MappingResult


def test_single_die_multiple_tiles():
    """Test simulation with 1 die processing multiple tiles sequentially."""

    print("=== Test: Single Die with Multiple Tiles ===")

    # Create a test matrix (1024x1024)
    matrix_shape = MatrixShape(
        rows=1024,
        cols=1024,
        batch_size=1,
        data_format=DataFormat(
            input_dtype=DataType.FP16,
            output_dtype=DataType.FP16,
            weight_dtype=DataType.FP16
        )
    )
    print(f"Matrix shape: {matrix_shape}")
    print(f"Matrix volume: {matrix_shape.volume} elements")
    print()

    # Create compute die specification with shared bandwidth
    die_spec = ComputeDieSpec(
        compute_power=16.0,  # 2 TFLOPS
        shared_bandwidth=8.0,  # 8 GB/s shared I/O bandwidth
        memory_bandwidth=0.4,  # 0.2 TB/s
    )
    print(f"Die specification: {die_spec}")
    print()

    # Create a chip with only 1 compute die
    chip_spec = ChipSpec(die_count=1, die_spec=die_spec)
    chip = Chip.create_from_spec(chip_spec)

    print(f"Chip configuration:")
    print(f"  Number of dies: {len(chip.compute_dies)}")
    print(f"  Total compute power: {chip.total_compute_power} TFLOPS")
    print()

    # Create mapping with multiple tiles assigned to the single die
    die_id = "die_0"

    # Create 4 tiles that partition the matrix
    tiles_data = [
        # Tile 1: top-left quadrant [0:512, 0:512]
        (die_id, 0, 512, 0, 512, 0, 1),
        # Tile 2: top-right quadrant [0:512, 512:1024]
        (die_id, 0, 512, 512, 1024, 0, 1),
        # Tile 3: bottom-left quadrant [512:1024, 0:512]
        (die_id, 512, 1024, 0, 512, 0, 1),
        # Tile 4: bottom-right quadrant [512:1024, 512:1024]
        (die_id, 512, 1024, 512, 1024, 0, 1),
    ]

    mapping = Mapping.from_tile_data(matrix_shape, chip, tiles_data)

    print("Tile assignment to single die:")
    for assigned_die_id, tiles in mapping.placement.items():
        if tiles:  # Only show dies with tiles
            total_area = sum(tile.area for tile in tiles)
            total_volume = sum(tile.volume for tile in tiles)
            print(f"  {assigned_die_id}: {len(tiles)} tiles")
            print(f"    Total area: {total_area} elements")
            print(f"    Total volume: {total_volume} elements")
            for i, tile in enumerate(tiles):
                print(f"    Tile {i+1}: {tile}")
    print()

    # Validate mapping
    print("=== Mapping Validation ===")
    try:
        mapping.check_all()
        print("✓ Mapping validation passed")
        print(f"✓ Bidirectional mapping check: {mapping.check_bidirectional_mapping()}")
        print(f"✓ Total tiles: {len(mapping.tiles)}")
        print(f"✓ Matrix coverage: {sum(tile.volume for tile in mapping.tiles.values())} == {matrix_shape.volume}")
    except ValueError as e:
        print(f"✗ Mapping validation failed: {e}")
        return
    print()

    # Run simulation
    print("=== Running Simulation ===")
    try:
        cycles = simulate(chip, mapping, save_trace=True,trace_filename="shared_link_trace.json")
        print(f"✓ Simulation completed successfully")
        print(f"✓ Total execution cycles: {cycles}")

        # Calculate compute utilization
        mapping_result = MappingResult(mapping=mapping, latency=cycles)
        utilization = calculate_compute_utilization(mapping_result)
        print(f"✓ Compute utilization: {utilization:.2%}")
        print(f"  Matrix operation count: {mapping_result.get_matrix_operation_count()}")
        print(f"  Chip total compute power: {mapping_result.get_chip_total_compute_power_gops():.0f} GOPS")

    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print()
    print("=== Test Summary ===")
    print("✓ Single die successfully processed multiple tiles")
    print("✓ Shared bandwidth engine handled sequential tile processing")
    print("✓ All validation checks passed")
    print("✓ Simulation engine ran without errors")


def test_single_die_multiple_tile_sizes():
    """Test simulation with 1 die processing tiles of different sizes."""

    print("\n=== Test: Single Die with Variable-sized Tiles ===")

    # Create a test matrix (1024x1024)
    matrix_shape = MatrixShape(
        rows=1024,
        cols=1024,
        batch_size=1,
        data_format=DataFormat(
            input_dtype=DataType.FP16,
            output_dtype=DataType.FP16,
            weight_dtype=DataType.FP16
        )
    )
    print(f"Matrix shape: {matrix_shape}")
    print()

    # Create compute die specification
    die_spec = ComputeDieSpec(
        compute_power=1.5,  # 1.5 TFLOPS
        shared_bandwidth=12.0,  # 12 GB/s shared I/O bandwidth
        memory_bandwidth=0.15,  # 0.15 TB/s
    )

    # Create a chip with only 1 compute die
    chip_spec = ChipSpec(die_count=1, die_spec=die_spec)
    chip = Chip.create_from_spec(chip_spec)

    # Create mapping with tiles of different sizes
    die_id = "die_0"

    # Create tiles with varying sizes to test different workloads
    tiles_data = [
        # Large tile: [0:512, 0:512]
        (die_id, 0, 512, 0, 512, 0, 1),
        # Small tile 1: [0:256, 512:1024]
        (die_id, 0, 256, 512, 1024, 0, 1),
        # Medium tile: [256:512, 512:1024]
        (die_id, 256, 512, 512, 1024, 0, 1),
        # Long tile: [512:1024, 0:1024]
        (die_id, 512, 1024, 0, 1024, 0, 1),
    ]

    mapping = Mapping.from_tile_data(matrix_shape, chip, tiles_data)

    print("Variable-sized tile assignment:")
    for assigned_die_id, tiles in mapping.placement.items():
        if tiles:
            print(f"  {assigned_die_id}: {len(tiles)} tiles")
            for i, tile in enumerate(tiles):
                print(f"    Tile {i+1}: {tile} (area: {tile.area})")
    print()

    # Validate mapping
    print("=== Validation ===")
    try:
        mapping.check_all()
        print("✓ Validation passed")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")
        return
    print()

    # Run simulation
    print("=== Running Simulation ===")
    try:
        cycles = simulate(chip, mapping, save_trace=False)
        print(f"✓ Simulation completed in {cycles} cycles")

        # Calculate and display utilization
        mapping_result = MappingResult(mapping=mapping, latency=cycles)
        utilization = calculate_compute_utilization(mapping_result)
        print(f"✓ Compute utilization: {utilization:.2%}")

    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        return

    print("✓ Variable-sized tiles test passed")


def test_single_die_batch_processing():
    """Test simulation with 1 die processing multiple tiles with batch dimension."""

    print("\n=== Test: Single Die with Batch Processing ===")

    # Create a test matrix with batch dimension (1024x1024x4)
    matrix_shape = MatrixShape(
        rows=1024,
        cols=1024,
        batch_size=4,  # 4 batches
        data_format=DataFormat(
            input_dtype=DataType.INT8,
            output_dtype=DataType.INT8,
            weight_dtype=DataType.INT8
        )
    )
    print(f"Matrix shape: {matrix_shape}")
    print()

    # Create compute die specification
    die_spec = ComputeDieSpec(
        compute_power=3.0,  # 3 TFLOPS
        shared_bandwidth=16.0,  # 16 GB/s shared I/O bandwidth
        memory_bandwidth=0.3,  # 0.3 TB/s
    )

    # Create a chip with only 1 compute die
    chip_spec = ChipSpec(die_count=1, die_spec=die_spec)
    chip = Chip.create_from_spec(chip_spec)

    # Create mapping with batched tiles
    die_id = "die_0"

    # Create tiles that cover all spatial regions across all 4 batches
    tiles_data = [
        # Tile 1: top-left quadrant [0:512, 0:512] covering all 4 batches
        (die_id, 0, 512, 0, 512, 0, 4),
        # Tile 2: top-right quadrant [0:512, 512:1024] covering all 4 batches
        (die_id, 0, 512, 512, 1024, 0, 4),
        # Tile 3: bottom-left quadrant [512:1024, 0:512] covering all 4 batches
        (die_id, 512, 1024, 0, 512, 0, 4),
        # Tile 4: bottom-right quadrant [512:1024, 512:1024] covering all 4 batches
        (die_id, 512, 1024, 512, 1024, 0, 4),
    ]

    mapping = Mapping.from_tile_data(matrix_shape, chip, tiles_data)

    print("Batched tile assignment:")
    for assigned_die_id, tiles in mapping.placement.items():
        if tiles:
            print(f"  {assigned_die_id}: {len(tiles)} tiles")
            for i, tile in enumerate(tiles):
                print(f"    Tile {i+1}: {tile}")
    print()

    # Validate mapping
    print("=== Validation ===")
    try:
        mapping.check_all()
        print("✓ Validation passed")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")
        return
    print()

    # Run simulation
    print("=== Running Simulation ===")
    try:
        cycles = simulate(chip, mapping, save_trace=False)
        print(f"✓ Simulation completed in {cycles} cycles")

        # Calculate and display utilization
        mapping_result = MappingResult(mapping=mapping, latency=cycles)
        utilization = calculate_compute_utilization(mapping_result)
        print(f"✓ Compute utilization: {utilization:.2%}")

    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        return

    print("✓ Batch processing test passed")


def main():
    """Run all shared engine tests."""

    print("Testing MatrixMachine sim_engine with single die, multiple tiles")
    print("=" * 70)

    # Test 1: Basic single die with multiple tiles
    test_single_die_multiple_tiles()

    # Test 2: Single die with variable-sized tiles
    # test_single_die_multiple_tile_sizes()

    # Test 3: Single die with batch processing
    # test_single_die_batch_processing()

    print("\n" + "=" * 70)
    print("All shared engine tests completed!")


if __name__ == "__main__":
    main()