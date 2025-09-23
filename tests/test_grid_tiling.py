#!/usr/bin/env python3

"""
Test example demonstrating the grid-based tiling strategy.
"""

import sys
import os
# Add parent directory to path to import matrixmachine package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matrixmachine.description import MatrixShape, ComputeDieSpec, ComputeDie, Chip, ChipSpec
from matrixmachine.strategy.trivial import TrivialTilingStrategy


def main():
    """Demonstrate the grid-based tiling strategy."""

    # Create a test matrix (10x8)
    matrix_shape = MatrixShape(rows=10, cols=8)
    print(f"Matrix shape: {matrix_shape.rows}x{matrix_shape.cols}")
    print(f"Matrix area: {matrix_shape.area()}")
    print()

    # Create compute dies specification
    die_spec = ComputeDieSpec(
        compute_power=1.0,  # 1 TFLOPS
        input_bandwidth=10.0,  # 10 GB/s
        output_bandwidth=10.0,  # 10 GB/s
    )

    # Create a chip with 3 compute dies
    chip_spec = ChipSpec(die_count=3, die_spec=die_spec)
    chip = Chip(spec=chip_spec)

    # Add compute dies to chip
    for i in range(3):
        die = ComputeDie(
            die_id=f"die_{i}",
            spec=die_spec,
            meta={"frequency": "1GHz", "topology": "mesh"},
        )
        chip.add_die(die)

    print(f"Chip configuration:")
    print(f"  Number of dies: {len(chip.compute_dies)}")
    print(f"  Total compute power: {chip.total_compute_power} TFLOPS")
    print(f"  Total bandwidth: {chip.total_bandwidth} GB/s")
    print()

    # Create grid tiling strategy
    strategy = TrivialTilingStrategy()

    # Example 1: Fixed grid mapping (3x2 grid)
    print("=== Example 1: Fixed 3x2 Grid ===")
    mapping1 = strategy.create_mapping(matrix_shape, chip, grid_rows=3, grid_cols=2)

    print("Tile distribution:")
    for die_id, tiles in mapping1.placement.items():
        total_area = sum(tile.area for tile in tiles)
        print(f"  {die_id}: {len(tiles)} tiles, total area: {total_area}")
        for tile in tiles:
            print(f"    {tile.tile_id}: [{tile.row0}:{tile.row1}, {tile.col0}:{tile.col1}]")
    print()

    # Example 2: Balanced mapping (automatic grid calculation)
    print("=== Example 2: Balanced Mapping (Automatic Grid) ===")
    mapping2 = strategy.create_balanced_mapping(matrix_shape, chip)

    print("Tile distribution:")
    for die_id, tiles in mapping2.placement.items():
        total_area = sum(tile.area for tile in tiles)
        print(f"  {die_id}: {len(tiles)} tiles, total area: {total_area}")
        for tile in tiles:
            print(f"    {tile.tile_id}: [{tile.row0}:{tile.row1}, {tile.col0}:{tile.col1}]")
    print()

    # Example 3: Balanced mapping with max tile area constraint
    print("=== Example 3: Balanced Mapping with Max Tile Area = 15 ===")
    mapping3 = strategy.create_balanced_mapping(matrix_shape, chip, max_tile_area=15)

    print("Tile distribution:")
    for die_id, tiles in mapping3.placement.items():
        total_area = sum(tile.area for tile in tiles)
        print(f"  {die_id}: {len(tiles)} tiles, total area: {total_area}")
        for tile in tiles:
            print(f"    {tile.tile_id}: [{tile.row0}:{tile.row1}, {tile.col0}:{tile.col1}]")
    print()

    # Validation summary
    print("=== Validation Summary ===")
    print(f"Example 1 - Number of tiles: {len(mapping1.tiles)}")
    print(f"Example 1 - Validation passed: {mapping1.check_bidirectional_mapping()}")
    print(f"Example 2 - Number of tiles: {len(mapping2.tiles)}")
    print(f"Example 2 - Validation passed: {mapping2.check_bidirectional_mapping()}")
    print(f"Example 3 - Number of tiles: {len(mapping3.tiles)}")
    print(f"Example 3 - Validation passed: {mapping3.check_bidirectional_mapping()}")


if __name__ == "__main__":
    main()
