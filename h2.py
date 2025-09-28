#!/usr/bin/env python3
"""Test script for MatrixMachine simulation with 4 ComputeDie configuration."""

from matrixmachine.core.description import ComputeDieSpec, ChipSpec, Chip, MatrixShape
from matrixmachine.core.sim_engine import simulate
from matrixmachine.core.utils import MappingResult, calculate_compute_utilization
from matrixmachine.strategy.trivial import TrivialTilingStrategy


def main():
    # Create ComputeDie specifications
    # Each die: compute_power=1.2T FLOPS, memory_bandwidth=400GB/s, input/output_bandwidth=6GB/s
    die_spec = ComputeDieSpec(
        compute_power=1.2,  # TFLOPS
        input_bandwidth=3,  # GB/s
        output_bandwidth=3,  # GB/s
        memory_bandwidth=0.4  # TB/s (400 GB/s = 0.4 TB/s)
    )

    # Create chip with 4 compute dies
    chip_spec = ChipSpec(die_count=4, die_spec=die_spec)
    chip = Chip.create_from_spec(chip_spec)

    print("=== Hardware Configuration ===")
    print(f"Number of compute dies: {len(chip.compute_dies)}")
    print(f"Per die - Compute power: {die_spec.compute_power} TFLOPS")
    print(f"Per die - Memory bandwidth: {die_spec.memory_bandwidth} TB/s")
    print(f"Per die - Input bandwidth: {die_spec.input_bandwidth} GB/s")
    print(f"Per die - Output bandwidth: {die_spec.output_bandwidth} GB/s")
    print(f"Total chip compute power: {chip.total_compute_power} TFLOPS")
    print()

    # Define matrix shape: 4k x 4k with batch size 8
    matrix_shape = MatrixShape(rows=2048, cols=2048, batch_size=8)

    print("=== Matrix Configuration ===")
    print(f"Matrix dimensions: {matrix_shape.rows} x {matrix_shape.cols}")
    print(f"Batch size: {matrix_shape.batch_size}")
    print(f"Total matrix volume: {matrix_shape.volume()} elements")
    print()

    # Create tiling strategy and mapping with 2x2 grid
    strategy = TrivialTilingStrategy()
    mapping = strategy.create_mapping(
        matrix_shape=matrix_shape,
        chip=chip,
        grid_rows=4,
        grid_cols=4,
        batch_splits=2
    )

    print("=== Tiling Configuration ===")
    print(f"Grid layout: 2x2 tiles")
    print(f"Total tiles: {len(mapping.tiles)}")

    # Display tile assignments
    print("\nTile assignments:")
    for die_id, tiles in mapping.placement.items():
        print(f"  {die_id}: {len(tiles)} tiles")
        for tile in tiles:
            print(f"    {tile.tile_id}: [{tile.row0}:{tile.row1}, {tile.col0}:{tile.col1}, {tile.batch0}:{tile.batch1}]")
            print(f"      Shape: {tile.rows} x {tile.cols} x {tile.batches}, Volume: {tile.volume}")
    print()

    # Run simulation
    print("=== Running Simulation ===")
    latency_cycles = simulate(chip, mapping, save_trace=True)

    print(f"Simulation completed in {latency_cycles} cycles")
    print()

    # Calculate and display utilization metrics
    mapping_result = MappingResult(mapping=mapping, latency=latency_cycles)
    compute_utilization = calculate_compute_utilization(mapping_result)

    print("=== Performance Results ===")
    print(f"Total simulation latency: {latency_cycles} cycles")
    print(f"Matrix operation count: {mapping_result.get_matrix_operation_count()} operations")
    print(f"Total chip compute power: {mapping_result.get_chip_total_compute_power_gops()} GOPS")
    print(f"Total available compute: {mapping_result.get_chip_total_compute_power_gops() * latency_cycles} GOPS*cycles")
    print(f"Compute utilization: {compute_utilization:.4f} ({compute_utilization * 100:.2f}%)")
    print()

    # Additional analysis per die
    print("=== Per-Die Analysis ===")
    die_volumes = mapping.die_volumes()
    for die_id, volume in die_volumes.items():
        utilization_per_die = volume / (die_spec.compute_power * 1000 * latency_cycles) if latency_cycles > 0 else 0
        print(f"{die_id}: {volume} operations, utilization: {utilization_per_die:.4f} ({utilization_per_die * 100:.2f}%)")

    print(f"\nTrace file saved as: main_trace.json")
    print("View the trace at: ui.perfetto.dev")


if __name__ == "__main__":
    main()