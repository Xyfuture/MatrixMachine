from matrixmachine.description import (
    MatrixShape,
    ComputeDieSpec,
    ChipSpec,
    Chip,
    Mapping,
)
from matrixmachine.strategy.trivial import TrivialTilingStrategy
from matrixmachine.sim_engine import simulate


def sim_test():
    # Create a matrix shape (4096x4096) with batch_size=1 (default for GEMV)
    matrix_shape = MatrixShape(rows=4096, cols=4096, batch_size=1)
    print(f"Matrix shape: {matrix_shape.rows}x{matrix_shape.cols}x{matrix_shape.batch_size}")
    print(f"Matrix area: {matrix_shape.area()}")
    print(f"Matrix volume: {matrix_shape.volume()}")

    # Create compute dies specification
    die_spec = ComputeDieSpec(
        compute_power=128,  # 128 TFLOPS
        input_bandwidth=128,  # 128 GB/s
        output_bandwidth=128,  # 128 GB/s
        memory_bandwidth=1.0,  # 1.0 TB/s
    )

    # Create a chip with 8 compute dies
    chip_spec = ChipSpec(die_count=8, die_spec=die_spec)
    chip = Chip.create_from_spec(chip_spec)

    print(f"Chip configuration:")
    print(f"  Number of dies: {len(chip.compute_dies)}")
    print(f"  Total compute power: {chip.total_compute_power} TFLOPS")
    print(f"  Total bandwidth: {chip.total_bandwidth} GB/s")
    print()

    # Create trivial tiling strategy
    strategy = TrivialTilingStrategy()

    # Create mapping using balanced strategy
    mapping = strategy.create_balanced_mapping(matrix_shape, chip)

    print("Tile distribution:")
    for die_id, tiles in mapping.placement.items():
        total_area = sum(tile.area for tile in tiles)
        print(f"  {die_id}: {len(tiles)} tiles, total area: {total_area}")

    print()

    # Validation
    print("Validation Summary:")
    print(f"Number of tiles: {len(mapping.tiles)}")
    print(f"Validation passed: {mapping.check_bidirectional_mapping()}")

    # Run simulation using the new simulate function
    print("Starting simulation...")
    running_cycles = simulate(chip, mapping, save_trace=True)
    
    print(f"Simulation completed! Total running cycles: {running_cycles}")


if __name__ == "__main__":
    sim_test()