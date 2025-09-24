from matrixmachine.core.description import (
    MatrixShape,
    ComputeDieSpec,
    ChipSpec,
    Chip,
    Mapping,
    TileAssignmentInput,
)
from matrixmachine.core.sim_engine import simulate
from matrixmachine.core.utils import MappingResult, calculate_compute_utilization
from typing import List


def create_manual_mappings():
    """
    Create two manual mappings for a 4096x4096 matrix with batch_size=64:
    1. First mapping: 1024x1024 tiles with batch_size=64 (16 tiles total)
    2. Second mapping: 1024x1024 tiles with batch_size=32 (32 tiles total)
    Both mappings distribute tiles evenly across 8 compute dies.
    """

    # Create matrix shape: 4096x4096 with batch_size=48
    matrix_shape = MatrixShape(rows=4096, cols=4096, batch_size=48)
    print(f"Matrix shape: {matrix_shape.rows}x{matrix_shape.cols}x{matrix_shape.batch_size}")
    print(f"Matrix volume: {matrix_shape.volume()}")
    print()

    # Create compute die specification (128GB/s I/O, 4TB/s memory, 128TOPS)
    die_spec = ComputeDieSpec(
        compute_power=128,  # 128 TOPS
        input_bandwidth=128,  # 128 GB/s
        output_bandwidth=128,  # 128 GB/s
        memory_bandwidth=4,  # 4 TB/s = 4000 GB/s
    )

    # Create chip with 8 compute dies
    chip_spec = ChipSpec(die_count=8, die_spec=die_spec)
    chip = Chip.create_from_spec(chip_spec)

    print(f"Chip configuration:")
    print(f"  Number of dies: {len(chip.compute_dies)}")
    print(f"  Total compute power: {chip.total_compute_power} TOPS")
    print(f"  Total I/O bandwidth: {chip.total_input_bandwidth + chip.total_output_bandwidth} GB/s")
    print(f"  Total memory bandwidth: {chip.total_memory_bandwidth} GB/s")
    print()

    # Create first mapping: 1024x1024 tiles with batch_size=48
    mapping1 = create_mapping_batch_48(matrix_shape, chip)

    # Create second mapping: 1024x1024 tiles with batch_size=24
    mapping2 = create_mapping_batch_24(matrix_shape, chip)

    # Print statistics for both mappings
    print_mapping_statistics("First Mapping (batch_size=48)", mapping1)
    print_mapping_statistics("Second Mapping (batch_size=24)", mapping2)

    return mapping1, mapping2


def create_mapping_batch_48(matrix_shape: MatrixShape, chip: Chip) -> Mapping:
    """
    Create mapping with 1024x1024 tiles and batch_size=48.
    This creates 4x4x1 = 16 tiles total (4 row tiles, 4 col tiles, 1 batch tile).
    """
    tile_data: List[TileAssignmentInput] = []
    die_ids = list(chip.compute_dies.keys())
    die_count = len(die_ids)

    tile_idx = 0

    # Create 4x4 grid of 1024x1024 tiles with full batch dimension
    for row_idx in range(4):  # 4096 / 1024 = 4
        for col_idx in range(4):  # 4096 / 1024 = 4
            row0 = row_idx * 1024
            row1 = (row_idx + 1) * 1024
            col0 = col_idx * 1024
            col1 = (col_idx + 1) * 1024
            batch0 = 0
            batch1 = 48  # Full batch dimension

            # Assign to die in round-robin fashion
            die_id = die_ids[tile_idx % die_count]

            tile_data.append((die_id, row0, row1, col0, col1, batch0, batch1))
            tile_idx += 1

    return Mapping.from_tile_data(matrix_shape, chip, tile_data)


def create_mapping_batch_24(matrix_shape: MatrixShape, chip: Chip) -> Mapping:
    """
    Create mapping with 1024x1024 tiles and batch_size=24.
    This creates 4x4x2 = 32 tiles total (4 row tiles, 4 col tiles, 2 batch tiles).
    """
    tile_data: List[TileAssignmentInput] = []
    die_ids = list(chip.compute_dies.keys())
    die_count = len(die_ids)

    tile_idx = 0

    # Create 4x4x2 grid of 1024x1024 tiles with half batch dimension
    for row_idx in range(4):  # 4096 / 1024 = 4
        for col_idx in range(4):  # 4096 / 1024 = 4
            for batch_idx in range(2):  # 48 / 24 = 2
                row0 = row_idx * 1024
                row1 = (row_idx + 1) * 1024
                col0 = col_idx * 1024
                col1 = (col_idx + 1) * 1024
                batch0 = batch_idx * 24
                batch1 = (batch_idx + 1) * 24

                # Assign to die in round-robin fashion
                die_id = die_ids[tile_idx % die_count]

                tile_data.append((die_id, row0, row1, col0, col1, batch0, batch1))
                tile_idx += 1

    return Mapping.from_tile_data(matrix_shape, chip, tile_data)


def print_mapping_statistics(title: str, mapping: Mapping):
    """Print detailed statistics for a mapping."""
    print(f"=== {title} ===")
    print(f"Total tiles: {len(mapping.tiles)}")
    print(f"Matrix volume: {mapping.matrix.volume()}")
    print(f"Validation passed: {mapping.check_bidirectional_mapping()}")

    # Print die loads and volumes
    die_loads = mapping.die_loads()
    die_volumes = mapping.die_volumes()

    print("Die distribution:")
    for die_id in sorted(die_loads.keys()):
        tiles_count = die_loads[die_id]
        volume = die_volumes[die_id]
        print(f"  {die_id}: {tiles_count} tiles, volume: {volume}")

        # Print details of tiles on this die
        tiles = mapping.tiles_of_die(die_id)
        for tile in tiles:
            print(f"    {tile.tile_id}: [{tile.row0}:{tile.row1}, {tile.col0}:{tile.col1}, {tile.batch0}:{tile.batch1}] volume={tile.volume}")

    print()


def run_simulations():
    """Run simulations for both mappings and save trace files."""
    mapping1, mapping2 = create_manual_mappings()

    # Additional validation
    try:
        mapping1.check_all()
        print("✓ First mapping validation passed")
    except ValueError as e:
        print(f"✗ First mapping validation failed: {e}")
        return

    try:
        mapping2.check_all()
        print("✓ Second mapping validation passed")
    except ValueError as e:
        print(f"✗ Second mapping validation failed: {e}")
        return

    print("=" * 60)
    print("RUNNING SIMULATIONS")
    print("=" * 60)

    # Run simulation for first mapping (batch_size=48)
    print("Running simulation for first mapping (batch_size=48)...")
    cycles1 = simulate(mapping1.chip, mapping1, save_trace=False)
    print(f"✓ First mapping simulation completed! Total cycles: {cycles1}")

    # Save trace for first mapping
    from matrixmachine.core.sim_engine import SimChip
    sim_chip1 = SimChip(mapping1.chip)
    sim_chip1.set_mapping(mapping1)
    sim_chip1.run_sim()
    sim_chip1.save_trace_file("trace_batch48.json")
    print(f"  Trace saved to: trace_batch48.json")
    print()

    # Run simulation for second mapping (batch_size=24)
    print("Running simulation for second mapping (batch_size=24)...")
    cycles2 = simulate(mapping2.chip, mapping2, save_trace=False)
    print(f"✓ Second mapping simulation completed! Total cycles: {cycles2}")

    # Save trace for second mapping
    sim_chip2 = SimChip(mapping2.chip)
    sim_chip2.set_mapping(mapping2)
    sim_chip2.run_sim()
    sim_chip2.save_trace_file("trace_batch24.json")
    print(f"  Trace saved to: trace_batch24.json")
    print()

    # Compare results
    print("=" * 60)
    print("SIMULATION COMPARISON")
    print("=" * 60)
    print(f"First mapping (batch_size=48):  {cycles1} cycles")
    print(f"Second mapping (batch_size=24): {cycles2} cycles")
    print(f"Difference: {abs(cycles1 - cycles2)} cycles")

    if cycles1 < cycles2:
        print(f"First mapping is faster by {cycles2 - cycles1} cycles ({((cycles2 - cycles1) / cycles2 * 100):.2f}%)")
    elif cycles2 < cycles1:
        print(f"Second mapping is faster by {cycles1 - cycles2} cycles ({((cycles1 - cycles2) / cycles1 * 100):.2f}%)")
    else:
        print("Both mappings have the same performance")

    print()

    # Print utilization analysis
    print("=" * 60)
    print("UTILIZATION ANALYSIS")
    print("=" * 60)
    print_utilization_analysis("First Mapping", mapping1, cycles1)
    print_utilization_analysis("Second Mapping", mapping2, cycles2)

    print("Trace files can be viewed at: https://ui.perfetto.dev/")


def print_utilization_analysis(title: str, mapping: Mapping, total_cycles: int):
    """Print utilization analysis using utils.MappingResult."""
    mapping_result = MappingResult(mapping=mapping, latency=total_cycles)

    print(f"\n=== {title} ===")
    print(f"Total simulation time: {total_cycles} cycles")
    print(f"Matrix operation count: {mapping_result.get_matrix_operation_count():,}")
    print(f"Total compute power: {mapping_result.get_chip_total_compute_power_gops()} TOPS")
    print(f"Compute utilization: {mapping_result.get_compute_utilization()*100:.1f}%")

    # Basic load balancing info
    die_loads = mapping.die_loads()
    die_volumes = mapping.die_volumes()

    print(f"Load distribution:")
    for die_id in sorted(die_loads.keys()):
        tiles_count = die_loads[die_id]
        volume = die_volumes[die_id]
        print(f"  {die_id}: {tiles_count} tiles, volume: {volume:,}")

    # Load balancing metric
    volumes = list(die_volumes.values())
    if volumes:
        min_vol = min(volumes)
        max_vol = max(volumes)
        load_balance = (min_vol / max_vol * 100) if max_vol > 0 else 100
        print(f"Load balancing: {load_balance:.1f}% (volume-based)")


def main():
    """Run the manual mapping example with simulations."""
    run_simulations()


if __name__ == "__main__":
    main()