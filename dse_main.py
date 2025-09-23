from matrixmachine.description import MatrixShape, ComputeDieSpec, ChipSpec, Chip
from matrixmachine.strategy.grid import GridTilingStrategy
from matrixmachine.sim_engine import SimChip
from matrixmachine.utils import MappingResult, calculate_compute_utilization


def main() -> None:
    # Keep matrix and hardware config identical to main.py
    matrix_shape = MatrixShape(rows=4096*2, cols=4096*2, batch_size=1)

    die_spec = ComputeDieSpec(
        compute_power=128,  # 128 TFLOPS
        input_bandwidth=128,  # 128 GB/s
        output_bandwidth=128,  # 128 GB/s
    )

    chip_spec = ChipSpec(die_count=8, die_spec=die_spec)
    chip = Chip.create_from_spec(chip_spec)

    print("Matrix:", matrix_shape.rows, "x", matrix_shape.cols, "x", matrix_shape.batch_size)
    print("Chip:")
    print("  dies:", len(chip.compute_dies))
    print("  total compute:", chip.total_compute_power, "TFLOPS")
    print("  total bandwidth:", chip.total_bandwidth, "GB/s")
    print()

    # Provide a compact grid pool to bound search time on large matrices
    # Now grid_pool specifies (num_rows_blocks, num_cols_blocks) instead of tile sizes
    grid_pool = [
        (8, 8),    # 8x8 blocks = 512x512 tiles for 4096x4096 matrix
        (16, 8),   # 16x8 blocks = 256x512 tiles
        (8, 16),   # 8x16 blocks = 512x256 tiles
        (16, 16),  # 16x16 blocks = 256x256 tiles
        (32, 16),  # 32x16 blocks = 128x256 tiles
        (16, 32),  # 16x32 blocks = 256x128 tiles
        (32, 32),  # 32x32 blocks = 128x128 tiles
        (4, 4),    # 4x4 blocks = 1024x1024 tiles
        (2, 2),    # 2x2 blocks = 2048x2048 tiles
        (1, 1),    # 1x1 block = full matrix
        (1,8),
        (2,8),
        (4,8),
        (2,4),
    ]

    strategy = GridTilingStrategy(stage2_max_per_rect=2)
    best_mapping, best_cycles, stats = strategy.dse_best_mapping(
        matrix_shape, chip, grid_pool=grid_pool
    )

    print("DSE finished:")
    print("  candidates:", stats.get("candidates", 0))
    print("  best cycles:", best_cycles)
    print("  tiles:", len(best_mapping.tiles))
    print("  valid bimap:", best_mapping.check_bidirectional_mapping())

    # Calculate and display compute utilization
    mapping_result = MappingResult(mapping=best_mapping, latency=float(best_cycles))
    utilization = calculate_compute_utilization(mapping_result)
    print(f"  compute utilization: {utilization:.4f} ({utilization*100:.2f}%)")
    print()

    print("Tile distribution (per die):")
    for die_id, tiles in best_mapping.placement.items():
        area = sum(t.area for t in tiles)
        print(f"  {die_id}: {len(tiles)} tiles, area={area}")
        # Print each tile with coordinates and size
        for t in tiles:
            print(
                f"    {t.tile_id}: [r:{t.row0}:{t.row1}, c:{t.col0}:{t.col1}] "
                f"size={t.rows}x{t.cols} area={t.area}"
            )

        # Also print a compact summary by tile size
        size_hist = {}
        for t in tiles:
            key = (t.rows, t.cols)
            size_hist[key] = size_hist.get(key, 0) + 1
        print("    size histogram:")
        for (rh, cw), cnt in sorted(size_hist.items()):
            print(f"      {rh}x{cw}: {cnt}")

    # Save Perfetto trace for the best candidate
    trace_path = "trace.json"
    sim_chip = SimChip(chip)
    sim_chip.set_mapping(best_mapping)
    sim_chip.run_sim()
    sim_chip.save_trace_file(trace_path)
    print(f"Saved trace to {trace_path}")


if __name__ == "__main__":
    main()
